import torch
import torch.nn.functional as F
import numpy as np
import copy
import os
import wandb

from tqdm import tqdm
import pickle
from utils.metrics import get_statistics
from utils.metrics import PA
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score

class Trainer:
    def __init__(self, args, logger, train_loader, test_loader):
        self.args = args
        self.logger = logger
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.threshold_function_map = {
            "oracle": self.oracle_thresholding,
        }

    def train(self, dataset, dataloader):
        pass

    @torch.no_grad()
    def infer(self):
        result = {}
        self.model.eval()
        gt = self.test_loader.dataset.y
        anomaly_scores = self.calculate_anomaly_scores()

        # thresholding
        threshold = self.get_threshold(gt=gt, anomaly_scores=anomaly_scores)
        result.update({"Threshold": threshold})

        # AUROC
        s = anomaly_scores - threshold
        logit = 1/(1+np.exp(-s)) # (N, )
        pred_prob = np.zeros((len(logit), 2))
        pred_prob[:, 0], pred_prob[:, 1] = 1-logit, logit
        wandb.sklearn.plot_roc(gt, pred_prob)
        auc = roc_auc_score(gt, anomaly_scores)
        result.update({"AUC": auc})

        # F1
        pred = (anomaly_scores > threshold).astype(int)
        acc = accuracy_score(gt, pred)
        p = precision_score(gt, pred, zero_division=1)
        r = recall_score(gt, pred, zero_division=1)
        f1 = f1_score(gt, pred, zero_division=1)

        result.update({
            "Accuracy": acc,
            "Precision": p,
            "Recall": r,
            "F1": f1,
        })
        wandb.sklearn.plot_confusion_matrix(gt, pred, labels=["normal", "abnormal"])

        # F1-PA
        pa_pred = PA(gt, pred)
        acc = accuracy_score(gt, pa_pred)
        p = precision_score(gt, pa_pred, zero_division=1)
        r = recall_score(gt, pa_pred, zero_division=1)
        f1 = f1_score(gt, pa_pred, zero_division=1)
        result.update({
            "Accuracy (PA)": acc,
            "Precision (PA)": p,
            "Recall (PA)": r,
            "F1 (PA)": f1,
        })
        wandb.sklearn.plot_confusion_matrix(gt, pa_pred, labels=["normal", "abnormal"])
        return result

    def get_threshold(self, gt, anomaly_scores):
        self.logger.info(f"thresholding with algorithm: {self.args.thresholding}")
        return self.threshold_function_map[self.args.thresholding](gt, anomaly_scores)

    def checkpoint(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.model.to(self.args.device)

    @staticmethod
    def save_dictionary(dictionary, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(dictionary, f)

    @staticmethod
    def reduce(anomaly_scores, stride=1, mode="mean", window_anomaly=False):
        '''
        :param anomaly_scores
        :param stride
        :param mode: how to reduce
        :return: (B,) tensor that indicates anomaly score of each point
        '''
        B, L, C = anomaly_scores.shape

        if window_anomaly:
            if mode=="mean":
                out = anomaly_scores.mean(axis=(-1, -2)) # (B, L, C) => (B,)
            elif mode=="max":
                out = anomaly_scores.max(axis=(-1, -2))  # (B, L, C) => (B,)
        else:
            T = (B-1)*stride+L
            out = np.zeros((T, L))
            if mode=="mean":
                x = anomaly_scores.mean(axis=-1) # (B, L, C) => (B, L)
                for i in range(L):
                    out[i*stride: i*stride+B, i] = x[:, i]
                out = np.true_divide(out.sum(axis=1), (out!=0).sum(axis=1)) # get mean except 0
            elif mode=="max":
                x = anomaly_scores.max(axis=-1)  # (B, L, C) => (B, L)
                for i in range(L):
                    out[i*stride: i*stride+B, i] = x[:, i]
                out = out.max(axis=-1)
            else:
                raise NotImplementedError
        return out

    @torch.no_grad()
    def oracle_thresholding(self, gt, anomaly_scores):
        '''
        Find the threshold according to Youden's J statistic,
        which maximizes (tpr-fpr)
        '''
        self.logger.info("Oracle Thresholding")
        fpr, tpr, thresholds = roc_curve(gt, anomaly_scores)
        J = tpr - fpr
        idx = np.argmax(J)
        best_threshold = thresholds[idx]
        self.logger.info(f"Best threshold found at: {best_threshold}, with fpr: {fpr[idx]}, tpr: {tpr[idx]}")
        return best_threshold

class ReconModelTrainer(Trainer):
    def __init__(self, args, logger, train_loader, test_loader):
        super(ReconModelTrainer, self).__init__(args, logger, train_loader, test_loader)

    def train(self):
        wandb.watch(self.model, log="all", log_freq=100)
        best_train_stats = None
        for epoch in range(1, self.args.epochs + 1):
            # train
            train_stats = self.train_epoch()
            self.logger.info(f"epoch {epoch} | train_stats: {train_stats}")
            self.checkpoint(os.path.join(self.args.checkpoint_path, f"epoch{epoch}.pth"))

            if best_train_stats is None or train_stats < best_train_stats:
                self.logger.info(f"Saving best results @epoch{epoch}")
                self.checkpoint(os.path.join(self.args.checkpoint_path, f"best.pth"))
                best_train_stats = train_stats
        return

    def train_epoch(self):
        self.model.train()
        log_freq = len(self.train_loader) // self.args.log_freq
        train_summary = 0.0
        for i, batch_data in enumerate(self.train_loader):
            train_log = self._process_batch(batch_data)
            if (i+1) % log_freq == 0:
                self.logger.info(f"{train_log}")
                wandb.log(train_log)
            train_summary += train_log["summary"]
        train_summary /= len(self.train_loader)
        return train_summary

    def calculate_anomaly_scores(self):
        recon_errors = self.calculate_recon_errors()
        anomaly_scores = self.reduce(recon_errors)
        return anomaly_scores

class SklearnModelTrainer(Trainer):
    def __init__(self, args, logger, train_loader, test_loader):
        super(SklearnModelTrainer, self).__init__(args, logger, train_loader, test_loader)

    def train(self):
        X = self.train_loader.dataset.x
        self.model.fit(X)

    def infer(self):
        result = {}
        self.model.eval()
        X, gt = self.test_loader.dataset.X, self.test_loader.dataset.y

        # In sklearn, 1 is normal and -1 is abnormal. convert to 1 (abnormal) or 0 (normal)
        yhat = (self.model.predict(X) == -1)

        # F1
        acc = accuracy_score(gt, yhat)
        p = precision_score(gt, yhat, zero_division=1)
        r = recall_score(gt, yhat, zero_division=1)
        f1 = f1_score(gt, yhat, zero_division=1)

        result.update({
            "Accuracy": acc,
            "Precision": p,
            "Recall": r,
            "F1": f1,
        })

        wandb.log({"Confusion Matrix": wandb.plot.confusion_matrix(gt, pred)})

        # F1-PA
        pa_pred = PA(gt, yhat)
        acc = accuracy_score(gt, pa_pred)
        p = precision_score(gt, pa_pred, zero_division=1)
        r = recall_score(gt, pa_pred, zero_division=1)
        f1 = f1_score(gt, pa_pred, zero_division=1)
        result.update({
            "Precision (PA)": p,
            "Recall (PA)": r,
            "F1 (PA)": f1,
        })

        wandb.log({"Confusion Matrix (PA)": wandb.plot.confusion_matrix(gt, pa_pred)})

        return result

    def checkpoint(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)