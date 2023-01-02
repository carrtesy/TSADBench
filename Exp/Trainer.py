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
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

class Trainer:
    def __init__(self, args, logger, train_loader, test_loader):
        self.args = args
        self.logger = logger
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.threshold_function_map = {
            "oracle": self.oracle_thresholding,
        }

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
        tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()

        result.update({
            "Accuracy": acc,
            "Precision": p,
            "Recall": r,
            "F1": f1,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        })
        wandb.sklearn.plot_confusion_matrix(gt, pred, labels=["normal", "abnormal"])

        # F1-PA
        pa_pred = PA(gt, pred)
        acc = accuracy_score(gt, pa_pred)
        p = precision_score(gt, pa_pred, zero_division=1)
        r = recall_score(gt, pa_pred, zero_division=1)
        f1 = f1_score(gt, pa_pred, zero_division=1)
        tn, fp, fn, tp = confusion_matrix(gt, pa_pred).ravel()
        result.update({
            "Accuracy (PA)": acc,
            "Precision (PA)": p,
            "Recall (PA)": r,
            "F1 (PA)": f1,
            "tn (PA)": tn,
            "fp (PA)": fp,
            "fn (PA)": fn,
            "tp (PA)": tp,
        })
        wandb.sklearn.plot_confusion_matrix(gt, pa_pred, labels=["normal", "abnormal"])
        return result

    def get_threshold(self, gt, anomaly_scores):
        self.logger.info(f"thresholding with algorithm: {self.args.thresholding}")
        return self.threshold_function_map[self.args.thresholding](gt, anomaly_scores)

    def checkpoint(self, filepath):
        self.logger.info(f"checkpointing: {filepath} @Trainer - torch.save")
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        self.logger.info(f"loading: {filepath} @Trainer - torch.load_state_dict")
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
        Find the threshold that maximizes f1-score
        '''
        self.logger.info("Oracle Thresholding")
        P, N = (gt == 1).sum(), (gt == 0).sum()
        fpr, tpr, thresholds = roc_curve(gt, anomaly_scores)

        fp = np.array(fpr * N, dtype=int)
        tn = np.array(N - fp, dtype=int)
        tp = np.array(tpr * P, dtype=int)
        fn = np.array(P - tp, dtype=int)

        eps = 1e-6
        precision = tp / np.maximum(tp + fp, eps)
        recall = tp / np.maximum(tp + fn, eps)
        f1 = 2 * (precision * recall) / np.maximum(precision + recall, eps)
        idx = np.argmax(f1)
        best_threshold = thresholds[idx]
        self.logger.info(f"Best threshold found at: {best_threshold}, "
                         f"with fpr: {fpr[idx]}, tpr: {tpr[idx]}\n"
                         f"tn: {tn[idx]} fn: {fn[idx]}\n"
                         f"fp: {fp[idx]} tp: {tp[idx]}")
        return best_threshold
