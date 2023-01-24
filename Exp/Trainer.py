import torch
import torch.nn.functional as F
import numpy as np
import copy
import os
import wandb

from tqdm import tqdm
import pickle
from utils.metrics import get_auroc, get_summary_stats
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

    def infer(self):
        result = {}
        gt = self.test_loader.dataset.y
        anomaly_scores = self.calculate_anomaly_scores()

        # thresholding
        threshold = self.get_threshold(gt=gt, anomaly_scores=anomaly_scores)
        result.update({"Threshold": threshold})

        # get eval metrics
        metrics = self.get_metrics(gt=gt, anomaly_scores=anomaly_scores, threshold=threshold)
        result.update(metrics)

        return result

    def get_threshold(self, gt, anomaly_scores):
        self.logger.info(f"thresholding with algorithm: {self.args.thresholding}")
        return self.threshold_function_map[self.args.thresholding](gt, anomaly_scores)

    def get_metrics(self, gt, anomaly_scores, threshold):
        result = {}
        # AUROC
        auc = get_auroc(gt, anomaly_scores, threshold)
        result.update({"AUC": auc})

        # F1
        pred = (anomaly_scores > threshold).astype(int)
        metrics = get_summary_stats(gt, pred)
        result.update(metrics)
        wandb.sklearn.plot_confusion_matrix(gt, pred, labels=["normal", "abnormal"])

        # F1-PA
        pa_pred = PA(gt, pred)
        pa_metrics = get_summary_stats(gt, pa_pred, desc="_PA")
        result.update(pa_metrics)
        wandb.sklearn.plot_confusion_matrix(gt, pa_pred, labels=["normal", "abnormal"])

        return result

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
