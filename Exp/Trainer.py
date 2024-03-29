import torch
import numpy as np
import os
import wandb
import matplotlib
import matplotlib.pyplot as plt

import pickle
from utils.metrics import get_auroc, get_summary_stats
from utils.metrics import PA
from sklearn.metrics import roc_curve

matplotlib.rcParams['agg.path.chunksize'] = 10000

class Trainer:
    def __init__(self, args, logger, train_loader, test_loader):
        self.args = args
        self.logger = logger
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.threshold_function_map = {
            "oracle": self.oracle_thresholding,
        }


    def train(self):
        pass


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

        # visualize anomaly scores, and save with np
        self.plot_anomaly_scores(gt, anomaly_scores, threshold)
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


    def plot_anomaly_scores(self, gt, anomaly_scores, threshold):
        '''
        plot anomaly score with gt & threshold, and save it to {args.plots} dir.
        '''
        # plot
        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        ax.set_title("Anomaly Score Visualization")
        ax.plot(anomaly_scores, label="anomaly scores")
        ax.axhline(y=threshold, color="black", label="threshold")
        s, e = None, None
        for i, label in enumerate(gt):
            if label == 1 and s is None:
                s = i
            elif label == 0 and s is not None:
                e = i - 1
                if (e - s) > 0:
                    ax.axvspan(s, e, facecolor='red', alpha=0.5)
                else:
                    ax.plot(s, anomaly_scores[i], 'ro')
                s, e = None, None
        ax.legend()
        plt.savefig(os.path.join(self.args.plot_path, "anomaly_vis.png"))

        # save
        with open(os.path.join(self.args.output_path, "anomaly_scores.npy"), 'wb') as f:
            np.save(f, anomaly_scores)
        with open(os.path.join(self.args.output_path, "gt.npy"), 'wb') as f:
            np.save(f, gt)
        with open(os.path.join(self.args.output_path, "threshold.pkl"), 'wb') as f:
            pickle.dump(threshold, f)


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
    def reduce(anomaly_scores, stride=1, mode="all"):
        '''
        :param anomaly_scores
        :param stride
        :param mode: how to reduce (all: (T, ), except_channel: (T, C))
        :return: (B,) tensor that indicates anomaly score of each point
        '''
        B, L, C = anomaly_scores.shape

        T = (B-1)*stride+L
        out = np.zeros((T, L, C))
        for i in range(L):
            out[i*stride: i*stride+B, i] = anomaly_scores[:, i]
        out = np.true_divide(out.sum(axis=1), (out!=0).sum(axis=1)) # get mean except 0
        if mode == "all":
            out = out.mean(axis=-1)  # (T, C) => (T, )
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
