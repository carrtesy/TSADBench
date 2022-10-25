import torch
import torch.nn.functional as F
import numpy as np
import copy

from tqdm import tqdm
import pickle
from utils.metrics import get_statistics

class Trainer:
    def __init__(self, args):
        self.args = args

    def train(self, dataset, dataloader):
        pass

    def infer(self, dataset, dataloader):
        pass

    @torch.no_grad()
    def infer(self, dataset, dataloader):
        pass

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


    def get_result(self, y, anomaly_scores, samples=100):
        '''
        :param y:
        :param anomaly_scores:
        :param samples:
        :return: F1, Precision, Recall of (y, yhat) and (y, yhat_PA)
        '''
        best_result = {}
        m, M = anomaly_scores.min(), anomaly_scores.max()

        # F1 SCORE
        threshold_iterator = tqdm(
            np.linspace(m, M, num=samples),
            total=len(np.linspace(m, M, num=samples)),
            desc="Thresholding F1",
            leave=True
        )

        for i, threshold in enumerate(threshold_iterator):
            ypred = anomaly_scores > threshold
            cm, a, p, r, f1 = get_statistics(y, ypred)
            if "F1" not in best_result or best_result["F1"] < f1:
                best_result.update(
                    {
                        "Threshold": threshold,
                        "Confusion Matrix": cm,
                        "Precision": p,
                        "Recall": r,
                        "F1": f1,
                    }
                )

        # F1-PA SCORE
        threshold_iterator = tqdm(
            np.linspace(m, M, num=samples),
            total=len(np.linspace(m, M, num=samples)),
            desc="Thresholding F1-PA",
            leave=True
        )

        for i, threshold in enumerate(threshold_iterator):
            ypred = anomaly_scores > threshold
            ypred_pa = self.PA(y, ypred)
            cm, a, p, r, f1 = get_statistics(y, ypred_pa)
            if "F1-PA" not in best_result or best_result["F1-PA"] < f1:
                best_result.update(
                    {
                        "Threshold-PA": threshold,
                        "Confusion Matrix-PA": cm,
                        "Precision-PA": p,
                        "Recall-PA": r,
                        "F1-PA": f1,
                    }
                )

        return best_result

    @staticmethod
    def PA(y, y_pred):
        '''
        Point-Adjust Algorithm
        https://github.com/thuml/Anomaly-Transformer/blob/main/solver.py
        '''
        anomaly_state = False
        y_pred_pa = copy.deepcopy(y_pred)
        for i in range(len(y)):
            if y[i] == 1 and y_pred_pa[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if y[j] == 0:
                        break
                    else:
                        if y_pred_pa[j] == 0:
                            y_pred_pa[j] = 1
                for j in range(i, len(y)):
                    if y[j] == 0:
                        break
                    else:
                        if y_pred_pa[j] == 0:
                            y_pred_pa[j] = 1
            elif y[i] == 0:
                anomaly_state = False
            if anomaly_state:
                y_pred_pa[i] = 1

        return y_pred_pa
