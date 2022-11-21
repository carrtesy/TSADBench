import torch
import torch.nn.functional as F
import numpy as np
import copy

from tqdm import tqdm
import pickle
from utils.metrics import get_statistics
from utils.metrics import PA

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

    @torch.no_grad()
    def oracle_thresholding(self, gt, anomaly_scores, samples=100, point_adjust=False):
        '''
        Find the threshold that gives best F1.
        '''

        m, M = anomaly_scores.min(), anomaly_scores.max()
        threshold_iterator = tqdm(
            np.linspace(m, M, num=samples),
            total=len(np.linspace(m, M, num=samples)),
            desc="Thresholding",
            leave=True
        )

        best_threshold, best_score = None, None
        for i, threshold in enumerate(threshold_iterator):
            pred = anomaly_scores > threshold
            if point_adjust:
                pred = PA(gt, pred)
            cm, a, p, r, f1 = get_statistics(gt, pred)
            if best_score is None or best_score < f1:
                best_threshold, best_score = threshold, f1

        return best_threshold

    @staticmethod
    def get_pointwise_result(gt, anomaly_scores, threshold):
        pred = (anomaly_scores > threshold).astype(int)
        cm, a, p, r, f1 = get_statistics(gt, pred)
        result = {
            "Threshold": threshold,
            "Confusion Matrix": cm.ravel(),
            "Precision": p,
            "Recall": r,
            "F1": f1,
        }
        return result

    @staticmethod
    def get_PA_result(gt, anomaly_scores, threshold):
        pred = (anomaly_scores > threshold).astype(int)
        pa_pred = PA(gt, pred)
        cm, a, p, r, f1 = get_statistics(gt, pa_pred)
        result = {
            "Threshold (PA)": threshold,
            "Confusion Matrix (PA)": cm.ravel(),
            "Precision (PA)": p,
            "Recall (PA)": r,
            "F1 (PA)": f1,
        }
        return result

class SklearnModelTrainer(Trainer):
    def __init__(self, args):
        super(SklearnModelTrainer, self).__init__(args)

    def train(self, dataset, dataloader):
        X = dataset.x
        self.model.fit(X)

    def infer(self, dataset, dataloader):
        X, y = dataset.x, dataset.y

        # In sklearn, 1 is normal and -1 is abnormal. convert to 1 (abnormal) or 0 (normal)
        yhat = self.model.predict(X)
        yhat = (yhat == -1)

        result = {}
        # F1
        cm, a, p, r, f1 = get_statistics(y, yhat)
        result.update({
            "Confusion Matrix": cm.ravel(),
            "Precision": p,
            "Recall": r,
            "F1": f1,
        })

        # F1-PA
        yhat_pa = self.PA(y, yhat)
        cm, a, p, r, f1 = get_statistics(y, yhat_pa)
        result.update({
            "Confusion Matrix-PA": cm.ravel(),
            "Precision-PA": p,
            "Recall-PA": r,
            "F1-PA": f1,
        })

        return result

    def checkpoint(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)