import torch
import torch.nn.functional as F
import numpy as np

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
    def reduce(anomaly_scores, mode="mean"):
        '''
        :param anomaly_scores
        :param mode: how to reduce
        :return: (B,) tensor that indicates anomaly score of each point
        '''
        B, L, C = anomaly_scores.shape
        out = np.zeros((B+L-1, L))
        if mode=="mean":
            x = anomaly_scores.mean(axis=-1) # (B, L, C) => (B, L)
            for i in range(L):
                out[i: i+B, i] = x[:, i]
            out = np.true_divide(out.sum(axis=1), (out!=0).sum(axis=1)) # get mean except 0
        elif mode=="last_step_mean":
            x = anomaly_scores.mean(axis=-1)  # (B, L, C) => (B, L)
            for i in range(L):
                out[i: i + B, i] = x[:, i]
            out = out[:, -1]
        elif mode=="max":
            x = anomaly_scores.max(axis=-1)  # (B, L, C) => (B, L)
            for i in range(L):
                out[i: i+B, i] = x[:, i]
            out = out.max(axis=-1)
        else:
            raise NotImplementedError
        return out

    @staticmethod
    def get_best_f1(y, anomaly_scores, samples=100):
        best_result = None
        m, M = anomaly_scores.min(), anomaly_scores.max()

        threshold_iterator = tqdm(
            np.linspace(m, M, num=samples),
            total=len(np.linspace(m, M, num=samples)),
            desc="thresholding",
            leave=True
        )

        for i, threshold in enumerate(threshold_iterator):
            ypred = anomaly_scores > threshold
            cm, a, p, r, f1 = get_statistics(y, ypred)
            if best_result is None or best_result["f1"] < f1:
                best_result = {
                    "threshold": threshold,
                    "cm": cm,
                    "p": p,
                    "r": r,
                    "f1": f1,
                }
        return best_result
