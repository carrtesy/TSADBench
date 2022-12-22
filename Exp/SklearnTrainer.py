from Exp.Trainer import Trainer
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