from Exp.Trainer import Trainer
import torch
import torch.nn.functional as F
import numpy as np
import copy
import os
import wandb

from tqdm import tqdm
import pickle
from utils.metrics import PA
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score

class SklearnModelTrainer(Trainer):
    def __init__(self, args, logger, train_loader, test_loader):
        super(SklearnModelTrainer, self).__init__(args, logger, train_loader, test_loader)

    def train(self):
        X = self.train_loader.dataset.x
        self.logger.info(f"training {self.args.model.name}")
        self.model.fit(X)
        self.logger.info(f"training complete.")
        self.checkpoint(os.path.join(self.args.checkpoint_path, f"best.pth"))

    def infer(self):
        result = {}
        X, gt = self.test_loader.dataset.x, self.test_loader.dataset.y

        # In sklearn, 1 is normal and -1 is abnormal. convert to 1 (abnormal) or 0 (normal)
        pred = (self.model.predict(X) == -1)

        # F1
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

    def checkpoint(self, filepath):
        self.logger.info(f"checkpointing to {filepath}")
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filepath):
        self.logger.info(f"loading from {filepath}")
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)