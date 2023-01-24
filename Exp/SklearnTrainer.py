from Exp.Trainer import Trainer
import torch
import torch.nn.functional as F
import numpy as np
import copy
import os
import wandb

from tqdm import tqdm
import pickle
from utils.metrics import PA, get_summary_stats
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

        # get eval metrics
        metrics = self.get_metrics(gt, pred)
        result.update(metrics)

        return result

    def get_metrics(self, gt, pred):
        result = {}

        # F1
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
        self.logger.info(f"checkpointing to {filepath}")
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)


    def load(self, filepath):
        self.logger.info(f"loading from {filepath}")
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)