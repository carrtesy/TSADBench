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

class ReconModelTrainer(Trainer):
    def __init__(self, args, logger, train_loader, test_loader):
        super(ReconModelTrainer, self).__init__(args, logger, train_loader, test_loader)

    def train(self):
        wandb.watch(self.model, log="all", log_freq=100)

        train_iterator = tqdm(
            range(1, self.args.epochs + 1),
            total=self.args.epochs,
            desc="training epochs",
            leave=True
        )

        best_train_stats = None

        for epoch in train_iterator:
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