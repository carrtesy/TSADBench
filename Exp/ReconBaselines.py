import wandb

# Trainer
from Exp.Trainer import Trainer, ReconModelTrainer

# models
from models.AE import AE
from models.VAE import VAE

# utils
from utils.metrics import get_statistics
import os

# others
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
import pickle

class AE_Trainer(ReconModelTrainer):
    def __init__(self, args, logger, train_loader, test_loader):
        super(AE_Trainer, self).__init__(args=args, logger=logger, train_loader=train_loader, test_loader=test_loader)

        self.model = AE(
            input_size=self.args.window_size * self.args.num_channels,
            latent_space_size=args.latent_dim,
        ).to(self.args.device)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)

    def _process_batch(self, batch_data) -> dict:
        X = batch_data[0].to(self.args.device)
        B, L, C = X.shape

        Xhat = self.model(X.reshape(B, L*C))
        Xhat = Xhat.reshape(B, L, C)
        self.optimizer.zero_grad()
        loss = F.mse_loss(Xhat, X)
        loss.backward()
        self.optimizer.step()

        out = {
            "loss": loss.item(),
            "summary": loss.item(),
        }
        return out

    @torch.no_grad()
    def infer(self):
        self.model.eval()
        y = self.test_loader.dataset.y
        recon_errors = self.calculate_recon_errors()    # (B, L, C)
        recon_errors = self.reduce(recon_errors)    # (T, )
        anomaly_scores = recon_errors
        result = self.get_result(y, anomaly_scores)
        return result

    def calculate_recon_errors(self):
        '''
        :param dataloader: eval dataloader
        :return:  returns (B, L, C) recon loss tensor
        '''
        eval_iterator = tqdm(
            self.test_loader,
            total=len(self.test_loader),
            desc="infer",
            leave=True
        )
        recon_errors = []
        for i, batch_data in enumerate(eval_iterator):
            X = batch_data[0].to(self.args.device)
            Xhat = self.model(X)
            recon_error = F.mse_loss(Xhat, X, reduction='none').to("cpu")
            recon_errors.append(recon_error)
        recon_errors = np.concatenate(recon_errors, axis=0)
        return recon_errors


class VAE_Trainer(ReconModelTrainer):
    def __init__(self, args, logger, train_loader, test_loader):
        super(VAE_Trainer, self).__init__(args=args, logger=logger, train_loader=train_loader, test_loader=test_loader)

        self.model = VAE(
            input_size=self.args.window_size * self.args.num_channels,
            latent_space_size=args.latent_dim,
        ).to(self.args.device)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)

    @staticmethod
    def gaussian_prior_KLD(mu, logvar):
        return -0.5 * torch.sum(1+logvar-mu.pow(2)-logvar.exp())

    def _process_batch(self, batch_data) -> dict:
        X = batch_data[0].to(self.args.device)
        B, L, C = X.shape

        Xhat, mu, logvar = self.model(X.reshape(B, L*C))
        Xhat = Xhat.reshape(B, L, C)
        self.optimizer.zero_grad()
        recon_loss = F.mse_loss(Xhat, X)
        KLD_loss = self.gaussian_prior_KLD(mu, logvar)
        loss = recon_loss + KLD_loss
        loss.backward()
        self.optimizer.step()

        out = {
            "recon_loss": recon_loss.item(),
            "KLD_loss": KLD_loss.item(),
            "total_loss": loss.item(),
            "summary": loss.item(),
        }
        return out

    @torch.no_grad()
    def infer(self):
        self.model.eval()
        y = self.test_loader.dataset.y
        recon_errors = self.calculate_recon_errors()    # (B, L, C)
        recon_errors = self.reduce(recon_errors)    # (T, )
        anomaly_scores = recon_errors
        result = self.get_result(y, anomaly_scores)
        return result

    def calculate_recon_errors(self):
        '''
        :param dataloader: eval dataloader
        :return:  returns (B, L, C) recon loss tensor
        '''
        eval_iterator = tqdm(
            self.test_loader,
            total=len(self.test_loader),
            desc="infer",
            leave=True
        )
        recon_errors = []
        for i, batch_data in enumerate(eval_iterator):
            X = batch_data[0].to(self.args.device)
            Xhat = self.model(X)
            recon_error = F.mse_loss(Xhat, X, reduction='none').to("cpu")
            recon_errors.append(recon_error)
        recon_errors = np.concatenate(recon_errors, axis=0)
        return recon_errors