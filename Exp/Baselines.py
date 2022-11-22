# Trainer
from Exp.Trainer import Trainer

# models
from models.LSTMEncDec import LSTMEncDec
from models.USAD import USAD


# utils
from utils.metrics import get_statistics

# others
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
import pickle

class LSTMEncDec_Trainer(Trainer):
    def __init__(self, args, train_loader, test_loader):
        super(LSTMEncDec_Trainer, self).__init__(args=args, train_loader=train_loader, test_loader=test_loader)

        self.model = LSTMEncDec(
            input_dim=self.args.num_channels,
            latent_dim=self.args.latent_dim,
            window_size=self.args.window_size,
        ).to(self.args.device)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)

    def train(self, dataset, dataloader):
        self.model.train()
        train_iterator = tqdm(
            self.train_loader,
            total=len(self.train_loader),
            desc="training",
            leave=True
        )

        train_summary = 0.0
        for i, batch_data in enumerate(train_iterator):
            train_log = self._process_batch(batch_data)
            train_iterator.set_postfix(train_log)
            train_summary += train_log["summary"]
        train_summary /= len(train_iterator)
        return train_summary

    def _process_batch(self, batch_data):
        X = batch_data[0].to(self.args.device)
        B, L, C = X.shape

        Xhat = self.model(X)
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
        mu, var = np.mean(recon_errors), np.cov(recon_errors)
        e = (recon_errors - mu)
        anomaly_scores = e @ var @ e.T
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

class USAD_Trainer(Trainer):
    def __init__(self, args, train_loader, test_loader):
        super(USAD_Trainer, self).__init__(args=args, train_loader=train_loader, test_loader=test_loader)

        # assert moving average output to be same as input
        assert self.args.dsr % 2
        self.moving_avg = torch.nn.AvgPool1d(
            kernel_size=self.args.dsr,
            stride=1,
            padding=(self.args.dsr - 1)//2,
            count_include_pad=False,
        ).to(self.args.device)

        self.model = USAD(
            input_size=args.window_size * args.num_channels,
            latent_space_size=args.latent_dim,
        ).to(self.args.device)

        self.optimizer1 = torch.optim.Adam(params=self.model.parameters(), lr=args.lr)
        self.optimizer2 = torch.optim.Adam(params=self.model.parameters(), lr=args.lr)
        self.epoch = 0


    def train(self):
        self.model.train()
        train_iterator = tqdm(
            self.train_loader,
            total=len(self.train_loader),
            desc="training",
            leave=True
        )

        train_summary = 0.0
        for i, batch_data in enumerate(train_iterator):
            train_log = self._process_batch(batch_data, self.epoch+1)
            train_iterator.set_postfix(train_log)
            train_summary += train_log["summary"]

        train_summary /= len(train_iterator)
        self.epoch += 1
        return train_summary

    def _process_batch(self, batch_data, epoch):
        X = batch_data[0].to(self.args.device)
        X = self.moving_avg(X.transpose(-1, -2)).transpose(-1, -2)

        B, L, C = X.shape

        # AE1
        z = self.model.encoder(X.reshape(B, L*C))
        Wt1p = self.model.decoder1(z).reshape(B, L, C)
        Wt2p = self.model.decoder2(z).reshape(B, L, C)
        Wt2dp = self.model.decoder2(self.model.encoder(Wt1p.reshape(B, L*C))).reshape(B, L, C)

        self.optimizer1.zero_grad()
        loss_AE1 = (1 / epoch) * F.mse_loss(X, Wt1p) + (1 - (1 / epoch)) * F.mse_loss(X, Wt2dp)
        loss_AE1.backward()
        self.optimizer1.step()

        # AE2
        z = self.model.encoder(X.reshape(B, L*C))
        Wt1p = self.model.decoder1(z).reshape(B, L, C)
        Wt2p = self.model.decoder2(z).reshape(B, L, C)
        Wt2dp = self.model.decoder2(self.model.encoder(Wt1p.reshape(B, L*C))).reshape(B, L, C)

        self.optimizer2.zero_grad()
        loss_AE2 = (1 / epoch) * F.mse_loss(X, Wt2p) - (1 - (1 / epoch)) * F.mse_loss(X, Wt2dp)
        loss_AE2.backward()
        self.optimizer2.step()

        out = {
            "loss_AE1": loss_AE1.item(),
            "loss_AE2": loss_AE2.item(),
            "summary": loss_AE1.item() + loss_AE2.item()
        }
        return out

    @torch.no_grad()
    def infer(self):
        self.model.eval()
        y = self.test_loader.dataset.y
        anomaly_scores = self.calculate_anomaly_scores(self.test_loader)
        anomaly_scores = self.reduce(anomaly_scores)
        result = self.get_result(y, anomaly_scores)
        return result

    def calculate_anomaly_scores(self):
        '''
        :return:  returns (B, L, C) recon loss tensor
        '''
        eval_iterator = tqdm(
            self.test_loader,
            total=len(self.test_loader),
            desc="infer",
            leave=True
        )
        anomaly_scores = []
        for i, batch_data in enumerate(eval_iterator):
            X = batch_data[0].to(self.args.device)
            anomaly_score = self.anomaly_score_calculator(X, self.args.alpha, self.args.beta).to("cpu")
            anomaly_scores.append(anomaly_score)
        anomaly_scores = np.concatenate(anomaly_scores, axis=0)
        return anomaly_scores

    def anomaly_score_calculator(self, Wt, alpha=0.5, beta=0.5):
        '''
        :param Wt: model input
        :param alpha: low detection sensitivity
        :param beta: high detection sensitivity
        :return: recon error (B, L, C)
        '''
        B, L, C = Wt.shape
        z = self.model.encoder(Wt.reshape(B, L*C))
        Wt1p = self.model.decoder1(z).reshape(B, L, C)
        Wt2dp = self.model.decoder2(self.model.encoder(Wt1p.reshape(B, L*C))).reshape(B, L, C)
        return alpha * F.mse_loss(Wt, Wt1p, reduction='none') + beta * F.mse_loss(Wt, Wt2dp, reduction='none')
