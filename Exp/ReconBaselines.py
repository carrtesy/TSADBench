import wandb

# Trainer
from Exp.ReconTrainer import ReconModelTrainer

# models
from models.LSTMEncDec import LSTMEncDec
from models.LSTMVAE import LSTMVAE
from models.USAD import USAD
from models.OmniAnomaly import OmniAnomaly

# others
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class LSTMEncDec_Trainer(ReconModelTrainer):
    def __init__(self, args, logger, train_loader, test_loader):
        super(LSTMEncDec_Trainer, self).__init__(args=args, logger=logger, train_loader=train_loader, test_loader=test_loader)

        self.model = LSTMEncDec(
            input_dim=self.args.num_channels,
            window_size=self.args.window_size,
            latent_dim=self.args.model.latent_dim,
            num_layers=self.args.model.num_layers,
            dropout=self.args.model.dropout,
        ).to(self.args.device)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)


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


    def calculate_anomaly_scores(self):
        # get train statistics.
        train_recon_errors = self.calculate_recon_errors(dataloader=self.train_loader) # (B, L, C)
        train_recon_errors = self.reduce(train_recon_errors, mode="except_channel") # (T_train, C)
        train_error_mu, train_error_cov = np.mean(train_recon_errors, axis=0), np.cov(train_recon_errors.T)

        # test statistics
        test_recon_errors = self.calculate_recon_errors(dataloader=self.test_loader)
        B, L, C = test_recon_errors.shape
        test_recon_errors = self.reduce(test_recon_errors, mode="except_channel") # (T_test, C)

        # anomaly scores
        r = test_recon_errors - train_error_mu # e-mu, (T, C)
        ic = np.linalg.pinv(train_error_cov) if C > 1 else np.array([[train_error_cov]]) # inverse of covariance matrix, (C, C)
        anomaly_scores = np.einsum("TC,CC,TC->T", r, ic, r)
        return anomaly_scores


    @torch.no_grad()
    def calculate_recon_errors(self, dataloader):
        '''
        :param dataloader (self.train or self.test)
        :return:  returns (B, L, C) recon loss tensor
        '''
        self.model.eval()
        iterator = tqdm(
            dataloader,
            total=len(dataloader),
            desc="calculating reconstruction errors",
            leave=True
        )
        recon_errors = []
        for i, batch_data in enumerate(iterator):
            X = batch_data[0].to(self.args.device)
            Xhat = self.model(X)
            recon_error = F.mse_loss(Xhat, X, reduction='none').to("cpu")
            recon_errors.append(recon_error)
        recon_errors = np.concatenate(recon_errors, axis=0)
        return recon_errors



class LSTMVAE_Trainer(ReconModelTrainer):
    def __init__(self, args, logger, train_loader, test_loader):
        super(LSTMVAE_Trainer, self).__init__(args=args, logger=logger, train_loader=train_loader, test_loader=test_loader)
        self.model = LSTMVAE(
            input_dim=self.args.num_channels,
            hidden_dim=self.args.model.hidden_dim,
            z_dim=self.args.model.z_dim,
            n_layers=self.args.model.n_layers,
        ).to(self.args.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)


    @staticmethod
    def gaussian_prior_KLD(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


    def _process_batch(self, batch_data) -> dict:
        X = batch_data[0].to(self.args.device)
        B, L, C = X.shape
        Xhat, mu, logvar = self.model(X)

        self.optimizer.zero_grad()
        recon_loss = F.mse_loss(Xhat, X)
        KLD_loss = self.gaussian_prior_KLD(mu, logvar)
        loss = recon_loss + self.args.model.beta * KLD_loss
        loss.backward()
        self.optimizer.step()

        out = {
            "recon_loss": recon_loss.item(),
            "KLD_loss": KLD_loss.item(),
            "total_loss": loss.item(),
            "summary": loss.item(),
        }
        return out


    def calculate_recon_errors(self):
        eval_iterator = tqdm(
            self.test_loader,
            total=len(self.test_loader),
            desc="calculating reconstruction errors",
            leave=True
        )

        recon_errors = []
        for i, batch_data in enumerate(eval_iterator):
            X = batch_data[0].to(self.args.device)
            B, L, C = X.shape
            Xhat, _, _ = self.model(X)
            recon_error = F.mse_loss(Xhat, X, reduction='none').to("cpu")
            recon_errors.append(recon_error)
        recon_errors = np.concatenate(recon_errors, axis=0)
        return recon_errors



class USAD_Trainer(ReconModelTrainer):
    def __init__(self, args, logger, train_loader, test_loader):
        super(USAD_Trainer, self).__init__(args=args, logger=logger, train_loader=train_loader, test_loader=test_loader)

        # assert moving average output to be same as input
        assert self.args.model.dsr % 2
        self.moving_avg = torch.nn.AvgPool1d(
            kernel_size=self.args.model.dsr,
            stride=1,
            padding=(self.args.model.dsr - 1)//2,
            count_include_pad=False,
        ).to(self.args.device)

        self.model = USAD(
            input_size=args.window_size * args.num_channels,
            latent_space_size=args.model.latent_dim,
        ).to(self.args.device)

        self.optimizer1 = torch.optim.Adam(params=self.model.parameters(), lr=args.lr)
        self.optimizer2 = torch.optim.Adam(params=self.model.parameters(), lr=args.lr)
        self.epoch = 0


    def train_epoch(self):
        self.model.train()
        log_freq = len(self.train_loader) // self.args.log_freq
        train_summary = 0.0
        for i, batch_data in enumerate(self.train_loader):
            train_log = self._process_batch(batch_data, self.epoch+1)
            if (i+1) % log_freq == 0:
                self.logger.info(f"{train_log}")
                wandb.log(train_log)
            train_summary += train_log["summary"]
        train_summary /= len(self.train_loader)
        self.epoch += 1
        return train_summary


    def _process_batch(self, batch_data, epoch) -> dict:
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


    def calculate_recon_errors(self):
        '''
        :return:  returns (B, L, C) recon loss tensor
        '''
        eval_iterator = tqdm(
            self.test_loader,
            total=len(self.test_loader),
            desc="calculating reconstruction errors",
            leave=True
        )
        recon_errors = []
        for i, batch_data in enumerate(eval_iterator):
            X = batch_data[0].to(self.args.device)
            recon_error = self.recon_error_criterion(X, self.args.model.alpha, self.args.model.beta).to("cpu")
            recon_errors.append(recon_error)
        recon_errors = np.concatenate(recon_errors, axis=0)
        return recon_errors


    def recon_error_criterion(self, Wt, alpha=0.5, beta=0.5):
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



class OmniAnomaly_Trainer(ReconModelTrainer):
    def __init__(self, args, logger, train_loader, test_loader):
        super(OmniAnomaly_Trainer, self).__init__(args=args, logger=logger, train_loader=train_loader, test_loader=test_loader)

        self.model = OmniAnomaly(
            in_dim=self.args.num_channels,
            hidden_dim=self.args.model.hidden_dim,
            z_dim=self.args.model.z_dim,
            dense_dim=self.args.model.dense_dim,
            out_dim=self.args.num_channels,
        ).to(self.args.device)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)
        self.beta = 0
        self.iter = 0


    @staticmethod
    def gaussian_prior_KLD(mu, logvar):
        return -0.5 * torch.sum(1+logvar-mu.pow(2)-logvar.exp())


    def _process_batch(self, batch_data) -> dict:
        X = batch_data[0].to(self.args.device)
        B, L, C = X.shape

        Xhat, mu, logvar = self.model(X)
        self.optimizer.zero_grad()
        recon_loss = F.mse_loss(Xhat, X)
        KLD_loss = self.gaussian_prior_KLD(mu, logvar)
        self.beta = min(self.iter * self.args.model.beta, 1)
        loss = recon_loss + self.beta * KLD_loss
        loss.backward()
        self.optimizer.step()
        self.iter += 1

        out = {
            "beta": self.beta,
            "recon_loss": recon_loss.item(),
            "KLD_loss": KLD_loss.item(),
            "total_loss": loss.item(),
            "summary": loss.item(),
        }
        return out


    def calculate_recon_errors(self):
        '''
        :param dataloader: eval dataloader
        :return:  returns (B, L, C) recon loss tensor
        '''
        eval_iterator = tqdm(
            self.test_loader,
            total=len(self.test_loader),
            desc="calculating reconstruction errors",
            leave=True
        )
        recon_errors = []
        for i, batch_data in enumerate(eval_iterator):
            X = batch_data[0].to(self.args.device)
            B, L, C = X.shape
            Xhat, _, _ = self.model(X)
            recon_error = F.mse_loss(Xhat, X, reduction='none').to("cpu")
            recon_errors.append(recon_error)
        recon_errors = np.concatenate(recon_errors, axis=0)
        return recon_errors
