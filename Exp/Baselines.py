import wandb

# Trainer
from Exp.Trainer import Trainer

# models
from models.AnomalyTransformer import AnomalyTransformer
from pyod.models.deep_svdd import DeepSVDD
from models.DAGMM import DAGMM
from models.THOC import THOC

# utils
from utils.metrics import PA
from utils.optim import adjust_learning_rate
from utils.custom_loss import my_kl_loss
from utils.metrics import PA
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score
from utils.tools import to_var # for DAGMM.

# others
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import datetime

from tqdm import tqdm
import pickle

class RandomModel_Trainer(Trainer):
    def __init__(self, args, logger, train_loader, test_loader):
        super(RandomModel_Trainer, self).__init__(args=args, logger=logger, train_loader=train_loader, test_loader=test_loader)
        self.threshold_function_map = {
            "oracle": self.oracle_thresholding,
        }


    def train(self):
        return


    def load(self, filepath):
        return


    def calculate_anomaly_scores(self):
        shape = self.test_loader.dataset.y.shape
        return np.random.rand(*shape)


class AnomalyTransformer_Trainer(Trainer):
    def __init__(self, args, logger, train_loader, test_loader):
        super(AnomalyTransformer_Trainer, self).__init__(args=args, logger=logger, train_loader=train_loader, test_loader=test_loader)
        self.model = AnomalyTransformer(
            win_size=self.args.window_size,
            enc_in=self.args.num_channels,
            c_out=self.args.num_channels,
            e_layers=self.args.model.e_layers,
        ).to(self.args.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=args.lr)
        self.threshold_function_map = {
            "oracle": self.oracle_thresholding,
            "anomaly_transformer": self.anomaly_transformer_thresholding,
        }
        self.criterion = nn.MSELoss()


    # code referrence:
    # https://github.com/thuml/Anomaly-Transformer/blob/72a71e5f0847bd14ba0253de899f7b0d5ba6ee97/solver.py#L130
    def train(self):
        wandb.watch(self.model, log="all", log_freq=100)
        self.model.train()
        time_now = time.time()
        train_steps = len(self.train_loader)

        for epoch in range(self.args.epochs):
            iter_count = 0
            loss1_list = []
            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.args.device)
                output, series, prior, _ = self.model(input)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.args.window_size)).detach()))
                                    + torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.args.window_size)).detach(), series[u])))
                    prior_loss += (torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.args.window_size)),series[u].detach()))
                                   + torch.mean(my_kl_loss(series[u].detach(), (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.args.window_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.args.model.k * series_loss).item())
                loss1 = rec_loss - self.args.model.k * series_loss
                loss2 = rec_loss + self.args.model.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.epochs - epoch) * train_steps - i)
                    self.logger.info(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            self.logger.info(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(loss1_list)
            self.logger.info(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} ")
            adjust_learning_rate(self.optimizer, epoch + 1, self.args.lr)
            self.logger.info(f'Updating learning rate to {self.optimizer.param_groups[0]["lr"]}')

            self.checkpoint(os.path.join(self.args.checkpoint_path, f"best.pth"))


    def calculate_anomaly_scores(self):
        temperature = self.args.model.temperature
        criterion = nn.MSELoss(reduce=False)

        attens_energy = []
        for i, (input_data, labels) in enumerate(self.test_loader):
            input = input_data.float().to(self.args.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.args.window_size)).detach()) * temperature
                    prior_loss = my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.args.window_size)), series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.args.window_size)).detach()) * temperature
                    prior_loss += my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.args.window_size)), series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)

            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        return test_energy


    @torch.no_grad()
    def anomaly_transformer_thresholding(self, gt, anomaly_scores):
        temperature = self.args.model.temperature
        criterion = nn.MSELoss(reduce=False)

        # (1) statistics on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.args.device)
            output, series, prior, _ = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.args.window_size)).detach()) * temperature
                    prior_loss = my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.args.window_size)), series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.args.window_size)).detach()) * temperature
                    prior_loss += my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.args.window_size)), series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        # test loader is set as threshold loader
        # see: https://github.com/thuml/Anomaly-Transformer/issues/32
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.test_loader):
            input = input_data.float().to(self.args.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.args.window_size)).detach()) * temperature
                    prior_loss = my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.args.window_size)), series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.args.window_size)).detach()) * temperature
                    prior_loss += my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.args.window_size)), series[u].detach()) * temperature
            # Metric
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.model.anomaly_ratio)
        return threshold


# code reference:
# https://github.com/danieltan07/dagmm/blob/master/solver.py
class DAGMM_Trainer(Trainer):
    def __init__(self, args, logger, train_loader, test_loader):
        super(DAGMM_Trainer, self).__init__(args=args, logger=logger, train_loader=train_loader, test_loader=test_loader)
        self.model = DAGMM(
            n_gmm=self.args.model.gmm_k,
            input_dim=self.args.window_size*self.args.num_channels,
            latent_dim=self.args.model.latent_dim,

        ).to(self.args.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)
        self.threshold_function_map = {
            "oracle": self.oracle_thresholding,
            "DAGMM": self.DAGMM_thresholding,
        }
        self.criterion = nn.MSELoss()


    def train(self):
        wandb.watch(self.model, log="all", log_freq=100)
        self.model.train()

        # Start training
        self.ap_global_train = np.array([0, 0, 0])
        best_train_stats = None
        for epoch in range(1, self.args.epochs+1):
            train_stats = self.train_epoch()
            self.logger.info(f"epoch {epoch} | train_stats: {train_stats}")
            # Save model checkpoints
            if best_train_stats is None or train_stats < best_train_stats:
                self.logger.info(f"Saving best results @epoch{epoch}")
                self.checkpoint(os.path.join(self.args.checkpoint_path, f"best.pth"))
                best_train_stats = train_stats
        return


    def train_epoch(self):
        log_freq = max(len(self.train_loader) // self.args.log_freq, 1)
        train_summary = 0.0
        for i, (input_data, labels) in enumerate(self.train_loader):
            B, L, C = input_data.shape
            input_data = to_var(input_data.reshape(B, L*C))
            total_loss, sample_energy, recon_error, cov_diag = self._process_batch(input_data)
            # Logging
            loss = {
                'recon_error': recon_error.item(),
                'cov_diag': cov_diag.item(),
                'sample_energy': sample_energy.item(),
                'summary': total_loss.data.item(),
            }

            if (i+1) % log_freq == 0:
                self.logger.info(f"{loss}")
                wandb.log(loss)
            train_summary += loss["summary"]
        train_summary /= len(self.train_loader)
        return train_summary


    def _process_batch(self, batch_data):
        enc, dec, z, gamma = self.model(batch_data)
        total_loss, sample_energy, recon_error, cov_diag = self.model.loss_function(
            batch_data, dec, z, gamma, self.args.model.lambda_energy, self.args.model.lambda_cov_diag
        )
        self.model.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.model.grad_clip)
        self.optimizer.step()
        return total_loss, sample_energy, recon_error, cov_diag


    @torch.no_grad()
    def calculate_anomaly_scores(self):
        # get test energy
        test_energy = []
        test_labels = []
        test_z = []
        for it, (input_data, labels) in enumerate(self.test_loader):
            B, L, C = input_data.shape
            input_data = to_var(input_data.reshape(B, L * C))
            enc, dec, z, gamma = self.model(input_data)
            sample_energy, cov_diag = self.model.compute_energy(z, size_average=False)
            test_energy.append(sample_energy.data.cpu().numpy())
            test_z.append(z.data.cpu().numpy())
            test_labels.append(labels.numpy())
        test_energy = np.concatenate(test_energy, axis=0)
        init_pred = test_energy[0].repeat(self.args.window_size - 1)
        test_energy = np.concatenate((init_pred, test_energy))
        return test_energy


    @torch.no_grad()
    def DAGMM_thresholding(self, gt, anomaly_scores):
        self.model.eval()

        # train energy
        N = 0
        mu_sum = 0
        cov_sum = 0
        gamma_sum = 0

        for it, (input_data, labels) in enumerate(self.train_loader):
            B, L, C = input_data.shape
            input_data = to_var(input_data.reshape(B, L * C))
            enc, dec, z, gamma = self.model(input_data)
            phi, mu, cov = self.model.compute_gmm_params(z, gamma)

            batch_gamma_sum = torch.sum(gamma, dim=0)

            gamma_sum += batch_gamma_sum
            mu_sum += mu * batch_gamma_sum.unsqueeze(-1)  # keep sums of the numerator only
            cov_sum += cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)  # keep sums of the numerator only

            N += input_data.size(0)

        train_phi = gamma_sum / N
        train_mu = mu_sum / gamma_sum.unsqueeze(-1)
        train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        self.logger.info("N:", N)
        self.logger.info("phi :\n", train_phi)
        self.logger.info("mu :\n", train_mu)
        self.logger.info("cov :\n", train_cov)

        train_energy = []
        train_labels = []
        train_z = []
        for it, (input_data, labels) in enumerate(self.train_loader):
            B, L, C = input_data.shape
            input_data = to_var(input_data.reshape(B, L * C))

            enc, dec, z, gamma = self.model(input_data)
            sample_energy, cov_diag = self.model.compute_energy(z, phi=train_phi, mu=train_mu, cov=train_cov,
                                                                size_average=False)

            train_energy.append(sample_energy.data.cpu().numpy())
            train_z.append(z.data.cpu().numpy())
            train_labels.append(labels.numpy())

        train_energy = np.concatenate(train_energy, axis=0)
        train_z = np.concatenate(train_z, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)

        # test energy
        test_energy = []
        test_labels = []
        test_z = []
        for it, (input_data, labels) in enumerate(self.test_loader):
            B, L, C = input_data.shape
            input_data = to_var(input_data.reshape(B, L * C))

            enc, dec, z, gamma = self.model(input_data)
            sample_energy, cov_diag = self.model.compute_energy(z, size_average=False)
            test_energy.append(sample_energy.data.cpu().numpy())
            test_z.append(z.data.cpu().numpy())
            test_labels.append(labels.numpy())

        test_energy = np.concatenate(test_energy, axis=0)
        test_z = np.concatenate(test_z, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        combined_labels = np.concatenate([train_labels, test_labels], axis=0)

        # thresholding
        thresh = np.percentile(combined_energy, 100 - 20)
        self.logger.info("Threshold :", thresh)
        return thresh


class DeepSVDD_Trainer(Trainer):
    def __init__(self, args, logger, train_loader, test_loader):
        super(DeepSVDD_Trainer, self).__init__(args, logger, train_loader, test_loader)
        self.model = DeepSVDD()


    def train(self):
        X = self.train_loader.dataset.X
        self.logger.info(f"training {self.args.model.name}")
        self.model.fit(X)
        self.logger.info(f"training complete.")
        self.checkpoint(os.path.join(self.args.checkpoint_path, f"best.pth"))


    def infer(self):
        result = {}
        X, gt = self.test_loader.dataset.X, self.test_loader.dataset.y

        pred = self.model.predict(X)

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


class THOC_Trainer(Trainer):
    def __init__(self, args, logger, train_loader, test_loader):
        super(THOC_Trainer, self).__init__(args=args, logger=logger, train_loader=train_loader, test_loader=test_loader)
        self.model = THOC(
            C=self.args.num_channels,
            W=self.args.window_size,
            n_hidden=self.args.model.hidden_dim,
            device=self.args.device
        ).to(self.args.device)
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.args.lr, weight_decay=self.args.model.L2_reg)


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
            if (i + 1) % log_freq == 0:
                self.logger.info(f"{train_log}")
                wandb.log(train_log)
            train_summary += train_log["summary"]
        train_summary /= len(self.train_loader)
        return train_summary


    def _process_batch(self, batch_data) -> dict:
        out = dict()
        X = batch_data[0].to(self.args.device)
        B, L, C = X.shape

        anomaly_score, loss_dict = self.model(X)
        for k in loss_dict:
            out.update({k: loss_dict[k].item()})

        self.optimizer.zero_grad()
        loss = loss_dict["L_THOC"] + self.args.model.LAMBDA_orth * loss_dict["L_orth"] + self.args.model.LAMBDA_TSS * loss_dict["L_TSS"]
        loss.backward()
        self.optimizer.step()

        out.update({
            "summary": loss.item(),
        })
        return out


    def calculate_anomaly_scores(self):
        eval_iterator = tqdm(
            self.test_loader,
            total=len(self.test_loader),
            desc="calculating reconstruction errors",
            leave=True
        )

        anomaly_scores = []
        for i, batch_data in enumerate(eval_iterator):
            X = batch_data[0].to(self.args.device)
            B, L, C = X.shape
            anomaly_score, loss_dict = self.model(X)
            anomaly_scores.append(anomaly_score)

        anomaly_scores = torch.cat(anomaly_scores, dim=0)
        init_pred = anomaly_scores[0].repeat(self.args.window_size-1)
        anomaly_scores = torch.cat((init_pred, anomaly_scores)).cpu().numpy()
        return anomaly_scores