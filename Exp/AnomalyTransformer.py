# Trainer
from Exp.Trainer import Trainer

# models
from models.AnomalyTransformer import AnomalyTransformer

# utils
from utils.metrics import get_statistics
from utils.metrics import PA
from utils.optim import adjust_learning_rate
from utils.custom_loss import my_kl_loss


# others
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os

from tqdm import tqdm
import pickle


class AnomalyTransformer_Trainer(Trainer):
    def __init__(self, args, train_loader, test_loader):
        super(AnomalyTransformer_Trainer, self).__init__(args=args, train_loader=train_loader, test_loader=test_loader)
        self.model = AnomalyTransformer(
            win_size=self.win_size,
            enc_in=args.num_channels,
            c_out=args.num_channels,
            e_layers=3,
        ).to(self.args.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=args.lr)
        self.num_epochs = self.args.epochs
        self.device = self.args.device
        self.threshold_function_map = {
            "oracle": self.oracle_thresholding,
            "anomaly_transformer": self.anomaly_transformer_thresholding,
        }

    # code referrence:
    # https://github.com/thuml/Anomaly-Transformer/blob/72a71e5f0847bd14ba0253de899f7b0d5ba6ee97/solver.py#L130
    def train(self):
        self.model.train()
        time_now = time.time()
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []
            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} ".format(
                    epoch + 1, train_steps, train_loss))

            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    # code referrence:
    # https://github.com/thuml/Anomaly-Transformer/blob/72a71e5f0847bd14ba0253de899f7b0d5ba6ee97/solver.py#L207
    @torch.no_grad()
    def infer(self):
        result = {}
        self.model.eval()
        gt = self.test_loader.dataset.y
        anomaly_scores = self.calculate_anomaly_scores()

        # F1
        threshold = self.get_threshold(gt=gt, anomaly_scores=anomaly_scores)
        pred = (anomaly_scores > threshold).astype(int)
        cm, a, p, r, f1 = get_statistics(gt, pred)
        result.update({
            "Threshold": threshold,
            "Confusion Matrix": cm.ravel(),
            "Precision": p,
            "Recall": r,
            "F1": f1,
        })

        # F1-PA
        threshold = self.get_threshold(gt=gt, anomaly_scores=anomaly_scores, point_adjust=True)
        pred = (anomaly_scores > threshold).astype(int)
        pa_pred = PA(gt, pred)
        cm, a, p, r, f1 = get_statistics(gt, pa_pred)
        result.update({
            "Threshold (PA)": threshold,
            "Confusion Matrix (PA)": cm.ravel(),
            "Precision (PA)": p,
            "Recall (PA)": r,
            "F1 (PA)": f1,
        })

        return result

    def calculate_anomaly_scores(self, temperature, criterion):
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.test_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)

            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        return test_energy

    def get_threshold(self, gt, anomaly_scores, point_adjust=False):
        print(f"thresholding with algorithm: {self.args.thresholding}")
        return self.threshold_function_map[self.args.thresholding](gt, anomaly_scores, point_adjust)

    @torch.no_grad()
    def anomaly_transformer_thresholding(self, gt, anomaly_scores, point_adjust):
        temperature = 50
        criterion = nn.MSELoss(reduce=False)

        # (1) statistics on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

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
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            # Metric
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        return threshold