import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
import pickle
from metrics import get_statistics

from models.USAD import USAD
from models.MAE import MAE
from sklearn.ensemble import IsolationForest


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
        self.model = self.model.load_state_dict(torch.load(filepath))
        self.model.to(self.args.device)

    @staticmethod
    def save_dictionary(dictionary, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(dictionary, f)

    @staticmethod
    def reduce(anomaly_scores, mode="mean"):
        B, L, C = anomaly_scores.shape
        out = np.zeros((B+L-1, L))
        if mode=="mean":
            x = anomaly_scores.mean(axis=-1) # (B, L, C) => (B, L)
            for i in range(L):
                out[i: i+B, i] = x[:, i]
            out = np.true_divide(out.sum(axis=1), (out!=0).sum(axis=1)) # get mean except zeros
        elif mode=="max":
            x = anomaly_scores.max(axis=-1)  # (B, L, C) => (B, L)
            for i in range(L):
                out[i: i+B, i] = x[:, i]
            out = out.max(axis=-1)
        else:
            raise NotImplementedError
        return out

    @staticmethod
    def get_best_f1(y, anomaly_scores, samples=1000):
        best_result = None
        m, M = anomaly_scores.min(), anomaly_scores.max()
        for threshold in np.linspace(m, M, num=samples):
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

class USAD_Trainer(Trainer):
    def __init__(self, args):
        super(USAD_Trainer, self).__init__(args=args)
        self.model = USAD(
            input_size=args.input_dim,
            latent_space_size=args.latent_dim,
        ).to(self.args.device)
        self.optimizer1 = torch.optim.Adam(params=self.model.parameters(), lr=args.lr)
        self.optimizer2 = torch.optim.Adam(params=self.model.parameters(), lr=args.lr)
        self.epoch = 0

    def train(self, dataset, dataloader):
        self.model.train()
        train_loss = 0.0
        train_iterator = tqdm(
            dataloader,
            total=len(dataloader),
            desc="training",
            leave=True
        )

        for i, batch_data in enumerate(train_iterator):
            loss = self._process_batch(batch_data, self.epoch+1)
            train_iterator.set_postfix({"train_loss": float(loss),})

        train_loss = train_loss / len(dataloader)
        self.epoch += 1
        return train_loss

    def _process_batch(self, batch_data, epoch):
        batch_data = batch_data[0].to(self.args.device)

        # AE1
        z = self.model.encoder(batch_data)
        Wt1p = self.model.decoder1(z)
        Wt2p = self.model.decoder2(z)
        Wt2dp = self.model.decoder2(self.model.encoder(Wt1p))

        self.optimizer1.zero_grad()
        loss_AE1 = (1 / epoch) * F.mse_loss(batch_data, Wt1p) + (1 - (1 / epoch)) * F.mse_loss(batch_data, Wt2dp)
        loss_AE1.backward()
        self.optimizer1.step()

        # AE2
        z = self.model.encoder(batch_data)
        Wt1p = self.model.decoder1(z)
        Wt2p = self.model.decoder2(z)
        Wt2dp = self.model.decoder2(self.model.encoder(Wt1p))

        self.optimizer2.zero_grad()
        loss_AE2 = (1 / epoch) * F.mse_loss(batch_data, Wt2p) - (1 - (1 / epoch)) * F.mse_loss(batch_data, Wt2dp)
        loss_AE2.backward()
        self.optimizer2.step()

        return loss_AE1 + loss_AE2

    @torch.no_grad()
    def infer(self, dataset, dataloader):
        self.model.eval()
        y = dataset.y
        anomaly_scores = self.calculate_anomaly_scores(dataloader)
        anomaly_scores = self.reduce(anomaly_scores)
        result = self.get_best_f1(y, anomaly_scores)
        return result

    def calculate_anomaly_scores(self, dataloader):
        '''
        :param dataloader: eval dataloader
        :return:  returns (B, L, C) recon loss tensor
        '''
        eval_iterator = tqdm(
            dataloader,
            total=len(dataloader),
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
        z = self.model.encoder(Wt)
        Wt1p = self.model.decoder1(z)
        Wt2dp = self.model.decoder2(self.model.encoder(Wt1p))
        return alpha * F.mse_loss(Wt, Wt1p, reduction='none') + beta * F.mse_loss(Wt, Wt2dp, reduction='none')





class MAE_Trainer(Trainer):
    def __init__(self, args):
        super(MAE_Trainer, self).__init__(args=args)
        self.model = MAE(
            input_size=args.input_dim,
            enc_dim=args.enc_dim,
            dec_dim=args.dec_dim,
            mask_ratio=args.mask_ratio,
        ).to(self.args.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=args.lr)
        self.loss_fn = torch.nn.MSELoss()

    def train(self, dataset, dataloader):
        self.model.train()
        train_loss = 0.0
        train_iterator = tqdm(
            dataloader,
            total=len(dataloader),
            desc="training",
            leave=True
        )

        for i, batch_data in enumerate(train_iterator):
            loss = self._process_batch(batch_data)
            train_iterator.set_postfix({"train_loss": float(loss),})

        train_loss = train_loss / len(dataloader)
        return train_loss

    def _process_batch(self, batch_data):
        X = batch_data[0].to(self.args.device)
        Xhat = self.model(X)
        self.optimizer.zero_grad()
        loss = self.loss_fn(Xhat, X)
        loss.backward()
        self.optimizer.step()
        return loss

    @torch.no_grad()
    def infer(self, dataset, dataloader):
        self.model.eval()
        X, y = dataset.x, dataset.y
        anomaly_scores = self.calculate_anomaly_scores(dataloader)
        anomaly_scores = self.reduce(anomaly_scores)
        result = self.get_best_f1(y, anomaly_scores)
        return result

    def calculate_anomaly_scores(self, dataloader):
        '''
        :param dataloader: eval dataloader
        :return:  returns (B, L, C) recon loss tensor
        '''
        eval_iterator = tqdm(
            dataloader,
            total=len(dataloader),
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


class IsolationForest_Trainer(Trainer):
    def __init__(self, args):
        super(IsolationForest_Trainer, self).__init__(args)
        self.model = IsolationForest()

    def train(self, dataset, dataloader):
        X = dataset.x
        self.model.fit(X)

    def infer(self, dataset, dataloader):
        X, y = dataset.x, dataset.y
        yhat = self.model.predict(X)
        cm, a, p, r, f1 = get_statistics(y, yhat)
        result = {
            "cm": cm,
            "p": p,
            "r": r,
            "f1": f1,
        }
        return result

    def checkpoint(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
