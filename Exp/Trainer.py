import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pickle
from metrics import get_statistics

from Models.USAD import USAD

class USAD_Trainer:
    def __init__(self, args):
        self.args = args
        self.model = USAD(
            input_size=args.input_dim,
            latent_space_size=args.latent_dim,
        ).to(self.args.device)
        self.optimizer1 = torch.optim.Adam(params=self.model.parameters(), lr=args.lr)
        self.optimizer2 = torch.optim.Adam(params=self.model.parameters(), lr=args.lr)
        self.epoch = 0

    def train(self, dataloader):
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

    def checkpoint(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def save_dictionary(self, dictionary, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(dictionary, f)

    def load(self, filepath):
        self.model = self.model.load_state_dict(torch.load(filepath))
        self.model.to(self.args.device)

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
    def infer(self, dataloader):
        self.model.eval()
        eval_iterator = tqdm(
            dataloader,
            total=len(dataloader),
            desc="infer",
            leave=True
        )

        # recon errors: returns (B, L, C) recon loss tensor
        recon_errors = []
        for i, batch_data in enumerate(eval_iterator):
            batch_data = batch_data[0].to(self.args.device)
            recon_error = self.calculate_recon_error(batch_data, self.args.alpha, self.args.beta).to("cpu")
            recon_errors.append(recon_error)
        recon_errors = np.concatenate(recon_errors, axis=0)

        # final anomaly score: returns (Number of total timesteps, ) tensor
        anomaly_scores = self.recon_errors_to_anomaly_scores(recon_errors, mode="mean")

        out = {
            "anomaly_scores": anomaly_scores,
            "recon_errors": recon_errors,
        }
        return out

    def calculate_recon_error(self, Wt, alpha=0.5, beta=0.5):
        '''
        :param Wt: model input
        :param alpha: low detection sensitivity
        :param beta: high detection sensitivity
        :return: recon error (B, L, C)
        '''
        z = self.model.encoder(Wt)
        Wt1p = self.model.decoder1(z)
        Wt2dp = self.model.decoder2(self.model.encoder(Wt1p))
        return alpha * F.mse_loss(Wt, Wt1p, reduce=False) + beta * F.mse_loss(Wt, Wt2dp, reduce=False)

    def recon_errors_to_anomaly_scores(self, recon_errors, mode="mean"):
        B, L, C = recon_errors.shape
        out = np.zeros((B+L-1, L))
        if mode=="mean":
            x = recon_errors.mean(axis=-1) # (B, L, C) => (B, L)
            for i in range(L):
                out[i: i+B, i] = x[:, i]
            out = np.true_divide(out.sum(axis=1), (out!=0).sum(axis=1)) # get mean except zeros
        elif mode=="max":
            x = recon_errors.max(axis=-1)  # (B, L, C) => (B, L)
            for i in range(L):
                out[i: i+B, i] = x[:, i]
            out = out.max(axis=-1)
        else:
            raise NotImplementedError
        return out

    def get_best_f1(self, y, anomaly_scores, samples=100):
        best_result = None
        m, M = anomaly_scores.min(), anomaly_scores.max()
        for threshold in np.linspace(m, M, num=1000):
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