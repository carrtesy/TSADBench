'''
LSTMVAE
Daehyung Park, Yuuna Hoshi, Charles C. Kemp:
A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-Based Variational Autoencoder. IEEE Robotics Autom. Lett. 3(2): 1544-1551 (2018)

Implementation by Dongmin Kim (tommy.dm.kim@gmail.com)
repo available at: https://github.com/carrtesy/LSTMVAE-Pytorch
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LSTMVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, n_layers):
        super(LSTMVAE, self).__init__()

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )
        self.mu, self.logvar = nn.Linear(hidden_dim, z_dim), nn.Linear(hidden_dim, z_dim)

        self.decoder = nn.LSTM(
            input_size=z_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )
        self.final = nn.Linear(hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.rand_like(std)
        z = mu + noise * std
        return z

    def forward(self, X):
        '''
        :param X: (B, W, C)
        :return: Losses,
        '''
        B, W, C = X.shape
        enc, _ = self.encoder(X)
        mu, logvar = self.mu(enc), self.logvar(enc)
        z = self.reparameterize(mu, logvar)
        dec, _ = self.decoder(z)
        out = self.final(dec)
        return out, mu, logvar

if __name__ == "__main__":
    B, L, C = 64, 12, 51
    ip = torch.randn((B, L, C))
    model = LSTMVAE(input_dim=C, hidden_dim=128, z_dim=3, n_layers=2)
    model(ip)