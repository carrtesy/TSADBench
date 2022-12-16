'''
Naive VAE implementation by:
Dongmin Kim (tommy.dm.kim@gmail.com)
'''

import torch
import torch.nn as nn
from models.EncDec import MLPEncoder, MLPDecoder

class VAE(nn.Module):
    def __init__(self, input_size, latent_space_size):
        super(VAE, self).__init__()
        self.encoder = MLPEncoder(input_size=input_size, latent_space_size=latent_space_size)
        self.decoder = MLPDecoder(input_size=input_size, latent_space_size=latent_space_size)
        self.mu = nn.Linear(latent_space_size, latent_space_size)
        self.logvar = nn.Linear(latent_space_size, latent_space_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def forward(self, x):
        enc_out = self.encoder(x)
        mu, logvar = self.mu(enc_out), self.logvar(enc_out)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z)
        return out, mu, logvar
