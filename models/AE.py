'''
Naive AE implementation by:
Dongmin Kim (tommy.dm.kim@gmail.com)
'''

import torch
import torch.nn as nn
from models.EncDec import MLPEncoder, MLPDecoder

class AE(nn.Module):
    def __init__(self, input_size, latent_space_size):
        super(AE, self).__init__()
        self.encoder = MLPEncoder(input_size=input_size, latent_space_size=latent_space_size)
        self.decoder = MLPDecoder(input_size=input_size, latent_space_size=latent_space_size)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out