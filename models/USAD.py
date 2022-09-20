import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, latent_space_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, input_size // 2)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(input_size // 2, input_size // 4)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(input_size // 4, latent_space_size)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        out = self.relu3(x)
        return out

class Decoder(nn.Module):
    def __init__(self, input_size, latent_space_size):
        super().__init__()
        self.linear1 = nn.Linear(latent_space_size, input_size // 4)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(input_size // 4, input_size // 2)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(input_size // 2, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        out = self.sigmoid(x)
        return out

class USAD(nn.Module):
    def __init__(self, input_size, latent_space_size):
        super().__init__()
        self.encoder = Encoder(input_size, latent_space_size)
        self.decoder1 = Decoder(input_size, latent_space_size)
        self.decoder2 = Decoder(input_size, latent_space_size)

    def forward(self, x):
        z = self.encoder(x)
        AE1_out, AE2_out = self.decoder1(z), self.decoder2(z)
        return AE1_out, AE2_out