import torch
import torch.nn as nn
from torch.autograd import Variable

class PlanarNormalizingFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.u = Variable(torch.randn(1), requires_grad = True)
        self.linear = nn.Linear(dim, dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = x + self.u.to(x.device) * self.tanh(self.linear(x))
        return out


class Qnet(nn.Module):
    def __init__(self, in_dim, hidden_dim, z_dim, dense_dim):
        super(Qnet, self).__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.z_dim = z_dim
        self.e_dim = hidden_dim
        self.dense_dim = dense_dim
        self.dense = nn.Linear(self.z_dim + self.e_dim, self.dense_dim)
        self.linear_mu = nn.Linear(self.dense_dim, self.z_dim)
        self.linear_sigma = nn.Linear(self.dense_dim, self.z_dim)
        self.softplus = nn.Softplus()
        self.pnf = PlanarNormalizingFlow(dim=self.z_dim)

    def forward(self, x):
        B, W, F = x.shape

        z = torch.zeros(B, W, self.z_dim).to(x.device)
        mu, logvar = torch.zeros(B, W, self.z_dim).to(x.device), torch.zeros(B, W, self.z_dim).to(x.device)

        e_t = None
        z_t = torch.zeros(B, 1, self.z_dim).to(x.device)
        for t in range(W):
            # GRU
            x_t = x[:, t, :].unsqueeze(1)
            _, e_t = self.gru(x_t, e_t)

            # Dense
            dense_input = torch.cat((z_t, torch.transpose(e_t, 0, 1)),  axis = 2)
            dense_output = self.dense(dense_input)

            # mu and sigma sampling
            mu_t, logvar_t = self.linear_mu(dense_output), self.softplus(self.linear_sigma(dense_output))
            mu[:, t, :], logvar[:, t, :] = mu_t.squeeze(1), logvar_t.squeeze(1)
            std = torch.exp(0.5 * logvar_t)
            z_t = mu_t + std * torch.rand_like(logvar_t)

            # latent z
            z_t = self.pnf(z_t)
            z[:, t, :] = z_t.squeeze(1)

        return z, mu, logvar


class Pnet(nn.Module):
    def __init__(self, z_dim, hidden_dim, dense_dim, out_dim):
        super(Pnet, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.dense_dim = dense_dim
        self.out_dim = out_dim

        self.gru = nn.GRU(z_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.LGSSM = nn.Linear(z_dim, z_dim)
        self.dense = nn.Linear(self.hidden_dim, self.dense_dim)
        self.linear_mu = nn.Linear(self.dense_dim, self.out_dim)
        self.linear_sigma = nn.Linear(self.dense_dim, self.out_dim)
        self.softplus = nn.Softplus()


    def forward(self, z):
        B, W, F = z.shape
        out = torch.zeros(B, W, self.out_dim).to(z.device)
        d_t = None

        for t in range(W):
            z_t_1 = z[:, t, :].unsqueeze(1)
            z_t = self.LGSSM(z_t_1)
            _, d_t = self.gru(z_t, d_t)

            dense_output = self.dense(torch.transpose(d_t, 0, 1))
            mu, sigma = self.linear_mu(dense_output), self.softplus(self.linear_sigma(dense_output))
            out_t = mu + sigma * torch.rand_like(sigma)

            out[:, t, :] = out_t.squeeze(1)

        return out


class OmniAnomaly(nn.Module):
    def __init__(self, in_dim, hidden_dim, z_dim, dense_dim, out_dim):
        super(OmniAnomaly, self).__init__()
        self.qnet = Qnet(in_dim=in_dim, hidden_dim=hidden_dim, z_dim=z_dim, dense_dim=dense_dim)
        self.pnet = Pnet(z_dim=z_dim, hidden_dim=hidden_dim, dense_dim=dense_dim, out_dim=out_dim)

    def forward(self, x):
        z, mu, logvar = self.qnet(x)
        out = self.pnet(z)
        return out, mu, logvar

if __name__ == "__main__":
    batch_size = 64
    window_size = 3
    channel_size = 51

    dense_dim = 512
    z_dim = 256


    q = Qnet(in_dim = channel_size, hidden_dim = 1024, z_dim = z_dim, dense_dim = dense_dim)
    p = Pnet(z_dim = z_dim, hidden_dim=1024, dense_dim = dense_dim, out_dim = channel_size)

    sample_x = torch.randn(batch_size, window_size, channel_size)
    z = q(sample_x)
    x_recon = p(z)

