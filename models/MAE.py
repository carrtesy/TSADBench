import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class MAE(nn.Module):
    def __init__(self, input_length, num_channels,
                 enc_dim, dec_dim,
                 enc_num_layers, dec_num_layers,
                 mask_ratio):
        super(MAE, self).__init__()

        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.enc_embed = nn.Linear(num_channels, enc_dim)
        self.dec_embed = nn.Linear(enc_dim, dec_dim)

        self.time_embedding = nn.Embedding(input_length, enc_dim)
        self.mask_token = -1# -torch.ones(1, 1, dec_dim)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.enc_dim,
                nhead=4,
            ),
            num_layers=enc_num_layers,
        )

        self.decoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.dec_dim,
                nhead=4,
            ),
            num_layers=dec_num_layers,
        )
        self.mask_ratio = mask_ratio
        self.final = nn.Linear(dec_dim, num_channels)

    def encode(self, x):
        B, L, C = x.shape
        x = self.enc_embed(x)
        x += self.time_embedding.weight.repeat(B, 1, 1)  # Positional Embedding (B, L, D)
        latent = self.encoder(x)
        return latent

    def decode(self, x):
        x = self.decoder(x)
        out = self.final(x)
        return out

    def forward(self, x):
        B, L, C = x.shape
        latent = self.encode(x)
        out = self.decode(latent)
        return out


class MAEv1(nn.Module):
    def __init__(self, input_length, num_channels, enc_dim, dec_dim, mask_ratio):
        super(MAE, self).__init__()

        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.enc_embed = nn.Linear(num_channels, enc_dim)
        self.dec_embed = nn.Linear(enc_dim, dec_dim)

        self.time_embedding = nn.Embedding(input_length, enc_dim)
        #self.time_interpolator = Interpolator(in_dim=int(input_length*(1-mask_ratio)), out_dim=input_length)
        self.mask_token = -torch.ones(1, 1, dec_dim)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.enc_dim,
                nhead=4,
            ),
            num_layers=1,
        )
        self.final = nn.Linear(dec_dim, num_channels)

        self.decoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.dec_dim,
                nhead=4,
            ),
            num_layers=1,
        )
        self.mask_ratio = mask_ratio

    def encode(self, x):
        B, L, C = x.shape
        x = self.enc_embed(x)
        x += self.time_embedding.weight.repeat(B, 1, 1) # Positional Embedding (B, L, D)
        x_masked, mask, ids_restore = self.random_time_masking(x, self.mask_ratio)
        latent = self.encoder(x_masked)
        return latent, mask, ids_restore

    def restore(self, x, ids_restore):
        # project to decoder embedding space
        x = self.dec_embed(x)

        # append mask tokens to sequence
        B, L_masked, dec_dim = x.shape[0], ids_restore.shape[1] - x.shape[1], x.shape[2]
        mask_tokens = self.mask_token.repeat(x.shape[0], L_masked, 1).to(x.device)
        x_ = torch.cat([x, mask_tokens], dim=1)
        out = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, dec_dim))

        return out
        #return x, out

    def decode(self, x):
        x = self.decoder(x)
        out = self.final(x)
        return out

    def forward(self, x):
        B, L, C = x.shape
        latent, mask, ids_restore = self.encode(x)

        #proj_latent, restored_latent = self.restore(latent, ids_restore)
        masked_latent = self.restore(x)

        #interpolated_latent = self.time_interpolator(proj_latent.transpose(1, -1)).transpose(1, -1)
        #latent_interpolation_loss = self.masked_mse(interpolated_latent, restored_latent, mask_val=0)


        #out = self.decode(interpolated_latent)
        return out
        #return out, latent_interpolation_loss

    @staticmethod
    def random_masking(x, mask_ratio):
        '''
        mask {mask_ratio} percent of timestep.
        :param x: (B, L, C, D)
        :param mask_ratio
        :return:
        '''
        B, L, C, D = x.shape
        x = x.reshape(B, L, C*D)
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    @staticmethod
    def random_time_masking(x, mask_ratio):
        '''
        mask {mask_ratio} percent of timestep.
        :param x: (B, T, D)
        :param mask_ratio
        :return:
        '''
        B, T, D = x.shape
        len_keep = int(T * (1 - mask_ratio))

        # sample random noise
        noise = torch.rand(B, T).to(x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, T]).to(x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        x_masked = torch.gather(x, dim=1, index=ids_keep.reshape(B, -1, 1).repeat(1, 1, D))

        return x_masked, mask, ids_restore

    @staticmethod
    def masked_mse(preds, labels, mask_val=0):
        mask = (labels != mask_val)
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = (preds - labels) ** 2
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)

class STMAE(nn.Module):
    def __init__(self, input_length, num_channels, enc_dim, dec_dim, mask_ratio):
        super(STMAE, self).__init__()

        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.enc_embed = nn.Linear(1, enc_dim)
        self.dec_embed = nn.Linear(enc_dim, dec_dim)

        self.node_embedding = nn.Embedding(num_channels, enc_dim)
        self.time_embedding = nn.Embedding(input_length, enc_dim)
        self.time_interpolator = Interpolator(in_dim=int(input_length*(1-mask_ratio)), out_dim=input_length)
        self.mask_token = torch.zeros(1, 1, dec_dim)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.enc_dim,
                nhead=4,
            ),
            num_layers=1,
        )
        self.final = nn.Linear(dec_dim, 1)

        self.decoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.dec_dim,
                nhead=4,
            ),
            num_layers=1,
        )
        self.mask_ratio = mask_ratio

    def encode(self, x, mask_ratio):
        B, L, C = x.shape
        x = self.enc_embed(x.unsqueeze(-1))
        x += self.node_embedding.weight.repeat(B, 1, 1).unsqueeze(1) # (B, 1, C, D)
        x += self.time_embedding.weight.repeat(B, 1, 1).unsqueeze(2) # (B, L, 1, D)
        #x_masked, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        x_masked, mask, ids_restore = self.random_time_masking(x, self.mask_ratio)

        latent = self.encoder(x_masked)
        return latent, mask, ids_restore

    def restore(self, x, ids_restore):
        # project to decoder embedding space
        x = self.dec_embed(x)

        # append mask tokens to sequence
        B, L_masked, dec_dim = x.shape[0], ids_restore.shape[1] - x.shape[1], x.shape[2]
        mask_tokens = self.mask_token.repeat(x.shape[0], L_masked, 1).to(x.device)
        x_ = torch.cat([x, mask_tokens], dim=1)
        out = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, dec_dim))

        return x, out

    def decode(self, x):
        x = self.decoder(x)
        out = self.final(x)
        return out

    def forward(self, x):
        B, L, C = x.shape
        latent, mask, ids_restore = self.encode(x, mask_ratio=self.mask_ratio)

        proj_latent, restored_latent = self.restore(latent, ids_restore)
        restored_latent = restored_latent.reshape(B, L, C, self.dec_dim)

        proj_latent = proj_latent.reshape(B, int((1-self.mask_ratio)*L), C, self.dec_dim) # (B, (1-mask_ratio)*L*C, D) =>  (B, (1-mask_ratio)*L, C, D)
        interpolated_latent = self.time_interpolator(proj_latent.transpose(1, -1)).transpose(1, -1)
        latent_interpolation_loss = self.masked_mse(interpolated_latent, restored_latent, mask_val=0)

        interpolated_latent = interpolated_latent.reshape(B, L*C, self.dec_dim)
        out = self.decode(interpolated_latent)
        out = out.reshape(B, L, C)
        return out, latent_interpolation_loss

    @staticmethod
    def random_masking(x, mask_ratio):
        '''
        mask {mask_ratio} percent of timestep.
        :param x: (B, L, C, D)
        :param mask_ratio
        :return:
        '''
        B, L, C, D = x.shape
        x = x.reshape(B, L, C*D)
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    @staticmethod
    def random_time_masking(x, mask_ratio):
        '''
        mask {mask_ratio} percent of timestep.
        :param x: (B, T, C, D)
        :param mask_ratio
        :return:
        '''
        B, T, C, D = x.shape
        len_keep = int(T * (1 - mask_ratio))

        # sample random noise
        noise = torch.rand(B, T).to(x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, T]).to(x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # repeat spatially
        mask = torch.repeat_interleave(mask, repeats=C, dim=1)

        # get spatially extended restoring index
        spatial_ids = torch.repeat_interleave(ids_restore, repeats=C, dim=1) * C
        ids_restore = spatial_ids + torch.arange(C).reshape(1, -1).repeat(B, T).to(x.device)

        x_masked = torch.gather(x, dim=1, index=ids_keep.reshape(B, -1, 1, 1).repeat(1, 1, C, D))
        x_masked = rearrange(x_masked, 'n l_k t d -> n (l_k t) d')

        return x_masked, mask, ids_restore

    @staticmethod
    def masked_mse(preds, labels, mask_val=0):
        mask = (labels != mask_val)
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = (preds - labels) ** 2
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)


class Interpolator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim // 4)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(out_dim // 4, out_dim // 2)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(out_dim // 2, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        out = self.linear3(x)
        return out
