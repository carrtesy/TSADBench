import torch
import torch.nn as nn

class MAE(nn.Module):
    def __init__(self, input_size, enc_dim, dec_dim, mask_ratio):
        super(MAE, self).__init__()

        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.enc_embed = nn.Linear(1, enc_dim)
        self.dec_embed = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.final = nn.Linear(dec_dim, 1)

        nn.init.xavier_normal_(self.mask_token)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.enc_dim,
                nhead=4,
            ),
            num_layers=1,
        )
        self.decoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.dec_dim,
                nhead=4,
            ),
            num_layers=1,
        )
        self.mask_ratio = mask_ratio

    def encode(self, x, mask_ratio):
        x_masked, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        x_masked = self.enc_embed(x_masked)
        latent = self.encoder(x_masked)
        return latent, mask, ids_restore

    def decode(self, x, ids_restore):
        # project to decoder embedding space
        x = self.dec_embed(x)

        # append mask tokens to sequence
        B, L_masked, dec_dim = x.shape[0], ids_restore.shape[1] - x.shape[1], x.shape[2]
        mask_tokens = self.mask_token.repeat(x.shape[0], L_masked, 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, dec_dim))

        # decode
        x = self.decoder(x)
        out = self.final(x)
        return out

    def forward(self, x):
        B, L, C = x.shape
        latent, mask, ids_restore = self.encode(x, mask_ratio=self.mask_ratio)
        out = self.decode(latent, ids_restore)
        out = out.reshape(B, L, C)
        return out

    def random_masking(self, x, mask_ratio):
        '''
        :param x: (B, T, C)
        :param mask_ratio:
        :return:
        '''
        B, T, C = x.shape
        L = T*C
        x = x.reshape(B, L, 1)
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore