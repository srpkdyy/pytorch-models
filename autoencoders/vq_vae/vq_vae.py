import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, std=0.01)
        nn.init.constant_(m.bias, 0)


class VQVAE(nn.Module):
    def __init__(self, channels=3, k_cat=512, dim=256, beta=0.25):
        super().__init__()

        self.encoder = nn.Sequential(
            Downsampler(channels, dim),
            nn.ReLU(),
            Downsampler(dim, dim),
            ResBlock(dim),
            ResBlock(dim)
        )

        self.quantizer = Quantizer(k_cat, dim, beta)

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            Upsampler(dim, dim),
            nn.ReLU(),
            Upsampler(dim, channels)
        )

        self.apply(init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x, loss = self.quantizer(x)
        x = self.decoder(x)

        return {'recon': x, 'vq_loss': loss}


class Downsampler(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv = nn.Conv2d(in_dim, out_dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsampler(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv = nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x):
        return x + self.block(x)


class Quantizer(nn.Module):
    def __init__(self, k_cat, dim, beta):
        super().__init__()

        self.emb = nn.Embedding(k_cat, dim)
        self.beta = beta

        nn.init.uniform_(self.emb.weight, -1/k_cat, 1/k_cat)

    def forward(self, x):
        b, c, h, w = x.shape

        ze = rearrange(x, 'b c h w -> (b h w) c').contiguous()
        ej = self.emb.weight

        with torch.no_grad():
            l2 = ze.pow(2).sum(-1, keepdims=True) + \
                 ej.pow(2).sum(-1) - 2 * \
                 torch.einsum('bc,kc->bk', ze, ej)
            
        nearest_ids = l2.argmin(dim=-1)

        zq = self.emb(nearest_ids)
        zq = rearrange(zq, '(b h w) c -> b c h w', b=b, h=h).contiguous()

        vq_loss = F.mse_loss(x.detach(), zq) + self.beta * F.mse_loss(x, zq.detach())

        zq = (zq - x).detach() + x

        return zq, vq_loss


if __name__ == '__main__':
    import torch
    vqvae = VQVAE()
    x = torch.rand(1, 3, 224, 224)
    print(vqvae(x))

