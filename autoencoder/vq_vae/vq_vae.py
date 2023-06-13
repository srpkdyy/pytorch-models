import torch.nn as nn


class VQVAE(nn.Module):
    def __init__(self, channels=3, k_cat=512, dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            Downsampler(channels, dim),
            Downsampler(dim, dim),
            ResBlock(dim),
            ResBlock(dim)
        )

        self.quantizer = Quantizer(k_cat, dim)

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            Upsampler(dim, dim),
            Upsampler(dim, channels)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.quantizer(x)
        x = self.decoder(x)

        return x


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
    def __init__(self, k_cat, dim):
        super().__init__()

        self.emb = nn.Embedding(k_cat, dim)

    def forward(self, x):
        return x

