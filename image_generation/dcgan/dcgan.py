import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


@torch.no_grad()
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class UpBlock(nn.Module):
    def __init__(self, dim, out_dim=None):
        super().__init__()

        out_dim = out_dim or dim
        self.conv = nn.ConvTranspose2d(dim, out_dim, 4, 2, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Generator(nn.Module):
    def __init__(self, z_dim, img_size):
        super().__init__()

        self.z_dim = z_dim
        self.img_size = img_size

        self.init_layer = nn.Sequential(
            Rearrange('b c -> b c 1 1'),
            nn.ConvTranspose2d(self.z_dim, 1024, 4, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.ups = nn.Sequential(
            UpBlock(1024, 512),
            UpBlock(512, 256),
            UpBlock(256, 128)
        )
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.apply(init_weights)

    def forward(self, x):
        x = self.init_layer(x)
        x = self.ups(x)
        x = self.final_layer(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, dim, out_dim=None):
        super().__init__()

        out_dim = out_dim or dim
        self.conv = nn.Conv2d(dim, out_dim, 4, 2, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_dim)
        self.act = nn.LeakyReLU(2e-1)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Discriminator(nn.Module):
    def __init__(self, z_dim, img_size):
        super().__init__()

        self.z_dim = z_dim
        self.img_size = img_size

        self.downs = nn.Sequential(
            DownBlock(3, 128),
            DownBlock(128, 256),
            DownBlock(256, 512),
            DownBlock(512, 1024)
        )
        self.final_layer = nn.Sequential(
            nn.Conv2d(1024, 1, 4, 1, bias=False),
            nn.Flatten(start_dim=0),
            nn.Sigmoid()
        )

        self.apply(init_weights)

    def forward(self, x):
        x = self.downs(x)
        x = self.final_layer(x)
        return x

