import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


@torch.no_grad()
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.normal_(m.weight, 0.0, 0.01)


class UpBlock2D(nn.Module):
    def __init__(self, in_dim, out_dim, is_first_block=False):
        super().__init__()

        pad = 0 if is_first_block else 1

        self.conv = nn.ConvTranspose2d(in_dim, out_dim, 4, 2, pad, bias=False)
        self.norm = nn.BatchNorm2d(out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class UpBlock3D(nn.Module):
    def __init__(self, in_dim, out_dim, is_first_block=False):
        super().__init__()

        ksize = (2, 4, 4) if is_first_block else (4, 4, 4)
        pad = 0 if is_first_block else 1

        self.conv = nn.ConvTranspose3d(in_dim, out_dim, ksize, 2, pad, bias=False)
        self.norm = nn.BatchNorm3d(out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()

        dims = [z_dim, 512, 256, 128, 64]

        ups_2d = [Rearrange('b c -> b c 1 1')]
        ups_3d = [Rearrange('b c -> b c 1 1 1')]
        for i in range(len(dims) - 1):
            in_d, out_d = dims[i:i+2]
            ups_2d.append(UpBlock2D(in_d, out_d, i==0))
            ups_3d.append(UpBlock3D(in_d, out_d, i==0))
        self.ups_2d = nn.Sequential(*ups_2d)
        self.ups_3d = nn.Sequential(*ups_3d)

        self.background = nn.Sequential(
            UpBlock2D(dims[-1], 3),
            nn.Tanh(),
        )
        self.foreground = nn.Sequential(
            UpBlock3D(dims[-1], 3),
            nn.Tanh()
        )
        self.mask = nn.Sequential(
            UpBlock3D(dims[-1], 1),
            nn.Sigmoid()
        )

        self.apply(init_weights)

    def forward(self, z):
        b = self.background(self.ups_2d(z))
        b = b.unsqueeze(2)

        h_3d = self.ups_3d(z)
        f = self.foreground(h_3d)
        m = self.mask(h_3d)

        out = m * f + (1 - m) * b
        return out


class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv = nn.Conv3d(in_dim, out_dim, 4, 2, 1, bias=False)
        self.norm = nn.BatchNorm3d(out_dim)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super().__init__()

        dims = [channels, 64, 128, 256, 512]

        downs = []
        for i in range(len(dims) - 1):
            downs.append(DownBlock(dims[i], dims[i+1]))
        self.downs = nn.Sequential(*downs)

        self.classifier = nn.Sequential(
            nn.Conv3d(dims[-1], 1, (2, 4, 4), 2, bias=False),
            nn.Flatten()
        )

        self.apply(init_weights)

    def forward(self, x):
        x = self.downs(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    g = Generator()
    d = Discriminator()

    import torch
    z = torch.rand(1, 100)
    out = g(z)
    print('z', out.shape)

    img = torch.rand(1, 3, 32, 64, 64)
    out = d(img)
    print('real/fake', out.shape)


