import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        return z


class Upsampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


