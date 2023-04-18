import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReLU(),
        )
        self.relu = nn.ReLU()

        if in_dim == out_dim:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_dim, out_dim, 1)

    def forward(self, x):
        h = self.block(x)
        x = self.shortcut(x)
        out = self.relu(h + x)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()


