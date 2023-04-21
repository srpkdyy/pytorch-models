import torch.nn as nn
from functools import partial


Conv2d = partial(nn.Conv2d, bias=False)


ARCH_CFG = {
    '18': [2, 2, 2, 2],
    '34': [3, 4, 6, 3],
    '50': [3, 4, 6, 3],
    '101': [3, 4, 23, 3],
    '152': [3, 8, 36, 3],
}


class BasicBlock(nn.Module):
    def __init__(self, in_dim, out_dim, downsample=False):
        super().__init__()

        self.stride = 2 if downsample else 1

        self.block = nn.Sequential(
            Conv2d(in_dim, out_dim, 3, self.stride, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            Conv2d(out_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
        )
        self.relu = nn.ReLU()

        if in_dim == out_dim:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = Conv2d(in_dim, out_dim, 1, self.stride)

    def forward(self, x):
        h = self.block(x)
        y = self.relu(h + self.shortcut(x))
        return y


class Bottleneck(BasicBlock):
    def __init__(self, in_dim, out_dim, downsample):
        super().__init__(in_dim, out_dim, downsample)

        btn_dim = out_dim // 4

        self.block = nn.Sequential(
            Conv2d(in_dim, btn_dim, 1, self.stride),
            nn.BatchNorm2d(btn_dim),
            nn.ReLU(),
            Conv2d(btn_dim, btn_dim, 3, padding=1),
            nn.BatchNorm2d(btn_dim),
            nn.ReLU(),
            Conv2d(btn_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
        )


class ResNet(nn.Module):
    def __init__(self, arch_type='18', n_classes=1000):
        super().__init__()

        self.init_conv = Conv2d(3, 64, 7, stride=2, padding=3),
        self.norm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

        cfg = ARCH_CFG[arch_type]
        if arch_type in ('18', '34'):
            Block = BasicBlock
            in_dim, out_dim = 64, 64
        else:
            Block = Bottleneck
            in_dim, out_dim = 64, 256

        convx = []

        for depth, n_layers in enumerate(cfg):
            for i in range(n_layers):
                downsample = i == 0 and depth >= 1
                convx.append(Block(in_dim, out_dim, downsample))
                in_dim = out_dim

            out_dim = in_dim * 2

        self.convx = nn.Sequential(*convx)

        # Fully conv
        self.avgpool = nn.AvgPool2d((7, 7))
        self.fc = Conv2d(in_dim, n_classes, 1)
        self.flatten = nn.Flatten()

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.convx(x)

        x = self.avgpool(x)
        x = self.fc(x)
        x = self.flatten(x)
        return x


if __name__ == '__main__':
    for arch in ('18', '34', '50', '101', '152'):
        model = ResNet(arch)
        print(f'Type: {arch}, Params: {sum([p.numel() for p in model.parameters()])}')
