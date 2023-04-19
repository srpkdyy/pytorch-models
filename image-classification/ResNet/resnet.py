import torch.nn as nn


ARCH_CFG = {
    '18': [2, 2, 2, 2],
    '34': [3, 4, 6, 3],
    '50': [3, 4, 6, 3],
    '101': [3, 4, 23, 3],
    '152': [3, 8, 36, 3],
}


class BasicBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
        )
        self.relu = nn.ReLU()

        if in_dim == out_dim:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_dim, out_dim, 1)

    def forward(self, x):
        h = self.block(x)
        y = self.relu(h + self.shortcut(x))
        return y


class Bottleneck(BasicBlock):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim)

        btn_dim = out_dim // 4

        self.block = nn.Sequential(
            nn.Conv2d(in_dim, btn_dim, 1),
            nn.BatchNorm2d(btn_dim),
            nn.ReLU(),
            nn.Conv2d(btn_dim, btn_dim, 3, padding=1),
            nn.BatchNorm2d(btn_dim),
            nn.ReLU(),
            nn.Conv2d(btn_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
        )


class ResNet(nn.Module):
    def __init__(self, arch_type='18'):
        super().__init__()

        self.init_conv = nn.Conv2d(3, 64, 7, stride=2, padding=3),
        self.pool = nn.MaxPool2d(3, stride=2, padding=1),

        cfg = ARCH_CFG[arch_type]
        if arch_type in ('18', '34'):
            Block = BasicBlock
            in_dim, out_dim = 64, 64
        else:
            Block = Bottleneck
            in_dim, out_dim = 64, 256

        convx = []

        for n_layers in cfg:
            for _ in range(n_layers):
                convx.append(Block(in_dim, out_dim))
                in_dim = out_dim

            out_dim = in_dim * 2

        self.convx = nn.Sequential(*convx)

        self.avgpool = nn.AvgPool2d((7, 7))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_dim, 1000)


    def init_weights(self):
        # ref 13.
        pass

    def forward(self, x):
        x = self.init_conv(x)
        x = self.pool(x)

        x = self.convx(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    for arch in ('18', '34', '50', '101', '152'):
        model = ResNet(arch)
        print(f'Type: {arch}, Params: {sum([p.numel() for p in model.parameters()])}')
