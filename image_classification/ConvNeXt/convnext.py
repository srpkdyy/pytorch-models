import torch.nn as nn
from functools import partial
from einops.layers.torch import Rearrange


ARCH_CFG = {
    'T': {
        'C': [96, 192, 384, 768],
        'B': [3, 3, 9, 3],
    },
    'S': {
        'C': [96, 192, 384, 768],
        'B': [3, 3, 27, 3],
    },
    'B': {
        'C': [128, 256, 512, 1024],
        'B': [3, 3, 27, 3],
    },
    'L': {
        'C': [192, 384, 768, 1536],
        'B': [3, 3, 27, 3],
    },
    'XL': {
        'C': [256, 512, 1024, 2048],
        'B': [3, 3, 27, 3],
    },
}


Conv2d = partial(nn.Conv2d, bias=True)


class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.down = nn.Sequential(
            Rearrange('b c h w -> b h w c'),
            nn.LayerNorm(in_dim),
            Rearrange('b h w c -> b c h w'),
            Conv2d(in_dim, out_dim, 2, 2),
        )

    def forward(self, x):
        return self.down(x)


class ConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        mid_dim = dim * 4

        self.block = nn.Sequential(
            Conv2d(dim, dim, 7, padding=3, groups=dim),
            Rearrange('b c h w -> b h w c'),
            nn.LayerNorm(dim),
            Rearrange('b h w c -> b c h w'),

            Conv2d(dim, mid_dim, 1),
            nn.GELU(),

            Conv2d(mid_dim, dim, 1),
        )


    def forward(self, x):
        out = self.block(x) + x
        return out


class ConvNeXt(nn.Module):
    def __init__(self, arch_type='T', n_classes=1000):
        super().__init__()

        cfg = ARCH_CFG[arch_type]
        n_blocks = cfg['B']
        dims = cfg['C']
        dim = dims[0]

        self.stem = nn.Sequential(
            Conv2d(3, dim, 4, stride=4),
            Rearrange('b c h w -> b h w c'),
            nn.LayerNorm(dim),
            Rearrange('b h w c -> b c h w'),
        )

        blocks = []

        for stage, n_blk in enumerate(n_blocks):
            for i in range(n_blk):
                if i == 0 and stage > 0:
                    blocks.append(Downsample(dim, dims[stage]))
                    dim = dims[stage]

                blocks.append(ConvBlock(dim))

        self.blocks = nn.Sequential(*blocks)

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(dim),
            nn.Linear(dim, n_classes)
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.output(x)
        return x


if __name__ == '__main__':
    import torch
    inputs = torch.rand(1, 3, 224, 224)
    model = ConvNeXt()
    print(model)
    print(model(inputs).shape)
    print(f'ConvNeXt-T params: {sum([p.numel() for p in model.parameters()])}')

