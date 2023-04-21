import copy
import torch
import torch.nn as nn
from functools import partial


Conv1 = partial(nn.Conv2d, kernel_size=1, stride=1, padding=0)
Conv3 = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1)
Conv7 = partial(nn.Conv2d, kernel_size=7, stride=1, padding=0)
ReLU = partial(nn.ReLU, inplace=True)
Pool = partial(nn.MaxPool2d, kernel_size=2, stride=2)


A_ARCH = [          # Out ch
    [Conv3],        # 64
    [Conv3],        # 128
    [Conv3, Conv3], # 256
    [Conv3, Conv3], # 512
    [Conv3, Conv3], # 512
]


# Local Response Normalization
# https://dl.acm.org/doi/pdf/10.1145/3065386
class LRN(nn.Module):
    def __init__(self, dim, k=2, n=5, alpha=1e-4, beta=0.75):
        super().__init__()

        self.dim = dim
        self.k = k
        self.n = n
        self.alpha = alpha
        self.beta = beta

        self.kernel = nn.Conv2d(dim, dim, n, padding=n//2, groups=dim, bias=False)
        self.kernel.weight.requires_grad = False
        self.kernel.weight.data.fill_(1.0)

    def forward(self, x):
        a = x ** 2
        norm = self.kernel(a)
        x = x / (self.k + self.alpha * norm) ** self.beta
        return x


def make_arch(net_type):
    arch = copy.deepcopy(A_ARCH)

    if 'LRN' in net_type:
        arch[0].append(LRN)
        return arch

    if net_type >= 'B':
        arch[0].append(Conv3)
        arch[1].append(Conv3)

    if net_type == 'C':
        arch[2].append(Conv1)
        arch[3].append(Conv1)
        arch[4].append(Conv1)

    if net_type >= 'D':
        arch[2].append(Conv3)
        arch[3].append(Conv3)
        arch[4].append(Conv3)

    if net_type == 'E':
        arch[2].append(Conv3)
        arch[3].append(Conv3)
        arch[4].append(Conv3)

    return arch


class VGG(nn.Module):
    def __init__(self, net_type: str, n_classes: int):
        super().__init__()

        self.n_classes = n_classes

        in_dim = 3
        out_dim = 64
        convnet = []
        convnet_arch = make_arch(net_type)

        for block in convnet_arch:
            for layer in block:
                if isinstance(layer, LRN):
                    convnet.append(layer(in_dim))
                else:
                    convnet.append(layer(in_dim, out_dim))
                    convnet.append(ReLU())

                in_dim = out_dim

            convnet.append(Pool())
            out_dim = min(out_dim * 2, 512)

        self.convnet = nn.Sequential(*convnet)

        self.fc = nn.Sequential(
            Conv7(512, 4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            Conv1(4096, 4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            Conv1(4096, self.n_classes),
            nn.Flatten()
        )

        self.apply(self.init_weights)

    def set_weights_with_trained_A(self, weights_typeA):
        pass

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.convnet(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    net_types = ['A', 'A-LRN', 'B', 'C', 'D', 'E']

    for nt in net_types:
        model = VGG(nt, 1000)
        input = torch.rand(1, 3, 224, 224)
        output = model(input)
        params = sum([m.numel() for m in model.parameters()])
        print(f'Model: {nt}, Params: {params/1e+6}M')

