import os
import itertools
import dataclasses

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision import transforms as TF
from torchvision.datasets import LSUN
from torchvision.utils import make_grid

from dcgan import Generator, Discriminator


torch.backends.cudnn.benchmark = True


@dataclasses.dataclass(frozen=True)
class Config:
    # Dataset
    root: str = 'data'
    batch_size: int = 128
    img_size: int = 64
    workers: int = 4

    # Optimizer
    lr: float = 2e-4
    beta1: float = 0.5

    # Model
    z_dim: int = 100

    # Training
    epoch: int = 1

    # Log
    nrow: int = 8

cfg = Config()


def main():
    ar = Accelerator(log_with='wandb')
    ar.init_trackers('DCGAN', config=cfg)

    device = ar.device
    lr = cfg.lr * ar.num_processes

    ds = LSUN(
        cfg.root,
        ['bedroom_train'],
        transform=TF.Compose([
            TF.Resize(cfg.img_size),
            TF.CenterCrop(cfg.img_size),
            TF.ToTensor(),
            TF.Normalize([0.5]*3, [0.5]*3)
        ])
    )
    dl = DataLoader( 
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True
    )

    G = Generator(cfg.z_dim, cfg.img_size).to(device)
    D = Discriminator(cfg.z_dim, cfg.img_size).to(device)

    criterion = nn.BCELoss().to(device)
    G_optimizer = optim.Adam(G.parameters(), lr, betas=(cfg.beta1, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr, betas=(cfg.beta1, 0.999))

    dl, G, D, G_optimizer, D_optimizer = ar.prepare(
        dl, G, D, G_optimizer, D_optimizer
    )

    real_label = torch.ones(cfg.batch_size, device=device)
    fake_label = torch.zeros(cfg.batch_size, device=device)

    for epoch in range(cfg.epoch):
        G.train()
        D.train()

        for i, (imgs, label) in enumerate(dl):
            # Update Discriminator with Real
            D_real_outputs = D(imgs)

            D_optimizer.zero_grad()
            D_real_loss = criterion(D_real_outputs, real_label)
            ar.backward(D_real_loss)

            # Update Discriminator with Fake
            noise = 2 * torch.rand(cfg.batch_size, cfg.z_dim, device=device) - 1
            D_fake_outputs = D(G(noise).detach())

            D_fake_loss = criterion(D_fake_outputs, fake_label)
            ar.backward(D_fake_loss)
            D_optimizer.step()

            # Update Generator
            noise = 2 * torch.rand(cfg.batch_size, cfg.z_dim, device=device) - 1
            samples = G(noise)
            outputs = D(samples)

            G_optimizer.zero_grad()
            G_loss = criterion(outputs, real_label)
            ar.backward(G_loss)
            G_optimizer.step()

            log = {
                'G_loss': G_loss.mean().item(),
                'D_real_loss': D_real_loss.mean().item(),
                'D_fake_loss': D_fake_loss.mean().item()
            }
            ar.print(f'Iter:{i}, ', log)

        log.update({
            'Generated samples': wandb.Image(make_grid(
                samples[:cfg.nrow**2], nrow=cfg.nrow,
                value_range=(-1, 1), normalize=True
            ))
        })
        ar.log(log, step=epoch)

    ar.end_training()


if __name__ == '__main__':
    main()

