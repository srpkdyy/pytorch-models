import os
import dataclasses
from accelerate import Accelerator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as TF
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

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
    logdir: str = 'logs'

cfg = Config()


def main():
    os.makedirs(cfg.logdir, exist_ok=True)

    ar = Accelerator(log_with='wandb')
    ar.init_trackers('DCGAN', config=cfg)
    device = ar.device

    ds = ImageFolder(
        cfg.root,
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
        pin_memory=True,
        drop_last=True
    )

    G = Generator(cfg.z_dim, cfg.img_size).to(device)
    D = Discriminator(cfg.z_dim, cfg.img_size).to(device)

    criterion = nn.BCELoss().to(device)
    G_optimizer = optim.Adam(G.parameters(), cfg.lr, betas=(cfg.beta1, 0.999))
    D_optimizer = optim.Adam(D.parameters(), cfg.lr, betas=(cfg.beta1, 0.999))

    dl, G, D, G_optimizer, D_optimizer = ar.prepare(
        dl, G, D, G_optimizer, D_optimizer
    )

    real_label = torch.zeros(cfg.batch_size, device=device)
    fake_label = torch.ones(cfg.batch_size, device=device)

    for epoch in range(cfg.epoch):
        for i, (imgs, label) in enumerate(dl):
            # Update Discriminator with Real
            D_real_outputs = D(imgs)

            D_optimizer.zero_grad()
            D_real_loss = criterion(D_real_outputs, real_label)
            ar.backward(D_real_loss)

            # Update Discriminator with Fake
            noise = torch.rand(cfg.batch_size, cfg.z_dim, device=device)
            D_fake_outputs = D(G(noise).detach())

            D_fake_loss = criterion(D_fake_outputs, fake_label)
            ar.backward(D_fake_loss)
            D_optimizer.step()

            # Update Generator
            noise = torch.rand(cfg.batch_size, cfg.z_dim, device=device)
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
            ar.log(log, step=i)
            ar.print(log)

    ar.end_training()
    save_image(G_outputs[:64], f'{cfg.logdir}/epoch{epoch}.png', nrow=8, value_range=(-1, 1))


if __name__ == '__main__':
    main()

