import os
import dataclasses
import itertools

import accelerate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as TF
from ezdl.datasets import VideoReader
from ezdl.utils import make_grid_video, save_gif

from vgan import Generator, Discriminator


accelerate.utils.set_seed(42)
torch.backends.cudnn.benchmark = True


@dataclasses.dataclass(frozen=True)
class Config:
    # Dataset
    root: str = 'data'
    batch_size: int = 64
    frame_size: int = 64
    workers: int = 8

    # Loss function
    lam: float = 0.1

    # Optimizer
    lr: float = 0.0002
    beta1: float = 0.5

    # Model
    z_dim: int = 100

    # Training
    epoch: int = 100

    # Log
    nrow: int = 4
    logdir = 'logs'

cfg = Config()


def main():
    os.makedirs(cfg.logdir, exist_ok=True)

    ar = accelerate.Accelerator(log_with='wandb')
    ar.init_trackers('VGAN', config=cfg)

    device = ar.device
    lr = cfg.lr * ar.num_processes

    ds = VideoReader(
        cfg.root,
        size=64,
        frames=32,
    )
    dl = DataLoader( 
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True
    )

    G = Generator(cfg.z_dim)
    D = Discriminator()

    G = nn.SyncBatchNorm.convert_sync_batchnorm(G)
    D = nn.SyncBatchNorm.convert_sync_batchnorm(D)

    criterion = nn.BCEWithLogitsLoss().to(device)
    G_optimizer = optim.Adam(G.parameters(), lr, betas=(cfg.beta1, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr, betas=(cfg.beta1, 0.999))

    dl, G, D, G_optimizer, D_optimizer = ar.prepare(
        dl, G, D, G_optimizer, D_optimizer
    )

    real_label = torch.ones(cfg.batch_size, device=device)
    fake_label = torch.zeros(cfg.batch_size, device=device)

    step = itertools.count()

    for epoch in range(cfg.epoch):
        G.train()
        D.train()

        for i, (videos, label) in enumerate(dl):
            # Update Discriminator with Real
            D_real_outputs = D(videos)

            D_optimizer.zero_grad()
            D_real_loss = criterion(D_real_outputs, real_label)
            ar.backward(D_real_loss)

            # Update Discriminator with Fake
            noise = torch.randn(cfg.batch_size, cfg.z_dim, device=device)
            samples, mask = G(noise)
            D_fake_outputs = D(samples.detach())

            D_fake_loss = criterion(D_fake_outputs, fake_label)
            ar.backward(D_fake_loss)
            D_optimizer.step()

            # Update Generator
            outputs = D(samples)

            G_optimizer.zero_grad()
            G_loss = criterion(outputs, real_label) + cfg.lam * torch.mean(mask)
            ar.backward(G_loss)
            G_optimizer.step()

            log = {
                'G_loss': ar.gather(G_loss).mean().item(),
                'D_real_loss': ar.gather(D_real_loss).mean().item(),
                'D_fake_loss': ar.gather(D_fake_loss).mean().item()
            }
            ar.log(log, step=next(step))

        ar.print(log)

        G.eval()
        with torch.no_grad():
            samples, _ = G(torch.randn(cfg.nrow**2, cfg.z_dim))

        save_gif(make_grid_video(
            samples, nrow=cfg.nrow, pad=1
        ), f'{cfg.logdir}/{epoch}.gif', fps=4)

    ar.end_training()


if __name__ == '__main__':
    main()

