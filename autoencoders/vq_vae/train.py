import os
import dataclasses

import torch
import accelerate
import torch.optim as optim
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms as TF
from torchvision.datasets import ImageNet
from torchvision.utils import make_grid, save_image

from vq_vae import VQVAE


torch.backends.cudnn.benchmark = True
accelerate.utils.set_seed(42)


def cycle(dl):
    while True:
        for data in dl:
            yield data


@dataclasses.dataclass(frozen=True)
class Config:
    # Dataset
    root: str = 'data'
    batch_size: int = 128
    resize: int = 128
    crop_size: int = 128
    workers: int = 8

    # Loss
    beta: float = 0.25

    # Optimizer
    lr: float = 2e-4
    
    # Model
    K: int = 512
    dim: int = 256

    # Training
    steps: int = 250000

    # Logging
    logdir: str = 'results'
    log_interval_steps: int = 10000
    nrow: int = 4


cfg = Config()


def main():
    os.makedirs(cfg.logdir, exist_ok=True)

    ar = Accelerator(log_with='wandb')
    ar.init_trackers('VQ-VAE', config=cfg)

    device = ar.device
    n_devices = ar.num_processes

    transform = TF.Compose([
        TF.Resize(cfg.resize),
        TF.CenterCrop(cfg.crop_size),
        TF.ToTensor(),
        TF.Normalize([0.5], [0.5])
    ])
    train_ds = ImageNet(cfg.root, 'train', transform)
    valid_ds = ImageNet(cfg.root, 'val', transform)

    dl_kwargs = {
        'batch_size': cfg.batch_size,
        'num_workers': cfg.workers,
        'pin_memory': True
    }
    train_dl = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    valid_dl = DataLoader(valid_ds, shuffle=False, **dl_kwargs)

    model = VQVAE(k_cat=cfg.K, dim=cfg.dim, beta=cfg.beta)

    optimizer = optim.Adam(model.parameters(), cfg.lr * n_devices)

    train_dl, valid_dl, model, optimizer = ar.prepare(
        train_dl, valid_dl, model, optimizer
    )

    train_dl = cycle(train_dl)

    steps = cfg.steps // n_devices
    for step in tqdm(range(steps), disable=not ar.is_local_main_process):

        model.train()

        imgs, _ = next(train_dl)

        outs = model(imgs)

        loss = F.mse_loss(imgs, outs['recon']) + outs['vq_loss']

        optimizer.zero_grad()
        ar.backward(loss)
        optimizer.step()

        if step % (cfg.log_interval_steps // n_devices) == 0:
            train_loss = ar.gather(loss).mean().item()

            model.eval()

            with torch.no_grad():
                imgs, _ = next(iter(valid_dl))

                outs = model(imgs)

                loss = F.mse_loss(imgs, outs['recon']) + outs['vq_loss']

                recon = ar.gather_for_metrics(outs['recon'])
                valid_loss = ar.gather_for_metrics(loss).mean().item()

            if ar.is_local_main_process:
                recon = recon[:cfg.nrow ** 2]
                save_image(
                    recon, f'{cfg.logdir}/recon_{step}.png',
                    nrow=cfg.nrow, normalize=True, value_range=(-1, 1)
                )

            log = {
                'Train loss': train_loss,
                'Valid loss': valid_loss
            }
            ar.log(log, step=step)

    ar.wait_for_everyone()
    ar.save(ar.unwrap_model(model).state_dict(), 'vqvae.pth')
    ar.end_training()


if __name__ == '__main__':
    main()

