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

from vgg import VGG


torch.backends.cudnn.benchmark = True


@dataclasses.dataclass(frozen=True)
class Config:
    # Dataset
    root: str = 'data'
    batch_size: int = 256
    resize: int = 256
    crop_size: int = 224
    workers: int = 8 

    # Optimizer
    lr: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 5e-4
    
    # Scheduler
    factor = 0.1

    # Model
    net_type: str = 'A'
    n_classes: int  = 1000

    # Training
    epoch: int = 74

    # Log
    logdir: str = 'logs'

cfg = Config()


def main():
    os.makedirs(cfg.logdir, exist_ok=True)

    ar = Accelerator(log_with='wandb')
    ar.init_trackers('VGG', config=cfg)

    device = ar.device
    cfg.lr *= ar.num_processes

    ds = ImageFolder(
        cfg.root,
        transform=TF.Compose([
            TF.Resize(cfg.resize),
            TF.RandomCrop(cfg.crop_size),
            TF.RandomHorizontalFlip(),
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
    )

if __name__ == '__main__':
    main()

