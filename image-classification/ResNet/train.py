import os
import dataclasses
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms as TF
from torchvision.datasets import ImageNet
from torchvision.utils import save_image
from torchmetrics import Accuracy

from resnet import ResNet


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
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0001
    
    # Scheduler
    factor = 0.1

    # Model
    arch_type: str = '18' # [18, 34, 50, 101, 154]
    n_classes: int = 1000

    # Training
    iters: int = 600000

    # Log
    log_interval: int = 100


cfg = Config()


def cycle(train_dl):
    while True:
        for data in train_dl:
            yield data


def main():
    ar = Accelerator(log_with='wandb')
    ar.init_trackers('ResNet', config=cfg)

    device = ar.device
    ndevice = ar.num_processes
    lr = cfg.lr * ndevice
    iters = cfg.iters // ndevice

    tf = TF.Compose([
        TF.Resize(cfg.resize),
        TF.RandomCrop(cfg.crop_size),
        TF.RandomHorizontalFlip(),
        TF.ToTensor(),
        # per-pixel mean subtracted. ref-21
        # color augmentation
        TF.Normalize([0.485, 0.456, 0.406],
                     [0.229, 0.224, 0.225])
    ])
    train_ds = ImageNet(cfg.root, 'train', transform=tf)
    valid_ds = ImageNet(cfg.root, 'val', transform=tf)

    kwargs = {
        'batch_size': cfg.batch_size,
        'num_workers': cfg.workers,
        'pin_memory': True
    }

    train_dl = DataLoader(train_ds, shuffle=True, **kwargs)
    valid_dl = DataLoader(valid_ds, shuffle=False, **kwargs)

    model = ResNet(cfg.arch_type)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.factor
    )

    train_dl, valid_dl, model, optimizer, scheduler = ar.prepare(
        train_dl, valid_dl, model, optimizer, scheduler
    )
    train_dl = cycle(train_dl)

    model.train()
    valid_acc_metric = Accuracy('multiclass', num_classes=cfg.n_classes).to(device)

    for step in tqdm(range(iters), disable=not ar.is_local_main_process):
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        ar.backward(loss)
        optimizer.step()

        train_loss = loss.item()
        scheduler.step(loss.item())

        log = {
            'Train Loss': train_loss,
            'Learning rate': optimizer.param_groups[0]['lr']
        }
        ar.log(log, step=step)

        if step % cfg.log_interval == 0:
            model.eval()
            valid_acc_metric.reset()

            for images, labels in tqdm(valid_dl, desc='Valid', disable=not ar.is_local_main_process):
                # 10 crop settings [21]
                with torch.no_grad():
                    outputs = model(images)
                preds = outputs.argmax(dim=1)
                valid_acc_metric.update(preds, labels)

            valid_acc = valid_acc_metric.compute().item()

            log = {
                'Valid Acc': valid_acc,
            }
            ar.log({'Valid Acc': valid_acc}, step=step)
            ar.print('Valid', log)

            model.train()

    ar.end_training()


if __name__ == '__main__':
    main()


