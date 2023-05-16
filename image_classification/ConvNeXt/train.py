import os
import dataclasses
import itertools
from functools import partial

import torch
import accelerate
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from tqdm import tqdm
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader
from torchvision import transforms as TF
from torchvision.datasets import ImageNet
from torchvision.utils import save_image
from torchmetrics import Accuracy

from convnext import ConvNeXt


torch.backends.cudnn.benchmark = True
accelerate.utils.set_seed(42)


@dataclasses.dataclass(frozen=True)
class Config:
    # Dataset
    root: str = 'data'
    batch_size: int = 4096
    resize: tuple = 256
    crop_size: int = 224
    test_size: int = 224
    workers: int = 8

    # Loss
    smooth_rate: float = 0.1

    # Optimizer
    lr: float = 4e-3
    momentum: float = 0.9
    weight_decay: float = 0.05
    
    # Scheduler
    warmup_epoch: int = 20

    # Model
    arch_type: str = 'T' # [T, S, B, L, XL]
    n_classes: int = 1000

    # Training
    epoch: int = 300


cfg = Config()


def main():
    ar = Accelerator(log_with='wandb', split_batches=True)
    ar.init_trackers('ConvNeXt', config=cfg)

    device = ar.device

    normalize = TF.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_ds = ImageNet(
        cfg.root,
        'train',
        transform=TF.Compose([
            TF.Resize(cfg.resize),
            TF.RandomCrop(cfg.crop_size),
            TF.ToTensor(),
            normalize,
        ])
    )
    valid_ds = ImageNet(
        cfg.root,
        'val',
        transform=TF.Compose([
            TF.Resize(cfg.test_size),
            TF.CenterCrop(cfg.test_size),
            TF.ToTensor(),
            normalize,
        ])
    )

    kwargs = {
        'batch_size': cfg.batch_size,
        'num_workers': cfg.workers,
        'pin_memory': True
    }
    train_dl = DataLoader(train_ds, shuffle=True, **kwargs)
    valid_dl = DataLoader(valid_ds, shuffle=False, **kwargs)

    model = ConvNeXt(cfg.arch_type, cfg.n_classes)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.smooth_rate)
    optimizer = optim.AdamW(
        model.parameters(),
        cfg.lr,
        weight_decay=cfg.weight_decay
    )
    scheduler = CosineLRScheduler(
        optimizer,
        cfg.epoch,
        warmup_t=cfg.warmup_epoch,
        warmup_lr_init=cfg.lr/100,
    )

    train_dl, valid_dl, model, optimizer, scheduler = ar.prepare(
        train_dl, valid_dl, model, optimizer, scheduler
    )

    train_pbar = partial(tqdm, train_dl, desc='Train', disable=not ar.is_local_main_process)
    valid_pbar = partial(tqdm, valid_dl, desc='Valid', disable=not ar.is_local_main_process)

    top1_acc_metric = Accuracy('multiclass', num_classes=cfg.n_classes).to(device)
    top5_acc_metric = Accuracy('multiclass', num_classes=cfg.n_classes, top_k=5).to(device)

    for epoch in range(cfg.epoch):

        model.train()

        for images, labels in train_pbar():

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            ar.backward(loss)
            optimizer.step()

        train_loss = ar.gather(loss).mean().item()
        log = {
            'Train Loss': train_loss,
            'Learning rate': optimizer.param_groups[0]['lr']
        }
        ar.log(log, step=epoch)

        model.eval()
        top1_acc_metric.reset()
        top5_acc_metric.reset()

        for images, labels in valid_pbar():
            with torch.no_grad():
                outputs = model(images)
            preds = outputs.argmax(dim=1)

            top1_acc_metric.update(preds, labels)
            top5_acc_metric.update(outputs, labels)

        top1_acc = top1_acc_metric.compute().item()
        top5_acc = top5_acc_metric.compute().item()

        log = {
            'Top1 Acc': top1_acc,
            'Top5 Acc': top5_acc,
        }
        ar.log(log, step=epoch)

        scheduler.step(epoch)

    ar.end_training()


if __name__ == '__main__':
    main()

