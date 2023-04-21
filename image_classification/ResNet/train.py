import os
import dataclasses
import itertools

import torch
import accelerate
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
accelerate.utils.set_seed(42)


@dataclasses.dataclass(frozen=True)
class Config:
    # Dataset
    root: str = 'data'
    batch_size: int = 256
    resize: tuple = (256, 480)
    crop_size: int = 224
    workers: int = 8

    # Optimizer
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0001
    
    # Scheduler
    factor = 0.1

    # Model
    arch_type: str = '18' # [18, 34, 50, 101, 152]
    n_classes: int = 1000

    # Training
    iters: int = 600000

    # Log
    log_interval: int = 500


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

    normalize = TF.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_ds = ImageNet(
        cfg.root,
        'train',
        transform=TF.Compose([
            TF.RandomChoice([TF.Resize(s) for s in cfg.resize])
            TF.RandomCrop(cfg.crop_size),
            TF.RandomHorizontalFlip(),
            TF.ToTensor(),
            normalize,
        ])
    )
    valid_ds = ImageNet(
        cfg.root,
        'val',
        transform=TF.Compose([
            TF.Resize(cfg.resize[0]),
            TF.CenterCrop(cfg.crop_size),
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

    model = ResNet(cfg.arch_type, cfg.n_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=cfg.factor
    )

    train_dl, valid_dl, model, optimizer, scheduler = ar.prepare(
        train_dl, valid_dl, model, optimizer, scheduler
    )

    train_dl = cycle(train_dl)
    iters_pbar = tqdm(range(iters), desc='Train', disable=not ar.is_local_main_process)
    valid_pbar = tqdm(valid_dl, desc='Valid', disable=not ar.is_local_main_process)

    top1_acc_metric = Accuracy('multiclass', num_classes=cfg.n_classes).to(device)
    top5_acc_metric = Accuracy('multiclass', num_classes=cfg.n_classes, top_k=5).to(device)

    for step in iters_pbar:
        model.train()

        images, labels = next(train_dl)

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
        ar.log(log, step=step)

        if step % cfg.log_interval == 0:
            model.eval()

            top1_acc_metric.reset()
            top5_acc_metric.reset()

            for images, labels in valid_pbar:
                with torch.no_grad():
                    outputs = model(images)
                preds = outputs.argmax(dim=1)

                top1_acc_metric.update(preds, labels)
                top5_acc_metric.update(outputs, labels)

            top1_acc = top1_acc_metric.compute().item()
            top5_acc = top1_acc_metric.compute().item()

            scheduler.step(top1_acc)

            log = {
                'Top1 Acc': top1_acc,
                'Top5 Acc': top5_acc,
            }
            ar.log(log, step=step)
            ar.print(log)

    ar.end_training()


if __name__ == '__main__':
    main()


