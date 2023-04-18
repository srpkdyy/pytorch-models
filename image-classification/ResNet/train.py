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


@dataclasses.dataclass
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
    n_classes: int = 1000

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

    tf = TF.Compose([
        TF.Resize(cfg.resize),
        TF.RandomCrop(cfg.crop_size),
        TF.RandomHorizontalFlip(),
        TF.ToTensor(),
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

    model = VGG(cfg.net_type, cfg.n_classes)

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

    step = itertools.count()
    valid_acc_metric = Accuracy('multiclass', num_classes=cfg.n_classes).to(device)

    for epoch in range(cfg.epoch):
        model.train()

        for images, labels in tqdm(train_dl, desc='Train', disable=not ar.is_local_main_process):
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            ar.backward(loss)
            optimizer.step()

            ar.log({'Train Loss': loss.item()}, step=next(step))

        model.eval()
        valid_acc_metric.reset()

        for images, labels in tqdm(valid_dl, desc='Valid', disable=not ar.is_local_main_process):
            with torch.no_grad():
                outputs = model(images)
            preds = outputs.argmax(dim=1)
            valid_acc_metric.update(preds, labels)

        valid_acc = valid_acc_metric.compute().item()
        scheduler.step(valid_acc)

        log = {
            'Valid Acc': valid_acc,
            'learning rate': optimizer.param_groups[0]['lr'],
        }
        ar.log(log, step=epoch)
        ar.print(log)

    ar.end_training()


if __name__ == '__main__':
    main()


