import os
import dataclasses
from accelerate import Accelerator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as TF
from torchvision.datasets import ImageNet
from torchvision.utils import save_image
from torchmetrics.classification.accuracy import Accuracy

from vgg import VGG


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

    transform = TF.Compose([
        TF.Resize(cfg.resize),
        TF.RandomCrop(cfg.crop_size),
        TF.RandomHorizontalFlip(),
        TF.ToTensor(),
        TF.Normalize([0.485, 0.456, 0.406],
                     [0.229, 0.224, 0.225])
    ])
    train_ds = ImageNet(cfg.root, 'train', transform)
    valid_ds = ImageNet(cfg.root, 'valid', transform)

    kwargs = {
        'batch_size': cfg.batch_size,
        'shuffle': True,
        'num_workers': cfg.workers,
        'pin_memory': True
    }
    train_dl = DataLoader(train_ds, **kwargs)
    valid_dl = DataLoader(valid_ds, **kwargs)

    model = VGG(cfg.net_type, cfg.n_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=cfg.factor
    )

    train_dl, valid_dl, model, optimizer, scheduler = ar.prepare(
        train_dl, valid_dl, model, optimizer, scheduler
    )

    for epoch in range(cfg.epoch):
        model.train()
        train_loss = 0

        for images, labels in train_dl:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            ar.backward(loss)
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        acc = Accuracy('multiclass', num_classes=cfg.n_classes)

        for images, labels in valid_dl:
            with torch.no_grad():
                outputs = model(images)
            preds = outputs.argmax(dim=1)
            acc.update(preds, labels)
        valid_acc = acc.compute()

        scheduler.step(valid_acc)

        log = {
            'Train Loss': train_loss,
            'Valid Acc': valid_acc
        }
        ar.log(log, step=epoch)
        ar.print(log)

    ar.end_training()


if __name__ == '__main__':
    main()

