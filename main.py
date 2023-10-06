'''Import libraries'''
import os, yaml
from easydict import EasyDict

from dataloaders.seed import SEEDDataset
from models.tsception import TSCeption

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.backends.cudnn as cudnn
from torchvision import transforms

from models.classifier import ClassifierTrainer
from sklearn.model_selection import KFold


with open('dataloaders/seed.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    args = EasyDict(config)

args.lr = float(args.lr)
args.weight_decay = float(args.weight_decay)

cudnn.benchmark = True
cudnn.fastest = True

dataset = SEEDDataset(args)

k_fold = KFold(n_splits=args.k_folds, split_path='results/split')

for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = TSCeption(args)

    trainer = ClassifierTrainer(model=model,
                                num_classes=args.num_classes, 
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                accelerator="gpu",
                                devices=args.devices)
    trainer.fit(train_loader,
            val_loader,
            max_epochs=args.EPOCHS,
            default_root_dir='results/train/{i}',
            callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)],
            enable_progress_bar=True,
            enable_model_summary=True,
            limit_val_batches=0.0)

    score = trainer.test(val_loader,
            enable_progress_bar=True,
            enable_model_summary=True)[0]
    print(f'Fold {i} test accuracy: {score["test_accuracy"]:.4f}')
