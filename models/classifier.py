import logging
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

_EVALUATE_OUTPUT = List[Dict[str, float]]  

log = logging.getLogger(__name__)


def classification_metrics(metric_list: List[str], num_classes: int):
    allowed_metrics = ['precision', 'recall', 'f1score', 'accuracy']

    for metric in metric_list:
        if metric not in allowed_metrics:
            raise ValueError(
                f"{metric} is not allowed. Please choose 'precision', 'recall', 'f1_score', 'accuracy'"
            )
    metric_dict = {
        'accuracy':
        torchmetrics.Accuracy(task='multiclass',
                              num_classes=num_classes,
                              top_k=1),
        'precision':
        torchmetrics.Precision(task='multiclass',
                               average='macro',
                               num_classes=num_classes),
        'recall':
        torchmetrics.Recall(task='multiclass',
                            average='macro',
                            num_classes=num_classes),
        'f1score':
        torchmetrics.F1Score(task='multiclass',
                             average='macro',
                             num_classes=num_classes)
    }
    metrics = [metric_dict[name] for name in metric_list]
    return MetricCollection(metrics)


class ClassifierTrainer(pl.LightningModule):
  
    def __init__(self,
                 devices,
                 model: nn.Module,
                 num_classes: int,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 accelerator: str = "cpu",
                 metrics: List[str] = ["accuracy"]):

        super().__init__()
        self.model = model

        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay

        self.devices = devices
        self.accelerator = accelerator
        self.metrics = metrics

        self.ce_fn = nn.CrossEntropyLoss()

        self.init_metrics(metrics, num_classes)

    def init_metrics(self, metrics: List[str], num_classes: int) -> None:
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

        self.train_metrics = classification_metrics(metrics, num_classes)
        self.val_metrics = classification_metrics(metrics, num_classes)
        self.test_metrics = classification_metrics(metrics, num_classes)

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            max_epochs: int = 300,
            *args,
            **kwargs) -> Any:

        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             max_epochs=max_epochs,
                             *args,
                             **kwargs)
        return trainer.fit(self, train_loader, val_loader)

    def test(self, test_loader: DataLoader, *args,
             **kwargs) -> _EVALUATE_OUTPUT:
        r'''
        Args:
            test_loader (DataLoader): Iterable DataLoader for traversing the test data batch (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
        '''
        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             *args,
                             **kwargs)
        return trainer.test(self, test_loader)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.ce_fn(y_hat, y)

        # log to prog_bar
        self.log("train_loss",
                 self.train_loss(loss),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)

        for i, metric_value in enumerate(self.train_metrics.values()):
            self.log(f"train_{self.metrics[i]}",
                     metric_value(y_hat, y),
                     prog_bar=True,
                     on_epoch=False,
                     logger=False,
                     on_step=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_loss",
                 self.train_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        for i, metric_value in enumerate(self.train_metrics.values()):
            self.log(f"train_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        # print the metrics
        str = "\n[Train] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("train_"):
                str += f"{key}: {value:.3f} "
        print(str + '\n')

        # reset the metrics
        self.train_loss.reset()
        self.train_metrics.reset()

    def validation_step(self, batch: Tuple[torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.ce_fn(y_hat, y)

        self.val_loss.update(loss)
        self.val_metrics.update(y_hat, y)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val_loss",
                 self.val_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        for i, metric_value in enumerate(self.val_metrics.values()):
            self.log(f"val_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        # print the metrics
        str = "\n[Val] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("val_"):
                str += f"{key}: {value:.3f} "
        print(str + '\n')

        self.val_loss.reset()
        self.val_metrics.reset()

    def test_step(self, batch: Tuple[torch.Tensor],
                  batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.ce_fn(y_hat, y)

        self.test_loss.update(loss)
        self.test_metrics.update(y_hat, y)
        return loss

    def on_test_epoch_end(self) -> None:
        self.log("test_loss",
                 self.test_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        for i, metric_value in enumerate(self.test_metrics.values()):
            self.log(f"test_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        # print the metrics
        str = "\n[Test] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("test_"):
                str += f"{key}: {value:.3f} "
        print(str + '\n')

        self.test_loss.reset()
        self.test_metrics.reset()

    def configure_optimizers(self):
        parameters = list(self.model.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        optimizer = torch.optim.Adam(trainable_parameters,
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        return optimizer

    def predict_step(self,
                     batch: Tuple[torch.Tensor],
                     batch_idx: int,
                     dataloader_idx: int = 0):
        x, y = batch
        y_hat = self(x)
        return y_hat