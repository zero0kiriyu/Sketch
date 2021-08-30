from hwdb_datamodule import HWDBDataModule
from typing import Any, List
from pytorch_lightning.trainer.trainer import Trainer

import torch
from pytorch_lightning import LightningModule, profiler
from torch.utils.data import dataloader
from torchmetrics.classification.accuracy import Accuracy
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from pytorch_lightning.utilities.cli import LightningCLI
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from flash.image import ImageClassifier
from LSR import LSR


class HWDBClassification(LightningModule):
    def __init__(
        self,
        batch_size=512,
        num_classes=7356,
        lr=3e-4,
        max_epoch=20,
        alpha=0.2,
        **kwargs
    ):
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.batch_size = batch_size
        self.lr = lr
        self.max_epoch = max_epoch
        self.alpha = alpha
        self.model = ImageClassifier(
            backbone="resnet18", num_classes=num_classes, pretrained=False
        )
        self.train_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        self.criterion = LSR()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        #lambda_ = np.random.beta(self.alpha, self.alpha)
        #index = torch.randperm(x.size(0)).cuda(non_blocking=True)
        #mix_x = lambda_ * x + (1 - lambda_) * x[index, :]
        #y_a, y_b = y, y[index]

        #logits = self.forward(mix_x)

        #loss = lambda_ * self.criterion(logits, y_a) + (1 - lambda_) * self.criterion(
        #    logits, y_b
        #)
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        acc = self.train_accuracy(preds, targets)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        acc = self.test_accuracy(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=1e-4,
        )
        #optimizer = torch.optim.SGD(self.model.parameters(),lr = self.lr,momentum=0.9,weight_decay=5e-4)

        lr_scheduler = {
            "scheduler": OneCycleLR(
                optimizer,
                8e-3,
                epochs=self.max_epoch,
                steps_per_epoch=int(3118477 / self.batch_size),
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


def cli_main():
    batch_size = 9216
    lr = 4e-3
    max_epoch = 10
    logger = TensorBoardLogger("logs/", name="classification")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    dataloader = HWDBDataModule(batch_size=batch_size)

    trainer = Trainer(
        precision=16,
        gpus=1,
        min_epochs=1,
        max_epochs=max_epoch,
        benchmark=True,
        logger=logger,
        #stochastic_weight_avg=True,
        callbacks=[lr_monitor],
    )

    classifier = HWDBClassification(lr=lr, batch_size=batch_size,)
    trainer.fit(classifier, dataloader)


if __name__ == "__main__":
    cli_main()
