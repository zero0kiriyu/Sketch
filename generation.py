import torch.nn as nn
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
from flash.image import ImageEmbedder
from LSR import LSR

import sys
sys.path.insert(0,'/home/featurize/DifferentiableSketching')
# I don't know why, but it has to be present before some import path

from model_base import SinglePassSimpleLineDecoder

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, state=None):
        x = self.encoder(x)

        return self.decoder(x)
    
class VariationalAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, latent_size):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.latent_size = latent_size  # note enc will be made to emit 2x latent size output & we'll split

    # Sampling function (using the reparameterisation trick)
    def _sample(self, mu, log_sigma2):
        if self.training:
            eps = torch.randn(mu.shape, device=mu.device)
            return mu + torch.exp(log_sigma2 / 2) * eps
        else:
            return mu

    def get_feature(self, x):
        return self.encoder(x)[:, 0:self.latent_size]

    def forward(self, x):
        x = self.encoder(x)
        z = self._sample( x[:, 0:self.latent_size], x[:, self.latent_size:])
        images = self.decoder(z)

        return images

class SketchGenerate(nn.Module):
    def __init__(self,latent_size = 512,nlines=12):
        super(SketchGenerate,self).__init__()
        
        self.encoder = ImageEmbedder(backbone="resnet18",embedding_dim=latent_size)
        for param in self.encoder.parameters():
            param.requires_grad = True

        self.decoder = SinglePassSimpleLineDecoder(nlines=nlines,input=latent_size,hidden=512, hidden2=1024, sz=64)
        self.ae = AutoEncoder(self.encoder, self.decoder)
        
    def forward(self,x):
        return self.ae(x)

    
class HWDBGeneration(LightningModule):
    def __init__(
        self,
        batch_size=512,
        latent_size=512,
        nlines=12,
        lr=3e-4,
        max_epoch=20,
        **kwargs
    ):
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.batch_size = batch_size
        self.lr = lr
        self.max_epoch = max_epoch
        self.model = SketchGenerate(latent_size=latent_size,nlines=nlines)
        self.train_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        output = self.forward(batch[0])
        output = torch.cat([output] * 3, dim=1)
        loss = self.criterion(output, batch[0])
        return loss, output, batch[0]

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, training_step_outputs):
        preds,targets = training_step_outputs[1],training_step_outputs[2]

        self.logger.experiment.add_image("train/input", targets[0].permute(1,2,0),self.current_epoch)
        self.logger.experiment.add_image("train/output", preds[0].permute(1,2,0),self.current_epoch)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        # if self.is_last_batch:
        #     self.logger.experiment.add_image("tesst/input", targets[0].permute(1,2,0),self.current_epoch)
        #     self.logger.experiment.add_image("tesst/output", preds[0].permute(1,2,0),self.current_epoch)
        #self.log("tesst/input",,prog_bar=False,on_epoch=True, on_step=False, logger=True)
        #self.log("test/iutput",preds[0].permute(1,2,0),prog_bar=False,on_epoch=True, on_step=False,logger=True)
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
    batch_size = 3072
    lr = 4e-3
    max_epoch = 10
    logger = TensorBoardLogger("logs/", name="generation")
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

    generation = HWDBGeneration(lr=lr, batch_size=batch_size,)
    trainer.fit(generation, dataloader)


if __name__ == "__main__":
    cli_main()
