from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from hwdb_dataset import HWDBDataset
from pl_bolts.datamodules.async_dataloader import AsynchronousLoader
# import pandas as pd
# from sklearn import preprocessing
# import numpy as np


class HWDBDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/home/featurize/data/HWDB",
        batch_size: int = 512,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers= persistent_workers

        # self.dims is returned when you call datamodule.size()
        self.dims = (3, 224, 224)

        self.data_train: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 7356

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        self.data_train = HWDBDataset(
            self.data_dir, "train"
        )
        self.data_test = HWDBDataset(
            self.data_dir,"test"
        )

    def train_dataloader(self):
        return AsynchronousLoader(DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor = self.prefetch_factor,
            persistent_workers= self.persistent_workers,
            shuffle=True,)
        )

    def test_dataloader(self):
        return AsynchronousLoader(DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor = self.prefetch_factor,
            persistent_workers= self.persistent_workers,
            shuffle=False,)
        )
    
if __name__ == "__main__":
    datamodule = HWDBDataModule()
    datamodule.setup()
    count = 0
    for (x,y) in datamodule.train_dataloader():
        count += 1
        if count == 100:
            break
    
