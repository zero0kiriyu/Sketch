import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import lmdb
import numpy as np
import os
from turbojpeg import TurboJPEG
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt


class HWDBDataset(Dataset):
    def __init__(self, path, dataset):
        self.dataset = dataset
        self.path = path
        self.env = lmdb.open(os.path.join(self.path, "lmdb"))
        self.jpeg = TurboJPEG()
        f = np.load(os.path.join(self.path, "preprocess.npz"), allow_pickle=True)
        self.label2id = f["label2id"].item()
        self.id2label = f["id2label"].item()
        if self.dataset == "train":
            self.datalist = f["train_filelist"]
            self.transform = A.Compose(
                [
                    A.Resize(56, 56),
                    A.PadIfNeeded(64,64),
                    A.HorizontalFlip(),
                    # A.RandomBrightness(),
                    ToTensorV2(),
                ]
            )
        elif self.dataset == "test":
            self.datalist = f["test_filelist"]
            self.transform = A.Compose([A.Resize(56, 56),A.PadIfNeeded(64,64), ToTensorV2(),])
        else:
            print("error!")

    def __getitem__(self, index):
        row = self.datalist[index]
        with self.env.begin() as txn:
            buffer = txn.get(row["name"].encode())
        arr = self.jpeg.decode(buffer)
        img = self.transform(image=arr)["image"]
        return img / 255.0, self.label2id[row["label"]]

    def __len__(self):
        return len(self.datalist)
