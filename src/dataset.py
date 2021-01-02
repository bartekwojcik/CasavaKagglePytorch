import json
import os

import numpy as np
import pandas as pd
import torch
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
import pytorch_lightning as pl
from PIL import Image


class CasavaDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        img_dir: str,
        encoder: OneHotEncoder,
        transform=None,
    ):

        self.dataframe = dataframe  # pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        #self.encoder = encoder

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.dataframe.iloc[idx]
        img_id = row["image_id"]
        img_name = os.path.join(self.img_dir, img_id)
        image = io.imread(img_name)
        pil_image = Image.fromarray(image)


        image = self.transform(pil_image)

        label = row["label"]

        #encoded_label = self.encoder.transform(np.array([[label]]))
        #encoded_label = encoded_label.reshape(-1,)
        # sample = {
        #     "image": image,
        #     "label": label,
        # }  # , 'sample_weight':sample_weight}

        return image,label


class CasavaDataModule(pl.LightningDataModule):

    def __init__(self,
                 training_samples_weights,
                 batch_size,
                 train_dataset,
                 val_dataset,
                 test_dataset,
                 data_loader_workers
                 ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.data_loader_workers = data_loader_workers
        self.batch_size = batch_size
        self.training_samples_weights = training_samples_weights

    def train_dataloader(self):
        weights = self.training_samples_weights
        sampler = WeightedRandomSampler(weights, len(weights))
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.data_loader_workers,
                          sampler= sampler
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,num_workers=self.data_loader_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,num_workers=self.data_loader_workers)
