import json
import os

import numpy as np
import pandas as pd
import torch
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from torch.utils.data import Dataset


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
        self.encoder = encoder

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.dataframe.iloc[idx]
        img_id = row["image_id"]
        img_name = os.path.join(self.img_dir, img_id)
        image = io.imread(img_name)

        if self.transform:
            # image = self.transform(image)
            # old_image = image
            image = self.transform(image=image)["image"]

        label = row["label"]

        encoded_label = self.encoder.transform(np.array([[label]]))
        encoded_label = encoded_label.reshape(-1,)
        sample = {
            "image": image,
            "label": encoded_label,
        }  # , 'sample_weight':sample_weight}

        return sample
