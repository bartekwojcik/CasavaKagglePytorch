import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader

from src.dataset import CasavaDataset


def split_dataset(df, test_size: float = 0.15):
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df["label_string"]
    )
    return train_df, test_df


def load_classes_from_json(json_dir: str):
    with open(json_dir) as f:
        classes_dict = json.load(f)
    return classes_dict


def prepare_datasets(
    csv_dir: str, json_dir: str, images_dir: str, train_transform, test_transform
):
    classes_dict = load_classes_from_json(json_dir)
    df = pd.read_csv(csv_dir)



    df["label_string"] = df["label"].apply(lambda x: str(x))

    class_weights = compute_class_weight(
        "balanced", classes=np.unique(df["label_string"]), y=df["label_string"]
    )



    # df['sample_weight'] = samples_weights

    train_df, test_df = split_dataset(df, test_size=0.15)
    test_df, val_df = split_dataset(test_df, test_size=0.5)

    encoder = OneHotEncoder()
    encoder.sparse = False
    encoder.fit(df[["label"]])

    training_samples_weights = compute_sample_weight(class_weight='balanced', y=train_df['label_string'])

    train_dataset = CasavaDataset(train_df, images_dir, encoder, train_transform)
    val_dataset = CasavaDataset(val_df, images_dir, encoder, test_transform)
    test_dataset = CasavaDataset(test_df, images_dir, encoder, test_transform)

    training_samples_weights = torch.DoubleTensor(training_samples_weights)

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        class_weights,
        training_samples_weights,
        classes_dict,
        encoder,
    )
