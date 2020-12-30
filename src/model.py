import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms, models
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import f1, accuracy
from torchsummary import summary
from src.dataset import CasavaDataset


class CasvaModel(pl.LightningModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        class_weights,
        num_classes: int = 5,
        image_dims=(3, 28, 28),
        learning_rate=2e-4,
        batch_size=128,
    ):

        super().__init__()
        self.batch_size = batch_size
        self.class_weights = torch.from_numpy(class_weights)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.img_dims = image_dims

        self._prepare_model(self.num_classes, freeze_base_network=True)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)

    def _prepare_model(self, num_classes, freeze_base_network):

        resnet = models.resnet18(pretrained=True)

        if freeze_base_network:
            for param in resnet.parameters():
                param.requires_grad = False  # todo make it change for fine tuning

        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=num_ftrs, out_features=128),
            nn.Linear(in_features=128, out_features=num_classes),
        )

        self.model = resnet
        summary(resnet, self.img_dims)

    def forward(self, x):
        result = self.model(x)
        # result = F.sigmoid(d)
        return result

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]

        output = self(x)
        loss = self.criterion(output, y.float())

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]
        output = self(x)
        loss = self.criterion(output, y.float())

        temp_logits = torch.sigmoid(output)
        f1_value = f1(temp_logits, y, self.num_classes, average="macro")

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1", f1_value, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
