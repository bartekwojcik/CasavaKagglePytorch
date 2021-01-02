import multiprocessing

from src.data_preparation import prepare_datasets
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from albumentations.pytorch import ToTensorV2
from torchvision import transforms, models
import albumentations as A
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import f1, accuracy
from torchsummary import summary

from src.dataset import CasavaDataset, CasavaDataModule
from src.model import CasvaModel
from src.viz import viz_batch
from src.testing import get_test_model_predictions, test_with_metrics
from pytorch_lightning.callbacks import ModelCheckpoint
from src.infer import infer_casava
from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.models.self_supervised.simclr.transforms import (
    SimCLREvalDataTransform, SimCLRTrainDataTransform)


def start_training(
    csv_file: str,
    json_file: str,
    images_dir: str,
    test_images_dir:str,
    training_max_epochs: int,
    finetuning_max_epochs: int,
    img_size: int,
    batch_size: int,
    learning_rate: float,
    use_gpus:bool,
    viz_datasets: bool = True,
):

    IMG_SIZE_NO_CHANNEL = (img_size, img_size)  # todo pass it as argument
    IMG_SIZE_CHANNEL = (3, *IMG_SIZE_NO_CHANNEL)

    train_transform = SimCLRTrainDataTransform(input_height=img_size)

    test_transform = SimCLREvalDataTransform(input_height=img_size)

    (
        train_dataset,
        val_dataset,
        test_dataset,
        class_weights,
        training_samples_weights,
        classes_dict,
        encoder,
    ) = prepare_datasets(
        csv_file, json_file, images_dir, train_transform, test_transform
    )

    if viz_datasets:
        for name, ds in [
            ("train", train_dataset),
            ("validation", val_dataset),
            ("test", test_dataset),
        ]:
            viz_batch(ds, 10, title=name)



    ##############################






    dm = CasavaDataModule(training_samples_weights=training_samples_weights,
                          batch_size=batch_size,
                          train_dataset=train_dataset,
                          val_dataset=val_dataset,
                          test_dataset=test_dataset,
                          data_loader_workers=multiprocessing.cpu_count()
                          )

    # model
    gpus = 1 if use_gpus else 0
    model = SimCLR(num_samples=len(train_dataset),
                   batch_size=dm.batch_size,
                   gpus=gpus,
                   dataset='mycifar?',
                   max_epochs=1,
                   warmup_epochs=0
                   )

    # fit
    trainer = pl.Trainer()
    trainer.fit(model, datamodule=dm)

    # # data
    # dm = CIFAR10DataModule(num_workers=0)
    # dm.train_transforms = SimCLRTrainDataTransform(32)
    # dm.val_transforms = SimCLREvalDataTransform(32)
    #
    # # model
    # model = SimCLR(num_samples=dm.num_samples, batch_size=dm.batch_size, dataset='cifar10',gpus=0)
    #
    # # fit
    # trainer = pl.Trainer()
    # trainer.fit(model, datamodule=dm)


    #############################################

    # model = CasvaModel(
    #     num_classes=len(classes_dict),
    #     class_weights=class_weights,
    #     training_samples_weights=training_samples_weights,
    #     image_dims=IMG_SIZE_CHANNEL,
    #     learning_rate=learning_rate,
    #     train_dataset=train_dataset,
    #     val_dataset=val_dataset,
    #     test_dataset=test_dataset,
    #     batch_size=batch_size,
    # )
    #
    # checkpoint_callback = ModelCheckpoint(monitor='val_loss',
    #                                       save_top_k=1,
    #                                       verbose=True,
    #                                       mode='min'
    #                                       )
    #
    # trainer = pl.Trainer(
    #     gpus=1 if use_gpus else 0,
    #     max_epochs=training_max_epochs,
    #     progress_bar_refresh_rate=20,
    #     callbacks=[checkpoint_callback]
    # )
    # trainer.fit(model)
    # trainer.test(model)
    #
    # # unfreeze model
    # model.unfreeze()
    # for param in model.model.parameters():
    #     param.requires_grad = True
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    # print("unfreezed all weights")
    #
    # fine_tuner = pl.Trainer(
    #     gpus=1 if use_gpus else 0,
    #     max_epochs=finetuning_max_epochs,
    #     progress_bar_refresh_rate=20,
    #     callbacks=[checkpoint_callback]
    # )
    #
    # fine_tuner.fit(model)
    # fine_tuner.test(model)

    ################################

    #todo this will have to be rewritten?

    preds, targets = get_test_model_predictions(
        model, test_dataset, batch_size=batch_size
    )

    test_with_metrics(preds, targets, classes_dict, threshold=0.5)

    infer_casava(model,test_images_dir,test_transform,encoder)

    print("done")

