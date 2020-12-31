from torch.utils.data import DataLoader
import torch
import numpy as np
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_test_model_predictions(model, test_dataset, batch_size):

    predictions, targets = [], []

    loader = DataLoader(test_dataset, batch_size=batch_size)

    for i_batch, sample_batched in enumerate(loader):
        images = sample_batched["image"].to(model.device)
        true_labels = sample_batched["label"].numpy().astype(np.uint8)

        outputs = model(images)
        results = torch.sigmoid(outputs).detach().numpy()
        predictions.append(results)
        targets.append(true_labels)

    stacked_preds = np.concatenate(predictions, axis=0)
    stacked_targets = np.concatenate(targets, axis=0)

    return stacked_preds, stacked_targets


def test_with_metrics(preds, targets, class_dict, threshold=0.5):

    rounded_preds = np.where(preds > threshold, 1, 0)

    print(metrics.classification_report(rounded_preds, targets))

    mcms = metrics.multilabel_confusion_matrix(targets, rounded_preds)

    for class_idx, cf in enumerate(mcms):
        class_name = class_dict[str(class_idx)]
        print_confusion_matrix(cf, class_name)


def print_confusion_matrix(confusion_matrix, class_names, fontsize=14):

    df_cm = pd.DataFrame(confusion_matrix,)

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False,)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha="right", fontsize=fontsize
    )
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=0, ha="right", fontsize=fontsize
    )
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    plt.title("Class - " + class_names)
    plt.show()
