import matplotlib.pyplot as plt
import numpy as np


def viz_batch(ds, n_images, title):
    a = 5
    for i in range(n_images):
        sample = ds[i]
        x = sample["image"].numpy()
        y = sample["label"]
        img = x
        label = y
        plt.imshow(np.transpose(img, axes=(1, 2, 0)))
        plt.title(f"Set:{title}. Label:{label}")
        plt.tight_layout()
        plt.show()
