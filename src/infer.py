from pathlib import Path
from skimage import io
import os
import torch
import numpy as np
import pandas as pd

def infer_casava(model,img_dir, test_transforms,decoder):



    img_files = [str(x) for x in Path(img_dir).glob("*")]

    column1 = []
    column2 = []
    model.eval()


    for img_name in img_files:
        img_id = os.path.basename(img_name)
        image = io.imread(img_name)
        image = test_transforms(image=image)["image"]
        image = image.unsqueeze(dim=0)
        image = image.to(model.device)

        output = model(image)
        results = torch.sigmoid(output).detach().numpy()
        index = results.argmax()

        hot_one = np.zeros((5,))
        hot_one[index] = 1
        label = decoder.inverse_transform(hot_one.reshape(1,-1)).item()

        column1.append(img_id)
        column2.append(label)

    data = {'image_id':column1,
            'label':column2
            }

    df = pd.DataFrame(data)

    df.to_csv('submission.csv',index=False,)







