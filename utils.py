import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil

def seprate_dataset(csv_path, image_path): 
    csv_file = pd.read_csv(csv_path)

    # -1/여자 ↔️ 1/남자 
    csv_file = csv_file[['image_id', 'Male']]

    for __, row in csv_file[['image_id', 'Male']].iterrows():
        if (row.Male == -1):
            shutil.move(image_path+row['image_id'], './celeba-gender-dataset/female')
        else: 
            shutil.move(image_path+row['image_id'], './celeba-gender-dataset/male')

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) 