### General utility functions and variables 



###############
### IMPORTS ###
###############

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image



###############
### MAPPING ###
###############

STATE_FARM_CLASSES = {
    'c0': 'safe driving',
    'c1': 'texting - right',
    'c2': 'talking on the phone - right',
    'c3': 'texting - left',
    'c4': 'talking on the phone - left',
    'c5': 'operating the radio',
    'c6': 'drinking',
    'c7': 'reaching behind',
    'c8': 'hair and makeup',
    'c9': 'talking to passenger'
}



######################
### DATA FUNCTIONS ###
######################

def load_driver_img_list() -> pd.DataFrame:
    DRIVER_IMG_LIST_PATH = 'data/state-farm-distracted-driver-detection/driver_imgs_list.csv'
    df = pd.read_csv(DRIVER_IMG_LIST_PATH)
    return df

def get_img_data(
    img_path: str,
    target_size: tuple = (224, 224),
    return_tensor: bool = False
):
    """
    Read an image from disk.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f'No image found at {img_path}')
    
    # Load and ensure RGB
    img = Image.open(img_path).convert('RGB')

    if target_size:
        img = img.resize(target_size)

    return np.array(img)

def plot_img(img_array: np.array, label_code: str = None, class_map: dict = None):
    """
    Display an image.
    """
    plt.imshow(img_array)

    if label_code:
        title_str = label_code
        if class_map:
            title_str += f": {class_map.get(label_code, 'Unknown')}"
        plt.title(title_str)

    plt.axis('off')
    plt.show()