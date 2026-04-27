### General utility functions and variables 



###############
### IMPORTS ###
###############

import os
import torch
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

def get_img(
    img_path: str,
    return_array: bool = False,
):
    """
    Read an image from disk.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f'No image found at {img_path}')
    
    # Load and ensure RGB
    img = Image.open(img_path).convert('RGB')

    if return_array:
        return np.array(img)
    return img

def plot_img(
    img_array: np.array = None,
    img_tensor: torch.tensor = None,
    label_code: str = None,
    class_map: dict = None
):
    """
    Display an image.
    """
    if (img_array is None and img_tensor is None) or (img_array and img_tensor):
        raise ValueError("Please pass exactly one of 'img_array' and 'img_tensor'.")

    if img_array:
        img = img_array

    else:
        # Move to CPU and convert to array
        img = img_tensor.clone().detach().cpu().numpy()

        # Transpose from (C, H, W) to (H, W, C)
        img = img.transpose(1, 2, 0)

        # Un-normalize
        mean = np.array([.485, .456, .406])
        std = np.array([.229, .224, .225])
        img = std * img + mean

        # Clip to 0-1 range
        img = np.clip(img, 0, 1)

    plt.imshow(img)

    if label_code:
        title_str = label_code
        if class_map:
            title_str += f": {class_map.get(label_code, 'Unknown')}"
        plt.title(title_str)

    plt.axis('off')
    plt.show()



############
### MISC ###
############

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device

def data_to_device(images, labels, device):
    return images.to(device), labels.to(device)
