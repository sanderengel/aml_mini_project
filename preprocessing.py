### Pre-processing



###############
### IMPORTS ###
###############

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import get_img



#####################
### DATASET CLASS ###
#####################

class DistractedDriverDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, root_dir: str, transformer = None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transformer = transformer
        self.label_map = {f'c{i}': i for i in range(10)}

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Get info from the dataframe
        row = self.dataframe.iloc[idx]
        img_name = row['img']
        label_code = row['classname']

        # Construct path
        img_path = os.path.join(self.root_dir, label_code, img_name)

        # Load image
        image = get_img(img_path)
        label = self.label_map[label_code]

        if self.transformer:
            image = self.transformer(image)

        return image, label



###################################
### DATA TRANSFORMER AND LOADER ###
###################################

def _get_transformer(distort: bool = False):
    # Standard ImageNet normalization used for pre-trained models
    norm_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    if distort:
        return transforms.Compose([
            # Distort image
            transforms.RandomRotation(15),                           # Rotate
            transforms.RandomResizedCrop(224, scale = (0.7, 0.9)),   # Crop
            transforms.ColorJitter(brightness = .2, contrast = 0.2), # Distort color

            # Convert to tensor and normalize
            transforms.ToTensor(),
            transforms.Normalize(*norm_stats),

            # Randomly erase parts of the image
            transforms.RandomErasing(p = 0.5, scale = (0.02, 0.2))
        ])
    else:
        return transforms.Compose([
            # Clean resize
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(*norm_stats)
        ])


def get_data_loader(
    metadata: pd.DataFrame,
    root_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    distort: bool = True
):
    transformer = _get_transformer(distort)
    ds = DistractedDriverDataset(metadata, root_dir, transformer)
    loader = DataLoader(ds, batch_size = batch_size, shuffle = shuffle, num_workers = 2)
    return loader
