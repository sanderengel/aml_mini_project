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

def _get_transformer(aug_params: dict = None):
    # Standard ImageNet normalization used for pre-trained models
    norm_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Apply augmentation if params passed
    if aug_params:
        rotation = aug_params.get('rotation', 0)
        crop_scale = aug_params.get('crop_scale', 1.0)
        color_jitter = aug_params.get('color_jitter', 0.0)
        erasing_prob = aug_params.get('erasing_prob', 0.0)

        steps = []
        if rotation > 0:
            steps.append(transforms.RandomRotation(rotation))
        if crop_scale < 1.0:
            steps.append(transforms.RandomResizedCrop(224, scale = (crop_scale, 1.0)))
        else:
            steps.append(transforms.Resize((224, 224)))
        if color_jitter > 0:
            steps.append(transforms.ColorJitter(brightness = color_jitter, contrast = color_jitter))
        steps += [transforms.ToTensor(), transforms.Normalize(*norm_stats)]
        if erasing_prob > 0:
            steps.append(transforms.RandomErasing(p = erasing_prob, scale = (0.02, 0.2)))
        return transforms.Compose(steps)
    
    # If no augmentation, only resize and normalize
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
    aug_params: dict = None
):
    transformer = _get_transformer(aug_params)
    ds = DistractedDriverDataset(metadata, root_dir, transformer)
    loader = DataLoader(ds, batch_size = batch_size, shuffle = shuffle, num_workers = 2)
    return loader
