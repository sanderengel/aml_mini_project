### Model trainer



###############
### IMPORTS ###
###############

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from utils import load_driver_img_list, get_device, data_to_device
from model import get_model
from preprocessing import get_data_loader



#####################
### LOAD METADATA ###
#####################

driver_img_list = load_driver_img_list()



#################
### FUNCTIONS ###
#################



def _train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs = 10
):
    device = get_device()
    model.to(device)

    best_val_loss = float('inf')

    for epoch in range(epochs):

        ## Training
        model.train() # Set to training mode
        for images, labels in train_loader:
            images, labels = data_to_device(images, labels, device)

            # Get loss
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Add to loss
            train_loss += loss.item() * images.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        ## Validation
        model.eval() # Set to evalation
        val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = data_to_device(images, labels, device)

                # Get loss
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                # Get predictions
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader.datasets)
        val_acc = correct / len(val_loader.dataset)

        print(f'Epoch {epoch + 1}: train Loss: {avg_train_loss:.4f}, val loss {avg_val_loss:.4f}, val acc: {val_acc:.4f}')

        # Save best model to checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_state_dict(), 'best_driver_model.pth')
            print('Checkpoint saved.')

    return model



################
### TRAINING ###
################

gkf = GroupKFold(n_splits = 5)
for train_idx, val_idx in gkf.split(
    driver_img_list['img'],
    driver_img_list['classname'],
    groups = driver_img_list['subject']
):
    train = driver_img_list.iloc[train_idx]
    val = driver_img_list.iloc[val_idx]

    # TODO: Initialize parameters for _train_model() and call it

    model = get_model()
    train_loader = get_data_loader(train)
    val_loader = get_data_loader(val)

    _train_model()


### TODO:
# 1. Dataset Class: we need a dataset class with data augmentation (random rotation and color jitter, etc)
# 2. Loss function (Cross Entropy)
# 3. Optimizer (Adam / SGD with Momentum)