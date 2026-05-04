### Model trainer



###############
### IMPORTS ###
###############

import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from utils import load_driver_img_list, get_device, data_to_device
from model import get_model
from preprocessing import get_data_loader



#################
### FUNCTIONS ###
#################

def _train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs = 5,
    checkpoint_path = None
):
    device = get_device()
    model.to(device)

    best_val_loss = float("inf")
    best_metrics = None
    epoch_history = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = data_to_device(images, labels, device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, batch {batch_idx}/{len(train_loader)}, loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = data_to_device(images, labels, device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        print(
            f"Epoch {epoch + 1}: "
            f"train loss: {avg_train_loss:.4f}, "
            f"val loss: {avg_val_loss:.4f}, "
            f"val acc: {val_acc:.4f}"
        )

        epoch_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_acc': val_acc,
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_metrics = (avg_train_loss, avg_val_loss, val_acc)
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
                print("Checkpoint saved.")

    return best_metrics, epoch_history



################
### TRAINING ###
################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Quick smoke test with minimal data/configs')
    args = parser.parse_args()

    driver_img_list = load_driver_img_list()
    train_dir = 'data/state-farm-distracted-driver-detection/imgs/train'

    n_epochs = 1 if args.test else 10

    RESULTS_DIR = 'data/results/'
    SEARCH_RESULTS_PATH = RESULTS_DIR + 'aug_search_results.csv'
    EPOCH_RESULTS_PATH = RESULTS_DIR + 'aug_epoch_results.csv'
    os.makedirs(RESULTS_DIR, exist_ok = True)

    if args.test:
        driver_img_list = driver_img_list.iloc[:128]
        cut = int(len(driver_img_list) * 0.8)
        splits = [(list(range(cut)), list(range(cut, len(driver_img_list))))]
        aug_grid = [(0, 1.0, 0.0, 0.0), (20, 0.8, 0.2, 0.5)]
    else:
        gkf = GroupKFold(n_splits = 5)
        splits = list(gkf.split(
            driver_img_list['img'],
            driver_img_list['classname'],
            groups = driver_img_list['subject']
        ))
        # One-at-a-time design: vary one parameter at a time, hold others at baseline
        aug_grid = [
            (0,  1.0, 0.0, 0.0),  # Baseline: no augmentation
            (10, 1.0, 0.0, 0.0),  # Light rotation only
            (20, 1.0, 0.0, 0.0),  # Heavy rotation only
            (0,  0.8, 0.0, 0.0),  # Crop only
            (0,  1.0, 0.2, 0.0),  # Color jitter only
            (0,  1.0, 0.0, 0.5),  # Random erasing only
            (20, 0.8, 0.2, 0.5),  # All augmentations
        ]

    all_results = pd.read_csv(SEARCH_RESULTS_PATH).to_dict('records') if os.path.exists(SEARCH_RESULTS_PATH) else []
    all_epoch_results = pd.read_csv(EPOCH_RESULTS_PATH).to_dict('records') if os.path.exists(EPOCH_RESULTS_PATH) else []

    for rotation, crop_scale, color_jitter, erasing_prob in aug_grid:
        aug_params = dict(
            rotation = rotation,
            crop_scale = crop_scale,
            color_jitter = color_jitter,
            erasing_prob = erasing_prob
        )
        print(f"\nConfig: {aug_params}")

        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"  Fold {fold + 1}")

            train = driver_img_list.iloc[train_idx]
            val = driver_img_list.iloc[val_idx]

            model = get_model(num_classes = 10)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-4)

            train_loader = get_data_loader(train, root_dir = train_dir, aug_params = aug_params)
            val_loader = get_data_loader(val, root_dir = train_dir, shuffle = False)

            metrics, epoch_history = _train_model(
                model = model,
                train_loader = train_loader,
                val_loader = val_loader,
                criterion = criterion,
                optimizer = optimizer,
                epochs = n_epochs
            )
            fold_metrics.append(metrics)

            for row in epoch_history:
                all_epoch_results.append({**aug_params, 'fold': fold + 1, **row})

        n = len(fold_metrics)
        all_results.append({
            **aug_params,
            'avg_train_loss': sum(m[0] for m in fold_metrics) / n,
            'avg_val_loss': sum(m[1] for m in fold_metrics) / n,
            'avg_val_acc': sum(m[2] for m in fold_metrics) / n,
        })

        # Save incrementally after each config so progress is not lost on crashes
        pd.DataFrame(all_results).to_csv(SEARCH_RESULTS_PATH, index = False)
        pd.DataFrame(all_epoch_results).to_csv(EPOCH_RESULTS_PATH, index = False)
        print(f"  Results saved ({len(all_results)}/{len(aug_grid)} configs done)")

    print(f'\nFinal results saved to {SEARCH_RESULTS_PATH} and {EPOCH_RESULTS_PATH}')
