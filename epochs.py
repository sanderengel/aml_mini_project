### Model trainer

###############
### IMPORTS ###
###############

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

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
    epochs=20,
    fold=0
):
    device = get_device()
    model.to(device)

    val_losses_per_epoch = []

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

        val_losses_per_epoch.append(avg_val_loss)

        print(
            f"Fold {fold}, Epoch {epoch + 1}: "
            f"train loss: {avg_train_loss:.4f}, "
            f"val loss: {avg_val_loss:.4f}, "
            f"val acc: {val_acc:.4f}"
        )

    return val_losses_per_epoch


def train_final_model(model, train_loader, criterion, optimizer, epochs):
    device = get_device()
    model.to(device)

    for epoch in range(epochs):
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

        avg_train_loss = train_loss / len(train_loader.dataset)

        print(
            f"Final model epoch {epoch + 1}/{epochs}: "
            f"train loss: {avg_train_loss:.4f}"
        )

    torch.save(model.state_dict(), "data/models/resnet18_final_all_data.pth")
    print("Saved final model to data/models/resnet18_final_all_data.pth")


################
### TRAINING ###
################

if __name__ == '__main__':
    driver_img_list = load_driver_img_list()
    train_dir = 'data/imgs/train'

    gkf = GroupKFold(n_splits=3)

    all_fold_losses = []
    max_epochs = 3

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(
            driver_img_list["img"],
            driver_img_list["classname"],
            groups=driver_img_list["subject"]
        )
    ):
        print(f"\nTraining fold {fold + 1}")

        train = driver_img_list.iloc[train_idx]
        val = driver_img_list.iloc[val_idx]

        model = get_model(num_classes=10)

        train_loader = get_data_loader(
            train,
            root_dir=train_dir,
            batch_size=32,
            shuffle=True,
            distort=True
        )

        val_loader = get_data_loader(
            val,
            root_dir=train_dir,
            batch_size=32,
            shuffle=False,
            distort=False
        )

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )

        val_losses = _train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=max_epochs,
            fold=fold + 1
        )

        all_fold_losses.append(val_losses)

    # ---- Compute best epoch ----
    avg_val_losses = np.mean(all_fold_losses, axis=0)
    best_epoch = int(np.argmin(avg_val_losses) + 1)

    print("\nAverage validation loss per epoch:")
    for epoch, loss in enumerate(avg_val_losses, start=1):
        print(f"Epoch {epoch}: {loss:.4f}")

    # ---- Plot validation loss ----
    epochs = range(1, max_epochs + 1)

    plt.figure(figsize=(8, 5))

    for i, fold_losses in enumerate(all_fold_losses):
        plt.plot(epochs, fold_losses, alpha=0.4, label=f"Fold {i + 1}")

    plt.plot(
        epochs,
        avg_val_losses,
        linewidth=3,
        label="Average validation loss"
    )

    plt.axvline(
        best_epoch,
        linestyle="--",
        label=f"Best epoch = {best_epoch}"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title("Validation loss evolution across epochs")
    plt.legend()
    plt.grid(True)

    plt.savefig("data/models/validation_loss_evolution.png", dpi=300)
    plt.show()

    print(f"\nBest epoch based on K-fold validation loss: {best_epoch}")

    # ---- Train final model ----
    print("\nTraining final model on all data")

    final_model = get_model(num_classes=10)

    final_loader = get_data_loader(
        driver_img_list,
        root_dir=train_dir,
        batch_size=32,
        shuffle=True,
        distort=True
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        final_model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    train_final_model(
        model=final_model,
        train_loader=final_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=best_epoch
    )