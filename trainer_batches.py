import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold

from utils import load_driver_img_list, get_device, data_to_device
from model import get_model
#from preprocessing import get_data_loader
from temp_pre import DriverDataset
from temp_pre import get_data_loader

driver_img_list = load_driver_img_list()


def _train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs=5
):
    device = get_device()
    model.to(device)

    best_val_loss = float("inf")

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
                print(f"Epoch {epoch + 1}, batch {batch_idx}/{len(train_loader)}, loss: {loss.item():.4f}")

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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "data/models/best_driver_model.pth")
            print("Checkpoint saved.")

    return model


gkf = GroupKFold(n_splits=5)

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

    train_loader = get_data_loader(train)
    val_loader = get_data_loader(val)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    _train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=5
    )

    