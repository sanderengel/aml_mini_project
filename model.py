### Model set-up

###############
### IMPORTS ###
###############
from torchvision import models
from torchvision.models import ResNet18_Weights
import torch.nn as nn


#############
### MODEL ###
#############

def get_model(num_classes=10, freeze_backbone=False):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Get number of features from ResNet backbone
    in_features = model.fc.in_features

    # Optional: freeze backbone
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classification head
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )

    return model

