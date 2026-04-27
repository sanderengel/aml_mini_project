### Model set-up

###############
### IMPORTS ###
###############
from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision.models import EfficientNet_B0_Weights
import torch.nn as nn


#############
### MODEL ###
#############

def get_model(num_classes=10):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Get number of features from ResNet backbone
    in_features = model.fc.in_features

    # Replace classification head
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )

    return model


# def get_model(num_classes=10):
#     model = models.efficientnet_b0(
#         weights=EfficientNet_B0_Weights.DEFAULT
#     )

#     in_features = model.classifier[1].in_features

#     model.classifier = nn.Sequential(
#         nn.Dropout(0.3),
#         nn.Linear(in_features, num_classes)
#     )

#     return model

