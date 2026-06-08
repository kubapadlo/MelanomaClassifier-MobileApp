import torch.nn as nn
from torchvision import models

def get_model(num_classes=2, unfreeze_layers=1):
    if unfreeze_layers not in range(5):
        raise ValueError(f"unfreeze_layers musi być w 0–4, otrzymano: {unfreeze_layers}")

    model = models.resnet50(weights="DEFAULT")

    for param in model.parameters():
        param.requires_grad = False

    layers = [model.layer4, model.layer3, model.layer2, model.layer1]
    for layer in layers[:unfreeze_layers]:
        for param in layer.parameters():
            param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model