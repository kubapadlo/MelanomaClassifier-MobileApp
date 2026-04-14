# src/model.py
import torch.nn as nn
from torchvision import models

def get_model(num_classes=2):
    model = models.resnet50(weights='DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model