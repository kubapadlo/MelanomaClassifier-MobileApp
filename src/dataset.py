import kagglehub
from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np

DATASET_ID = "hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images"

def get_paths():
    base_path = kagglehub.dataset_download(DATASET_ID)
    train_path = Path(base_path) / 'melanoma_cancer_dataset' / 'train'
    test_path  = Path(base_path) / 'melanoma_cancer_dataset' / 'test'
    return train_path, test_path

def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),                          # crop zamiast zwykłego resize
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1),                    # maskuje losowe fragmenty
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transforms, test_transforms


def get_dataloaders(batch_size=32):
    train_path, test_path = get_paths()
    train_tf, test_tf = get_transforms()

    train_dataset = datasets.ImageFolder(root=train_path, transform=train_tf)
    test_dataset  = datasets.ImageFolder(root=test_path,  transform=test_tf)


    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False,  num_workers=4, pin_memory=True)

    print(f"Klasy: {train_dataset.classes}")
    print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")

    return train_loader, test_loader