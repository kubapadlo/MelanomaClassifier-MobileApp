import logging
from pathlib import Path

import kagglehub
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

DATASET_ID = "hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.1),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, eval_tf


def get_dataloaders(batch_size=32, val_split=0.15, num_workers=2):
    base = Path(kagglehub.dataset_download(DATASET_ID)) / "melanoma_cancer_dataset"
    train_tf, eval_tf = get_transforms()

    full_train = datasets.ImageFolder(root=base / "train", transform=train_tf)
    test_ds    = datasets.ImageFolder(root=base / "test",  transform=eval_tf)

    val_size   = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    # val używa eval_tf
    val_ds.dataset = datasets.ImageFolder(root=base / "train", transform=eval_tf)

    logger.info("Klasy: %s | Train: %d | Val: %d | Test: %d",
                full_train.classes, train_size, val_size, len(test_ds))

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    return make_loader(train_ds, True), make_loader(val_ds, False), make_loader(test_ds, False)