import kagglehub
from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

DATASET_ID = "hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images"

def get_paths():
    base_path = kagglehub.dataset_download(DATASET_ID)
    train_path = Path(base_path) / 'melanoma_cancer_dataset' / 'train'
    test_path  = Path(base_path) / 'melanoma_cancer_dataset' / 'test'
    return train_path, test_path

def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
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
                              shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=2)

    return train_loader, test_loader