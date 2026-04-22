import torch
from src.dataset import get_dataloaders
from src.model import get_model
from src.train import run_training  
import wandb

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używam urządzenia: {device}")

    wandb.init(
        project="melanoma-detection", 
        name="resnet50-eksperyment-1",
        config={
            "learning_rate": 0.0001,
            "epochs": 4,
            "batch_size": 32,
            "model_architecture": "ResNet50",
            "optimizer": "Adam"
        }
    )
    config = wandb.config 

    train_loader, test_loader = get_dataloaders(batch_size=config.batch_size)

    model = get_model(num_classes=2).to(device)

    trained_model = run_training(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader, 
        num_epochs=config.epochs,
        lr=config.learning_rate,
        device=device
    )

    torch.save(trained_model.state_dict(), "model.pth")
    wandb.save("model.pth")
    wandb.finish()

if __name__ == "__main__":
    main()