import argparse 
import torch
import wandb
from src.dataset import get_dataloaders
from src.model import get_model
from src.train import run_training

def parse_args():
    parser = argparse.ArgumentParser(description="Trening modelu na czerniaka")
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate')
    parser.add_argument('--epochs', type=int, default=10, help='Liczba epok')
    parser.add_argument('--batch_size', type=int, default=32, help='Rozmiar paczki')
    parser.add_argument('--model_name', type=str, default='resnet50-test', help='Nazwa eksperymentu w W&B')
    return parser.parse_args()

def main():
    args = parse_args() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="melanoma-detection",
        name=args.model_name,
        config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        }
    )
    
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)
    model = get_model(num_classes=2).to(device)

    trained_model = run_training(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device
    )

    torch.save(trained_model.state_dict(), "model.pth")
    wandb.save("model.pth")
    wandb.finish()

if __name__ == "__main__":
    main()