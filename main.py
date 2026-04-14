import torch
from src.dataset import get_dataloaders
from src.model import get_model
from src.train import run_training  

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, _ = get_dataloaders(batch_size=32)
    
    model = get_model(num_classes=2).to(device)
    
    trained_model = run_training(
        model=model, 
        train_loader=train_loader, 
        num_epochs=10, 
        lr=0.0001, 
        device=device
    )
    
    torch.save(trained_model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()