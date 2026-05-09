import argparse
import torch
import wandb
from src.dataset import get_dataloaders
from src.model import get_model
from src.train import run_training


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="Trening modelu na czerniaka")
    parser.add_argument('--lr',              type=float, default=0.0001)
    parser.add_argument('--epochs',          type=int,   default=5)
    parser.add_argument('--batch_size',      type=int,   default=32)
    parser.add_argument('--patience',        type=int,   default=2)
    parser.add_argument('--unfreeze_layers', type=int,   default=1)
    parser.add_argument('--weight_decay',    type=float, default=1e-4)
    parser.add_argument('--seed',            type=int,   default=42)
    parser.add_argument('--model_name',      type=str,   default='model')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używam: {device}")

    wandb.init(
        project="melanoma-detection",
        name=args.model_name,
        config=vars(args)
    )

    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)
    model = get_model(num_classes=2, unfreeze_layers=args.unfreeze_layers).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametry: {total:,} total | {trainable:,} trenowalne")
    wandb.config.update({"trainable_params": trainable})

    run_training(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        args=args
    )

    wandb.finish()


if __name__ == "__main__":
    main()