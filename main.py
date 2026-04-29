import argparse
import torch
import wandb
from pathlib import Path
from src.dataset import get_dataloaders
from src.model import get_model
from src.train import run_training


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="Trening modelu na czerniaka")
    parser.add_argument('--lr',             type=float, default=0.0001)
    parser.add_argument('--epochs',         type=int,   default=20)
    parser.add_argument('--batch_size',     type=int,   default=32)
    parser.add_argument('--patience',       type=int,   default=5)
    parser.add_argument('--unfreeze_layers',type=int,   default=1)
    parser.add_argument('--weight_decay',   type=float, default=1e-4)
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--model_name',     type=str,   default='resnet50')
    parser.add_argument('--output_dir',     type=str,   default='checkpoints')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używam: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project="melanoma-detection",
        name=args.model_name,
        config=vars(args)  # przekazuje wszystkie args naraz
    )

    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)
    model = get_model(num_classes=2, unfreeze_layers=args.unfreeze_layers).to(device)

    # Loguj liczbę parametrów
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametry: {total:,} total | {trainable:,} trenowalne")
    wandb.config.update({"trainable_params": trainable})

    trained_model = run_training(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        patience=args.patience,
        weight_decay=args.weight_decay,
    )

    checkpoint_path = output_dir / f"{args.model_name}.pth"
    torch.save({
        "model_state_dict": trained_model.state_dict(),
        "config": vars(args),
    }, checkpoint_path)

    wandb.save(str(checkpoint_path))
    wandb.finish()
    print(f"Model zapisany: {checkpoint_path}")


if __name__ == "__main__":
    main()