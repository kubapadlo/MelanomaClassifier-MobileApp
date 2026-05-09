import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from pathlib import Path


def run_training(model, train_loader, test_loader, args, device="cpu"):

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    epochs_no_improve = 0

    best_model_path = Path("models") / f"{args.model_name}.pth"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            correct += (torch.max(outputs, 1)[1] == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        scheduler.step()

        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        print(f"Ep {epoch+1:02d}/{args.epochs} | Train {train_loss:.3f}/{train_acc:.1f}% | Val {val_loss:.3f}/{val_acc:.1f}%")

        wandb.log({"epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc,
                   "val_loss": val_loss, "val_acc": val_acc, "lr": scheduler.get_last_lr()[0]})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": vars(args),
                "val_acc": val_acc,
                "epoch": epoch
            }, best_model_path)
            print(f"  ✓ Nowy najlepszy model: {val_acc:.1f}%")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping po {epoch+1} epokach.")
                break

    checkpoint = torch.load(best_model_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs, 1)[1] == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), 100 * correct / total