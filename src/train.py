import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

def run_training(model, train_loader, test_loader, num_epochs=20, lr=0.0001,
                 device="cpu", patience=5, weight_decay=1e-4):

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0.0
    epochs_no_improve = 0

    epoch_bar = tqdm(range(num_epochs), desc="Trening", unit="epoka")

    for epoch in epoch_bar:
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        train_bar = tqdm(train_loader, desc=f"  Epoka {epoch+1} [train]", 
                         leave=False, unit="batch")

        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_bar.set_postfix(loss=f"{loss.item():.4f}", 
                                  acc=f"{100*correct/total:.1f}%")

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        scheduler.step()

        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}", train_acc=f"{train_acc:.1f}%",
            val_loss=f"{val_loss:.4f}",   val_acc=f"{val_acc:.1f}%"
        )

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss,     "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0]
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                tqdm.write(f"Early stopping po {epoch+1} epokach.")
                break

    model.load_state_dict(torch.load("best_model.pth"))
    return model


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    val_bar = tqdm(loader, desc="  Walidacja", leave=False, unit="batch")

    with torch.no_grad():
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            val_bar.set_postfix(acc=f"{100*correct/total:.1f}%")

    return total_loss / len(loader), 100 * correct / total