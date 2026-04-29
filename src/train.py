import torch
import torch.nn as nn
import torch.optim as optim
import wandb

def run_training(model, train_loader, test_loader, num_epochs=20, lr=0.0001, device="cpu", patience=5, weight_decay=1e-4):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing redukuje overfitting
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # --- TRAIN ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # stabilizuje trening
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        scheduler.step()

        # --- EVAL ---
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0]
        })

        # --- EARLY STOPPING + CHECKPOINT ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")  # zapisz najlepszy model
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping po {epoch+1} epokach.")
                break

    model.load_state_dict(torch.load("best_model.pth"))  # wróć do najlepszego
    return model


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss / len(loader), 100 * correct / total