import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

logger = logging.getLogger(__name__)

@dataclass
class Metrics:
    loss:        float
    accuracy:    float
    auc:         float
    f1:          float
    sensitivity: float
    specificity: float

    def __str__(self):
        return (f"loss={self.loss:.4f} | acc={self.accuracy:.1f}% | "
                f"AUC={self.auc:.4f} | F1={self.f1:.4f} | "
                f"sens={self.sensitivity:.4f} | spec={self.specificity:.4f}")


def evaluate(model, loader, criterion, device) -> Metrics:
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            all_probs.extend(torch.softmax(outputs, 1)[:, 1].cpu().numpy())
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    return Metrics(
        loss        = total_loss / len(loader),
        accuracy    = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels)),
        auc         = roc_auc_score(all_labels, all_probs),
        f1          = f1_score(all_labels, all_preds, average="macro", zero_division=0),
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0,
        specificity = tn / (tn + fp) if (tn + fp) else 0.0,
    )


def run_training(model, train_loader, val_loader, test_loader, args, device):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_auc  = 0.0
    no_improve    = 0
    best_path     = Path("models") / f"{args.model_name}.pth"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        # --- trening ---
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total   += labels.size(0)
        scheduler.step()

        train_acc = 100.0 * correct / total
        val_m     = evaluate(model, val_loader, criterion, device)

        logger.info("Ep %02d/%d | train loss=%.4f acc=%.1f%% | val %s",
                    epoch + 1, args.epochs, train_loss / len(train_loader), train_acc, val_m)
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss / len(train_loader),
                   "train_acc": train_acc, "lr": scheduler.get_last_lr()[0],
                   **{f"val_{k}": v for k, v in asdict(val_m).items()}})

        if val_m.auc > best_val_auc:
            best_val_auc = val_m.auc
            no_improve   = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_metrics": asdict(val_m),
                "epoch": epoch,
                "config": vars(args)  
            }, best_path)
            logger.info("  ✓ nowy best: AUC=%.4f", val_m.auc)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info("Early stopping po %d epokach.", epoch + 1)
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_m = evaluate(model, test_loader, criterion, device)
    logger.info("=== TEST === %s", test_m)
    wandb.log({f"test_{k}": v for k, v in asdict(test_m).items()})
    return model, test_m