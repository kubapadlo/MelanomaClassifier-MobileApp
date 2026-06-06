import argparse
import logging
import random
import sys

import numpy as np
import torch
import wandb

from src.dataset import get_dataloaders
from src.model import get_model
from src.train import run_training

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--epochs",          type=int,   default=20)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--patience",        type=int,   default=5)
    p.add_argument("--unfreeze_layers", type=int,   default=1)
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--model_name",      type=str,   default="melanoma_model")
    p.add_argument("--no_wandb",        action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Urządzenie: %s", device)

    wandb.init(project="melanoma-detection", name=args.model_name,
               config=vars(args), mode="disabled" if args.no_wandb else "online")

    try:
        train_loader, val_loader, test_loader = get_dataloaders(args.batch_size)
        model = get_model(num_classes=2, unfreeze_layers=args.unfreeze_layers).to(device)
        run_training(model, train_loader, val_loader, test_loader, args, device)
    except KeyboardInterrupt:
        logger.warning("Przerwano.")
        sys.exit(0)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()