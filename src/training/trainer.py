import argparse
import hashlib
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.config import settings
from src.constants import HORIZON, LOOKBACK
from src.models import PinballLoss, get_dataloaders, get_feature_columns, get_model

logger = logging.getLogger(__name__)

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()[:8]
        )
    except Exception:
        return "unknown"


def get_parquet_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        optimizer.zero_grad()
        if scaler:
            with autocast("cuda"):
                y_pred = model(x)
                loss = criterion(y_pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def train(
    parquet_path: Path,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    resume: Path | None = None,
    use_wandb: bool = True,
) -> Path:
    set_seed(settings.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    train_loader, val_loader, _ = get_dataloaders(parquet_path, batch_size=batch_size)

    input_dim = len(get_feature_columns())
    model = get_model(input_dim=input_dim).to(device)
    criterion = PinballLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    scaler = GradScaler("cuda") if device.type == "cuda" else None

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    wandb_run_id = None

    if resume and resume.exists():
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        wandb_run_id = checkpoint.get("wandb_run_id")
        logger.info(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")

    if use_wandb and settings.wandb_api_key:
        wandb.init(
            project=settings.wandb_project,
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "lookback": LOOKBACK,
                "horizon": HORIZON,
                "input_dim": input_dim,
                "seed": settings.seed,
            },
            id=wandb_run_id,
            resume="allow",
        )
        wandb.watch(model)

    settings.models_root.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = resume if resume and resume.exists() else None

    for epoch in range(start_epoch, epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        logger.info(
            f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
        )

        if use_wandb and settings.wandb_api_key:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            timestamp = datetime.now(UTC).strftime("%Y%m%d")
            checkpoint_path = settings.models_root / f"forecaster_v1.0_{timestamp}.pth"
            run_id = None
            if use_wandb and settings.wandb_api_key and wandb.run is not None:
                run_id = wandb.run.id
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "wandb_run_id": run_id,
                    "git_hash": get_git_hash(),
                    "parquet_hash": get_parquet_hash(parquet_path),
                    "seed": settings.seed,
                    "feature_columns": get_feature_columns(),
                },
                checkpoint_path,
            )
            best_checkpoint_path = checkpoint_path
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    if use_wandb and settings.wandb_api_key:
        wandb.finish()

    if best_checkpoint_path is None:
        msg = "Training produced no checkpoint (no epoch improved on the initial loss)"
        raise RuntimeError(msg)
    return best_checkpoint_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.parquet is None:
        processed_dir = settings.data_root / "processed"
        files = sorted(processed_dir.glob("grid_features_v*.parquet"))
        if not files:
            raise FileNotFoundError("No parquet files found. Run make preprocess first.")
        args.parquet = files[-1]

    train(args.parquet, args.epochs, args.batch_size, args.lr, args.resume, not args.no_wandb)


if __name__ == "__main__":
    main()
