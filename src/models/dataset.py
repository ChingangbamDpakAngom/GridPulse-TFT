import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from src.constants import HORIZON, LOOKBACK, TARGET_COL
from src.processing.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

# Columns that are not numeric model inputs.
NON_FEATURE_COLS = ["timestamp", "region_id", "gap_filled", TARGET_COL]


def get_feature_columns() -> list[str]:
    return [c for c in FEATURE_COLUMNS if c not in NON_FEATURE_COLS]


class SmartGridDataset(Dataset):
    """Sliding-window dataset over rows [start_idx, end_idx) of a feature parquet.

    Every window (lookback + horizon) is fully contained within the given row
    range, so datasets built on disjoint ranges cannot leak into each other.
    """

    def __init__(
        self,
        parquet_path: Path,
        start_idx: int,
        end_idx: int,
        feature_cols: list[str] | None = None,
    ):
        self.parquet_path = parquet_path
        self.feature_cols = feature_cols or get_feature_columns()
        self.table = pq.read_table(parquet_path)

        end_idx = min(end_idx, len(self.table))
        last_valid_start = end_idx - (LOOKBACK + HORIZON)
        self.indices = list(range(start_idx, max(start_idx, last_valid_start + 1)))
        logger.info(f"Dataset: {len(self.indices)} samples from {parquet_path}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row_idx = self.indices[idx]
        window = self.table.slice(row_idx, LOOKBACK + HORIZON).to_pandas()

        features = window[self.feature_cols].to_numpy(dtype=np.float32)
        target = window[TARGET_COL].to_numpy(dtype=np.float32)

        x = np.nan_to_num(features[:LOOKBACK], nan=0.0)
        y = np.nan_to_num(target[LOOKBACK : LOOKBACK + HORIZON], nan=0.0)

        return {"x": torch.from_numpy(x), "y": torch.from_numpy(y)}


def get_dataloaders(
    parquet_path: Path,
    batch_size: int = 64,
    train_split: float = 0.6,
    val_split: float = 0.2,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    table = pq.read_table(parquet_path)
    n = len(table)

    min_rows = 3 * (LOOKBACK + HORIZON)
    if n < min_rows:
        msg = f"Need at least {min_rows} rows for train/val/test windows, got {n}"
        raise ValueError(msg)

    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    # Each split owns a disjoint row range; windows never cross a boundary.
    train_dataset = SmartGridDataset(parquet_path, 0, train_end)
    val_dataset = SmartGridDataset(parquet_path, train_end, val_end)
    test_dataset = SmartGridDataset(parquet_path, val_end, n)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    return train_loader, val_loader, test_loader
