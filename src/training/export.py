import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import torch

from src.config import settings
from src.models import get_feature_columns, get_model

logger = logging.getLogger(__name__)


def export_to_onnx(checkpoint_path: Path, output_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    feature_columns = checkpoint.get("feature_columns") or get_feature_columns()
    input_dim = len(feature_columns)
    model = get_model(input_dim=input_dim)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dummy_input = torch.randn(1, 168, input_dim)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["historical_features"],
        output_names=["quantile_forecasts"],
        dynamic_axes={
            "historical_features": {0: "batch_size"},
            "quantile_forecasts": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    logger.info(f"Exported ONNX model to {output_path}")

    return {
        "version": "1.0",
        "checkpoint": str(checkpoint_path),
        "onnx": str(output_path),
        "trained_at": datetime.now(UTC).isoformat(),
        "git_hash": checkpoint.get("git_hash", "unknown"),
        "parquet_hash": checkpoint.get("parquet_hash", "unknown"),
        "seed": checkpoint.get("seed", 42),
        "feature_columns": feature_columns,
        "metrics": {"pinball_val": checkpoint.get("best_val_loss", 0.0)},
    }


def update_manifest(entry: dict):
    manifest_path = settings.models_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        with manifest_path.open() as f:
            manifest = json.load(f)
    else:
        manifest = {"models": []}

    manifest["models"].append(entry)
    manifest["models"].sort(key=lambda x: x["trained_at"], reverse=True)

    tmp = manifest_path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(manifest, f, indent=2)
    tmp.replace(manifest_path)

    logger.info(f"Updated manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.checkpoint is None:
        checkpoints = sorted(settings.models_root.glob("forecaster_v*.pth"))
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found. Run make train first.")
        args.checkpoint = checkpoints[-1]

    timestamp = datetime.now(UTC).strftime("%Y%m%d")
    if args.output is None:
        args.output = settings.models_root / f"forecaster_v1.0_{timestamp}.onnx"

    entry = export_to_onnx(args.checkpoint, args.output)
    update_manifest(entry)


if __name__ == "__main__":
    main()
