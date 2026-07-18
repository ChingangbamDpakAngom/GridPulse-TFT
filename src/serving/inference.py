"""Model loading and forecast generation.

Three-tier strategy: ONNX model -> seasonal-naive fallback -> 503 (no data).
Torch is intentionally not imported here; serving relies on onnxruntime only.
"""

import json
import logging
import re
import time
from datetime import UTC, timedelta
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from src.config import settings
from src.constants import HORIZON, LOOKBACK, TARGET_COL
from src.serving.copilot import get_copilot_recommendation
from src.serving.postprocess import build_forecast_quantiles, select_charge_windows
from src.serving.schemas import ForecastRequest, ServerResponseSchema

logger = logging.getLogger(__name__)

_SESSION: ort.InferenceSession | None = None
_SESSION_PATH: Path | None = None
_MANIFEST_ENTRY: dict | None = None


def load_latest_model() -> tuple[ort.InferenceSession, dict]:
    """Load the newest ONNX model listed in models/manifest.json (cached per path)."""
    global _SESSION, _SESSION_PATH, _MANIFEST_ENTRY

    manifest_path = settings.models_root / "manifest.json"
    if not manifest_path.exists():
        msg = f"Model manifest not found: {manifest_path}. Run `make train && make export`."
        raise FileNotFoundError(msg)

    manifest = json.loads(manifest_path.read_text())
    models = manifest.get("models", [])
    if not models:
        msg = "Model manifest is empty. Run `make train && make export`."
        raise FileNotFoundError(msg)

    entry = models[0]  # manifest is sorted newest-first by export.py
    onnx_path = Path(entry["onnx"])
    if not onnx_path.is_absolute():
        onnx_path = settings.project_root / onnx_path
    if not onnx_path.exists():
        msg = f"ONNX model missing on disk: {onnx_path}"
        raise FileNotFoundError(msg)

    if _SESSION is None or onnx_path != _SESSION_PATH:
        _SESSION = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        _SESSION_PATH = onnx_path
        _MANIFEST_ENTRY = entry
        logger.info(f"Loaded ONNX model {onnx_path}")

    return _SESSION, _MANIFEST_ENTRY


def _latest_parquet() -> Path:
    processed = settings.processed_root
    best: tuple[int, Path] | None = None
    if processed.exists():
        for f in processed.glob("grid_features_v*.parquet"):
            match = re.match(r"grid_features_v(\d+)\.parquet$", f.name)
            if match:
                version = int(match.group(1))
                if best is None or version > best[0]:
                    best = (version, f)
    if best is None:
        msg = "No processed feature parquet found. Run `make ingest && make preprocess`."
        raise FileNotFoundError(msg)
    return best[1]


def load_latest_features(region_id: str = "GB") -> pd.DataFrame:
    df = pd.read_parquet(_latest_parquet())
    if df.empty:
        msg = "Processed feature parquet is empty. Re-run `make preprocess` with fresh data."
        raise FileNotFoundError(msg)
    if "region_id" in df.columns and region_id in set(df["region_id"].dropna()):
        df = df[df["region_id"] == region_id]
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def predict_onnx(session: ort.InferenceSession, entry: dict, df: pd.DataFrame) -> np.ndarray:
    feature_cols = entry.get("feature_columns")
    if not feature_cols:
        msg = "Manifest entry has no feature_columns"
        raise ValueError(msg)
    if len(df) < LOOKBACK:
        msg = f"Need {LOOKBACK} hourly rows for the model input window, have {len(df)}"
        raise ValueError(msg)

    window = df.tail(LOOKBACK)
    x = window[feature_cols].to_numpy(dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0)[np.newaxis, :, :]  # (1, LOOKBACK, F)

    input_name = session.get_inputs()[0].name
    (output,) = session.run(None, {input_name: x})
    return output[0]  # (HORIZON, 3)


def seasonal_naive_forecast(df: pd.DataFrame, horizon: int = HORIZON) -> np.ndarray:
    """Hour-of-day climatology over the last 7 days with an empirical q10/q90 spread."""
    history = df.dropna(subset=[TARGET_COL])
    if history.empty:
        msg = "No historical intensity values available for fallback forecast"
        raise FileNotFoundError(msg)

    recent = history.tail(24 * 7).copy()
    recent["hour"] = recent["timestamp"].dt.hour
    by_hour_mean = recent.groupby("hour")[TARGET_COL].mean()
    by_hour_std = recent.groupby("hour")[TARGET_COL].std()
    overall_mean = float(recent[TARGET_COL].mean())
    overall_std = float(recent[TARGET_COL].std() or overall_mean * 0.15)

    last_ts = df["timestamp"].max()
    quantiles = np.zeros((horizon, 3), dtype=np.float32)
    for i in range(horizon):
        hour = (last_ts + timedelta(hours=i + 1)).hour
        q50 = float(by_hour_mean.get(hour, overall_mean))
        std = float(by_hour_std.get(hour) or overall_std)
        spread = 1.28 * std  # ~10th/90th percentile under normality
        quantiles[i] = [max(q50 - spread, 0.0), q50, q50 + spread]
    return quantiles


async def get_forecast(request: ForecastRequest) -> ServerResponseSchema:
    start = time.perf_counter()
    horizon = min(request.horizon_hours, HORIZON)

    df = load_latest_features(request.region_id)

    try:
        session, entry = load_latest_model()
        quantiles = predict_onnx(session, entry, df)
        model_version = f"onnx-{entry.get('version', '?')}"
    except Exception as e:
        if isinstance(e, FileNotFoundError):
            logger.warning(f"No ONNX model available ({e}); using seasonal-naive fallback")
        else:
            logger.exception("ONNX inference failed; using seasonal-naive fallback")
        quantiles = seasonal_naive_forecast(df)
        model_version = "seasonal-naive-fallback"

    quantiles = np.sort(np.asarray(quantiles, dtype=np.float64), axis=1)[:horizon]

    last_ts = df["timestamp"].max().to_pydatetime().astimezone(UTC)
    timestamps = [last_ts + timedelta(hours=i + 1) for i in range(horizon)]

    predictions = build_forecast_quantiles(quantiles, timestamps)
    windows = select_charge_windows(quantiles[:, 0], timestamps, horizon_hours=horizon)
    copilot = await get_copilot_recommendation(windows, quantiles[:, 1].tolist())

    latency_ms = (time.perf_counter() - start) * 1000
    return ServerResponseSchema(
        execution_latency_ms=round(latency_ms, 2),
        model_version=model_version,
        predictions=predictions,
        copilot_insight=copilot,
    )
