import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats

from src.config import settings

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "timestamp",
    "region_id",
    "intensity_actual",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "temperature_c",
    "wind_speed_ms",
    "cloud_cover_pct",
    "solar_irradiance_wm2",
    "gap_filled",
]


def add_cyclical_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col], utc=True)
    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    doy = ts.dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    return df


def add_lag_features(
    df: pd.DataFrame, target_col: str = "intensity_actual", lags: list[int] | None = None
) -> pd.DataFrame:
    if lags is None:
        lags = [1, 24, 168]
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}h"] = df.groupby("region_id")[target_col].shift(lag)
    return df


def estimate_solar_irradiance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "cloud_cover_pct" in df.columns:
        df["solar_irradiance_wm2"] = 1000 * (1 - df["cloud_cover_pct"] / 100).clip(0, 1)
    else:
        df["solar_irradiance_wm2"] = 0.0
    return df


def detect_drift(
    current: pd.DataFrame, previous: pd.DataFrame, columns: list[str], threshold: float = 0.15
) -> list[dict]:
    alerts = []
    for col in columns:
        if col not in current.columns or col not in previous.columns:
            continue
        curr_vals = current[col].dropna()
        prev_vals = previous[col].dropna()
        if len(curr_vals) < 10 or len(prev_vals) < 10:
            continue
        ks_stat, p_value = stats.ks_2samp(curr_vals, prev_vals)
        if ks_stat > threshold:
            alerts.append(
                {
                    "column": col,
                    "ks_statistic": float(ks_stat),
                    "p_value": float(p_value),
                    "threshold": threshold,
                    "current_mean": float(curr_vals.mean()),
                    "previous_mean": float(prev_vals.mean()),
                    "detected_at": datetime.now(UTC).isoformat(),
                }
            )
    return alerts


def build_features(grid_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    if grid_df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    grid_df = grid_df.copy()
    grid_df["timestamp"] = pd.to_datetime(grid_df["timestamp"], utc=True)
    grid_df = grid_df.sort_values(["region_id", "timestamp"]).reset_index(drop=True)

    grid_df = add_cyclical_features(grid_df)
    grid_df = add_lag_features(grid_df)

    if not weather_df.empty:
        weather_df = weather_df.copy()
        weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], utc=True)
        weather_df = weather_df.sort_values(["region_id", "timestamp"]).reset_index(drop=True)
        weather_df = estimate_solar_irradiance(weather_df)

        merged = pd.merge_asof(
            grid_df.sort_values("timestamp"),
            weather_df.sort_values("timestamp"),
            on="timestamp",
            by="region_id",
            direction="backward",
            tolerance=pd.Timedelta(hours=1),
        )
    else:
        merged = grid_df
        for col in [
            "temperature_c",
            "wind_speed_ms",
            "cloud_cover_pct",
            "solar_irradiance_wm2",
            "gap_filled",
        ]:
            if col not in merged.columns:
                merged[col] = np.nan

    if "gap_filled" not in merged.columns:
        merged["gap_filled"] = False

    for col in FEATURE_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan

    return merged[FEATURE_COLUMNS].reset_index(drop=True)


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")

    table = pa.Table.from_pandas(df)
    pq.write_table(table, tmp)
    tmp.replace(path)

    schema_hash = hashlib.sha256(str(table.schema).encode()).hexdigest()[:16]
    logger.info(f"Wrote {len(df)} rows to {path} (schema: {schema_hash})")
    return schema_hash


def get_previous_parquet(processed_dir: Path) -> pd.DataFrame | None:
    files = sorted(processed_dir.glob("grid_features_v*.parquet"))
    if len(files) < 2:
        return None
    return pd.read_parquet(files[-2])


def next_version(processed_dir: Path) -> int:
    versions = []
    for f in processed_dir.glob("grid_features_v*.parquet"):
        match = re.match(r"grid_features_v(\d+)\.parquet$", f.name)
        if match:
            versions.append(int(match.group(1)))
    return max(versions, default=0) + 1


def run_feature_engineering(
    grid_df: pd.DataFrame, weather_df: pd.DataFrame, version: int | None = None
) -> Path:
    processed_dir = settings.processed_root
    processed_dir.mkdir(parents=True, exist_ok=True)

    features_df = build_features(grid_df, weather_df)

    if version is None:
        version = next_version(processed_dir)

    output_path = processed_dir / f"grid_features_v{version}.parquet"
    schema_hash = write_parquet_atomic(features_df, output_path)

    prev_df = get_previous_parquet(processed_dir)
    if prev_df is not None:
        drift_cols = [
            "intensity_actual",
            "temperature_c",
            "wind_speed_ms",
            "cloud_cover_pct",
            "solar_irradiance_wm2",
        ]
        alerts = detect_drift(features_df, prev_df, drift_cols, settings.drift_ks_threshold)
        if alerts:
            log_alert = settings.logs_root / "drift_alerts.jsonl"
            log_alert.parent.mkdir(parents=True, exist_ok=True)
            with log_alert.open("a") as f:
                for alert in alerts:
                    alert["version"] = version
                    alert["schema_hash"] = schema_hash
                    f.write(json.dumps(alert) + "\n")
            logger.warning(f"Drift detected: {len(alerts)} columns exceeded threshold")

    return output_path


if __name__ == "__main__":
    import argparse

    from src.processing.cleaner import clean_raw_data

    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    end = datetime.now(UTC)
    start = end - pd.Timedelta(days=args.days)
    grid_df, weather_df = clean_raw_data(start, end)
    out = run_feature_engineering(grid_df, weather_df)
    print(f"Features written to {out}")
