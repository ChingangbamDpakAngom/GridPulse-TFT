import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
from pydantic import ValidationError

from src.config import settings
from src.ingestion.schemas import GridInboundSchema, WeatherInboundSchema

logger = logging.getLogger(__name__)

NATIONAL_REGION_ID = "GB"


class Cleaner:
    def __init__(self):
        self.raw_grid_root = settings.raw_grid_root
        self.raw_weather_root = settings.raw_weather_root

    def _read_grid_files(self, date: datetime) -> list[dict[str, Any]]:
        records = []
        date_dir = self.raw_grid_root / date.strftime("%Y/%m/%d")
        if not date_dir.exists():
            return records

        for file in sorted(date_dir.glob("*.json")):
            try:
                data = json.loads(file.read_text())
                # Ingestion wraps the API payload: {"intensity": {...}, "generation": ...}
                payload = data.get("intensity", data) if isinstance(data, dict) else data
                validated = GridInboundSchema.model_validate(payload)
                for point in validated.data:
                    intensity = point.intensity
                    records.append(
                        {
                            "timestamp": point.from_,
                            "region_id": NATIONAL_REGION_ID,
                            "intensity_actual": intensity.actual if intensity else None,
                            "intensity_forecast": intensity.forecast if intensity else None,
                        }
                    )
            except (ValidationError, json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse {file}: {e}")
        return records

    def _read_weather_files(self, date: datetime) -> list[dict[str, Any]]:
        records = []
        date_dir = self.raw_weather_root / date.strftime("%Y/%m/%d")
        if not date_dir.exists():
            return records

        for file in sorted(date_dir.glob("*.json")):
            try:
                data = json.loads(file.read_text())
                validated = WeatherInboundSchema.model_validate(data)
                for region in validated.regions:
                    for point in region.data:
                        records.append(
                            {
                                "timestamp": point.timestamp,
                                "region_id": region.region_id,
                                "temperature_c": point.temperature_c,
                                "wind_speed_ms": point.wind_speed_ms,
                                "wind_direction_deg": point.wind_direction_deg,
                                "precipitation_mm": point.precipitation_mm,
                                "cloud_cover_pct": point.cloud_cover_pct,
                                "humidity_pct": point.humidity_pct,
                                "pressure_hpa": point.pressure_hpa,
                            }
                        )
            except (ValidationError, json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse {file}: {e}")
        return records

    def _resample_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample to hourly means per region, inserting NaN rows for missing hours."""
        if df.empty:
            return df
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if "region_id" in df.columns:
            resampled = (
                df.set_index("timestamp")
                .groupby("region_id")[numeric_cols]
                .resample("1h")
                .mean()
                .reset_index()
            )
        else:
            resampled = df.set_index("timestamp")[numeric_cols].resample("1h").mean().reset_index()
        return resampled

    def _interpolate_gaps(self, df: pd.DataFrame, max_gap_hours: int = 2) -> pd.DataFrame:
        """Time-interpolate gaps up to max_gap_hours, forward-fill longer ones.

        Rows whose values were filled either way are flagged with gap_filled=True.
        """
        if df.empty:
            return df
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        def _fill_group(group: pd.DataFrame) -> pd.DataFrame:
            group = group.sort_values("timestamp").set_index("timestamp")
            numeric_cols = group.select_dtypes(include="number").columns
            was_nan = group[numeric_cols].isna().any(axis=1)
            group[numeric_cols] = group[numeric_cols].interpolate(
                method="time", limit=max_gap_hours
            )
            group[numeric_cols] = group[numeric_cols].ffill()
            is_filled = was_nan & group[numeric_cols].notna().all(axis=1)
            group["gap_filled"] = is_filled
            return group.reset_index()

        if "region_id" in df.columns:
            filled = pd.concat(
                [_fill_group(g) for _, g in df.groupby("region_id", sort=False)], ignore_index=True
            )
        else:
            filled = _fill_group(df)

        filled["gap_filled"] = filled["gap_filled"].fillna(value=False).astype(bool)  # noqa: FBT003
        return filled

    def process_day(self, date: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
        grid_records = self._read_grid_files(date)
        weather_records = self._read_weather_files(date)

        grid_df = pd.DataFrame(grid_records)
        weather_df = pd.DataFrame(weather_records)

        if not grid_df.empty:
            grid_df = self._resample_hourly(grid_df)
            grid_df = self._interpolate_gaps(grid_df)

        if not weather_df.empty:
            weather_df = self._resample_hourly(weather_df)
            weather_df = self._interpolate_gaps(weather_df)

        return grid_df, weather_df


def clean_raw_data(start_date: datetime, end_date: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaner = Cleaner()
    all_grid = []
    all_weather = []

    current = start_date
    while current <= end_date:
        grid_df, weather_df = cleaner.process_day(current)
        if not grid_df.empty:
            all_grid.append(grid_df)
        if not weather_df.empty:
            all_weather.append(weather_df)
        current += timedelta(days=1)

    grid_combined = pd.concat(all_grid, ignore_index=True) if all_grid else pd.DataFrame()
    weather_combined = pd.concat(all_weather, ignore_index=True) if all_weather else pd.DataFrame()

    # Duplicate hours can appear when multiple ingestion snapshots cover the same period.
    if not grid_combined.empty:
        grid_combined = (
            grid_combined.sort_values("timestamp")
            .drop_duplicates(subset=["region_id", "timestamp"], keep="last")
            .reset_index(drop=True)
        )
    if not weather_combined.empty:
        weather_combined = (
            weather_combined.sort_values("timestamp")
            .drop_duplicates(subset=["region_id", "timestamp"], keep="last")
            .reset_index(drop=True)
        )

    return grid_combined, weather_combined


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    end = datetime.now(UTC)
    start = end - timedelta(days=args.days)
    grid_df, weather_df = clean_raw_data(start, end)
    print(f"Grid: {len(grid_df)} rows, Weather: {len(weather_df)} rows")
