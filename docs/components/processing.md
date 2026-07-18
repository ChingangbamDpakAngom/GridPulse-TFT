# Processing Component

[[_TOC_]]

## Overview

Two-stage pipeline: **Cleaner** → **Feature Engineer** → Parquet.

## Module Map

| Path | Responsibility |
|------|----------------|
| `src/processing/cleaner.py` | Time-align, resample to hourly, gap fill |
| `src/processing/features.py` | Cyclical calendar, lag embeddings, drift detection |

## Cleaner (`cleaner.py`)

### Resampling

- Input: half-hourly grid + hourly weather
- Method: `resample("1h").mean()` on numeric columns
- Output: hourly timestamps (UTC, floor to hour)

### Gap Handling

| Gap Length | Action |
|------------|--------|
| ≤ 2 hours | Linear interpolation (`interpolate(method="time")`) |
| > 2 hours | Forward-fill + `gap_filled=True` flag column |

### Usage

```python
from src.processing.cleaner import clean_raw_data
from datetime import datetime, timedelta

end = datetime.utcnow()
start = end - timedelta(days=7)
grid_df, weather_df = clean_raw_data(start, end)
```

## Feature Engineer (`features.py`)

### Feature Groups

| Group | Columns |
|-------|---------|
| Identity | `timestamp`, `region_id` |
| Target | `intensity_actual` (nullable) |
| Cyclical calendar | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `doy_sin`, `doy_cos` |
| Lags | `lag_1h`, `lag_24h`, `lag_168h` (grouped by region) |
| Weather | `temperature_c`, `wind_speed_ms`, `cloud_cover_pct`, `solar_irradiance_wm2` |
| Quality | `gap_filled` (bool) |

### Cyclical Encoding

```python
hour_sin = sin(2π * hour / 24)
hour_cos = cos(2π * hour / 24)
# Similarly for day-of-week (7) and day-of-year (365.25)
```

### Lag Features

- Shifted within each `region_id` group
- `lag_1h` = previous hour actual
- `lag_24h` = same hour yesterday
- `lag_168h` = same hour last week

### Solar Irradiance Proxy

```python
solar_irradiance_wm2 = 1000 * (1 - cloud_cover_pct / 100).clip(0, 1)
```

### Drift Detection (KS Test)

After each feature build, compares new Parquet vs previous on:

- `intensity_actual`
- `temperature_c`
- `wind_speed_ms`
- `cloud_cover_pct`
- `solar_irradiance_wm2`

If KS-statistic > `DRIFT_KS_THRESHOLD` (default 0.15) → writes to `logs/drift_alerts.jsonl`.

### Atomic Parquet Write

```python
tmp = path.with_suffix(".parquet.tmp")
pq.write_table(table, tmp, compression="snappy")
tmp.replace(path)  # atomic on POSIX + Windows NTFS
```

### Usage

```python
from src.processing.features import run_feature_engineering

parquet_path = run_feature_engineering(grid_df, weather_df, version=1)
# → data/processed/grid_features_v1.parquet
```

## Schema Validation

Before write, validates columns match `FEATURE_COLUMNS` (26 cols). Mismatch → `SchemaMismatchError`, no bad file shipped.

## Related

- [[Ingestion|Ingestion Component]]
- [[Training|Training Component]]
- [[Architecture#layer-2-preprocessing-columnar-storage|Architecture: Layer 2]]