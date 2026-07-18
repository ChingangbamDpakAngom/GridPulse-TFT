import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.processing.cleaner import Cleaner


def test_cleaner_resample_hourly() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 30, tzinfo=UTC),
                datetime(2024, 1, 1, 1, 0, tzinfo=UTC),
            ],
            "intensity_actual": [100, 110, 120],
        }
    )
    cleaner = Cleaner()
    resampled = cleaner._resample_hourly(df)
    assert len(resampled) == 2


def test_cleaner_resample_preserves_region() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, 0, 0, tzinfo=UTC)] * 2,
            "region_id": ["london", "wales_s"],
            "temperature_c": [10.0, 12.0],
        }
    )
    cleaner = Cleaner()
    resampled = cleaner._resample_hourly(df)
    assert "region_id" in resampled.columns
    assert set(resampled["region_id"]) == {"london", "wales_s"}
    # Regions must not be averaged together
    assert len(resampled) == 2


def test_cleaner_interpolate_gaps() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 1, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 3, 0, tzinfo=UTC),
            ],
            "value": [100, 110, 140],
        }
    )
    cleaner = Cleaner()
    result = cleaner._interpolate_gaps(df, max_gap_hours=2)
    assert len(result) == 3
    assert np.isclose(result.iloc[2]["value"], 140)


def test_cleaner_interpolate_fills_nan() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 1, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 2, 0, tzinfo=UTC),
            ],
            "value": [100.0, np.nan, 140.0],
        }
    )
    cleaner = Cleaner()
    result = cleaner._interpolate_gaps(df, max_gap_hours=2)
    assert np.isclose(result.iloc[1]["value"], 120.0)
    assert bool(result.iloc[1]["gap_filled"]) is True
    assert bool(result.iloc[0]["gap_filled"]) is False


def test_cleaner_reads_ingested_grid_file(tmp_path, monkeypatch) -> None:
    date = datetime(2024, 1, 1, tzinfo=UTC)
    date_dir = tmp_path / "grid" / date.strftime("%Y/%m/%d")
    date_dir.mkdir(parents=True)
    payload = {
        "intensity": {
            "data": [
                {
                    "from": "2024-01-01T00:00Z",
                    "to": "2024-01-01T00:30Z",
                    "intensity": {"forecast": 100, "actual": 105, "index": "moderate"},
                }
            ]
        },
        "generation": {},
        "fetched_at": "2024-01-01T01:00:00+00:00",
    }
    (date_dir / "grid_0000_abc.json").write_text(json.dumps(payload))

    cleaner = Cleaner()
    cleaner.raw_grid_root = tmp_path / "grid"
    records = cleaner._read_grid_files(date)
    assert len(records) == 1
    assert records[0]["intensity_actual"] == 105
    assert records[0]["region_id"] == "GB"
