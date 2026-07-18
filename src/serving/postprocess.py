import logging
from datetime import datetime, timedelta

import numpy as np

from src.serving.schemas import ChargeWindow, ForecastQuantiles

logger = logging.getLogger(__name__)


def select_charge_windows(
    quantile_10: np.ndarray,
    timestamps: list[datetime],
    horizon_hours: int = 48,
    window_hours: int = 2,
    max_windows: int = 3,
    threshold_multiplier: float = 1.0,
) -> list[ChargeWindow]:
    if len(quantile_10) != horizon_hours:
        msg = f"Expected {horizon_hours} values, got {len(quantile_10)}"
        raise ValueError(msg)
    if horizon_hours < window_hours:
        return []

    historical_median = float(np.median(quantile_10))

    candidates = []
    for i in range(horizon_hours - window_hours + 1):
        window_vals = quantile_10[i : i + window_hours]
        avg_intensity = float(np.mean(window_vals))
        if avg_intensity < historical_median * threshold_multiplier:
            candidates.append((i, avg_intensity))

    candidates.sort(key=lambda x: x[1])

    selected = []
    used_hours = set()

    for start_idx, avg_intensity in candidates:
        window_hours_set = set(range(start_idx, start_idx + window_hours))
        if window_hours_set & used_hours:
            continue

        start_ts = timestamps[start_idx]
        end_ts = start_ts + timedelta(hours=window_hours)

        selected.append(
            ChargeWindow(
                start=start_ts, end=end_ts, expected_intensity_gco2_kwh=avg_intensity, rationale=""
            )
        )
        used_hours.update(window_hours_set)

        if len(selected) >= max_windows:
            break

    selected.sort(key=lambda w: w.start)
    logger.info(f"Selected {len(selected)} charge windows from {len(candidates)} candidates")
    return selected


def build_forecast_quantiles(
    quantiles: np.ndarray, timestamps: list[datetime]
) -> list[ForecastQuantiles]:
    if quantiles.shape != (len(timestamps), 3):
        msg = f"Expected ({len(timestamps)}, 3), got {quantiles.shape}"
        raise ValueError(msg)

    return [
        ForecastQuantiles(
            timestamp=timestamps[i],
            quantile_10=float(quantiles[i, 0]),
            quantile_50=float(quantiles[i, 1]),
            quantile_90=float(quantiles[i, 2]),
        )
        for i in range(len(timestamps))
    ]
