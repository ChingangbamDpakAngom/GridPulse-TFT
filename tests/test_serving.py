"""End-to-end serving tests: features -> parquet -> (ONNX) -> /forecast."""

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

import src.serving.inference as inference
from src.config import settings
from src.models.dataset import get_feature_columns
from src.models.tft import TemporalFusionLite
from src.processing.features import run_feature_engineering
from src.serving.api import app
from src.serving.schemas import ForecastRequest
from src.training.export import export_to_onnx, update_manifest

N_HOURS = 600


def _make_synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = [start + timedelta(hours=i) for i in range(N_HOURS)]
    hours = np.arange(N_HOURS)
    noise = np.random.default_rng(0).normal(0, 5, N_HOURS)
    intensity = 180 + 60 * np.sin(2 * np.pi * hours / 24) + noise

    grid_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "region_id": "GB",
            "intensity_actual": intensity,
            "intensity_forecast": intensity + 5,
        }
    )
    weather_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "region_id": "GB",
            "temperature_c": 10 + 5 * np.sin(2 * np.pi * hours / 24),
            "wind_speed_ms": 6.0,
            "cloud_cover_pct": 50.0,
        }
    )
    return grid_df, weather_df


@pytest.fixture
def data_env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_root", tmp_path / "data")
    monkeypatch.setattr(settings, "models_root", tmp_path / "models")
    monkeypatch.setattr(settings, "logs_root", tmp_path / "logs")
    # reset cached ONNX session between tests
    inference._SESSION = None
    inference._SESSION_PATH = None
    inference._MANIFEST_ENTRY = None

    grid_df, weather_df = _make_synthetic_frames()
    return run_feature_engineering(grid_df, weather_df)


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    async def _no_call(prompt: str):
        return None

    monkeypatch.setattr("src.serving.copilot.call_llm", _no_call)


async def test_forecast_seasonal_fallback(data_env):
    response = await inference.get_forecast(ForecastRequest())
    assert response.model_version == "seasonal-naive-fallback"
    assert len(response.predictions) == 48
    p = response.predictions[0]
    assert p.quantile_10 <= p.quantile_50 <= p.quantile_90
    assert "fallback" in response.copilot_insight.dispatch_strategy_summary.lower()


async def test_forecast_onnx_path(data_env, tmp_path):
    feature_cols = get_feature_columns()
    model = TemporalFusionLite(input_dim=len(feature_cols))
    checkpoint_path = tmp_path / "ckpt.pth"
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_columns": feature_cols,
            "best_val_loss": 1.0,
            "git_hash": "test",
            "parquet_hash": "test",
            "seed": 42,
        },
        checkpoint_path,
    )
    settings.models_root.mkdir(parents=True, exist_ok=True)
    onnx_path = settings.models_root / "forecaster_v1.0_test.onnx"
    entry = export_to_onnx(checkpoint_path, onnx_path)
    update_manifest(entry)

    response = await inference.get_forecast(ForecastRequest())
    assert response.model_version.startswith("onnx-")
    assert len(response.predictions) == 48
    for p in response.predictions:
        assert p.quantile_10 <= p.quantile_50 <= p.quantile_90


async def test_forecast_respects_horizon(data_env):
    response = await inference.get_forecast(ForecastRequest(horizon_hours=12))
    assert len(response.predictions) == 12


def test_api_healthz_degraded_without_model(data_env):
    with TestClient(app) as client:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "degraded"


def test_api_forecast_end_to_end(data_env):
    with TestClient(app) as client:
        resp = client.post("/forecast", json={"region_id": "GB", "horizon_hours": 48})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["predictions"]) == 48
        assert body["copilot_insight"]["dispatch_strategy_summary"]


def test_api_forecast_503_without_data(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_root", tmp_path / "empty")
    monkeypatch.setattr(settings, "models_root", tmp_path / "empty_models")
    with TestClient(app) as client:
        resp = client.post("/forecast", json={})
        assert resp.status_code == 503


def test_api_rejects_bad_horizon(data_env):
    with TestClient(app) as client:
        resp = client.post("/forecast", json={"horizon_hours": 500})
        assert resp.status_code == 422


def test_manifest_is_valid_json(data_env, tmp_path):
    settings.models_root.mkdir(parents=True, exist_ok=True)
    update_manifest({"version": "1.0", "onnx": "x.onnx", "trained_at": "2024-01-01T00:00:00+00:00"})
    manifest = json.loads((settings.models_root / "manifest.json").read_text())
    assert manifest["models"][0]["version"] == "1.0"
