# Quickstart

## Prerequisites

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or use pip
pip install uv
```

## 1. Clone & Install

```bash
git clone https://github.com/yourorg/gridpulse-tft
cd gridpulse-tft
uv pip install -e ".[dev]"
```

## 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys:
# - MET_OFFICE_API_KEY (required)
# - WANDB_API_KEY (optional, for experiment tracking)
# - LLM_API_KEY (optional, for copilot)
```

## 3. Ingest Data

```bash
# Fetch latest grid + weather data
make ingest
# Or manually:
python -m src.ingestion.grid_client
python -m src.ingestion.weather_client
```

Raw JSON lands in `data/raw/grid/YYYY/MM/DD/` and `data/raw/weather/YYYY/MM/DD/`.

## 4. Preprocess & Feature Engineer

```bash
make preprocess
# Or:
python -m src.processing.cleaner
python -m src.processing.features
```

Outputs `data/processed/grid_features_v1.parquet` with 26 columns (cyclical calendar, lags, weather).

## 5. Train Model

```bash
make train
# Or with options:
python -m src.training.trainer --epochs 50 --batch-size 64 --lr 1e-3
```

Checkpoints saved to `models/forecaster_v1.0_YYYYMMDD.pth`. Metrics logged to Weights & Biases.

## 6. Export to ONNX

```bash
make export
# Or:
python -m src.training.export
```

Creates `models/forecaster_v1.0_YYYYMMDD.onnx` and updates `models/manifest.json`.

## 7. Serve API

```bash
make serve
# Or:
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
```

Health check: `curl http://localhost:8000/healthz`

## 8. Query Forecast

```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"region_id": "GB", "horizon_hours": 48}'
```

Response includes quantile forecasts, optimal charge windows, and copilot summary.

## 9. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Open http://localhost:8501 for interactive charts.

## Make Targets Summary

| Target | Description |
|--------|-------------|
| `make ingest` | Fetch latest grid & weather data |
| `make preprocess` | Clean + feature engineer → Parquet |
| `make train` | Train TFT-lite (resume with `--resume`) |
| `make export` | PyTorch → ONNX + manifest update |
| `make serve` | Start FastAPI server |
| `make test` | Run pytest suite |
| `make lint` | Ruff check + format |
| `make precommit` | Secret scan + lint + test |