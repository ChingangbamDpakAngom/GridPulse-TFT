# GridPulse-TFT

**UK Smart Grid Carbon Intensity Forecaster** — an end-to-end time-series ML pipeline that ingests live UK grid carbon intensity and weather data, trains a TFT-lite multi-quantile PyTorch model, exports it to ONNX, and serves 48-hour probabilistic forecasts through FastAPI with a GenAI battery-dispatch copilot.

```
National Grid ESO ──┐
                    ├─► Ingestion ─► Cleaning & Features ─► Parquet
UK Met Office ──────┘                                          │
                                                               ▼
Dashboard ◄── Copilot (LLM) ◄── Post-process ◄── FastAPI ◄── TFT-lite ─► ONNX
```

## Features

- **Multi-quantile forecasts** (q10 / q50 / q90) over a 48-hour horizon for uncertainty-aware dispatch decisions
- **TFT-lite model** — LSTM encoder + temporal attention + quantile heads, sized for reliable ONNX export with dynamic batch axes
- **Pinball (quantile) loss** training with early stopping, AMP on CUDA, W&B experiment tracking, and reproducible seeds
- **Resilient ingestion** — retries with exponential backoff and jitter, atomic file writes, content-hash deduplication
- **Drift detection** — Kolmogorov–Smirnov tests on feature distributions between dataset versions
- **GenAI dispatch copilot** — DeepSeek or Gemini summarises charge windows, with timeout, 24 h response cache, and a deterministic template fallback
- **Streamlit dashboard** for quantile bands and recommended charge windows

## Project status

The full pipeline is wired end-to-end: ingestion → cleaning/features → training → ONNX export → serving. If no trained model is present, `/forecast` automatically serves a **seasonal-naive fallback** built from the latest processed data (`model_version: seasonal-naive-fallback`), and returns 503 only when no data exists at all. To try the UI without any keys, data, or model, run `python demo_dashboard.py` (mock API on :8001 + dashboard on :8501).

## Requirements

- Python **3.11+**
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- API keys (free tiers available): [UK Met Office DataHub](https://datahub.metoffice.gov.uk/), [Weights & Biases](https://wandb.ai/), and a DeepSeek **or** Gemini API key. The [Carbon Intensity API](https://carbonintensity.org.uk/) needs no key.

## Quickstart

```bash
git clone https://github.com/<you>/gridpulse-tft.git
cd gridpulse-tft

# 1. Create environment and install
uv venv && uv pip install -e ".[dev]"

# 2. Configure secrets
cp .env.example .env      # then fill in the API keys

# 3. Run the pipeline (one command)
make pipeline             # ingest -> preprocess -> train -> export

# ...or run the stages individually
make ingest               # fetch grid + weather data
make preprocess           # clean, feature-engineer, write parquet
make train                # train TFT-lite (logs to W&B)
make export               # export best checkpoint to ONNX + manifest

# 4. Serve and view
make serve                # FastAPI on :8000  (see Project status above)
streamlit run dashboard/app.py
```

`make pipeline` wraps [`scripts/run_pipeline.py`](scripts/run_pipeline.py), which runs every stage in order, stops at the first failure, and prints a per-stage timing summary. Useful flags (run it directly to use them): `--days N`, `--epochs N`, `--no-wandb`, `--skip-ingest`, `--skip-train`, `--skip-export`. If `MET_OFFICE_API_KEY` is unset, weather ingestion is skipped automatically and the model trains on grid data alone.

No keys yet? Try the self-contained demo:

```bash
python demo_dashboard.py   # mock API on :8001 + dashboard on :8501
```

## Configuration

All settings load from `.env` (see `.env.example`) via `pydantic-settings`:

| Variable | Default | Purpose |
|---|---|---|
| `GRID_ESO_API_BASE` | `https://api.carbonintensity.org.uk` | Carbon intensity API (no auth) |
| `MET_OFFICE_API_BASE` / `MET_OFFICE_API_KEY` | — | Weather forecasts |
| `WANDB_API_KEY` / `WANDB_PROJECT` | `gridpulse-tft` | Experiment tracking |
| `LLM_PROVIDER` / `LLM_API_KEY` / `LLM_MODEL` | `deepseek` | Copilot backend (`deepseek` or `gemini`) |
| `APP_HOST` / `APP_PORT` | `0.0.0.0:8000` | API bind address |
| `RATE_LIMIT_PER_MIN` | `60` | Per-client API rate limit |
| `LLM_TIMEOUT_SECONDS` | `5` | Copilot call timeout before template fallback |
| `DRIFT_KS_THRESHOLD` | `0.15` | KS statistic threshold for drift alerts |
| `SEED` | `42` | Reproducibility seed |

**Never commit `.env`** — it is gitignored; only `.env.example` is tracked.

## Project structure

```
src/
├── config.py          # pydantic-settings configuration
├── ingestion/         # async API clients (grid + weather) with retry/atomic writes
├── processing/        # cleaning, resampling, feature engineering, drift detection
├── models/            # TFT-lite nn.Module, dataset/dataloaders, pinball loss
├── training/          # training loop (W&B, AMP, early stopping) + ONNX export
└── serving/           # FastAPI app, post-processing, LLM copilot
dashboard/app.py       # Streamlit UI
tests/                 # pytest suite
docs/                  # MkDocs Material documentation
ARCHITECTURE.md        # full system design document
```

## Development

```bash
make test        # pytest
make lint        # ruff check + format check
make precommit   # lint + gitleaks secret scan
mkdocs serve     # docs at :8000
```

## Documentation

Full documentation lives in [`docs/`](docs/) (MkDocs Material) and the detailed design in [ARCHITECTURE.md](ARCHITECTURE.md).

## License

No license has been chosen yet. Until one is added, all rights are reserved.
