# Command Reference

## Make Targets

| Target | Description |
|--------|-------------|
| `make install` | Install deps in `.venv` via `uv pip install -e ".[dev]"` |
| `make ingest` | Run both grid + weather ingestion once |
| `make preprocess` | Clean + feature engineer → Parquet |
| `make train` | Train TFT-lite (resume with `--resume`) |
| `make export` | Export latest checkpoint to ONNX + update manifest |
| `make serve` | Start FastAPI on `0.0.0.0:8000` with reload |
| `make test` | Run `pytest -v` |
| `make lint` | `ruff check && ruff format --check` |
| `make precommit` | Secret scan + lint + test |
| `make logs` | `tail -f logs/app.jsonl logs/serving_errors.jsonl` |

## Direct Python Modules

```bash
# Ingestion
python -m src.ingestion.grid_client
python -m src.ingestion.weather_client

# Processing
python -m src.processing.cleaner --days 30
python -m src.processing.features

# Training
python -m src.training.trainer --parquet data/processed/grid_features_v1.parquet --epochs 50
python -m src.training.trainer --resume models/forecaster_v1.0_20260718.pth

# Export
python -m src.training.export --checkpoint models/forecaster_v1.0_20260718.pth

# Serving
python -m src.serving.api
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
```

## Dashboard

```bash
# Requires running API
streamlit run dashboard/app.py --server.port 8501
```

## Documentation

```bash
# Serve locally
mkdocs serve

# Build static site
mkdocs build --site-dir site

# Deploy to GitHub Pages
mkdocs gh-deploy
```

## Environment

```bash
# Show current config
python -c "from src.config import settings; print(settings.model_dump_json(indent=2))"

# Validate .env
python -c "from src.config import settings; missing=settings.validate_required(); print('Missing:', missing or 'none')"
```

## Docker (Optional)

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install -e ".[dev]"
COPY . .
CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t gridpulse-tft .
docker run -p 8000:8000 --env-file .env gridpulse-tft
```

## Related

- [[Quickstart|Quickstart]]
- [[Monitoring|Monitoring]]
- [[Troubleshooting|Troubleshooting]]