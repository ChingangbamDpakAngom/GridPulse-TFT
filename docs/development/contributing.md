# Contributing

## Setup

```bash
git clone https://github.com/yourorg/gridpulse-tft
cd gridpulse-tft
uv pip install -e ".[dev]"
cp .env.example .env  # fill in keys
make test
make lint
```

## Workflow

1. Create branch: `git checkout -b feat/your-feature`
2. Make changes with tests
3. Run `make precommit` (secret scan + lint + test)
4. Open PR with description linking issue

## Code Style

- **Formatter**: Ruff (100 char, double quotes)
- **Types**: Full annotations, `X \| None` not `Optional[X]`
- **Imports**: Ruff-organised (stdlib → third-party → local)
- **Docstrings**: NumPy style for public functions
- **No comments** unless explaining *why*, not *what*

## Testing

```bash
# All tests
make test

# Single file
pytest tests/test_cleaner.py -v

# With coverage
pytest --cov=src --cov-report=term-missing
```

### Test Categories

| Path | Purpose |
|------|---------|
| `tests/test_config.py` | Settings validation |
| `tests/test_cleaner.py` | Resample, gap-fill |
| `tests/test_models.py` | Forward pass, loss, features |

Add tests for new components in matching `test_*.py`.

## Adding a Region

1. Add region to `UK_REGIONS` in `src/ingestion/weather_client.py`
2. Add region ID mapping in `src/processing/features.py` if needed
3. Update `ForecastRequest` schema if new `region_id` format
4. Test ingestion: `python -m src.ingestion.weather_client`
5. Update dashboard region selector

## Extending Quantiles

Current: `[0.1, 0.5, 0.9]` (3 heads)

To add e.g. `[0.05, 0.1, 0.5, 0.9, 0.95]`:

1. `PinballLoss` quantiles list in `trainer.py` + `loss.py`
2. `TemporalFusionLite` `num_quantiles` param
3. `decoder` output size: `horizon × 5`
3. `postprocess.py` expects index 0 = lowest quantile
4. Export → manifest version bump
5. Dashboard chart: add q05/q95 bands

## Adding Weather Features

1. Parse in `weather_client.py` `_parse_met_office`
2. Add to `WeatherRecord` schema
3. Merge in `features.py` `build_features`
4. Add cyclical/lag if temporal
5. Update `FEATURE_COLUMNS` list
6. Re-run `make preprocess && make train && make export`

## Related

- [[Testing|Testing]]
- [[Extending|Extending]]
- [[Architecture|Architecture]]