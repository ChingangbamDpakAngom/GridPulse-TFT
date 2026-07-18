# Testing Guide

## Test Structure

```
tests/
├── test_config.py      # Settings validation
├── test_cleaner.py     # Resample, gap-fill logic
├── test_models.py      # Forward pass, loss, features
├── test_ingestion.py   # Schema validation (future)
├── test_serving.py     # API + fallback (future)
└── test_postprocess.py # Charge window logic (future)
```

## Running Tests

```bash
# All
make test

# Verbose
pytest -v

# With coverage
pytest --cov=src --cov-report=html

# Parallel
pytest -n auto

# Specific marker
pytest -m "not slow"
```

## Fixtures

```python
# conftest.py (auto-discovered)
import pytest
from src.config import Settings

@pytest.fixture
def test_settings(monkeypatch):
    monkeypatch.setenv("MET_OFFICE_API_KEY", "test")
    monkeypatch.setenv("WANDB_API_KEY", "test")
    monkeypatch.setenv("LLM_API_KEY", "test")
    return Settings()
```

## Async Tests

```python
import pytest
from src.ingestion.grid_client import GridClient

@pytest.mark.asyncio
async def test_grid_client_fetch(test_settings):
    client = GridClient()
    # Mock httpx.AsyncClient.get
    ...
    await client.close()
```

## Mocking External APIs

Use `httpx.MockTransport` or `respx`:

```python
import respx
import httpx

@respx.mock
async def test_weather_client():
    respx.get("https://api-metoffice.../forecast/hourly/uk").mock(
        return_value=httpx.Response(200, json={"features": []})
    )
    client = WeatherClient()
    result = await client.fetch_forecast("uk")
    assert result == []
```

## Snapshot Testing (ONNX)

```python
def test_onnx_output_matches_pytorch():
    import torch, onnxruntime as ort
    from src.models import get_model
    from src.training.export import export_to_onnx

    model = get_model(input_dim=20)
    model.eval()
    x = torch.randn(1, 168, 20)

    with torch.no_grad():
        pytorch_out = model(x).numpy()

    ort_session = ort.InferenceSession("model.onnx")
    onnx_out = ort_session.run(None, {"historical_features": x.numpy()})[0]

    assert np.allclose(pytorch_out, onnx_out, atol=1e-5)
```

## Performance Tests

Mark with `@pytest.mark.slow`:

```python
@pytest.mark.slow
def test_training_step_performance():
    # Runs 1 epoch, asserts < 30s on CPU
    ...
```

Run with: `pytest -m slow`

## CI Pipeline (GitHub Actions)

```yaml
# .github/workflows/ci.yml
- name: Test
  run: make test
- name: Lint
  run: make lint
- name: Secret scan
  run: gitleaks detect --no-git --source .
```

## Related

- [[Contributing|Contributing]]
- [[Extending|Extending]]