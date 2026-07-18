# Extending the System

## Add New Data Source

### 1. Create Client

```python
# src/ingestion/new_source_client.py
class NewSourceClient:
    async def fetch(self) -> NewSourceSchema: ...
    def _write_atomic(self, data, path): ...
    async def run_ingestion(self): ...
```

### 2. Define Schema

```python
# src/ingestion/schemas.py
class NewSourceRecord(BaseModel):
    timestamp: datetime
    value: float
    quality_flag: str | None

class NewSourceSchema(BaseModel):
    region_id: str
    data: list[NewSourceRecord]
```

### 3. Register in Cleaner

```python
# src/processing/cleaner.py
def _read_new_source_files(self, date): ...
```

### 4. Merge in Features

```python
# src/processing/features.py
def build_features(grid_df, weather_df, new_source_df): ...
```

## Add New Model Architecture

### 1. Implement `nn.Module`

```python
# src/models/my_model.py
class MyModel(nn.Module):
    def __init__(self, input_dim, **kwargs): ...
    def forward(self, x): ...
```

### 2. Register in `__init__.py`

```python
# src/models/__init__.py
from src.models.my_model import MyModel, get_my_model
__all__ += ["MyModel", "get_my_model"]
```

### 3. Update Trainer

```python
# src/training/trainer.py
from src.models import get_my_model

def train(..., model_type="tft", ...):
    if model_type == "my_model":
        model = get_my_model(input_dim=...)
```

### 4. Ensure ONNX Exportability

- No dynamic control flow
- No custom autograd functions
- Static shapes except batch axis
- Test: `torch.onnx.export(..., opset_version=17)`

## Custom Quantiles

See [[Contributing#Extending Quantiles|Contributing: Extending Quantiles]].

## New Fallback Tier

```python
# src/serving/api.py
async def tier_2_persistence_fallback(...):
    # e.g., climatology from historical parquet
    ...
```

Add to fallback chain in `get_forecast`.

## Multi-Region Serving

1. Train per-region models (or single model with region embedding)
2. Manifest: `{"models": [{"region": "GB", "onnx": "..."}, {"region": "FR", ...}]}`
3. `api.py`: load all ONNX sessions, route by `region_id`
4. Dashboard: region selector calls correct session

## Async Batch Inference

For high-throughput internal use:

```python
# src/serving/batch.py
async def batch_forecast(requests: list[ForecastRequest]) -> list[ServerResponseSchema]:
    # Gather unique (region, horizon) pairs
    # Run inference once per unique pair
    # Map results back to requests
    ...
```

## Related

- [[Contributing|Contributing]]
- [[Architecture|Architecture]]
- [[Model Architecture|Model Architecture]]