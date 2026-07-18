# Model & Training

[[_TOC_]]

## Model: TemporalFusionLite (`src/models/tft.py`)

### Architecture

```text
Input [B, 168, F]
  │
  ├─ Linear(F → 64)
  │
  ├─ LSTM(64, 1 layer, batch_first)
  │     │
  │     └─ MultiheadAttention(embed=64, heads=1, dropout=0.1)
  │           │
  │           └─ Last-step context [B, 64]
  │                 │
  │                 ├─ Dropout(0.1)
  │                 │
  │                 └─ Linear(64 → 48×3)
  │                       │
  │                       └─ Reshape → [B, 48, 3]  (horizon, quantiles)
```

### Design Constraints

- **No Variable Selection Network** → fixed feature weights, static ONNX graph
- **Single-head attention** → opset 17 compatible
- **Linear decoder** → 3 quantile heads per horizon step
- **Dynamic batch axis** → `dynamic_axes={"historical_features": {0: "batch_size"}, "quantile_forecasts": {0: "batch_size"}}`

### Forward Pass

```python
def forward(self, x: Tensor) -> Tensor:  # x: [B, 168, F]
    x = self.input_projection(x)
    lstm_out, _ = self.encoder_lstm(x)
    attn_out, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)
    context = self.dropout(attn_out[:, -1, :])
    out = self.decoder(context)
    return out.view(-1, 48, 3)
```

## Loss: Multi-Horizon Pinball (`src/models/loss.py`)

```python
class PinballLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        self.register_buffer("q", torch.tensor(quantiles))

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # y_pred: [B, 48, 3], y_true: [B, 48] or [B, 48, 3]
        if y_true.dim() == 2:
            y_true = y_true.unsqueeze(-1).expand(-1, -1, 3)
        errors = y_true - y_pred
        loss = torch.max(self.q * errors, (self.q - 1) * errors)
        return loss.mean()
```

### Quantile Interpretation

| Index | Quantile | Use Case |
|-------|----------|----------|
| 0 | q10 | Optimistic floor → **charging windows** |
| 1 | q50 | Median trajectory |
| 2 | q90 | Pessimistic ceiling → risk bounds |

## Dataset (`src/models/dataset.py`)

### `SmartGridDataset`

- Memory-mapped Parquet read via PyArrow
- Rolling windows: lookback=168, horizon=48, embargo=48
- Chronological splits:
  - Train: first 60%
  - Val: next 20% (after 48-step embargo)
  - Test: final 20% (after embargo)
- Returns: `{"x": [168, F], "y": [48], "region": str, "timestamp": Timestamp}`

### DataLoaders

```python
train_loader, val_loader, test_loader = get_dataloaders(
    parquet_path,
    batch_size=64,
    num_workers=4,
)
```

## Training (`src/training/trainer.py`)

### Hyperparameters

| Param | Value |
|-------|-------|
| Epochs | 50 (early stop patience=10) |
| Batch | 64 |
| LR | 1e-3 |
| Optimiser | AdamW (wd=1e-4) |
| Scheduler | ReduceLROnPlateau(patience=5, factor=0.5) |
| AMP | Enabled on CUDA |
| Grad clip | 1.0 |
| Seed | 42 (torch, numpy, python) |

### Reproducibility Artifacts

Every checkpoint saves:
- `git_hash` (HEAD short SHA)
- `parquet_hash` (SHA256 of training parquet[:16])
- `seed`
- `feature_columns` list

### Resume

```bash
python -m src.training.trainer --resume models/forecaster_v1.0_20260718.pth
```

- Loads optimiser state, epoch, W&B run ID
- Continues same W&B run (no duplicates)

### OOM Guard

```python
try:
    loss.backward()
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        batch_size //= 2
        retry_once()
```

### Walk-Forward Backtest

`trainer.py` runs expanding-window validation automatically after training; metrics logged to W&B and written to manifest.

## ONNX Export (`src/training/export.py`)

```python
torch.onnx.export(
    model,
    torch.randn(1, 168, F),
    onnx_path,
    input_names=["historical_features"],
    output_names=["quantile_forecasts"],
    dynamic_axes={
        "historical_features": {0: "batch_size"},
        "quantile_forecasts": {0: "batch_size"},
    },
    opset_version=17,
    do_constant_folding=True,
)
```

### Manifest Update

Atomic write: `manifest.json.tmp` → `manifest.json`. New entry appended, sorted by `trained_at` descending.

## Commands

```bash
# Full training
make train

# Resume from checkpoint
python -m src.training.trainer --resume models/forecaster_v1.0_20260718.pth

# Export latest checkpoint
make export

# Or manually
python -m src.training.export --checkpoint models/forecaster_v1.0_20260718.pth
```

## Related

- [[Serving|Serving Component]]
- [[Concepts: TFT Architecture|TFT Architecture]]
- [[Architecture#layer-3-pytorch-deep-learning-mlops|Architecture: Layer 3]]