# Troubleshooting

## Common Issues

### API Returns 503 "forecast_unavailable"

**Cause**: All fallback tiers exhausted (no ONNX, no parquet, no historical data).

**Checks**:

```bash
# 1. Manifest exists?
cat models/manifest.json

# 2. ONNX file on disk?
ls -la models/forecaster_v*.onnx

# 3. Processed parquet?
ls -la data/processed/grid_features_v*.parquet

# 4. Startup logs
grep "Failed to load model" logs/app.jsonl
```

**Fix**: Run `make preprocess && make train && make export`.

---

### ONNX InferenceError / Shape Mismatch

**Cause**: Feature column mismatch between training parquet and serving.

**Debug**:

```python
# In Python REPL
from src.models import get_feature_columns
print(get_feature_columns())
# Compare with parquet columns:
import pyarrow.parquet as pq
print(pq.read_schema("data/processed/grid_features_v1.parquet").names)
```

**Fix**: Re-run `make preprocess` → `make train` → `make export`.

---

### Drift Alerts Flooding Logs

**Cause**: `DRIFT_KS_THRESHOLD` too low or genuine distribution shift.

**Actions**:

1. Check `logs/drift_alerts.jsonl` for column + KS-stat
2. If seasonal (e.g. winter vs summer): raise threshold to 0.2
3. If sensor issue: investigate upstream API

```bash
# View recent alerts
tail -20 logs/drift_alerts.jsonl | jq .
```

---

### Copilot Always Shows Template Fallback

**Cause**: LLM timeout, auth error, or JSON parse failure.

**Debug**:

```bash
# Check serving logs
grep "LLM call failed" logs/app.jsonl

# Test LLM directly
python -c "
import httpx, os
r = httpx.post('https://api.deepseek.com/v1/chat/completions',
  headers={'Authorization': f'Bearer {os.getenv(\"LLM_API_KEY\")}'},
  json={'model': 'deepseek-chat', 'messages': [{'role': 'user', 'content': 'hi'}], 'timeout': 5})
print(r.status_code, r.text[:200])
"
```

**Fixes**:

- Verify `LLM_API_KEY` in `.env`
- Increase `LLM_TIMEOUT_SECONDS`
- Check provider status page

---

### Training OOM on GPU

**Cause**: Batch size too large for VRAM.

**Fix**: Trainer auto-halves batch size once. If persists:

```bash
python -m src.training.trainer --batch-size 32
```

Or force CPU:

```bash
CUDA_VISIBLE_DEVICES="" python -m src.training.trainer
```

---

### Ingestion Writes Zero Files

**Cause**: API key invalid, network blocked, or schema validation failure.

**Debug**:

```bash
# Run with verbose logging
python -m src.ingestion.grid_client 2>&1 | head -30
python -m src.ingestion.weather_client 2>&1 | head -30
```

Check:

- `MET_OFFICE_API_KEY` valid (test in browser)
- Corporate firewall allows `api-metoffice.apiconnect.ibmcloud.com`
- `data/raw/` writable by process user

---

### `make test` Fails on Config Import

**Cause**: `.env` missing required keys.

**Fix**:

```bash
cp .env.example .env
# Edit .env with test keys (any non-empty string works for unit tests)
```

---

### Dashboard Shows "API not reachable"

**Cause**: FastAPI not running or wrong `API_URL`.

**Fix**:

```bash
# Terminal 1
make serve

# Terminal 2
export API_URL=http://localhost:8000
streamlit run dashboard/app.py
```

---

## Debug Commands

| Task | Command |
|------|---------|
| Validate manifest | `python -m src.training.export --checkpoint models/forecaster_v1.0_latest.pth` |
| Inspect parquet schema | `python -c "import pyarrow.parquet as pq; print(pq.read_schema('data/processed/grid_features_v1.parquet'))"` |
| Test ONNX inference | `python -c "import onnxruntime as ort; import numpy as np; s=ort.InferenceSession('models/forecaster_v1.0_20260718.onnx'); print(s.run(None, {'historical_features': np.random.randn(1,168,20).astype('f4')})[0].shape)"` |
| View recent logs | `tail -f logs/app.jsonl | jq .` |
| Check drift alerts | `cat logs/drift_alerts.jsonl | jq .` |

## Related

- [[Monitoring|Monitoring]]
- [[Commands|Commands]]
- [[GitHub Issues](https://github.com/yourorg/gridpulse-tft/issues)]