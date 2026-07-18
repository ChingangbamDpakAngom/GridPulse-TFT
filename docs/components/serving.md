# Serving Component

[[_TOC_]]

## Overview

FastAPI application with three-tier fallback, deterministic post-processor, and GenAI copilot.

## Module Map

| Path | Responsibility |
|------|----------------|
| `src/serving/api.py` | FastAPI app, rate limiting, endpoints |
| `src/serving/postprocess.py` | Deterministic charge-window optimizer |
| `src/serving/copilot.py` | LLM client, timeout, cache, template fallback |
| `src/serving/schemas.py` | Request/response Pydantic contracts |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/healthz` | `{"status": "ok"|"degraded", "version": "0.1.0"}` |
| `POST` | `/forecast` | Main inference endpoint |

### Request

```json
{
  "region_id": "GB",
  "horizon_hours": 48
}
```

### Response (`ServerResponseSchema`)

```json
{
  "execution_latency_ms": 42.3,
  "model_version": "1.0",
  "predictions": [
    {"timestamp": "2026-07-18T01:00:00Z", "quantile_10": 120, "quantile_50": 185, "quantile_90": 260}
  ],
  "copilot_insight": {
    "optimized_charge_windows": [
      {"start": "2026-07-18T03:00:00Z", "end": "2026-07-18T05:00:00Z", "expected_intensity_gco2_kwh": 115, "rationale": "..."}
    ],
    "dispatch_strategy_summary": "Forecast shows..."
  }
}
```

## Three-Tier Fallback

| Tier | Trigger | Behaviour | `model_version` |
|------|---------|-----------|-----------------|
| 0 | Normal | ONNX inference → postprocess → copilot | `1.0` (from manifest) |
| 1 | ONNX error / NaN / shape mismatch | Seasonal-naive: last 48h actuals as q50, q10=0.85×q50, q90=1.15×q50 | `fallback:seasonal_naive_v1` |
| 2 | No historical actuals | Constant 200 gCO₂/kWh bands | `fallback:constant_v1` |
| 3 | All above fail | HTTP 503 `{error: "forecast_unavailable", retry_after_seconds: 300}` | — |

**Health endpoint** returns `degraded` if tier ≥ 1 active.

## Post-Processor (`postprocess.py`)

### Algorithm

1. Rank next 48h by `quantile_10` (lowest expected emissions)
2. Select up to 3 non-overlapping 2-hour windows where `mean(q10) < historical_median(q10)`
3. Return `ChargeWindow` objects (rationale filled later by copilot)

### Deterministic Guarantees

- No randomness, no LLM in window selection
- Same forecast → same windows
- Pure NumPy, < 1 ms latency

## GenAI Copilot (`copilot.py`)

### Flow

```
ChargeWindows + q50 forecast
        │
        ▼
Format prompt (JSON schema)
        │
        ▼
SHA-256(prompt) → 24h in-memory cache
        │
        ├─ HIT → return cached JSON
        │
        └─ MISS → call LLM (5s timeout)
                  │
                  ├─ SUCCESS + valid JSON → cache + return
                  │
                  └─ FAIL (timeout, schema error, empty) → template fallback
```

### Providers

| Provider | Endpoint | Model |
|----------|----------|-------|
| DeepSeek | `https://api.deepseek.com/v1/chat/completions` | `deepseek-chat` |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent` | `gemini-pro` |

### Template Fallback

```python
def template_fallback(windows, q50_forecast):
    rationale = [f"Low carbon window (q10={w.intensity:.0f} gCO₂/kWh)" for w in windows]
    summary = f"{len(windows)} windows identified. Median range {min(q50):.0f}–{max(q50):.0f}. LLM unavailable."
    return CopilotRecommendation(windows=windows, summary=summary)
```

## Rate Limiting

- Token bucket per client IP
- Default: 60 req/min (config `RATE_LIMIT_PER_MIN`)
- Exceeded → HTTP 429 `{error: "rate_limit_exceeded", retry_after: 60}`

## Request Guards

- Pydantic validation → HTTP 422 on bad JSON
- Max input tensor: 7 days × 14 features × float32
- NaN/Inf in input → HTTP 400

## Startup Checks

On boot, `api.py` validates:

1. `manifest.json` exists & valid JSON
2. Latest ONNX file exists on disk
3. At least one processed Parquet exists
4. All required `.env` keys present

Failure → process exits non-zero with checklist.

## Logging

- Structured JSON via `structlog` → `logs/app.jsonl`
- Per-request fields: `request_id`, `path`, `region_id`, `latency_ms`, `model_version`, `fallback_tier`, `copilot_status`

## Commands

```bash
# Start server
make serve
# Or:
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

# Test endpoint
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"region_id": "GB", "horizon_hours": 48}'
```

## Related

- [[Model & Training|Model & Training]]
- [[Dashboard|Dashboard]]
- [[Architecture#layer-4-production-serving-genai-copilot|Architecture: Layer 4]]