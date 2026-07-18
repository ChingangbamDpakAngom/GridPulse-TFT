# API Reference

## Base URL

```
http://localhost:8000
```

## Endpoints

### Health Check

```
GET /healthz
```

**Response 200**

```json
{
  "status": "ok" | "degraded"
}
```

- `ok`: All startup checks passed
- `degraded`: Model/parquet missing but server running (fallback active)

---

### Carbon Intensity Forecast

```
POST /forecast
Content-Type: application/json
```

**Request Body** (`ForecastRequest`)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `region_id` | string | No | `"GB"` | Region code (future multi-region) |
| `horizon_hours` | integer | No | `48` | Forecast horizon (1–168) |

```json
{
  "region_id": "GB",
  "horizon_hours": 48
}
```

**Response 200** (`ServerResponseSchema`)

```json
{
  "execution_latency_ms": 42.3,
  "model_version": "1.0",
  "predictions": [
    {
      "timestamp": "2026-07-18T01:00:00Z",
      "quantile_10": 45,
      "quantile_50": 120,
      "quantile_90": 280
    }
  ],
  "copilot_insight": {
    "optimized_charge_windows": [
      {
        "start": "2026-07-18T02:00:00Z",
        "end": "2026-07-18T04:00:00Z",
        "expected_intensity_gco2_kwh": 38.2,
        "rationale": "Overnight wind surplus, q10 below 50 gCO₂/kWh"
      }
    ],
    "dispatch_strategy_summary": "Charge 2-4 AM and 10 AM-12 PM when renewable penetration peaks..."
  }
}
```

**Response 422** – Validation error (Pydantic)

```json
{
  "detail": [
    {"loc": ["body", "horizon_hours"], "msg": "ensure this value is less than or equal to 168", "type": "value_error.number.not_le"}
  ]
}
```

**Response 429** – Rate limited

```json
{
  "error": "rate_limit_exceeded",
  "retry_after": 60
}
```

**Response 503** – All fallbacks exhausted (Tier 3)

```json
{
  "error": "forecast_unavailable",
  "retry_after_seconds": 300,
  "last_good_model_version": "1.0"
}
```

---

## Data Models

### ForecastQuantiles

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime (ISO 8601, UTC) | Forecast timestep |
| `quantile_10` | float | Optimistic floor (q10) |
| `quantile_50` | float | Median forecast |
| `quantile_90` | float | Pessimistic ceiling (q90) |

### ChargeWindow

| Field | Type | Description |
|-------|------|-------------|
| `start` | datetime | Window start (inclusive) |
| `end` | datetime | Window end (exclusive) |
| `expected_intensity_gco2_kwh` | float | Mean q10 across window |
| `rationale` | string | Copilot-generated rationale |

### CopilotRecommendation

| Field | Type | Description |
|-------|------|-------------|
| `optimized_charge_windows` | `ChargeWindow[]` | Selected windows |
| `dispatch_strategy_summary` | string | Natural-language strategy (≥50 chars) |

### ServerResponseSchema

| Field | Type | Description |
|-------|------|-------------|
| `execution_latency_ms` | float | End-to-end latency |
| `model_version` | string | Manifest version or fallback tag |
| `predictions` | `ForecastQuantiles[]` | Horizon quantiles |
| `copilot_insight` | `CopilotRecommendation` | Windows + summary |

---

## Error Handling

| Code | When | Retry? |
|------|------|--------|
| 422 | Invalid request body | No (fix payload) |
| 429 | Rate limit exceeded | Yes (after `retry_after`) |
| 500 | Unexpected server error | Yes (with backoff) |
| 503 | Tier-3 fallback exhausted | Yes (300s) |

Clients should implement exponential backoff on 5xx/429/503.

---

## Rate Limits

- 60 requests/minute per client IP
- Configurable via `RATE_LIMIT_PER_MIN` env var
- Headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `Retry-After`

---

## Example Client (Python)

```python
import httpx
from datetime import datetime

async def get_forecast(region="GB", horizon=48):
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            "http://localhost:8000/forecast",
            json={"region_id": region, "horizon_hours": horizon}
        )
        resp.raise_for_status()
        return resp.json()

# Usage
data = await get_forecast()
for p in data["predictions"]:
    print(f"{p['timestamp']}: q10={p['quantile_10']} q50={p['quantile_50']} q90={p['quantile_90']}")
```

---

## OpenAPI Schema

Available at `/openapi.json` or interactive docs at `/docs` (Swagger UI) and `/redoc`.