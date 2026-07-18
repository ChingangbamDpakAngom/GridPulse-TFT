# Monitoring & Observability

## Structured Logging

All logs: JSON Lines via `structlog` → `logs/app.jsonl` + stdout.

### Standard Fields

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | ISO8601 | Event time |
| `level` | string | `info`/`warning`/`error` |
| `event` | string | Human-readable message |
| `request_id` | uuid4 | Per-request correlation ID |
| `path` | string | HTTP path |
| `region_id` | string | Request region |
| `latency_ms` | float | End-to-end latency |
| `model_version` | string | Manifest version or fallback tag |
| `fallback_tier` | int | 0=ONNX, 1=seasonal, 2=cold, 3=503 |
| `copilot_status` | string | `ok`/`cached`/`timeout`/`template` |

### Example Request Log

```json
{
  "timestamp": "2026-07-18T10:15:23.456Z",
  "level": "info",
  "event": "forecast completed",
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "path": "/forecast",
  "region_id": "GB",
  "latency_ms": 38.2,
  "model_version": "1.0",
  "fallback_tier": 0,
  "copilot_status": "ok"
}
```

### Error Logs

Separate files for component isolation:

- `logs/serving_errors.jsonl` — inference exceptions + input tensor hash
- `logs/ingestion_errors.jsonl` — fetch/validation failures
- `logs/drift_alerts.jsonl` — KS-test threshold breaches

## Health Endpoint

```
GET /healthz
```

```json
{ "status": "ok" }          // all startup checks passed
{ "status": "degraded" }    // model/parquet missing, fallback active
```

Load balancers should route away from `degraded`.

## Prometheus Metrics (Optional)

Add `prometheus-fastapi-instrumentator` to expose `/metrics`:

```python
# In api.py
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

Key metrics:

- `http_requests_total{path,method,status}`
- `http_request_duration_seconds{path,method}`
- `forecast_latency_ms` (histogram)
- `fallback_tier_total{tier}`
- `copilot_status_total{status}`

## Alerting Rules (Prometheus)

```yaml
groups:
- name: gridpulse
  rules:
  - alert: APIHighLatency
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{path="/forecast"}[5m])) > 0.5
    for: 5m
    labels: {severity: warning}
    annotations:
      summary: "Forecast p95 latency > 500ms"

  - alert: FallbackActive
    expr: increase(fallback_tier_total{tier=~"1|2|3"}[5m]) > 0
    for: 2m
    labels: {severity: critical}
    annotations:
      summary: "Forecast fallback tier {{ $labels.tier }} active"

  - alert: DriftDetected
    expr: increase(drift_alerts_total[1h]) > 0
    labels: {severity: warning}
    annotations:
      summary: "Feature drift detected in {{ $labels.column }}"
```

## Grafana Dashboard (Suggested Panels)

1. **Request Rate** — `rate(http_requests_total[5m])` by status
2. **Latency** — p50/p95/p99 of `/forecast`
3. **Fallback Tier** — stacked bar `fallback_tier_total`
4. **Copilot Status** — pie `copilot_status_total`
5. **Drift Alerts** — timeline from `drift_alerts.jsonl`
6. **Model Version** — gauge showing active `model_version`

## Related

- [[Operations Commands|Commands]]
- [[Troubleshooting|Troubleshooting]]
- [[Architecture#86-logging-observability|Architecture: Logging]]