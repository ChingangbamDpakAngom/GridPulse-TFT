# ARCHITECTURE: UK SMART GRID CARBON INTENSITY FORECASTER

## 1. System Overview & Objectives
This system is an end-to-end, **resilient** time-series forecasting pipeline. It ingests live **half-hourly** carbon intensity metrics and regional weather forecasts across the UK (National Grid ESO publishes at 30-minute settlement periods; the previous "5-minute" wording was incorrect), processes them into optimized columnar storage, trains/evaluates a **TFT-lite** multi-quantile PyTorch model to handle uncertainty forecasting, exports it to an optimized ONNX graph, and serves inferences via an asynchronous FastAPI backend paired with a generative AI decision copilot.

The model mimics the spirit of the Temporal Fusion Transformer (recurrent encoder, temporal attention, per-horizon quantile heads) but is constrained to a footprint that is reliably exportable to ONNX with dynamic batch axes.

---

## 2. End-to-End System Flowchart
This flowchart explicitly describes data structures, boundaries, and communication protocols between components.

```mermaid
graph TD
    %% Layer 1: Ingestion
    subgraph L1 [Layer 1: Deterministic Data Ingestion]
        A1[Cron / Scheduler] -->|Triggers half-hourly| A2[grid_client.py]
        A1 -->|Triggers hourly| A3[weather_client.py]
        A2 -->|HTTP GET / JSON, retry x3| B1[National Grid ESO API]
        A3 -->|HTTP GET / JSON, retry x3| B2[UK Met Office API]
        B1 -->|Raw JSON payload| C1[schemas.py: GridInboundSchema]
        B2 -->|Raw JSON payload| C2[schemas.py: WeatherInboundSchema]
        C1 -->|Pydantic Validated Dict| D1[Write to data/raw/grid/YYYY/MM/DD/]
        C2 -->|Pydantic Validated Dict| D2[Write to data/raw/weather/YYYY/MM/DD/]
    end

    %% Layer 2: Preprocessing & Storage
    subgraph L2 [Layer 2: Preprocessing & Columnar Storage]
        D1 -->|Read raw JSON stream| E1[cleaner.py]
        D2 -->|Read raw JSON stream| E1
        E1 -->|Resample to hourly, interpolate gaps, forward-fill NaN| E2[features.py]
        E2 -->|Sine/Cosine Calendar & Lag Embedding| F1[DuckDB / PyArrow]
        F1 -->|Write Parquet, validate schema| F2[(data/processed/grid_features_vX.parquet)]
    end

    %% Layer 3: Model Training Core
    subgraph L3 [Layer 3: PyTorch Deep Learning & MLOps]
        F2 -->|Memory-mapped read| G1[dataset.py: SmartGridDataset]
        G1 -->|Sliding Windows [Batch, Lookback=168, Feats]| G2[dataset.py: DataLoader]
        G2 -->|Tensors| H1[trainer.py]
        H2[tft.py: TemporalFusionLite] -->|nn.Module| H1
        H3[loss.py: PinballLoss] -->|Custom Metric| H1
        H1 -->|Log metrics| I1[Weights & Biases API]
        H1 -->|Save Checkpoint| I2[models/forecaster_vX.Y_YYYYMMDD.pth]
        I2 -->|Trace Model Graph| J1[export.py]
        J1 -->|Freeze & Quantize| J2[(models/forecaster_vX.Y_YYYYMMDD.onnx)]
        J2 -->|Register entry| J3[(models/manifest.json)]
    end

    %% Layer 4: Production Serving & GenAI Copilot
    subgraph L4 [Layer 4: Asynchronous Serving Engine]
        J2 -->|On Startup: Load latest graph per manifest| K1[api.py: FastAPI Application]
        L1_Client[Frontend Dashboard] -->|HTTP POST JSON request| K1
        K1 -->|Extract Tensor Windows| K2[onnxruntime.InferenceSession]
        K2 -->|Predict Quantiles [0.1, 0.5, 0.9]| K3[postprocess.py: WindowOptimizer]
        K3 -->|Pick charge windows from q10 minima| K4[copilot.py]
        K5[LLM API: Gemini / DeepSeek] -->|System Prompt Context| K4
        K4 -->|Format Forecast + Window Summary| K5
        K5 -->|JSON Structured Summary| K1
    end

    %% Layer 5: Presentation Layer
    subgraph L5 [Layer 5: User Interface]
        K1 -->|JSON Payload: Inferences + Text| L1_Client
        L1_Client -->|Plotly Visual Shaded Bands| L2_UI[Interactive Charts & Map]
    end

    style F2 fill:#f9f,stroke:#333,stroke-width:2px
    style J2 fill:#f9f,stroke:#333,stroke-width:2px
    style I2 fill:#fff,stroke:#333,stroke-width:1px
    style K3 fill:#cff,stroke:#333,stroke-width:1px
```

---

## 3. Directory Layout Standards
The system must follow this modular structure. No implementation logic may be put inside the repository root.

```text
gridpulse-tft/
├── AGENTS.md                    # Runtime behavior instructions for opencode agents
├── ARCHITECTURE.md              # This document — single source of truth for the design
├── pyproject.toml               # Python package management via UV
├── Makefile                     # make ingest, preprocess, train, export, serve, test, lint, precommit
├── .env.example                 # Committed template documenting required env vars
├── .env                         # Local secrets — NEVER committed (gitignored)
├── .gitignore                   # Excludes .env, .pth, .onnx, logs/, raw data
├── data/
│   ├── raw/                     # Immutable JSON responses (gitignored)
│   │   ├── grid/
│   │   └── weather/
│   └── processed/               # Schema-enforced Parquet files (latest only is committed)
├── logs/                        # Structured logs (gitignored — app, drift_alerts, ingestion_errors, serving_errors)
├── models/
│   ├── manifest.json            # Registry of trained artifacts and versions (atomic writes)
│   └── *.pth / *.onnx           # Trained artifacts (gitignored)
├── src/
│   ├── __init__.py
│   ├── config.py                # Pydantic BaseSettings, loads .env, validates required keys at boot
│   ├── ingestion/
│   │   ├── grid_client.py       # Async client for National Grid ESO with retry + atomic writes
│   │   ├── weather_client.py    # Async client for Met Office with retry + atomic writes
│   │   └── schemas.py           # Inbound Pydantic v2 validation rules
│   ├── processing/
│   │   ├── cleaner.py           # Time-alignment, resample to hourly, forward-fill, schema gate
│   │   └── features.py          # Cyclical features, lag embeddings, drift detection (KS test)
│   ├── models/
│   │   ├── dataset.py           # PyTorch Dataset, lookback rolling windows, train/val/test split + embargo
│   │   ├── tft.py               # TemporalFusionLite (TFT-inspired lightweight net)
│   │   └── loss.py              # Tensorized multi-quantile pinball loss
│   ├── training/
│   │   ├── trainer.py           # Epoch training, gradient clipping, OOM guard, resume, walk-forward backtest
│   │   └── export.py            # PyTorch to ONNX + atomic manifest update
│   └── serving/
│       ├── api.py               # FastAPI, 3-tier fallback, /healthz, rate limit, request guards
│       ├── postprocess.py       # Deterministic charge-window optimizer (pre-LLM)
│       ├── copilot.py           # LLM client with timeout + template fallback + 24h cache
│       ├── fallback.py          # Seasonal-naive baseline generator
│       └── schemas.py           # Pydantic contract definition for UI exchange
└── tests/
    ├── test_ingestion.py
    ├── test_cleaner.py
    ├── test_dataset.py
    ├── test_onnx_latency.py
    ├── test_postprocess.py
    ├── test_fallback.py         # Tier-1/2/3 fallback behaviors
    └── test_config.py           # Required-keys validation at boot
```

---

## 4. Mathematical Engine Specification
To evaluate uncertainty, the model optimizer does not minimize Mean Squared Error (MSE). It minimizes **Multi-Horizon Pinball (Quantile) Loss**.

For a target value $y_t$, a predicted quantile $\hat{y}_t^{(q)}$ at quantile level $q \in \mathcal{Q}$, across a forecast horizon $T = 48$, the objective function is defined as:

$$\mathcal{L}_{\text{total}} = \frac{1}{|\mathcal{Q}| \cdot T} \sum_{q \in \mathcal{Q}} \sum_{t=1}^{T} \max\left(q(y_t - \hat{y}_t^{(q)}),\ (q - 1)(y_t - \hat{y}_t^{(q)})\right)$$

The model must explicitly track and evaluate three target quantiles:
*   $q = 0.1$: Lower boundary (high renewable surge, target charging window).
*   $q = 0.5$: Median trajectory.
*   $q = 0.9$: Upper boundary (high fossil utilization risk).

---

## 5. Explicit Data Contracts (Schemas)

### 5.1 Inbound Grid Contract (`src/ingestion/schemas.py`)
```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional

class CarbonIntensityRecord(BaseModel):
    from_time: datetime = Field(..., alias="from")
    to_time: datetime = Field(..., alias="to")
    # `None` MUST be preserved — 0 gCO2/kWh means "zero-carbon generation",
    # not "missing". The cleaner handles forward-fill, not the schema.
    intensity_actual: Optional[int] = Field(None, alias="actual")
    intensity_forecast: int = Field(..., alias="forecast")
    index: Optional[int] = None

class GridInboundSchema(BaseModel):
    region_id: int
    region_name: str
    data: List[CarbonIntensityRecord]
```

### 5.2 Inbound Weather Contract (`src/ingestion/schemas.py`)
```python
class WeatherRecord(BaseModel):
    timestamp: datetime
    temperature_c: float
    wind_speed_ms: float
    cloud_cover_pct: float
    solar_irradiance_wm2: Optional[float] = None

class WeatherInboundSchema(BaseModel):
    region_id: int
    region_name: str
    data: List[WeatherRecord]
```

### 5.3 Processed Feature Contract (`src/processing/features.py`)
The parquet written by Layer 2 MUST contain exactly these columns:

| Group | Columns |
|---|---|
| Identity | `timestamp` (UTC, hourly), `region_id` |
| Target | `intensity_actual` (nullable, forward-filled only inside training windows) |
| Cyclical calendar | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `doy_sin`, `doy_cos` |
| Lags | `lag_1h`, `lag_24h`, `lag_168h` |
| Weather exogenous | `temperature_c`, `wind_speed_ms`, `cloud_cover_pct`, `solar_irradiance_wm2` |

The resample granularity is **1 hour** (mean aggregation of the half-hourly raw within each hour). Gaps ≤ 2 hours are linearly interpolated; longer gaps are forward-filled and flagged with a `gap_filled` boolean column.

### 5.4 Outbound Serving Interface Contract (`src/serving/schemas.py`)
```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List

class ForecastQuantiles(BaseModel):
    timestamp: datetime
    quantile_10: float = Field(..., description="Optimistic low-carbon floor")
    quantile_50: float = Field(..., description="Median carbon forecast")
    quantile_90: float = Field(..., description="Pessimistic peak emissions ceiling")

class ChargeWindow(BaseModel):
    start: datetime
    end: datetime
    expected_intensity_gco2_kwh: float
    rationale: str

class CopilotRecommendation(BaseModel):
    optimized_charge_windows: List[ChargeWindow]
    dispatch_strategy_summary: str = Field(..., min_length=50)

class ServerResponseSchema(BaseModel):
    execution_latency_ms: float
    model_version: str
    predictions: List[ForecastQuantiles]
    copilot_insight: CopilotRecommendation
```

---

## 6. Serving-Layer Architecture

### 6.1 Post-Processor (`src/serving/postprocess.py`)
A **deterministic** module. Given the ONNX quantile forecast tensor `[horizon=48, 3]`, it:
1. Ranks the next-48-hour horizon by `quantile_10` (lowest expected emissions).
2. Selects up to N contiguous windows (default N=3, 2 hours each) where `quantile_10` is locally minimal and below the historical median.
3. Returns proto `ChargeWindow` objects (without a `rationale` string — that is the LLM's job).

The post-processor is the **single source of truth** for which windows get recommended. The LLM never chooses timestamps from raw tensors.

### 6.2 Copilot (`src/serving/copilot.py`)
Receives the structured forecast + deterministic windows from the post-processor, formats a contextual execution prompt for commercial battery storage scheduling, and calls the LLM with `response_format={"type": "json_object"}`. The LLM **only generates the NL `dispatch_strategy_summary` and per-window `rationale` strings**. It must never act as a direct time-series retriever and must never invent timestamps not present in the post-processor output.

---

## 7. MLOps, Optimization & Serving Guidelines

### 7.1 Data Windows
The PyTorch `Dataset` must structure multidimensional arrays using a rolling lookback window of **168 steps (7 days of 1-hour intervals)** to predict a forward target horizon of **48 steps (2 days of 1-hour intervals)**.

### 7.2 Model Architecture (`src/models/tft.py` — `TemporalFusionLite`)
Constrained TFT-inspired network for reliable ONNX export:
* **Encoder**: 1-layer LSTM (hidden=64) over the 168-step lookback.
* **Temporal attention**: single-head attention over encoder outputs (NOT full multi-head gated attention — keeps ONNX opset ≥17 export simple).
* **Decoder**: linear projection to 3 quantile heads per horizon, shape `[batch, 48, 3]`.
* No dynamic variable selection network (VSN) — feature weights are static learned parameters, ensuring a fixed ONNX graph.

### 7.3 Model Portability Rule (`src/training/export.py`)
ONNX export MUST use dynamic axes and opset 17:
```python
torch.onnx.export(
    model,
    dummy_input,                                   # shape [1, 168, n_features]
    output_path,                                    # models/forecaster_vX.Y_YYYYMMDD.onnx
    input_names=["historical_features"],
    output_names=["quantile_forecasts"],            # shape [batch, 48, 3]
    dynamic_axes={
        "historical_features": {0: "batch_size"},
        "quantile_forecasts": {0: "batch_size"},
    },
    opset_version=17,
)
```
After export, `export.py` must append an entry to `models/manifest.json`:
```json
{
  "version": "0.1.0",
  "checkpoint": "models/forecaster_v0.1.0_20260717.pth",
  "onnx": "models/forecaster_v0.1.0_20260717.onnx",
  "trained_at": "2026-07-17T10:00:00Z",
  "feature_columns": ["..."],
  "metrics": {"pinball_val": 12.3, "quantile_10_mae": 4.1, "quantile_90_mae": 5.0}
}
```
On API startup, `api.py` reads `manifest.json`, loads the entry with the latest `trained_at` timestamp, and exposes its `version` in every response as `model_version`.

### 7.4 Config & Secrets (`src/config.py`)
Use `pydantic-settings` BaseSettings loading from `.env`. Committed template `.env.example` documents every key. **`.env` is gitignored and never pushed.**

Required keys:
```
GRID_ESO_API_BASE=
MET_OFFICE_API_KEY=
MET_OFFICE_API_BASE=
WANDB_API_KEY=
WANDB_PROJECT=gridpulse-tft
LLM_PROVIDER=deepseek            # or gemini
LLM_API_KEY=
LLM_MODEL=deepseek-chat
APP_HOST=0.0.0.0
APP_PORT=8000
SEED=42                          # training reproducibility
RATE_LIMIT_PER_MIN=60            # per-client request budget
LLM_TIMEOUT_SECONDS=5            # hard timeout on copilot calls
DRIFT_KS_THRESHOLD=0.15          # alert threshold in features.py
```

### 7.5 Ingestion Reliability
Both `grid_client.py` and `weather_client.py` must:
* Use `httpx.AsyncClient` with a 10s connect / 30s read timeout.
* Retry transient failures (5xx, network) up to 3 times with exponential backoff (1s, 2s, 4s).
* Skip the write if a payload fails Pydantic validation; log via `structlog` and continue (never crash the scheduler).

### 7.6 Makefile Targets
```
make ingest        # run both clients once (manual / dry-run)
make preprocess    # clean + feature-engineer raw -> processed parquet
make train         # run trainer.py with config from .env (supports --resume)
make export        # export latest checkpoint to ONNX + atomic manifest update
make serve         # uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
make test          # pytest -v
make lint          # ruff check && ruff format --check
make precommit     # secret-scan via gitleaks (or grep fallback) + ruff + pytest -q
make logs          # tail logs/app.jsonl logs/serving_errors.jsonl (Ctrl-C to stop)
```

---

## 8. Security, Secrets & Crash Resilience

This section elevates the system from "MVP" to "safe to run unattended". Every external boundary (network, disk, GPU, LLM, scheduler) must assume failure is the default and degrade gracefully.

### 8.1 Secrets Management
- **Source of truth**: `.env` (gitignored). Loaded by `src/config.py` via `pydantic-settings` `BaseSettings`.
- **Template**: `.env.example` checked into the repo, documenting every required key with empty values and inline comments. **No real values ever** appear in `.env.example`.
- **Validation at boot**: `config.py` raises a typed `ConfigError` listing every missing/empty required key before the app starts. The app **never starts with partial secrets** — it fails loud and early.
- **Never log secrets**: `structlog` processor drops any field whose name matches `/key|token|secret|password|api_key/i`, replacing with `"***"`.
- **No secrets in code**: hardcoded API keys, even in tests, are forbidden. Tests use `monkeypatch` + fixture values.
- **Git guard**: `.gitignore` includes `.env`, `*.pem`, `*.key`, `.secrets/`. A pre-commit hook (`Makefile` `make precommit`) runs `gitleaks` if installed, else `grep -rIE "(api_key|secret|password)\s*=\s*[\"'][^\"']{8,}" --exclude-dir=.git --exclude=.env.example . || true`.

### 8.2 Ingestion Crash Resilience
- **Per-call isolation**: a failure in `grid_client.py` NEVER crashes `weather_client.py` and vice versa. Each client runs in its own `asyncio.TaskGroup` with `asyncio.TaskGroup` exception capture.
- **Retries**: 3 attempts, exponential backoff (1s, 2s, 4s) with jitter ±0.25s, only on `httpx.TimeoutException`, `httpx.NetworkError`, HTTP 5xx. 4xx responses are NOT retried (auth/config issue, not transient).
- **Disk write safety**: raw JSON is first written to `data/raw/<source>/<YYYY/MM/DD>/<ts>.json.tmp`, then `os.replace()`d to the final name (atomic on POSIX and Windows NTFS). A crash mid-write never produces a half-parsed payload.
- **Deduplication**: each payload file is content-addressed by `sha256(json_bytes)[:16]` in its filename. Re-runs within the same half-hour window skip if the dedup file exists.
- **Scheduler never crashes**: the cron entry wraps the whole job in `try/except Exception` and writes a structured error to `logs/ingestion_errors.jsonl` (gitignored). The next tick runs regardless.

### 8.3 Processing Crash Resilience
- **Parquet write**: same temp-then-rename pattern — `grid_features_vX.parquet.tmp` → `grid_features_vX.parquet`. A clean file is never partially overwritten.
- **Manifest atomic write**: `manifest.json` is written to `manifest.json.tmp` then `os.replace()`d. Concurrent `make export` and `make serve` never read a half-written manifest.
- **Schema validation gate**: `cleaner.py` validates the output parquet columns against the §5.3 contract BEFORE writing. A schema mismatch raises `SchemaMismatchError` and aborts the batch — no bad parquet ever ships.
- **Drift alert (NEW)**: after writing the parquet, `features.py` computes KS-statistic vs the previous day's parquet for `intensity_actual`, `wind_speed_ms`, `cloud_cover_pct`. If KS > 0.15 for any column, it writes an entry to `logs/drift_alerts.jsonl` with `column`, `ks_stat`, `previous_date`, `current_date`. The alert is informational — training does not auto-pause — but `trainer.py` reads the latest alert at startup and refuses to start if drift is in the top decile.

### 8.4 Training Crash Resilience
- **Checkpoint versioning**: every epoch writes `forecaster_vX.Y_YYYYMMDD_epoch{N:03d}.pth`. Only the best-val-loss checkpoint is renamed to `forecaster_vX.Y_YYYYMMDD.pth` at the end. Worst-case crash loses one epoch.
- **Resume support**: `trainer.py --resume <path>` loads optimizer state, epoch number, W&B run ID from the checkpoint. The next run continues the W&B run rather than spawning a duplicate.
- **OOM guard**: if `torch.cuda.OutOfMemoryError` (or `RuntimeError` matching `CUDA out of memory`) is raised, the trainer halves `batch_size`, logs the event, and retries the epoch once. A second OOM aborts with a clear error.
- **Seed & reproducibility (NEW)**: `torch.manual_seed`, `numpy.random.seed`, and `random.seed` are all set from `config.seed` (default 42). The seed, git commit hash (`subprocess.check_output(["git", "rev-parse", "HEAD"])`), and parquet SHA-256 are written into the checkpoint metadata and into the manifest entry.
- **Backtest protocol (NEW)**: `trainer.py` performs **expanding-window walk-forward validation** with chronological split:
  - Train: earliest 60% of timestamps
  - Validate: next 20%
  - Test (held out, evaluated once): final 20%
  - **Embargo**: 48-step gap between train end and validate start (prevents lookback leakage into horizon).
  - Metrics on the test split are written to `manifest.json` and are the only numbers reported externally.

### 8.5 Serving Crash Resilience & Fallbacks
This is the most critical section because serving is on the request hot path. Three fallback tiers, applied in order:

**Tier 0 — Normal path**
```
Request → feature extraction → ONNX inference → postprocess → copilot → response
```

**Tier 1 — ONNX inference fails (OOM, NaN, shape mismatch, ORT exception)**
- `api.py` catches `onnxruntime.RuntimeException` and any `np.isnan(output).any()`.
- Falls back to a **seasonal-naive baseline**: returns the last 48 hours of `intensity_actual` from the cache (see Tier 2) as `quantile_50`, with quantile_10 = `quantile_50 * 0.85` and quantile_90 = `quantile_50 * 1.15` as a fixed uncertainty band.
- The response sets `model_version = "fallback:seasonal_naive_v1"` so the client knows.
- The error is logged to `logs/serving_errors.jsonl` with the input tensor hash for later triage.
- HTTP 200 (not 500) — the client always gets a usable forecast.

**Tier 2 — Historical cache miss (no recent actuals available)**
- `api.py` keeps an in-memory LRU cache (1000 entries, keyed by `(region_id,Forecast_start_ts)`).
- On cold start or cache eviction, it reads the last available 48 hours from the latest parquet in `data/processed/`.
- If the parquet itself is unreadable, fall through to Tier 3.

**Tier 3 — Cascaded failure (everything above failed)**
- `api.py` returns an HTTP 503 with a structured body `{ "error": "forecast_unavailable", "retry_after_seconds": 300, "last_good_model_version": "..." }`.
- A health endpoint `/healthz` returns `degraded` instead of `ok` so upstream load balancers can route away.
- The server keeps running — one bad request never kills the process.

**LLM Copilot fallback** (independent of forecast fallback):
- `copilot.py` has a 5s hard timeout on the LLM call (`httpx.Timeout(5.0)`).
- On timeout, network error, JSON-schema-validation failure, or empty response, it returns a **templated summary**: `"Forecast for region {R}: {N} charge windows identified in next 48h; lowest q10 = {V} gCO2/kWh at {T}. NL summary unavailable — LLM degraded."`.
- The deterministic `ChargeWindow` objects from `postprocess.py` are always returned regardless of LLM state — the user always gets actionable windows.
- LLM responses are cached by SHA-256 of the post-processed window tuple (24h TTL). Identical forecasts within 24h return the cached summary without re-hitting the LLM.

**Startup safety**:
- On boot, `api.py` validates: (a) manifest exists and is valid JSON, (b) at least one ONNX file referenced by manifest exists on disk, (c) at least one parquet exists in `data/processed/`, (d) `.env` loaded with all required keys present. If any check fails, the server exits non-zero with a list of missing items. It does NOT start in a half-broken state.

**Request-level guards**:
- All request bodies validated by Pydantic; invalid JSON returns HTTP 422, never 500.
- Per-client rate limit: 60 req/min (token bucket in-process). Over-limit returns HTTP 429.
- Request input tensor shapes validated (max 7 days × 14 features `float32`). Oversized or NaN inputs return HTTP 400.

### 8.6 Logging & Observability
- `structlog` with JSON formatter, written to `logs/app.jsonl` (gitignored) and stdout.
- Every request logs: `request_id`, `path`, `region_id`, `latency_ms`, `model_version`, `fallback_tier` (0/1/2/3), `copilot_status` (ok/cached/timeout/template).
- No PII is ever expected (no user accounts in v1); if added later, redaction is mandatory.

---

## 9. Scope Boundaries (Non-Goals)
To keep the build tractable, the following are explicitly **out of scope** for v1:
* Real-time streaming (Kafka / RabbitMQ) — ingestion is scheduled batch.
* Multi-region graph models — each region is forecast independently in v1.
* Auto-retraining triggers — model retraining is a manual `make train && make export`.
* Frontend bundling — Layer 5 is a separate repo / static dashboard consuming the API.
