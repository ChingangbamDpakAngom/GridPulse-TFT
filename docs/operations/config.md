# Configuration Reference

## .env File

```bash
# Core APIs
GRID_ESO_API_BASE=https://api.carbonintensity.org.uk
MET_OFFICE_API_BASE=https://api-metoffice.apiconnect.ibmcloud.com/metoffice/production/v1
MET_OFFICE_API_KEY=your-met-office-key

# Experiment tracking
WANDB_API_KEY=your-wandb-key
WANDB_PROJECT=gridpulse-tft

# LLM Copilot
LLM_PROVIDER=deepseek          # deepseek | gemini
LLM_API_KEY=your-llm-key
LLM_MODEL=deepseek-chat

# Server
APP_HOST=0.0.0.0
APP_PORT=8000

# Reproducibility
SEED=42

# Rate limiting
RATE_LIMIT_PER_MIN=60

# LLM timeout (seconds)
LLM_TIMEOUT_SECONDS=5.0

# Drift detection
DRIFT_KS_THRESHOLD=0.15
```

## Validation

On startup, `src/config.py` validates **required** keys:

```python
required = ["MET_OFFICE_API_KEY", "WANDB_API_KEY", "LLM_API_KEY"]
```

Missing → `RuntimeError` with list. App never starts with partial config.

## Override Priority

1. `.env` file (gitignored)
2. Shell environment variables
3. Defaults in `Settings` class

## Secret Management

- **Never** commit `.env`
- `.env.example` documents all keys with empty values
- Pre-commit hook: `gitleaks` or `grep` scan
- `structlog` processor redacts any field matching `/key|token|secret|password|api_key/i`

## Feature Flags via Env

| Env | Effect |
|-----|--------|
| `LLM_PROVIDER=gemini` | Switch copilot to Google Gemini |
| `LLM_TIMEOUT_SECONDS=10` | Longer copilot timeout |
| `RATE_LIMIT_PER_MIN=120` | Double rate limit |
| `DRIFT_KS_THRESHOLD=0.1` | More sensitive drift alerts |

## Related

- [[Quickstart|Quickstart]]
- [[Architecture#74-config-secrets|Architecture: Config & Secrets]]