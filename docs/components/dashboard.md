# Dashboard (Streamlit)

## Overview

Non-technical web UI consuming `/forecast` API. Built with Streamlit + Plotly.

## Features

- **Live forecast chart**: Shaded quantile bands (q10–q90) + median line
- **Charge windows**: Highlighted green bars with rationale tooltips
- **Copilot summary**: Natural-language dispatch strategy
- **Model info**: Version, latency, fallback tier badge
- **Region selector**: GB + future multi-region
- **Auto-refresh**: Configurable interval (default 30 min)

## File Structure

```
dashboard/
├── app.py              # Main Streamlit app
├── components/
│   ├── chart.py        # Plotly figure builder
│   ├── sidebar.py      # Controls
│   └── styles.py       # CSS overrides
└── utils/
    ├── api.py          # Async HTTP client
    └── cache.py        # TTL cache decorator
```

## Running

```bash
# Terminal 1: API server
make serve

# Terminal 2: Dashboard
streamlit run dashboard/app.py --server.port 8501
```

Then open http://localhost:8501

## API Client (`dashboard/utils/api.py`)

```python
async def get_forecast(region: str, horizon: int) -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(f"{API_URL}/forecast", json={"region_id": region, "horizon_hours": horizon})
        resp.raise_for_status()
        return resp.json()
```

## Chart Component (`dashboard/components/chart.py`)

```python
def make_forecast_chart(predictions, charge_windows):
    fig = go.Figure()
    # q10–q90 shaded band
    fig.add_trace(go.Scatter(x=ts, y=q90, fill='tonexty', fillcolor='rgba(255,0,0,0.1)', line=dict(width=0), name='q90'))
    fig.add_trace(go.Scatter(x=ts, y=q10, fill='tonexty', fillcolor='rgba(0,255,0,0.1)', line=dict(width=0), name='q10'))
    # median line
    fig.add_trace(go.Scatter(x=ts, y=q50, mode='lines', line=dict(color='blue', width=2), name='median'))
    # charge windows
    for w in charge_windows:
        fig.add_vrect(x0=w['start'], x1=w['end'], fillcolor='green', opacity=0.2, annotation_text=w['rationale'][:30])
    return fig
```

## Screenshots

*(Add screenshots after first run)*

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `API_URL` | `http://localhost:8000` | FastAPI base URL |
| `REFRESH_SECONDS` | `1800` | Auto-refresh interval |
| `DEFAULT_REGION` | `GB` | Initial region |