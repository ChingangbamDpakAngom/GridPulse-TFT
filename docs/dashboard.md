# Streamlit Dashboard

[[_TOC_]]

## Overview

Interactive, non-technical dashboard at `dashboard/app.py`. Visualises quantile forecasts, charge windows, and copilot rationale.

## Launch

```bash
# Requires API running on localhost:8000
make serve &
streamlit run dashboard/app.py
# Opens http://localhost:8501
```

## Features

### Sidebar Controls

- **Region**: GB / England / Scotland / Wales
- **Horizon**: 6–168 hours (slider)
- **Auto-refresh**: 5-minute toggle
- **Manual refresh** button

### Main Panel

1. **Quantile Band Chart** (Altair)
   - Shaded q10–q90 band
   - Solid q50 median line
   - Hover tooltip with all three quantiles

2. **Charge Windows Timeline** (Altair bar)
   - Green bars for each optimal 2h window
   - Hover shows intensity + rationale

3. **Expandable Window Details**
   - Click to see copilot rationale per window

4. **Copilot Strategy Box**
   - Full `dispatch_strategy_summary` in info callout

5. **Raw JSON Expander**
   - Full API response for debugging

### Visual Design

- Colour-blind safe palette (viridis/green)
- Responsive width
- UTC timestamps with local-time hint

## Screenshots

*(Add screenshots to `docs/assets/` and reference here)*

## Configuration

Environment variables (optional):

| Var | Default | Description |
|-----|---------|-------------|
| `API_BASE` | `http://localhost:8000` | FastAPI base URL |
| `REFRESH_SECONDS` | `300` | Auto-refresh interval |

## Embedding in Other Apps

The dashboard is a standalone Streamlit app. To embed:

```python
import streamlit.components.v1 as components
components.iframe("http://localhost:8501", height=800)
```

## Related

- [[Serving|Serving Component]]
- [[Quickstart|Quickstart]]
- [[Architecture#layer-5-presentation-layer|Architecture: Layer 5]]