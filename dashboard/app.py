import os
import time

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_BASE = os.environ.get("GRIDPULSE_API_BASE", "http://localhost:8000")

st.set_page_config(page_title="GridPulse-TFT", page_icon="⚡", layout="wide")

st.title("⚡ UK Grid Carbon Intensity Forecaster")
st.caption("TFT-lite multi-quantile forecast + GenAI dispatch copilot")


@st.cache_data(ttl=300)
def fetch_forecast(region: str = "GB", horizon: int = 48):
    try:
        resp = requests.post(
            f"{API_BASE}/forecast", json={"region_id": region, "horizon_hours": horizon}, timeout=10
        )
        if resp.status_code == 200:
            return resp.json()
        st.error(f"API error {resp.status_code}: {resp.text}")
        return None
    except requests.exceptions.ConnectionError:
        st.warning("API not reachable. Start with `make serve`")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def plot_quantiles(data: dict):
    preds = data["predictions"]
    df = pd.DataFrame(
        [
            {
                "timestamp": pd.to_datetime(p["timestamp"]),
                "q10": p["quantile_10"],
                "q50": p["quantile_50"],
                "q90": p["quantile_90"],
            }
            for p in preds
        ]
    )

    base = alt.Chart(df).encode(x=alt.X("timestamp:T", title="Time (UTC)"))

    band = base.mark_area(opacity=0.2, color="steelblue").encode(
        y=alt.Y("q10:Q", title="Carbon Intensity (gCO₂/kWh)"),
        y2="q90:Q",
        tooltip=["timestamp:T", "q10:Q", "q50:Q", "q90:Q"],
    )

    median = base.mark_line(color="steelblue", strokeWidth=2).encode(
        y="q50:Q", tooltip=["timestamp:T", "q50:Q"]
    )

    return (band + median).properties(height=400).interactive()


def plot_charge_windows(data: dict):
    windows = data["copilot_insight"]["optimized_charge_windows"]
    if not windows:
        return None

    df = pd.DataFrame(
        [
            {
                "start": pd.to_datetime(w["start"]),
                "end": pd.to_datetime(w["end"]),
                "intensity": w["expected_intensity_gco2_kwh"],
                "rationale": w["rationale"],
            }
            for w in windows
        ]
    )

    return (
        alt.Chart(df)
        .mark_bar(opacity=0.6, color="green")
        .encode(
            x=alt.X("start:T", title="Window"),
            x2="end:T",
            y=alt.value(0),
            y2=alt.value(1),
            tooltip=["start:T", "end:T", "intensity:Q", "rationale:N"],
        )
        .properties(height=80)
    )


with st.sidebar:
    st.header("Controls")
    region = st.selectbox("Region", ["GB"], index=0)
    horizon = st.slider("Horizon (hours)", 6, 48, 48, step=6)
    auto_refresh = st.checkbox("Auto-refresh (5 min)", value=False)
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()

col1, col2 = st.columns([3, 1])
with col1:
    data = fetch_forecast(region, horizon)
with col2:
    st.metric("Model Version", "N/A" if not data else data.get("model_version", "N/A"))
    if data:
        st.metric("Latency (ms)", f"{data.get('execution_latency_ms', 0):.0f}")

if data:
    st.subheader("📊 48h Carbon Intensity Forecast (Quantiles)")
    st.altair_chart(plot_quantiles(data), use_container_width=True)

    st.subheader("⚡ Optimal Charge Windows")
    cw_chart = plot_charge_windows(data)
    if cw_chart:
        st.altair_chart(cw_chart, use_container_width=True)

        windows = data["copilot_insight"]["optimized_charge_windows"]
        for i, w in enumerate(windows, 1):
            label = (
                f"Window {i}: {w['start']} → {w['end']}  |  "
                f"{w['expected_intensity_gco2_kwh']:.0f} gCO₂/kWh"
            )
            with st.expander(label):
                st.write(w["rationale"])

    st.subheader("🤖 Copilot Dispatch Strategy")
    st.info(data["copilot_insight"]["dispatch_strategy_summary"])

    with st.expander("🔍 Raw JSON Response"):
        st.json(data)
else:
    st.info(
        "Start the API server: `make serve` "
        "(requires `make preprocess`; `make train && make export` for the ONNX model)"
    )

if auto_refresh:
    time.sleep(300)
    st.cache_data.clear()
    st.rerun()
