#!/usr/bin/env python
"""
Demo dashboard with mocked API - runs without real API keys or trained model.
"""
import json
import threading
import time
from datetime import datetime, timedelta, UTC
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import streamlit as st
from streamlit.web import cli as stcli
import sys

# ---- Mock FastAPI server ----
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def make_mock_forecast():
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    preds = []
    for h in range(48):
        ts = now + timedelta(hours=h)
        base = 120 + 80 * (0.5 + 0.5 * ((h % 24) / 12 - 1) ** 2)  # daily cycle
        preds.append({
            "timestamp": ts.isoformat(),
            "quantile_10": round(base * 0.6, 1),
            "quantile_50": round(base, 1),
            "quantile_90": round(base * 1.5, 1),
        })
    # charge windows at q10 minima
    q10s = [p["quantile_10"] for p in preds]
    median_q10 = sorted(q10s)[len(q10s)//2]
    windows = []
    for i in range(46):
        if sum(q10s[i:i+2])/2 < median_q10:
            windows.append({
                "start": preds[i]["timestamp"],
                "end": preds[i+2]["timestamp"],
                "expected_intensity_gco2_kwh": round(sum(q10s[i:i+2])/2, 1),
                "rationale": "Overnight wind surplus, q10 well below daily median"
            })
    windows = windows[:3]
    return {
        "execution_latency_ms": 23.4,
        "model_version": "demo-1.0",
        "predictions": preds,
        "copilot_insight": {
            "optimized_charge_windows": windows,
            "dispatch_strategy_summary": (
                f"Forecast shows {len(windows)} low-carbon windows in the next 48h. "
                f"Best charging: {windows[0]['start'][:16]}–{windows[0]['end'][:16]} "
                f"(~{windows[0]['expected_intensity_gco2_kwh']} gCO₂/kWh). "
                "Charge overnight when wind peaks; avoid 4–8 PM fossil ramp."
            )
        }
    }

MOCK_DATA = make_mock_forecast()

@app.get("/healthz")
def healthz():
    return {"status": "ok", "version": "demo"}

@app.post("/forecast")
def forecast(_req: dict):
    return MOCK_DATA

def run_mock_api():
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")

# ---- Streamlit dashboard (embedded) ----
DASHBOARD_CODE = '''
import requests, pandas as pd, altair as alt, streamlit as st
from datetime import datetime, timedelta

API = "http://127.0.0.1:8001"
st.set_page_config(page_title="GridPulse-TFT Demo", page_icon="⚡", layout="wide")
st.title("⚡ UK Grid Carbon Intensity Forecaster (Demo)")
st.caption("Mock data — no API keys or model required")

@st.cache_data(ttl=30)
def get_forecast():
    try:
        return requests.post(f"{API}/forecast", json={"region_id":"GB","horizon_hours":48}, timeout=5).json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None

data = get_forecast()
if not data:
    st.stop()

c1, c2 = st.columns([3,1])
with c1:
    st.metric("Model", data["model_version"])
with c2:
    st.metric("Latency", f'{data["execution_latency_ms"]:.0f} ms')

# Quantile chart
df = pd.DataFrame(data["predictions"])
df["timestamp"] = pd.to_datetime(df["timestamp"])
base = alt.Chart(df).encode(x=alt.X("timestamp:T", title="Time (UTC)"))
band = base.mark_area(opacity=0.2, color="steelblue").encode(y="quantile_10:Q", y2="quantile_90:Q")
line = base.mark_line(color="steelblue", strokeWidth=2).encode(y="quantile_50:Q", tooltip=["timestamp","quantile_10","quantile_50","quantile_90"])
st.altair_chart((band+line).properties(height=350).interactive(), use_container_width=True)

# Charge windows
st.subheader("⚡ Optimal Charge Windows")
for i, w in enumerate(data["copilot_insight"]["optimized_charge_windows"], 1):
    with st.expander(f"Window {i}: {w['start'][:16]} → {w['end'][:16]}  |  {w['expected_intensity_gco2_kwh']} gCO₂/kWh"):
        st.write(w["rationale"])

st.subheader("🤖 Copilot Dispatch Strategy")
st.info(data["copilot_insight"]["dispatch_strategy_summary"])

with st.expander("Raw JSON"):
    st.json(data)
'''

def run_dashboard():
    # Write dashboard to temp file and run streamlit
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode='w', suffix='_dashboard.py', delete=False) as f:
        f.write(DASHBOARD_CODE)
        dash_path = f.name
    sys.argv = ["streamlit", "run", dash_path, "--server.port=8501", "--server.headless=true"]
    try:
        stcli.main()
    finally:
        os.unlink(dash_path)

if __name__ == "__main__":
    print("Starting mock API on http://127.0.0.1:8001")
    api_thread = threading.Thread(target=run_mock_api, daemon=True)
    api_thread.start()
    time.sleep(2)  # wait for API to start
    print("Starting dashboard on http://localhost:8501")
    run_dashboard()