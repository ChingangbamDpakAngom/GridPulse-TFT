import hashlib
import json
import logging
import time
from datetime import datetime

import httpx

from src.config import settings
from src.serving.postprocess import ChargeWindow
from src.serving.schemas import CopilotRecommendation

logger = logging.getLogger(__name__)

LLM_CACHE = {}
CACHE_TTL = 86400
CACHE_MAX_ENTRIES = 256


def format_prompt(charge_windows: list[ChargeWindow], quantile_50_forecast: list[float]) -> str:
    windows_json = "\n".join(
        f"  - {w.start.isoformat()} to {w.end.isoformat()}: "
        f"{w.expected_intensity_gco2_kwh:.1f} gCO2/kWh"
        for w in charge_windows
    )

    return f"""You are a battery storage dispatch optimization copilot for UK grid operators.
Given the following carbon intensity forecast and recommended charge windows, provide:
1. A concise dispatch strategy summary (50-200 words)
2. A rationale for each charge window

Forecast Summary:
48h horizon, median carbon intensity ranges \
{min(quantile_50_forecast):.0f}-{max(quantile_50_forecast):.0f} gCO2/kWh

Recommended Charge Windows:
{windows_json}

Respond in JSON format:
{{
  "optimized_charge_windows": [
    {{
      "start": "ISO8601",
      "end": "ISO8601",
      "expected_intensity_gco2_kwh": 0.0,
      "rationale": "string"
    }}
  ],
  "dispatch_strategy_summary": "string"
}}"""


async def call_llm(prompt: str) -> dict | None:
    cache_key = hashlib.sha256(prompt.encode()).hexdigest()[:16]

    if cache_key in LLM_CACHE:
        cached, timestamp = LLM_CACHE[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            logger.info("Using cached LLM response")
            return cached

    timeout = httpx.Timeout(settings.llm_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            if settings.llm_provider == "deepseek":
                response = await client.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.llm_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": settings.llm_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.3,
                    },
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]

            elif settings.llm_provider == "gemini":
                response = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{settings.llm_model}:generateContent",
                    headers={"x-goog-api-key": settings.llm_api_key},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "responseMimeType": "application/json",
                            "temperature": 0.3,
                        },
                    },
                )
                response.raise_for_status()
                data = response.json()
                content = data["candidates"][0]["content"]["parts"][0]["text"]

            else:
                raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")

            result = json.loads(content)
            if len(LLM_CACHE) >= CACHE_MAX_ENTRIES:
                oldest = min(LLM_CACHE, key=lambda k: LLM_CACHE[k][1])
                del LLM_CACHE[oldest]
            LLM_CACHE[cache_key] = (result, time.time())
            return result

        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return None


def template_fallback(
    charge_windows: list[ChargeWindow], quantile_50_forecast: list[float]
) -> CopilotRecommendation:
    windows = []
    for w in charge_windows:
        windows.append(
            ChargeWindow(
                start=w.start,
                end=w.end,
                expected_intensity_gco2_kwh=w.expected_intensity_gco2_kwh,
                rationale=(
                    f"Low carbon window (q10={w.expected_intensity_gco2_kwh:.0f} gCO2/kWh). "
                    "Charge when renewables exceed grid average."
                ),
            )
        )

    summary = (
        f"Forecast for next 48h: median carbon intensity ranges "
        f"{min(quantile_50_forecast):.0f}-{max(quantile_50_forecast):.0f} gCO2/kWh. "
        f"{len(windows)} charge windows identified with q10 below historical median. "
        f"LLM summary unavailable - using template fallback."
    )

    return CopilotRecommendation(
        optimized_charge_windows=windows, dispatch_strategy_summary=summary
    )


async def get_copilot_recommendation(
    charge_windows: list[ChargeWindow], quantile_50_forecast: list[float]
) -> CopilotRecommendation:
    prompt = format_prompt(charge_windows, quantile_50_forecast)
    result = await call_llm(prompt)

    if result:
        windows = []
        for w_data in result.get("optimized_charge_windows", []):
            windows.append(
                ChargeWindow(
                    start=datetime.fromisoformat(w_data["start"].replace("Z", "+00:00")),
                    end=datetime.fromisoformat(w_data["end"].replace("Z", "+00:00")),
                    expected_intensity_gco2_kwh=w_data["expected_intensity_gco2_kwh"],
                    rationale=w_data.get("rationale", ""),
                )
            )
        return CopilotRecommendation(
            optimized_charge_windows=windows,
            dispatch_strategy_summary=result.get("dispatch_strategy_summary", ""),
        )

    return template_fallback(charge_windows, quantile_50_forecast)
