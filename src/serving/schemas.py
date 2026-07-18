from datetime import datetime

from pydantic import BaseModel, Field


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
    optimized_charge_windows: list[ChargeWindow]
    dispatch_strategy_summary: str = Field(..., min_length=50)


class ServerResponseSchema(BaseModel):
    execution_latency_ms: float
    model_version: str
    predictions: list[ForecastQuantiles]
    copilot_insight: CopilotRecommendation


class ForecastRequest(BaseModel):
    region_id: str = "GB"
    horizon_hours: int = Field(default=48, ge=2, le=48)
