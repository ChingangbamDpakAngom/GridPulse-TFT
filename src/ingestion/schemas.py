from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class IntensityData(BaseModel):
    forecast: int | None = None
    actual: int | None = None
    index: str | None = None


class GridDataPoint(BaseModel):
    from_: datetime = Field(alias="from")
    to: datetime = Field(alias="to")
    intensity: IntensityData | None = None
    generationmix: list[dict] | None = None

    @field_validator("intensity", mode="before")
    @classmethod
    def validate_intensity(cls, v):
        if v is None:
            return None
        if isinstance(v, dict):
            return IntensityData(**v)
        return v


class GridInboundSchema(BaseModel):
    data: list[GridDataPoint]


class WeatherDataPoint(BaseModel):
    timestamp: datetime
    temperature_c: float | None = None
    wind_speed_ms: float | None = None
    wind_direction_deg: float | None = None
    precipitation_mm: float | None = None
    cloud_cover_pct: float | None = None
    humidity_pct: float | None = None
    pressure_hpa: float | None = None


class WeatherRegionData(BaseModel):
    region_id: str
    data: list[WeatherDataPoint]


class WeatherInboundSchema(BaseModel):
    regions: list[WeatherRegionData]
    issued_at: datetime
