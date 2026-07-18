# Ingestion Component

[[_TOC_]]

## Overview

Resilient, scheduled data ingestion from two UK APIs:

- **National Grid ESO** — Carbon intensity (30-min settlement periods)
- **Met Office DataHub** — Weather forecasts (hourly, 16 regions)

## Module Map

| Path | Responsibility |
|------|----------------|
| `src/ingestion/grid_client.py` | ESO API client, retry, atomic write, dedup |
| `src/ingestion/weather_client.py` | Met Office client, multi-region fan-out |
| `src/ingestion/schemas.py` | Pydantic v2 inbound validation |

## National Grid ESO Client

```python
from src.ingestion.grid_client import GridClient

client = GridClient()
await client.run_ingestion()  # fetches last 24h, writes to data/raw/grid/
```

### Key Behaviours

- **Endpoint**: `GET /intensity/regionid/{region}` + `GET /generation`
- **Retries**: 3× on 5xx/timeout (1s, 2s, 4s + jitter)
- **Atomic write**: `data/raw/grid/YYYY/MM/DD/grid_HHMM_<hash>.json.tmp` → `.json`
- **Deduplication**: SHA-256(content)[:16] in filename; re-runs skip identical payload
- **Validation**: `GridInboundSchema` rejects malformed JSON before disk write
- **Isolation**: Runs in own `asyncio.TaskGroup`; failure never kills weather ingestion

## Met Office Client

```python
from src.ingestion.weather_client import WeatherClient

client = WeatherClient()
await client.run_ingestion()  # fans out to 16 UK regions
```

### Key Behaviours

- **Endpoint**: `GET /forecast/hourly/{region_id}` per region
- **Regions**: `uk`, `scotland_n`, `scotland_e`, `england_nw`, `england_ne`, `yorkshire_humber`, `wales_n`, `east_midlands`, `west_midlands`, `east_england`, `wales_s`, `south_east`, `london`, `south_west`, `south_england`
- **Auth**: `apikey` header from `MET_OFFICE_API_KEY`
- **Parsing**: Extracts `temperature`, `windSpeed`, `windDirection`, `precipitationRate`, `cloudCover`, `relativeHumidity`, `pressure`
- **Same resilience**: retry, atomic write, validation, isolation

## Data Contracts

### GridInboundSchema

```python
class CarbonIntensityRecord(BaseModel):
    from_: datetime = Field(alias="from")
    to: datetime
    intensity: IntensityValue  # {forecast: int, actual: int|None, index: str|None}

class GridInboundSchema(BaseModel):
    region_id: int
    region_name: str
    data: list[CarbonIntensityRecord]
```

### WeatherInboundSchema

```python
class WeatherRecord(BaseModel):
    timestamp: datetime
    temperature_c: float|None
    wind_speed_ms: float|None
    wind_direction_deg: float|None
    precipitation_mm: float|None
    cloud_cover_pct: float|None
    humidity_pct: float|None
    pressure_hpa: float|None

class WeatherRegionData(BaseModel):
    region_id: str
    data: list[WeatherRecord]

class WeatherInboundSchema(BaseModel):
    regions: list[WeatherRegionData]
    issued_at: datetime
```

## Scheduling (cron)

```cron
# Half-hourly grid ingestion
*/30 * * * * /path/to/venv/bin/python -m src.ingestion.grid_client

# Hourly weather ingestion
0 * * * * /path/to/venv/bin/python -m src.ingestion.weather_client
```

## Testing

```bash
# Unit tests
pytest tests/test_ingestion.py -v

# Dry-run ingestion
python -m src.ingestion.grid_client
python -m src.ingestion.weather_client
```

## Related

- [[Processing|Processing Component]]
- [[Architecture#layer-1-deterministic-data-ingestion|Architecture: Layer 1]]