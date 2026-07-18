import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from src.config import settings
from src.ingestion.schemas import WeatherDataPoint, WeatherInboundSchema, WeatherRegionData

logger = logging.getLogger(__name__)

# Representative coordinates per region for the Met Office site-specific point API.
UK_REGIONS: dict[str, tuple[float, float]] = {
    "uk": (52.48, -1.90),
    "scotland_n": (57.48, -4.22),
    "scotland_e": (55.95, -3.19),
    "england_nw": (53.48, -2.24),
    "england_ne": (54.98, -1.61),
    "yorkshire_humber": (53.80, -1.55),
    "wales_n": (53.23, -4.13),
    "east_midlands": (52.95, -1.15),
    "west_midlands": (52.48, -1.90),
    "east_england": (52.63, 1.30),
    "wales_s": (51.48, -3.18),
    "south_east": (51.27, 0.52),
    "london": (51.51, -0.13),
    "south_west": (50.72, -3.53),
    "south_england": (50.90, -1.40),
}


class WeatherClient:
    def __init__(self):
        headers = {"apikey": settings.met_office_api_key, "accept": "application/json"}
        self.client = httpx.AsyncClient(
            base_url=settings.met_office_api_base,
            timeout=httpx.Timeout(10.0, read=30.0),
            headers=headers,
        )

    @retry(wait=wait_exponential_jitter(initial=1, max=4), stop=stop_after_attempt(3), reraise=True)
    async def fetch_forecast(self, latitude: float, longitude: float) -> dict:
        resp = await self.client.get(
            "/hourly",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "excludeParameterMetadata": "true",
            },
        )
        resp.raise_for_status()
        return resp.json()

    def _parse_met_office(self, raw: dict) -> list[WeatherDataPoint]:
        parsed = []
        for feature in raw.get("features", []):
            time_series = feature.get("properties", {}).get("timeSeries", [])
            for ts in time_series:
                try:
                    # mslp is reported in Pa; convert to hPa when it looks like Pa.
                    pressure = ts.get("mslp")
                    if pressure is not None and pressure > 2000:
                        pressure = pressure / 100
                    parsed.append(
                        WeatherDataPoint(
                            timestamp=ts.get("time"),
                            temperature_c=ts.get("screenTemperature"),
                            wind_speed_ms=ts.get("windSpeed10m"),
                            wind_direction_deg=ts.get("windDirectionFrom10m"),
                            precipitation_mm=ts.get("precipitationRate"),
                            cloud_cover_pct=ts.get("totalCloudCover"),
                            humidity_pct=ts.get("screenRelativeHumidity"),
                            pressure_hpa=pressure,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to parse weather point: {e}")
        return parsed

    async def fetch_all_regions(self) -> WeatherInboundSchema:
        regions_data = []
        for region_name, (lat, lon) in UK_REGIONS.items():
            try:
                raw = await self.fetch_forecast(lat, lon)
                parsed = self._parse_met_office(raw)
                regions_data.append(WeatherRegionData(region_id=region_name, data=parsed))
            except Exception as e:
                logger.warning(f"Failed to fetch weather for {region_name}: {e}")

        return WeatherInboundSchema(regions=regions_data, issued_at=datetime.now(UTC))

    def _write_atomic(self, data: dict, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, default=str))
        tmp.replace(path)

    async def run_ingestion(self):
        try:
            data = await self.fetch_all_regions()
            if not data.regions:
                msg = "No weather regions fetched; check MET_OFFICE_API_KEY and API base URL"
                raise RuntimeError(msg)
            now = datetime.now(UTC)
            date_dir = settings.raw_weather_root / now.strftime("%Y/%m/%d")
            filepath = date_dir / f"weather_{now.strftime('%H%M')}.json"
            self._write_atomic(data.model_dump(mode="json"), filepath)
            logger.info(f"Saved weather data to {filepath}")
        except Exception:
            logger.exception("Weather ingestion failed")
            raise

    async def close(self):
        await self.client.aclose()


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if not settings.met_office_api_key:
        msg = "MET_OFFICE_API_KEY is not set. Add it to .env (see .env.example)."
        raise SystemExit(msg)
    client = WeatherClient()
    try:
        await client.run_ingestion()
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
