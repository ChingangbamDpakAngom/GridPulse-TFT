import asyncio
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from src.config import settings
from src.ingestion.schemas import GridInboundSchema

logger = logging.getLogger(__name__)


class GridClient:
    def __init__(self):
        self.client = httpx.AsyncClient(
            base_url=settings.grid_eso_api_base, timeout=httpx.Timeout(10.0, read=30.0)
        )

    @retry(wait=wait_exponential_jitter(initial=1, max=4), stop=stop_after_attempt(3), reraise=True)
    async def fetch_intensity(self, date: str | None = None) -> dict:
        """Fetch national half-hourly carbon intensity. `date` is YYYY-MM-DD (default: today)."""
        path = f"/intensity/date/{date}" if date else "/intensity/date"
        resp = await self.client.get(path)
        resp.raise_for_status()
        return resp.json()

    @retry(wait=wait_exponential_jitter(initial=1, max=4), stop=stop_after_attempt(3), reraise=True)
    async def fetch_generation(self) -> dict:
        resp = await self.client.get("/generation")
        resp.raise_for_status()
        return resp.json()

    def _dedup_key(self, payload: dict) -> str:
        content = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(content).hexdigest()[:16]

    def _write_atomic(self, data: dict, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, default=str))
        tmp.replace(path)

    async def run_ingestion(self):
        try:
            intensity_data = await self.fetch_intensity()
            generation_data = await self.fetch_generation()

            validated = GridInboundSchema.model_validate(intensity_data)

            now = datetime.now(UTC)
            combined = {
                "intensity": intensity_data,
                "generation": generation_data,
                "fetched_at": now.isoformat(),
                "validated": validated.model_dump(mode="json"),
            }

            dedup = self._dedup_key(combined["intensity"])
            date_dir = settings.raw_grid_root / now.strftime("%Y/%m/%d")
            filepath = date_dir / f"grid_{now.strftime('%H%M')}_{dedup}.json"

            self._write_atomic(combined, filepath)
            logger.info(f"Saved grid data to {filepath}")
        except Exception:
            logger.exception("Grid ingestion failed")
            raise

    async def close(self):
        await self.client.aclose()


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    client = GridClient()
    try:
        await client.run_ingestion()
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
