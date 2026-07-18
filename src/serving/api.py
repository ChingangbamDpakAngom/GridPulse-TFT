import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.serving.inference import get_forecast, load_latest_model
from src.serving.schemas import ForecastRequest, ServerResponseSchema

logger = logging.getLogger(__name__)

RATE_WINDOW = 60.0  # seconds
RATE_LIMIT: dict[str, list[float]] = {}
_RATE_PRUNE_EVERY = 1000
_rate_request_count = 0


async def rate_limit_middleware(request: Request, call_next):
    global _rate_request_count
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    # Periodically drop idle clients so the map cannot grow without bound.
    _rate_request_count += 1
    if _rate_request_count % _RATE_PRUNE_EVERY == 0:
        stale = [ip for ip, hits in RATE_LIMIT.items() if not hits or now - hits[-1] > RATE_WINDOW]
        for ip in stale:
            del RATE_LIMIT[ip]

    hits = [t for t in RATE_LIMIT.get(client_ip, []) if now - t < RATE_WINDOW]
    if len(hits) >= settings.rate_limit_per_min:
        RATE_LIMIT[client_ip] = hits
        return JSONResponse(
            status_code=429, content={"error": "rate_limit_exceeded", "retry_after": RATE_WINDOW}
        )

    hits.append(now)
    RATE_LIMIT[client_ip] = hits
    return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    try:
        load_latest_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"No model loaded at startup ({e}); will serve fallback forecasts")

    if not settings.processed_root.exists():
        logger.warning("No processed data directory found; /forecast will return 503")

    yield
    logger.info("Shutting down...")


app = FastAPI(title="GridPulse-TFT Carbon Intensity Forecaster", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.middleware("http")(rate_limit_middleware)


@app.get("/healthz")
async def health_check():
    status = "ok"
    try:
        load_latest_model()
    except Exception:
        status = "degraded"
    return {"status": status, "version": "0.1.0"}


@app.post("/forecast", response_model=ServerResponseSchema)
async def forecast_endpoint(request: ForecastRequest):
    try:
        return await get_forecast(request)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.exception("Forecast error")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/")
async def root():
    return {"service": "GridPulse-TFT", "version": "0.1.0", "docs": "/docs"}
