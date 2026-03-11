"""
services/ingest/ingest_svc/main.py

FastAPI ingest service for Dunetrace.

Run:
    cd services/ingest
    uvicorn ingest_svc.main:app --reload --port 8001

Docs:
    http://localhost:8001/docs
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ingest_svc.config import settings
from ingest_svc.db import init_pool, close_pool, ensure_schema
from ingest_svc.routers import ingest, health

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("dunetrace.ingest")


# Rate limiter

_RATE_LIMIT_REQUESTS = settings.RATE_LIMIT_REQUESTS
_RATE_LIMIT_WINDOW   = 60  # seconds

_rate_counters: dict[str, list[float]] = defaultdict(list)
_rate_lock = Lock()


def _is_rate_limited(ip: str) -> bool:
    now = time.monotonic()
    with _rate_lock:
        timestamps = _rate_counters[ip]
        # drop entries outside the window
        _rate_counters[ip] = [t for t in timestamps if now - t < _RATE_LIMIT_WINDOW]
        if len(_rate_counters[ip]) >= _RATE_LIMIT_REQUESTS:
            return True
        _rate_counters[ip].append(now)
        return False


# Lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting — auth_mode=%s", settings.AUTH_MODE)
    await init_pool()
    await ensure_schema()
    yield
    await close_pool()
    logger.info("Shutdown complete")


# App

def create_app() -> FastAPI:
    app = FastAPI(
        title="Dunetrace Ingest API",
        version="0.1.0",
        description="Receives agent instrumentation events from the Dunetrace SDK.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_dev else [],
        allow_methods=["POST", "GET"],
        allow_headers=["Content-Type", "X-Dunetrace-Agent"],
    )

    # Rate limiting + request timing
    @app.middleware("http")
    async def rate_limit_and_log(request: Request, call_next):
        # Only rate-limit the ingest endpoint
        if request.url.path == "/v1/ingest":
            ip = request.client.host if request.client else "unknown"
            if _is_rate_limited(ip):
                logger.warning("Rate limit exceeded. ip=%s", ip)
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Max 60 requests per minute."},
                )
        t = time.monotonic()
        response = await call_next(request)
        ms = (time.monotonic() - t) * 1000
        logger.info("%s %s %d %.1fms",
                    request.method, request.url.path, response.status_code, ms)
        return response

    app.include_router(ingest.router)
    app.include_router(health.router)

    return app


app = create_app()
