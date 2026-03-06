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
from contextlib import asynccontextmanager

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


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting — auth_mode=%s", settings.AUTH_MODE)
    await init_pool()
    await ensure_schema()
    yield
    await close_pool()
    logger.info("Shutdown complete")


# ── App ────────────────────────────────────────────────────────────────────────

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

    # Request timing log
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
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
