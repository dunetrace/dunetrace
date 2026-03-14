"""
Customer REST API. Read-only — serves runs, signals, and agent summaries.

Run:
    cd services/api
    uvicorn api_svc.main:app --reload --port 8002

Docs:
    http://localhost:8002/docs
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api_svc.config import settings
from api_svc.db.queries import init_pool, close_pool, check_db
from api_svc.routers import agents, runs, signals, insights
from api_svc.schemas import HealthResponse

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("dunetrace.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting — auth_mode=%s", settings.AUTH_MODE)
    await init_pool()
    yield
    await close_pool()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Dunetrace Customer API",
        version="0.1.0",
        description=(
            "Query your agent observability data: runs, events, failure signals, "
            "and AI-generated explanations."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_dev else ["https://app.dunetrace.io"],
        allow_methods=["GET"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        t = time.monotonic()
        response = await call_next(request)
        ms = (time.monotonic() - t) * 1000
        logger.info("%s %s %d %.1fms",
                    request.method, request.url.path, response.status_code, ms)
        return response

    app.include_router(agents.router)
    app.include_router(runs.router)
    app.include_router(signals.router)
    app.include_router(insights.router)

    @app.get("/health", response_model=HealthResponse, include_in_schema=False)
    async def health() -> HealthResponse:
        return HealthResponse(db=await check_db())

    return app


app = create_app()
