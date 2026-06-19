"""
Ingest API. Accepts event batches from the SDK and writes them to Postgres.

Run:
    cd services/ingest
    uvicorn ingest_svc.main:app --reload --port 8001

Docs:
    http://localhost:8001/docs
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ingest_svc.auth import is_trusted
from ingest_svc.config import settings
from ingest_svc.db import init_pool, close_pool, ensure_schema, verify_api_key, get_pool
from ingest_svc.rate_limiter import get_limiter
from ingest_svc.routers import ingest, health, otlp

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("dunetrace.ingest")


# Lifespan


async def _evict_loop() -> None:
    """Evict stale rate-limit windows every 2 minutes to prevent unbounded memory growth."""
    while True:
        await asyncio.sleep(120)
        get_limiter().evict_stale()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting — auth_mode=%s", settings.AUTH_MODE)
    await init_pool()
    await ensure_schema()
    evict_task = asyncio.create_task(_evict_loop())
    yield
    evict_task.cancel()
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
        allow_headers=[
            "Content-Type",
            "X-Dunetrace-Agent",
            "Authorization",
            "X-Dunetrace-Agent-Id",
            "X-Dunetrace-Agent-Version",
        ],
    )

    # Registered first = inner = runs AFTER rate_limit_and_log.
    # Trusted path (auth service): reads agent_id from header, sets RLS, skips DB.
    # Dev/direct path: resolves api_key from body → agent_id via DB lookup.
    @app.middleware("http")
    async def set_company_context(request: Request, call_next):
        if is_trusted(request):
            agent_id = request.headers.get("x-agent-id") or None
            request.state.agent_id = agent_id
            request.state.company_id = request.headers.get("x-company-id") or None
            if agent_id:
                pool = get_pool()
                if pool:
                    try:
                        async with pool.acquire() as conn:
                            async with conn.transaction():
                                await conn.execute(
                                    "SELECT set_config('app.current_agent_id', $1, true)",
                                    agent_id,
                                )
                    except Exception:
                        pass
            return await call_next(request)

        import json as _json

        agent_id: str | None = None
        try:
            if request.method == "POST" and request.url.path in ("/v1/ingest", "/v1/deploy"):
                body_bytes = await request.body()
                data = _json.loads(body_bytes) if body_bytes else {}
                api_key = data.get("api_key", "") or ""
                if api_key:
                    agent_id = await verify_api_key(api_key)
        except Exception:
            pass

        request.state.agent_id = agent_id
        request.state.company_id = None

        if agent_id:
            pool = get_pool()
            if pool:
                try:
                    async with pool.acquire() as conn:
                        async with conn.transaction():
                            await conn.execute(
                                "SELECT set_config('app.current_agent_id', $1, true)",
                                agent_id,
                            )
                except Exception:
                    pass
        return await call_next(request)

    # Registered second = outer = runs FIRST.
    # Trusted path: skip rate limiting (auth service already validated and may rate-limit itself).
    # Dev/direct path: parse body, rate-limit by key (or IP fallback).
    @app.middleware("http")
    async def rate_limit_and_log(request: Request, call_next):
        if is_trusted(request):
            t = time.monotonic()
            response = await call_next(request)
            ms = (time.monotonic() - t) * 1000
            logger.info(
                "%s %s %d %.1fms [proxy]",
                request.method,
                request.url.path,
                response.status_code,
                ms,
            )
            return response

        import json as _json

        api_key = ""
        if request.method == "POST" and request.url.path in ("/v1/ingest", "/v1/deploy"):
            try:
                body_bytes = await request.body()
                data = _json.loads(body_bytes) if body_bytes else {}
                api_key = data.get("api_key", "") or ""
            except Exception:
                pass

        if request.url.path in ("/v1/ingest", "/v1/deploy"):
            bucket = (
                api_key
                if api_key and not api_key.startswith("dt_dev_")
                else (request.client.host if request.client else "unknown")
            )
            limiter = get_limiter()
            allowed, retry_after = await limiter.is_allowed(bucket)
            if not allowed:
                bucket_id = hashlib.sha256(bucket.encode()).hexdigest()[:8]
                logger.warning("Rate limit exceeded. bucket=%s", bucket_id)
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded."},
                    headers={"Retry-After": str(retry_after)},
                )

        t = time.monotonic()
        response = await call_next(request)
        ms = (time.monotonic() - t) * 1000
        logger.info("%s %s %d %.1fms", request.method, request.url.path, response.status_code, ms)
        return response

    app.include_router(ingest.router)
    app.include_router(otlp.router)
    app.include_router(health.router)

    return app


app = create_app()
