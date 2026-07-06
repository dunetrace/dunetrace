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
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ingest_svc.auth import is_trusted
from ingest_svc.config import settings
from ingest_svc.db import (
    close_pool,
    ensure_schema,
    get_event_store,
    get_pool,
    init_pool,
    verify_api_key,
)
from ingest_svc.rate_limiter import _HEARTBEAT_INTERVAL, get_limiter
from ingest_svc.routers import ingest, health, otlp

_PRUNE_INTERVAL = 86400.0  # once a day

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


async def _rate_limit_heartbeat_loop() -> None:
    """Report this worker alive periodically so the rate limiter can divide
    the configured rpm across however many ingest workers/replicas are
    currently running — see rate_limiter.py's module docstring."""
    while True:
        await get_limiter()._heartbeat()
        await asyncio.sleep(_HEARTBEAT_INTERVAL)


async def _run_prune_once() -> None:
    """One retention pass. A DB error here must not kill the loop — retention
    is enforced on a best-effort basis, and a transient failure just means
    it's retried on the next tick."""
    try:
        dropped = await get_event_store().prune_old_events(settings.EVENT_RETENTION_DAYS)
        if dropped:
            logger.info("Pruned %d stale event partition(s)", dropped)
    except Exception as exc:
        logger.warning("prune_old_events failed: %s", exc)


async def _prune_loop() -> None:
    """Drop event partitions older than EVENT_RETENTION_DAYS once a day.

    prune_old_events() existed and was tested but was never actually called
    from anywhere — partitions were created forever and never reclaimed.
    Runs once immediately at startup (in case the service was down long
    enough for retention to matter right away), then every _PRUNE_INTERVAL.
    """
    while True:
        await _run_prune_once()
        await asyncio.sleep(_PRUNE_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting — auth_mode=%s", settings.AUTH_MODE)
    await init_pool()
    await ensure_schema()
    evict_task = asyncio.create_task(_evict_loop())
    heartbeat_task = asyncio.create_task(_rate_limit_heartbeat_loop())
    prune_task = asyncio.create_task(_prune_loop())
    yield
    evict_task.cancel()
    heartbeat_task.cancel()
    prune_task.cancel()
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
    # Trusted path (auth service): reads org_id/agent_id from headers, sets Postgres
    # session config for RLS (defined in dunetrace-cloud, not here), skips DB lookup.
    # Dev/direct path: resolves api_key from body → org_id via DB lookup.
    #
    # x-org-id is the current header name; x-customer-id is accepted as a fallback
    # for callers running an older cloud gateway build (pre-v0.5.0 naming) — see
    # docs/migrations/multi-tenancy-v0.5.0.md.
    @app.middleware("http")
    async def set_org_context(request: Request, call_next):
        if is_trusted(request):
            agent_id = request.headers.get("x-agent-id") or None
            org_id = request.headers.get("x-org-id") or request.headers.get("x-customer-id") or None
            request.state.agent_id = agent_id
            request.state.org_id = org_id
            if agent_id or org_id:
                pool = get_pool()
                if pool:
                    try:
                        async with pool.acquire() as conn:
                            async with conn.transaction():
                                if org_id:
                                    await conn.execute(
                                        "SELECT set_config('app.current_org_id', $1, true)", org_id
                                    )
                                if agent_id:
                                    await conn.execute(
                                        "SELECT set_config('app.current_agent_id', $1, true)",
                                        agent_id,
                                    )
                    except Exception as exc:
                        logger.warning("Failed to set org context: %s", exc)
            return await call_next(request)

        import json as _json

        agent_id: str | None = None
        org_id: str | None = None
        try:
            if request.method == "POST" and request.url.path in ("/v1/ingest", "/v1/deploy"):
                body_bytes = await request.body()
                data = _json.loads(body_bytes) if body_bytes else {}
                agent_id = data.get("agent_id")
                api_key = data.get("api_key", "") or ""
                if api_key:
                    org_id = await verify_api_key(api_key)
        except Exception as exc:
            logger.warning("Failed to set org context: %s", exc)

        request.state.agent_id = agent_id
        request.state.org_id = org_id

        if org_id:
            pool = get_pool()
            if pool:
                try:
                    async with pool.acquire() as conn:
                        async with conn.transaction():
                            await conn.execute(
                                "SELECT set_config('app.current_org_id', $1, true)", org_id
                            )
                            if agent_id:
                                await conn.execute(
                                    "SELECT set_config('app.current_agent_id', $1, true)",
                                    agent_id,
                                )
                except Exception as exc:
                    logger.warning("Failed to set org context: %s", exc)
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
        elif request.method == "POST" and request.url.path == "/v1/otlp/traces":
            # OTLP auth is a Bearer header, not a JSON body field — and the
            # body may be gzip-compressed protobuf, which there's no reason
            # to read/decode here just to find the rate-limit bucket.
            auth = request.headers.get("Authorization", "")
            api_key = auth[7:].strip() if auth.startswith("Bearer ") else ""

        if request.url.path in ("/v1/ingest", "/v1/deploy", "/v1/otlp/traces"):
            bucket = (
                api_key
                if api_key and not api_key.startswith("dt_dev_")
                else (request.client.host if request.client else "unknown")
            )
            limiter = get_limiter()
            allowed, retry_after = await limiter.is_allowed(bucket)
            if not allowed:
                logger.warning("Rate limit exceeded. retry_after=%ds", retry_after)
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
