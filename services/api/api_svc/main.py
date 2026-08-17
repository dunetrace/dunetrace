"""
Customer REST API — serves runs, signals, agent summaries, and key management.

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

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api_svc.auth import require_org
from api_svc.config import settings
from api_svc.db.queries import init_pool, close_pool, check_db
from api_svc.routers import (
    agents,
    custom_detectors,
    runs,
    signals,
    insights,
    issues,
    failure_patterns,
    policies,
    patterns,
    replay,
    slack,
    keys,
    orgs,
    integrations,
    elevenlabs,
    external_signals,
    conversations,
    calls,
    alert_integrations,
    linear_webhook,
    github_integration,
    performance_trends,
    packs,
    approvals,
    otel_receiver,
)
from api_svc.schemas import HealthResponse

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("dunetrace.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting — auth_mode=%s", settings.AUTH_MODE)
    if settings.is_dev:
        # Loud, not debug-level: in dev mode every endpoint is unauthenticated,
        # including the ones that mutate policies and spend LLM credit. Anyone
        # who can reach the port is effectively an admin of every org.
        logger.warning(
            "AUTH_MODE=%s — AUTHENTICATION IS DISABLED. Every endpoint is open, "
            "including policy writes and LLM-spending routes, and all requests "
            "resolve to the default org. Do not expose this port beyond "
            "localhost. Set AUTH_MODE=prod for any shared or public deployment.",
            settings.AUTH_MODE,
        )
    await init_pool()
    # Nudge a rate recheck if voice pricing defaults are >90 days stale (Phase 2.2).
    from api_svc.voice_pricing import check_pricing_staleness

    check_pricing_staleness()
    yield
    await close_pool()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Dunetrace Customer API",
        version=settings.APP_VERSION,
        description=(
            "Query your agent observability data: runs, events, failure signals, "
            "and AI-generated explanations."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_dev else ["https://app.dunetrace.io"],
        allow_methods=["GET", "POST", "DELETE", "PATCH"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        t = time.monotonic()
        response = await call_next(request)
        ms = (time.monotonic() - t) * 1000
        logger.info(
            "%s %s %d %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            ms,
        )
        return response

    _auth = [Depends(require_org)]
    app.include_router(agents.router, dependencies=_auth)
    app.include_router(custom_detectors.router, dependencies=_auth)
    app.include_router(runs.router, dependencies=_auth)
    app.include_router(replay.router, dependencies=_auth)
    app.include_router(signals.router, dependencies=_auth)
    app.include_router(insights.router, dependencies=_auth)
    app.include_router(performance_trends.router, dependencies=_auth)
    # No dependencies=_auth here, matching github_integration.router above —
    # GET /v1/packs is a static catalog with no org context, so it must stay
    # unauthenticated; the other three pack endpoints each declare their own
    # Depends(require_org) individually.
    app.include_router(packs.router)
    app.include_router(approvals.router, dependencies=_auth)
    app.include_router(issues.router, dependencies=_auth)
    app.include_router(failure_patterns.router, dependencies=_auth)
    app.include_router(patterns.router, dependencies=_auth)
    app.include_router(policies.router, dependencies=_auth)
    app.include_router(slack.router)  # uses Slack signature verification, not API key auth
    app.include_router(
        linear_webhook.router
    )  # uses Linear signature verification, not API key auth
    app.include_router(keys.router, dependencies=_auth)
    app.include_router(orgs.router, dependencies=_auth)
    app.include_router(integrations.router, dependencies=_auth)
    app.include_router(elevenlabs.router, dependencies=_auth)
    app.include_router(otel_receiver.router, dependencies=_auth)
    app.include_router(external_signals.router, dependencies=_auth)
    app.include_router(conversations.router, dependencies=_auth)
    app.include_router(calls.router, dependencies=_auth)
    app.include_router(alert_integrations.router, dependencies=_auth)
    # No router-level _auth here — /callback is GitHub's own browser
    # redirect (no Dunetrace API key on that request at all); the other
    # four endpoints in this router each declare their own
    # Depends(require_org) explicitly instead. See github_integration.py's
    # module docstring.
    app.include_router(github_integration.router)

    @app.get("/health", response_model=HealthResponse, include_in_schema=False)
    async def health() -> HealthResponse:
        return HealthResponse(db=await check_db(), version=settings.APP_VERSION)

    @app.get("/v1/config", include_in_schema=False)
    async def config():
        return {
            "github_configured": settings.github_configured,
        }

    return app


app = create_app()
