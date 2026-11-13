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

from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api_svc.auth import require_org
from api_svc.config import settings
from api_svc.db.queries import init_pool, close_pool, get_pool
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
from api_svc.schemas import HealthResponse, ReadyResponse
from dunetrace_schemas import metrics as dt_metrics
from dunetrace_schemas.migrations import CURRENT_SCHEMA_VERSION

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("dunetrace.api")

# ── Metrics ───────────────────────────────────────────────────────────────────
# Module level, not inside create_app(): the factories dedupe by name, but the
# metrics are process-wide and the tests build the app many times per process.
dt_metrics.register_standard("api", settings.APP_VERSION)
_REQUESTS = dt_metrics.counter(
    "dunetrace_api_requests_total",
    "HTTP requests served by the Customer API, by templated route.",
    ("path", "method", "status"),
)
_REQUEST_SECONDS = dt_metrics.histogram(
    "dunetrace_api_request_seconds",
    "Wall-clock time to serve one HTTP request, by templated route.",
    ("path", "method"),
    buckets=dt_metrics.DEFAULT_LATENCY_BUCKETS,
)


# Methods we serve. request.method is caller-controlled: both services run plain
# uvicorn (no [standard], so h11 parses the request line) and h11 accepts any
# RFC-7230 token as a method. _route_path already folds unknown PATHS into
# "other" precisely so a scanner cannot mint time series; method got no such
# treatment, so `ZZZ-SCAN-1 /nope` repeated with distinct tokens grew the
# registry without bound — one counter child plus a full histogram (12 buckets,
# sum and count) each, held for the life of the process and visible on the
# unauthenticated /metrics endpoint.
_KNOWN_METHODS = frozenset(
    {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "TRACE", "CONNECT"}
)


def _method_label(method: str) -> str:
    """The HTTP method if it is one we serve, else "other"."""
    return method if method in _KNOWN_METHODS else "other"


def _route_path(request: Request) -> str:
    """The route *template* ("/v1/runs/{run_id}"), never the concrete URL.

    Starlette stores the matched route on the scope once routing has run, so
    this is read after ``call_next``. Anything unmatched (404s, probes for
    paths that do not exist) is folded into one ``other`` label — the whole
    point is a bounded label set that a scanner cannot inflate.
    """
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) and path else "other"


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
    # init_pool applied + required the schema, so this is the version it left
    # behind; the gauge is what lets a scraper see a replica lagging a deploy.
    _, ready_info = await dt_metrics.db_ready(get_pool(), CURRENT_SCHEMA_VERSION)
    if ready_info.get("schema_version") is not None:
        dt_metrics.set_schema_version(ready_info["schema_version"])
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
        status = 500  # what the client sees if call_next raises
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            elapsed = time.monotonic() - t
            path = _route_path(request)
            method = _method_label(request.method)
            _REQUESTS.labels(path=path, method=method, status=str(status)).inc()
            _REQUEST_SECONDS.labels(path=path, method=method).observe(elapsed)
            logger.info(
                "%s %s %d %.1fms",
                request.method,
                request.url.path,
                status,
                elapsed * 1000,
            )

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

    # /health is liveness (never fails while the process answers), /ready is
    # readiness (503 until the DB answers at the schema this build needs), and
    # /metrics is the Prometheus scrape. All three are unauthenticated on
    # purpose — compose healthchecks and scrapers carry no API key — so they
    # are for the internal network only; the same is true of the
    # AUTH_MODE=dev warning above. None of them exposes tenant data.
    @app.get("/health", response_model=HealthResponse, include_in_schema=False)
    async def health() -> HealthResponse:
        """Liveness only. No DB round-trip, for the same reason as ingest's:
        a probe that waits on a pool connection reports a saturated-but-
        working process as dead, and an orchestrator acting on that restarts
        the one thing still draining the backlog. Dependency state is
        ``/ready``'s job."""
        return HealthResponse(version=settings.APP_VERSION)

    @app.get("/ready", response_model=ReadyResponse, include_in_schema=False)
    async def ready() -> Response:
        ok, info = await dt_metrics.db_ready(get_pool(), CURRENT_SCHEMA_VERSION)
        body = ReadyResponse(
            status="ok" if ok else "not_ready",
            version=settings.APP_VERSION,
            db=str(info.get("db", "unknown")),
            schema_version=info.get("schema_version"),
            required=int(info.get("required", CURRENT_SCHEMA_VERSION)),
            pool=info.get("pool"),
        )
        return JSONResponse(status_code=200 if ok else 503, content=body.model_dump())

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        payload, content_type = dt_metrics.render()
        return Response(content=payload, media_type=content_type)

    @app.get("/v1/config", include_in_schema=False)
    async def config():
        return {
            "github_configured": settings.github_configured,
        }

    return app


app = create_app()
