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
from starlette.requests import ClientDisconnect

from dunetrace_schemas import metrics as _metrics
from dunetrace_schemas.migrations import CURRENT_SCHEMA_VERSION
from ingest_svc.auth import is_trusted
from ingest_svc.body_limits import BodyTooLarge, read_bounded_body
from ingest_svc.config import settings
from ingest_svc.db import (
    close_pool,
    ensure_schema,
    get_event_store,
    get_pool,
    init_pool,
    retention_looks_stale,
    scrub_old_signal_evidence,
    verify_api_key,
)
from ingest_svc.rate_limiter import _HEARTBEAT_INTERVAL, get_limiter
from ingest_svc.routers import ingest, health, otlp

_PRUNE_INTERVAL = 86400.0  # once a day

# JSON endpoints whose body the middlewares below buffer in full (to find the
# rate-limit bucket and the org context). INGEST_MAX_BODY_BYTES is enforced on
# these before any of that buffering happens. /v1/otlp/traces is not here: the
# middleware never reads its body, and the route caps it itself with
# OTLP_MAX_BODY_BYTES (routers/otlp.py) using the same bounded reader.
_BODY_LIMITED_PATHS = frozenset({"/v1/ingest", "/v1/deploy"})

# Request-level metrics. Counted by the outermost middleware (count_requests in
# create_app) so a 429/413 short-circuit from rate_limit_and_log — which never
# reaches a route — is still a request. The persist-path metrics are in
# routers/ingest.py, next to the code they time.
_requests = _metrics.counter(
    "dunetrace_ingest_requests_total",
    "HTTP requests by route template, method and response status.",
    ("path", "method", "status"),
)
_rate_limited = _metrics.counter(
    "dunetrace_ingest_rate_limited_total",
    "Requests rejected with 429 by the per-key/per-agent rate limiter.",
)
_body_too_large = _metrics.counter(
    "dunetrace_ingest_body_too_large_total",
    "Requests rejected with 413 for a body over INGEST_MAX_BODY_BYTES.",
)

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


async def _otlp_maintenance_loop() -> None:
    """Drain the OTLP retry buffer (batches whose DB persist failed while the DB
    was down), flush receiver stat counters to the DB, and evict idle per-org
    span-rate buckets. Best-effort: a failing tick is logged and retried."""
    from ingest_svc.db import flush_otel_stats
    from ingest_svc.otel_stats import get_otel_stats
    from ingest_svc.otlp_limits import get_persist_retry, get_span_limiter

    while True:
        await asyncio.sleep(5.0)
        try:
            flushed = await get_persist_retry().retry_once()
            if flushed:
                logger.info("OTLP retry flushed %d buffered batch(es)", flushed)
            await flush_otel_stats(get_otel_stats().drain())
            get_span_limiter().evict_stale()
        except Exception as exc:
            logger.debug("OTLP maintenance tick failed: %s", exc)


async def _run_partition_topup_once() -> int:
    """Create the next few monthly event partitions. Best-effort like the other
    two passes: a failure here is retried on the next tick, and must not stop
    retention from running."""
    try:
        from ingest_svc.db.postgres import ensure_event_partitions

        return await ensure_event_partitions()
    except Exception as exc:
        logger.warning("event partition top-up failed: %s", exc)
        return 0


async def _run_prune_once() -> int:
    """One retention pass. A DB error here must not kill the loop — retention
    is enforced on a best-effort basis, and a transient failure just means
    it's retried on the next tick. Returns partitions dropped (0 on failure) —
    used by the manual /admin/prune-events endpoint to report back to the caller.

    Row-count and per-partition detail are logged inside prune_old_events()
    itself, where the data is already at hand (see db/postgres.py) — this
    wrapper only adds wall-clock timing, since that's specific to how often
    this loop runs, not to the prune logic itself.
    """
    t0 = time.monotonic()
    try:
        dropped = await get_event_store().prune_old_events(settings.EVENT_RETENTION_DAYS)
        elapsed_s = time.monotonic() - t0
        if dropped:
            logger.info("Retention pass took %.2fs, dropped %d partition(s)", elapsed_s, dropped)
        return dropped
    except Exception as exc:
        logger.warning("prune_old_events failed: %s", exc)
        return 0


async def _run_scrub_once() -> int:
    """One evidence-scrub pass. Same best-effort contract as _run_prune_once.

    Kept separate from the prune pass rather than folded into it: they operate
    on different tables with different mechanics (partition drop vs batched
    UPDATE), and a failure in one must not skip the other — the scrub is the
    only thing expiring raw content out of failure_signals.evidence, so losing
    it silently to an unrelated partition error is exactly the failure mode
    worth avoiding. Returns rows scrubbed (0 on failure).
    """
    t0 = time.monotonic()
    try:
        scrubbed = await scrub_old_signal_evidence(settings.EVENT_RETENTION_DAYS)
        if scrubbed:
            logger.info(
                "Evidence scrub took %.2fs, scrubbed %d signal(s)",
                time.monotonic() - t0,
                scrubbed,
            )
        return scrubbed
    except Exception as exc:
        logger.warning("scrub_old_signal_evidence failed: %s", exc)
        return 0


async def _prune_loop() -> None:
    """Expire aged data once a day: drop event partitions older than
    EVENT_RETENTION_DAYS, then strip content-bearing keys out of signal evidence
    past the same horizon.

    prune_old_events() existed and was tested but was never actually called
    from anywhere — partitions were created forever and never reclaimed.
    Runs once immediately at startup (in case the service was down long
    enough for retention to matter right away), then every _PRUNE_INTERVAL.

    Both passes share EVENT_RETENTION_DAYS deliberately, so raw content leaves
    `events` and `failure_signals.evidence` at the same moment — one content
    horizon to state and audit, rather than two that can drift apart.
    """
    while True:
        # Top up FIRST. Partitions were only ever created at startup, so a
        # long-uptime process silently began writing into events_default —
        # rows the pruner deliberately never drops, and which then block the
        # next restart from creating the month they belong to. See
        # ensure_event_partitions.
        await _run_partition_topup_once()
        await _run_prune_once()
        await _run_scrub_once()
        await asyncio.sleep(_PRUNE_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting — auth_mode=%s", settings.AUTH_MODE)
    if settings.is_dev:
        logger.warning(
            "AUTH_MODE=%s — AUTHENTICATION IS DISABLED. Anyone who can reach "
            "this port can write events into any org. Do not expose it beyond "
            "localhost. Set AUTH_MODE=prod for any shared or public deployment.",
            settings.AUTH_MODE,
        )
    await init_pool()
    await ensure_schema()
    # ensure_schema has already required CURRENT_SCHEMA_VERSION; this reads the
    # version it found so dunetrace_schema_version reports the real number.
    # db_ready never raises — a failed read leaves the gauge at 0.
    _, ready_info = await _metrics.db_ready(get_pool(), CURRENT_SCHEMA_VERSION)
    if ready_info.get("schema_version") is not None:
        _metrics.set_schema_version(ready_info["schema_version"])

    try:
        if await retention_looks_stale(settings.EVENT_RETENTION_DAYS):
            logger.warning(
                "Retention check: a partition already exceeds EVENT_RETENTION_DAYS=%d at "
                "startup — either this is the first startup after enabling retention on "
                "older data (harmless, the prune loop below will catch up momentarily), or "
                "the retention loop has been silently failing across restarts. Watch for "
                "'Retention pass' log lines after startup to confirm it catches up.",
                settings.EVENT_RETENTION_DAYS,
            )
    except Exception as exc:
        logger.warning("Retention staleness check failed (non-fatal): %s", exc)

    evict_task = asyncio.create_task(_evict_loop())
    heartbeat_task = asyncio.create_task(_rate_limit_heartbeat_loop())
    prune_task = asyncio.create_task(_prune_loop())
    otlp_maint_task = asyncio.create_task(_otlp_maintenance_loop())
    yield
    evict_task.cancel()
    heartbeat_task.cancel()
    prune_task.cancel()
    otlp_maint_task.cancel()
    await close_pool()
    logger.info("Shutdown complete")


# App


def _reject_body_too_large(request: Request, exc: BodyTooLarge) -> JSONResponse:
    """413 for a body over INGEST_MAX_BODY_BYTES. Logged at WARNING with the
    declared and received sizes and the client address — which is also the
    rate-limit bucket an unparseable body would land in, since the api_key
    inside it is never read — and never any of the body itself."""
    _body_too_large.inc()
    client = request.client.host if request.client else "unknown"
    logger.warning(
        "Request body too large; rejected with 413. path=%s declared=%s received=%d "
        "limit=%d client=%s",
        request.url.path,
        "absent" if exc.declared is None else exc.declared,
        exc.received,
        exc.limit,
        client,
    )
    return JSONResponse(
        status_code=413,
        content={"detail": f"Request body exceeds {exc.limit} bytes."},
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


def _path_label(request: Request, known_paths: frozenset[str]) -> str:
    """The ``path`` label for dunetrace_ingest_requests_total.

    The matched route's template when the router ran (so
    ``/admin/keys/{key_id}/agents/{agent_id}/quota``, never the literal ids);
    the literal path when a middleware short-circuited before routing (a 429
    or 413 on a known endpoint); else ``"other"`` — an unknown path is
    caller-controlled and must not mint a new time series per request.
    """
    template = getattr(request.scope.get("route"), "path", None)
    if isinstance(template, str) and template:
        return template
    raw = request.url.path
    return raw if raw in known_paths else "other"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Dunetrace Ingest API",
        version=settings.APP_VERSION,
        description="Receives agent instrumentation events from the Dunetrace SDK.",
        lifespan=lifespan,
    )
    _metrics.register_standard("ingest", settings.APP_VERSION)

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
    #
    # Resolves the caller's org/agent onto request.state so routes and logging
    # can use them. It used to ALSO issue
    #   SELECT set_config('app.current_org_id', $1, true)
    # for row-level security defined in dunetrace-cloud. That could never work
    # and has been removed: the third argument makes the setting
    # TRANSACTION-local, the transaction commits as the `async with` exits, the
    # connection is released immediately, and the route's insert acquires a
    # DIFFERENT connection from the pool. So the value was gone before anything
    # could read it, at a cost of one extra pool acquisition and two round-trips
    # on the hottest path in the system. A gateway that wants RLS has to set the
    # GUC on the same connection as the statement it guards, which means doing
    # it in the query layer, not in middleware.
    #
    # x-org-id is the current header name; x-customer-id is accepted as a fallback
    # for callers running an older cloud gateway build (pre-v0.5.0 naming).
    @app.middleware("http")
    async def set_org_context(request: Request, call_next):
        if is_trusted(request):
            agent_id = request.headers.get("x-agent-id") or None
            org_id = request.headers.get("x-org-id") or request.headers.get("x-customer-id") or None
            request.state.agent_id = agent_id
            request.state.org_id = org_id
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
        return await call_next(request)

    # Registered second = outer = runs FIRST.
    # Trusted path: skip rate limiting (auth service already validated and may rate-limit itself).
    # Dev/direct path: parse body, rate-limit by key (or IP fallback).
    @app.middleware("http")
    async def rate_limit_and_log(request: Request, call_next):
        # Body-size cap, first thing, on the trusted and direct paths alike:
        # this is the outermost middleware, and nothing below it may buffer
        # the body before the cap has been applied. Content-Length is checked
        # before a byte is read; a body that grows past the cap mid-stream
        # (no Content-Length, or a lying one) is cut off there. On success the
        # bounded read caches the body on the request, so request.body() below,
        # set_org_context and the route all still see it. The trusted gateway
        # is not exempt — memory safety belongs to the process doing the
        # buffering, whatever the upstream promised.
        if request.method == "POST" and request.url.path in _BODY_LIMITED_PATHS:
            try:
                await read_bounded_body(request, settings.INGEST_MAX_BODY_BYTES)
            except BodyTooLarge as exc:
                return _reject_body_too_large(request, exc)
            except ClientDisconnect:
                # Was swallowed by the `except Exception: pass` around the old
                # request.body() call and surfaced later as FastAPI's own 400
                # when the route tried to read the body; keep it a 400, just
                # earlier. Nobody is left to receive it, so no log noise.
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Client disconnected before the body was received."},
                )

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
        agent_id: str | None = None
        if request.method == "POST" and request.url.path in _BODY_LIMITED_PATHS:
            try:
                # Already read and cached by the bounded read above; this
                # returns the cached bytes without touching the stream.
                body_bytes = await request.body()
                data = _json.loads(body_bytes) if body_bytes else {}
                api_key = data.get("api_key", "") or ""
                agent_id = data.get("agent_id") or None
            except Exception:
                pass
        elif request.method == "POST" and request.url.path == "/v1/otlp/traces":
            # OTLP auth is a Bearer header, not a JSON body field — and the
            # body may be gzip-compressed protobuf, which there's no reason
            # to read/decode here just to find the rate-limit bucket. Agent
            # identity for per-agent sub-limiting is the same story: the
            # X-Dunetrace-Agent-Id header override (see otlp.py/CLAUDE.md) is
            # free to read, but the service.name resource-attribute fallback
            # would require decoding the body — skipped here, so an OTLP
            # trace relying on service.name alone only gets key-level limiting.
            auth = request.headers.get("Authorization", "")
            api_key = auth[7:].strip() if auth.startswith("Bearer ") else ""
            agent_id = request.headers.get("X-Dunetrace-Agent-Id") or None

        # OPTIONS is excluded: a CORS preflight carries no payload and does no
        # work, but the check was not method-gated, so every preflight spent a
        # token from the caller's bucket. Once exhausted the browser got a 429
        # for the preflight itself — and because CORSMiddleware sits INSIDE this
        # one, that response carries no Access-Control-Allow-Origin, so the
        # browser reports an opaque CORS failure rather than the rate limit that
        # actually happened. HEAD likewise transfers nothing.
        if request.method not in ("OPTIONS", "HEAD") and request.url.path in (
            "/v1/ingest",
            "/v1/deploy",
            "/v1/otlp/traces",
        ):
            bucket = (
                api_key
                if api_key and not api_key.startswith("dt_dev_")
                else (request.client.host if request.client else "unknown")
            )
            limiter = get_limiter()
            result = await limiter.is_allowed(bucket, agent_id)
            if not result.allowed:
                _rate_limited.inc()
                logger.warning("Rate limit exceeded. retry_after=%ds", result.retry_after)
                headers = {
                    "Retry-After": str(result.retry_after),
                    "X-RateLimit-Key-Remaining": str(result.key_remaining),
                }
                if result.agent_remaining is not None:
                    headers["X-RateLimit-Agent-Remaining"] = str(result.agent_remaining)
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded."},
                    headers=headers,
                )

        t = time.monotonic()
        response = await call_next(request)
        ms = (time.monotonic() - t) * 1000
        logger.info("%s %s %d %.1fms", request.method, request.url.path, response.status_code, ms)
        return response

    app.include_router(ingest.router)
    app.include_router(otlp.router)
    app.include_router(health.router)

    # Literal paths the path label may carry when no route matched (a
    # middleware short-circuit). Templated routes are excluded on purpose: a
    # literal /admin/keys/7/... would leak the id into the label; they are
    # only ever reported via scope["route"] once routing has run.
    known_paths = frozenset(
        r.path
        for r in app.routes
        if isinstance(getattr(r, "path", None), str) and "{" not in r.path
    )

    # Registered last = outermost = runs FIRST and sees every response,
    # including the 429/413 that rate_limit_and_log returns without calling
    # the next layer. An exception escaping the stack is the 500 Starlette's
    # ServerErrorMiddleware (above us) will turn it into.
    @app.middleware("http")
    async def count_requests(request: Request, call_next):
        try:
            response = await call_next(request)
        except Exception:
            _requests.labels(
                path=_path_label(request, known_paths),
                method=_method_label(request.method),
                status="500",
            ).inc()
            raise
        _requests.labels(
            path=_path_label(request, known_paths),
            method=_method_label(request.method),
            status=str(response.status_code),
        ).inc()
        return response

    return app


app = create_app()
