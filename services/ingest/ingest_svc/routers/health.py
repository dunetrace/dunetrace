"""GET /health (liveness), GET /ready (readiness), GET /metrics (Prometheus).

Three probes with three different questions:

* ``/health`` — is the process up and serving HTTP? Touches nothing else.
* ``/ready`` — can it do its job right now? Database reachable, shared schema
  at least at the version this build was written against. This is what the
  compose healthcheck and any load balancer should point at.
* ``/metrics`` — Prometheus exposition. No auth, hidden from OpenAPI: it is
  for an in-network scraper only and must not be published beyond that
  network (compose binds the ingest port to loopback).
"""

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse

from dunetrace_schemas import metrics as _metrics
from dunetrace_schemas.migrations import CURRENT_SCHEMA_VERSION
from ingest_svc.config import settings
from ingest_svc.db import get_pool
from ingest_svc.schemas import HealthResponse, ReadyResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, include_in_schema=False)
async def health() -> HealthResponse:
    """Liveness only. Deliberately no DB round-trip: a probe that waits on a
    pool connection reports a saturated-but-working process as dead, and an
    orchestrator acting on that restarts the one thing that was still draining
    the backlog. Dependency state is ``/ready``'s job."""
    return HealthResponse(version=settings.APP_VERSION)


@router.get("/ready", include_in_schema=False)
async def ready() -> JSONResponse:
    """Readiness: 200 when the DB answers ``SELECT 1`` and the applied schema
    version is at least ``CURRENT_SCHEMA_VERSION``, 503 with the same body
    otherwise. The body always carries the verdict's inputs so a failing probe
    can be read from the healthcheck log alone."""
    ok, info = await _metrics.db_ready(get_pool(), CURRENT_SCHEMA_VERSION)
    body = ReadyResponse(
        status="ok" if ok else "not_ready",
        version=settings.APP_VERSION,
        **info,
    )
    return JSONResponse(status_code=200 if ok else 503, content=body.model_dump())


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    body, content_type = _metrics.render()
    return Response(content=body, media_type=content_type)
