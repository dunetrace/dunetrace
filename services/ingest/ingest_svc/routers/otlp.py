"""
POST /v1/otlp/traces — OTLP/HTTP JSON trace receiver.

Accepts OpenTelemetry trace payloads and converts them to Dunetrace events
using the span → event mapper in otel.py.

Auth:
    Authorization: Bearer <api_key>   (same key used by the SDK)
    Dev mode (AUTH_MODE=dev): unauthenticated requests are accepted.

Agent identity:
    By default, service.name from each OTLP resourceSpan is used as agent_id.
    Override with the X-Dunetrace-Agent-Id header to force all spans in the
    request to a single agent_id (useful when service.name differs from the
    Dunetrace agent name).

Response:
    200 {}  — OTLP expects an empty JSON object on success.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status

from ingest_svc.config import settings
from ingest_svc.db import insert_events, verify_api_key
from ingest_svc.otel import otlp_to_events
from ingest_svc.schemas import IngestEvent

logger = logging.getLogger("dunetrace.ingest.otlp")
router = APIRouter()


@router.post(
    "/v1/otlp/traces",
    status_code=status.HTTP_200_OK,
    summary="OTLP/HTTP trace receiver",
    include_in_schema=False,
)
async def receive_otlp_traces(
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict:
    # ── Auth ──────────────────────────────────────────────────────────────────
    auth = request.headers.get("Authorization", "")
    api_key = auth[7:].strip() if auth.startswith("Bearer ") else ""

    resolved = await verify_api_key(api_key)
    if resolved is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key",
        )

    # ── Parse body ────────────────────────────────────────────────────────────
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    resource_spans = body.get("resourceSpans", [])
    if not resource_spans:
        return {}

    # ── Optional agent_id / version overrides ─────────────────────────────────
    agent_id_override = request.headers.get("X-Dunetrace-Agent-Id") or None
    agent_version_override = request.headers.get("X-Dunetrace-Agent-Version") or None

    batch_id = str(uuid.uuid4())
    span_count = sum(
        len(ss.get("spans", []))
        for rs in resource_spans
        for ss in rs.get("scopeSpans", [])
    )
    logger.info(
        "OTLP traces received. resources=%d spans=%d batch_id=%s",
        len(resource_spans),
        span_count,
        batch_id,
    )

    background_tasks.add_task(
        _persist_otlp,
        resource_spans,
        batch_id,
        agent_id_override,
        agent_version_override,
    )
    return {}


async def _persist_otlp(
    resource_spans: list,
    batch_id: str,
    agent_id_override: str | None,
    agent_version_override: str | None,
) -> None:
    try:
        event_dicts = otlp_to_events(
            resource_spans,
            agent_id_override=agent_id_override,
            agent_version_override=agent_version_override,
            batch_id=batch_id,
        )
        if not event_dicts:
            logger.debug("OTLP: no mappable spans in batch_id=%s", batch_id)
            return

        events = [IngestEvent(**e) for e in event_dicts]
        inserted = await insert_events(events, batch_id)
        logger.debug(
            "OTLP persisted. batch_id=%s events=%d inserted=%d",
            batch_id,
            len(events),
            inserted,
        )
    except Exception as exc:
        logger.error("OTLP persist failed. batch_id=%s error=%s", batch_id, exc)
