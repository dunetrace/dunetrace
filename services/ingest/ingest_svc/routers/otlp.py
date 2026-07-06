"""
POST /v1/otlp/traces — OTLP/HTTP trace receiver.

Accepts OpenTelemetry trace payloads (both application/json and
application/x-protobuf, gzip-compressed or not — see the real OTLP/HTTP
spec: https://opentelemetry.io/docs/specs/otlp/#otlphttp) and converts them
to Dunetrace events using the span → event mapper in otel.py.

application/x-protobuf is the default for most real OTel Collector configs
and for Python's own OTLPSpanExporter — supporting only JSON (as this
endpoint originally did) meant most real-world senders couldn't reach it.

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

import gzip
import json
import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status

from ingest_svc.auth import is_trusted
from ingest_svc.config import settings
from ingest_svc.db import get_event_store, verify_api_key
from ingest_svc.otel import otlp_to_events, protobuf_to_resource_spans
from ingest_svc.schemas import IngestEvent

logger = logging.getLogger("dunetrace.ingest.otlp")
router = APIRouter()


def _decode_body(raw: bytes, content_type: str, content_encoding: str) -> list[dict]:
    """Decompress if needed, then parse as protobuf or JSON depending on
    Content-Type. Raises ValueError on any malformed body — the caller
    turns that into a 400."""
    if content_encoding.lower() == "gzip":
        try:
            raw = gzip.decompress(raw)
        except Exception as exc:
            raise ValueError(f"invalid gzip body: {exc}") from exc

    if "application/x-protobuf" in content_type:
        try:
            return protobuf_to_resource_spans(raw)
        except Exception as exc:
            raise ValueError(f"invalid protobuf body: {exc}") from exc

    try:
        body = json.loads(raw) if raw else {}
    except Exception as exc:
        raise ValueError(f"invalid JSON body: {exc}") from exc
    return body.get("resourceSpans", [])


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
    if is_trusted(request):
        org_id = request.headers.get("x-org-id") or request.headers.get("x-customer-id", "")
        if not org_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Trusted request missing x-org-id",
            )
    else:
        auth = request.headers.get("Authorization", "")
        api_key = auth[7:].strip() if auth.startswith("Bearer ") else ""
        org_id = await verify_api_key(api_key)
        if org_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or inactive API key",
            )

    # ── Parse body (JSON or protobuf, optionally gzip-compressed) ─────────────
    raw_body = await request.body()
    content_type = request.headers.get("content-type", "")
    content_encoding = request.headers.get("content-encoding", "")
    try:
        resource_spans = _decode_body(raw_body, content_type, content_encoding)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not resource_spans:
        return {}

    # ── Optional agent_id / version overrides ─────────────────────────────────
    agent_id_override = request.headers.get("X-Dunetrace-Agent-Id") or None
    agent_version_override = request.headers.get("X-Dunetrace-Agent-Version") or None

    batch_id = str(uuid.uuid4())
    span_count = sum(
        len(ss.get("spans", [])) for rs in resource_spans for ss in rs.get("scopeSpans", [])
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
        org_id,
    )
    return {}


async def _persist_otlp(
    resource_spans: list,
    batch_id: str,
    agent_id_override: str | None,
    agent_version_override: str | None,
    org_id: str,
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
        inserted = await get_event_store().insert_events(events, batch_id, org_id)
        logger.debug(
            "OTLP persisted. batch_id=%s events=%d inserted=%d",
            batch_id,
            len(events),
            inserted,
        )
    except Exception as exc:
        logger.error("OTLP persist failed. batch_id=%s error=%s", batch_id, exc)
