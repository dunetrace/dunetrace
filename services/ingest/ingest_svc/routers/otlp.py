"""
POST /v1/otlp/traces — OTLP/HTTP trace receiver.

Accepts OpenTelemetry trace payloads (both application/json and
application/x-protobuf, gzip-compressed or not — see the real OTLP/HTTP
spec: https://opentelemetry.io/docs/specs/otlp/#otlphttp) and converts them
to Dunetrace events using the span → event mapper in otel.py.

application/x-protobuf is the default for most real OTel Collector configs
and for Python's own OTLPSpanExporter — supporting only JSON (as this
endpoint originally did) meant most real-world senders couldn't reach it.

Auth (any one of, checked in this order):
    Authorization: Bearer <api_key>        same key the SDK uses
    X-Dunetrace-API-Key: <api_key>         for collectors that set headers
    dunetrace.api_key resource attribute   for emitters that set only resource attrs
    Trusted gateway: x-internal-token + x-org-id
    Dev mode (ENV=dev): unauthenticated requests resolve to the 'default' org.

Missing or invalid auth is rejected with 401, and rejected requests never reach
mapping or storage. Auth-rejection logs are throttled so a misconfigured sender
can't flood them. A per-org kill switch (organizations.otel_ingestion_enabled)
disables ingestion for one org with 403, on top of rate limiting.

Agent identity:
    By default, service.name from each OTLP resourceSpan is used as agent_id.
    Override with the X-Dunetrace-Agent-Id header to force all spans in the
    request to a single agent_id (useful when service.name differs from the
    Dunetrace agent name).

Response:
    200 {}  — OTLP expects an empty JSON object on success.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
import zlib

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ingest_svc.auth import is_trusted
from ingest_svc.config import settings
from ingest_svc.db import (
    fetch_otel_ingestion_enabled,
    set_otel_ingestion_enabled,
    verify_api_key,
)
from ingest_svc.otel import _attr, otlp_to_events, protobuf_to_resource_spans
from ingest_svc.otel_stats import get_otel_stats
from ingest_svc.otlp_limits import (
    get_inflight_guard,
    get_persist_retry,
    get_span_limiter,
)
from ingest_svc.routers.ingest import _check_admin_key
from ingest_svc.schemas import IngestEvent

_SUPPORTED_CONTENT_TYPES = ("application/json", "application/x-protobuf")

logger = logging.getLogger("dunetrace.ingest.otlp")
router = APIRouter()

# Collapse auth-rejection logs to at most one line per window, with a running
# count, so a misconfigured collector retrying forever can't flood the log.
_AUTH_FAIL_LOG_WINDOW = 60.0
_last_auth_fail_log = 0.0
_auth_fail_count = 0

# Per-org enablement is cached so the accept path stays off the DB.
_ENABLE_CACHE_TTL = 300.0
_enable_cache: dict[str, tuple[bool, float]] = {}


class _OtlpTooLarge(Exception):
    """Body (compressed or decompressed) exceeds a configured limit. Mapped to a
    413 by the route."""


async def _read_bounded_body(request: Request, limit: int) -> bytes:
    """Read the request body, refusing to buffer more than `limit` bytes. Honors
    Content-Length when present, and also caps during streaming so a missing or
    lying Content-Length can't slip an oversized body past."""
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError:
            declared = -1
        if declared > limit:
            raise _OtlpTooLarge()
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise _OtlpTooLarge()
        chunks.append(chunk)
    return b"".join(chunks)


def _gunzip_bounded(raw: bytes, limit: int) -> bytes:
    """Decompress a gzip body, refusing to expand past `limit` bytes. Guards
    against gzip bombs (a few KB expanding to gigabytes). Raises _OtlpTooLarge
    when the limit is hit, ValueError when the body isn't valid gzip."""
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    out = bytearray()
    try:
        out += decompressor.decompress(raw, limit + 1)
        while decompressor.unconsumed_tail and len(out) <= limit:
            out += decompressor.decompress(decompressor.unconsumed_tail, limit + 1 - len(out))
        out += decompressor.flush()
    except Exception as exc:
        raise ValueError(f"invalid gzip body: {exc}") from exc
    if len(out) > limit:
        raise _OtlpTooLarge()
    return bytes(out)


def _decode_body(raw: bytes, content_type: str, content_encoding: str) -> list[dict]:
    """Decompress if needed, then parse as protobuf or JSON depending on
    Content-Type. Raises ValueError on a malformed body (caller: 400) or
    _OtlpTooLarge on a gzip bomb (caller: 413)."""
    if content_encoding.lower() == "gzip":
        raw = _gunzip_bounded(raw, settings.OTLP_MAX_DECOMPRESSED_BYTES)

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


# ── Auth helpers ─────────────────────────────────────────────────────────────────


def _header_api_key(request: Request) -> str:
    """API key from Authorization: Bearer <key>, or the X-Dunetrace-API-Key
    header. Returns "" when neither is present."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return request.headers.get("X-Dunetrace-API-Key", "").strip()


def _resource_api_key(resource_spans: list[dict]) -> str:
    """API key carried as a `dunetrace.api_key` resource attribute, for emitters
    that can set resource attributes but not request headers. First one wins."""
    for rs in resource_spans:
        val = _attr(rs.get("resource", {}).get("attributes", []), "dunetrace.api_key")
        if val:
            return str(val).strip()
    return ""


def _record_auth_failure(request: Request) -> None:
    global _last_auth_fail_log, _auth_fail_count
    _auth_fail_count += 1
    now = time.monotonic()
    if now - _last_auth_fail_log >= _AUTH_FAIL_LOG_WINDOW:
        client = request.client.host if request.client else "unknown"
        logger.warning(
            "OTLP auth rejected. count=%d window=%.0fs last_client=%s",
            _auth_fail_count,
            _AUTH_FAIL_LOG_WINDOW,
            client,
        )
        _last_auth_fail_log = now
        _auth_fail_count = 0


async def _ingestion_enabled(org_id: str) -> bool:
    """Per-org OTel enablement, cached for _ENABLE_CACHE_TTL. Fail-open, so a DB
    hiccup never silently drops OTLP traffic (see fetch_otel_ingestion_enabled)."""
    now = time.monotonic()
    cached = _enable_cache.get(org_id)
    if cached and now - cached[1] < _ENABLE_CACHE_TTL:
        return cached[0]
    enabled = await fetch_otel_ingestion_enabled(org_id)
    _enable_cache[org_id] = (enabled, now)
    return enabled


# ── Route ────────────────────────────────────────────────────────────────────────


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
    stats = get_otel_stats()
    trusted = is_trusted(request)
    org_id: str | None = None
    if trusted:
        org_id = request.headers.get("x-org-id") or request.headers.get("x-customer-id", "")
        if not org_id:
            stats.record_auth_failure()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Trusted request missing x-org-id",
            )

    # ── Read + parse body (JSON or protobuf, optionally gzip-compressed) ──────
    # Done up front: the spans are needed for mapping, and for the
    # dunetrace.api_key resource-attribute auth fallback below. The read is size-
    # bounded so an oversized body is rejected before it is buffered in full.
    content_type = request.headers.get("content-type", "")
    content_encoding = request.headers.get("content-encoding", "")

    base_ct = content_type.split(";", 1)[0].strip().lower()
    if base_ct and base_ct not in _SUPPORTED_CONTENT_TYPES:
        stats.record_rejected(org_id, "unsupported_media")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported content-type '{base_ct}'. Use application/json or "
                "application/x-protobuf."
            ),
        )

    try:
        raw_body = await _read_bounded_body(request, settings.OTLP_MAX_BODY_BYTES)
    except _OtlpTooLarge:
        stats.record_rejected(org_id, "oversized")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Request body exceeds {settings.OTLP_MAX_BODY_BYTES} bytes.",
        )
    try:
        resource_spans = _decode_body(raw_body, content_type, content_encoding)
    except _OtlpTooLarge:
        stats.record_rejected(org_id, "gzip_bomb")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Compressed body expands beyond the decompression limit.",
        )
    except ValueError as exc:
        stats.record_rejected(org_id, "malformed")
        raise HTTPException(status_code=400, detail=str(exc))

    # ── Auth (untrusted): header key, then resource-attribute key ─────────────
    if not trusted:
        org_id = await verify_api_key(_header_api_key(request))
        if org_id is None:
            resource_key = _resource_api_key(resource_spans)
            org_id = await verify_api_key(resource_key) if resource_key else None
        if org_id is None:
            stats.record_auth_failure()
            _record_auth_failure(request)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=(
                    "Missing or invalid API key. Provide it as 'Authorization: "
                    "Bearer <key>', an 'X-Dunetrace-API-Key' header, or a "
                    "'dunetrace.api_key' resource attribute."
                ),
            )

    # ── Per-org enablement (kill switch) ──────────────────────────────────────
    if not await _ingestion_enabled(org_id):
        stats.record_rejected(org_id, "disabled_org")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="OTel ingestion is disabled for this org.",
        )

    if not resource_spans:
        return {}

    span_count = sum(
        len(ss.get("spans", [])) for rs in resource_spans for ss in rs.get("scopeSpans", [])
    )

    # ── Per-org span rate limit (noisy-neighbor protection) ───────────────────
    allowed, retry_after = get_span_limiter().allow(org_id, span_count)
    if not allowed:
        # Per-org attribution is recorded in the stats table (shown on the
        # receiver dashboard); the org_id is deliberately kept out of this log
        # line since it is derived from the API key (avoids clear-text logging of
        # credential-derived data).
        stats.record_rate_limit_hit(org_id)
        logger.warning("OTLP span rate limit exceeded. spans=%d", span_count)
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Span rate limit exceeded."},
            headers={"Retry-After": str(retry_after)},
        )

    # ── Backpressure: shed load rather than grow an unbounded backlog ─────────
    if not get_inflight_guard().try_reserve():
        stats.record_rejected(org_id, "backpressure")
        logger.warning("OTLP backpressure: too many in-flight batches, shedding.")
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Receiver busy, retry shortly."},
            headers={"Retry-After": "1"},
        )

    # ── Optional agent_id / version overrides ─────────────────────────────────
    agent_id_override = request.headers.get("X-Dunetrace-Agent-Id") or None
    agent_version_override = request.headers.get("X-Dunetrace-Agent-Version") or None

    batch_id = str(uuid.uuid4())
    stats.record_received(org_id, span_count)
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
    # Reserved by the route before scheduling; always released here so the
    # backpressure counter can't leak.
    try:
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
        except Exception as exc:
            # A mapping/validation failure is a data problem, not a transient DB
            # one — log and drop, don't retry it forever.
            get_otel_stats().record_rejected(org_id, "mapping_error")
            logger.error("OTLP mapping failed. batch_id=%s error=%s", batch_id, exc)
            return

        get_otel_stats().record_translated(org_id, len(events))
        # DB failures go through the circuit breaker + retry buffer, so the
        # receiver keeps accepting while the DB is unreachable.
        await get_persist_retry().persist(events, batch_id, org_id)
        logger.debug("OTLP mapped. batch_id=%s events=%d", batch_id, len(events))
    finally:
        get_inflight_guard().release()


# ── Admin: per-org enablement ────────────────────────────────────────────────────


class OtelIngestionToggle(BaseModel):
    admin_key: str
    org_id: str  # in the body, not the path: this admin action targets any org
    enabled: bool


@router.put(
    "/admin/otel-ingestion",
    summary="Enable or disable OTel ingestion for an org (admin kill switch)",
    include_in_schema=False,
)
async def set_org_otel_ingestion(body: OtelIngestionToggle) -> dict:
    """Admin-only. Flips organizations.otel_ingestion_enabled for the org named in
    the body. Cross-org by design (an operator disabling an abusive org), so the
    target org_id is a body field, not a path param or the caller's own org.
    Takes effect immediately (the per-org cache entry is dropped here)."""
    _check_admin_key(body.admin_key)
    try:
        await set_otel_ingestion_enabled(body.org_id, body.enabled)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    _enable_cache.pop(body.org_id, None)
    logger.info(
        "OTel ingestion %s for org_id=%s", "enabled" if body.enabled else "disabled", body.org_id
    )
    return {"org_id": body.org_id, "otel_ingestion_enabled": body.enabled}
