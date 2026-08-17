"""OTel receiver observability (Phase 5). Per-org view of what the OTLP receiver
did: spans received, translated, rejected (by reason), auth failures, and rate-
limit hits, plus anomaly flags for the dashboard. Read-only over the
otel_receiver_stats table that ingest writes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from api_svc.auth import require_org
from api_svc.db.queries import (
    fetch_otel_receiver_stats,
    get_org_otel_ingestion_enabled,
    set_org_otel_ingestion_enabled,
)
from api_svc.otel_receiver_health import detect_anomalies, summarize_totals

router = APIRouter()


class OtelIngestionToggle(BaseModel):
    enabled: bool


@router.get(
    "/v1/orgs/otel-receiver/stats",
    summary="OTel receiver activity and health for the org",
    include_in_schema=False,
)
async def otel_receiver_stats(
    hours: int = Query(24, ge=1, le=168),
    org_id: str = Depends(require_org),
) -> dict:
    series = await fetch_otel_receiver_stats(org_id, hours)
    return {
        "org_id": org_id,
        "hours": hours,
        "series": series,
        "totals": summarize_totals(series),
        "anomalies": detect_anomalies(series),
    }


@router.get(
    "/v1/orgs/otel-receiver/enabled",
    summary="Whether OTel ingestion is enabled for the org",
    include_in_schema=False,
)
async def get_otel_ingestion_enabled(org_id: str = Depends(require_org)) -> dict:
    return {"enabled": await get_org_otel_ingestion_enabled(org_id)}


@router.put(
    "/v1/orgs/otel-receiver/enabled",
    summary="Enable or disable OTel ingestion for the org",
    include_in_schema=False,
)
async def set_otel_ingestion_enabled(
    body: OtelIngestionToggle,
    org_id: str = Depends(require_org),
) -> dict:
    await set_org_otel_ingestion_enabled(org_id, body.enabled)
    return {"enabled": body.enabled}
