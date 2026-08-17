"""ElevenLabs pull integration config (Phase 4.1). Bring-your-own API key per
org, encrypted at rest with the same Fernet infra as the Langfuse/LangSmith/
Braintrust config in routers/integrations.py. Kept in its own router (not folded
into integrations.py) because ElevenLabs is not an evaluation provider: it has
no endpoint_url (the base URL is fixed), it validates the key on save, and it
stores into the dedicated elevenlabs_integrations table.

The credential is encrypted then never returned once stored. GET exposes
configuration and health only, never the key. Only elevenlabs_worker (Phase
4.3) ever decrypts it, to call ElevenLabs' history API."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api_svc.auth import require_org
from api_svc.crypto import encrypt_credentials
from api_svc.db.queries import (
    analytics_cost_by_outcome,
    analytics_truncation_impact,
    analytics_voice_impact,
    delete_elevenlabs_integration,
    get_call_elevenlabs_cost,
    get_elevenlabs_integration_status,
    get_run_elevenlabs_generations,
    list_elevenlabs_generations,
    upsert_elevenlabs_integration,
)
from api_svc.elevenlabs_analytics import (
    FRUSTRATION_SIGNALS,
    summarize_cost_by_outcome,
    summarize_truncation_impact,
    summarize_voice_impact,
)
from api_svc.elevenlabs_client import (
    ElevenLabsAuthError,
    ElevenLabsUnreachable,
    validate_api_key,
)
from api_svc.voice_pricing import load_pricing, tts_cost_usd

logger = logging.getLogger("dunetrace.api.elevenlabs")
router = APIRouter(prefix="/v1/orgs/integrations", tags=["Integrations"])

# A correlation at or above this confidence is shown plainly; below it, the UI
# adds a "verify" indicator. char_time (0.70) falls below; voice_char_time (0.85)
# and the deterministic tiers are at or above.
_CONFIDENT_AT = 0.85


class ElevenLabsIntegrationRequest(BaseModel):
    api_key: str = Field(min_length=1)
    # Default 5 minutes, the conservative cadence the brief mandates. Floor is
    # 60s, not the 10s the evaluation providers allow: ElevenLabs history does
    # not change fast enough to justify polling harder, and a higher floor keeps
    # us comfortably clear of ElevenLabs' concurrency limits even across many
    # orgs.
    poll_interval_secs: int = Field(default=300, ge=60, le=3600)


class ElevenLabsIntegrationStatus(BaseModel):
    configured: bool
    poll_interval_secs: int | None = None
    enabled: bool | None = None
    last_polled_at: float | None = None
    last_success_at: float | None = None
    consecutive_failures: int | None = None


def _status_response(status: dict | None) -> ElevenLabsIntegrationStatus:
    if status is None:
        return ElevenLabsIntegrationStatus(configured=False)
    return ElevenLabsIntegrationStatus(
        configured=True,
        poll_interval_secs=status["poll_interval_secs"],
        enabled=status["enabled"],
        last_polled_at=status["last_polled_at"].timestamp() if status["last_polled_at"] else None,
        last_success_at=status["last_success_at"].timestamp()
        if status["last_success_at"]
        else None,
        consecutive_failures=status["consecutive_failures"],
    )


@router.post(
    "/elevenlabs",
    summary="Configure an ElevenLabs pull integration for this org",
    response_model=ElevenLabsIntegrationStatus,
    status_code=201,
)
async def set_elevenlabs_integration(
    body: ElevenLabsIntegrationRequest,
    org_id: str = Depends(require_org),
) -> ElevenLabsIntegrationStatus:
    # Encrypt first: it is local and fails fast if the master key is missing, so
    # a misconfigured server never makes an outbound call to ElevenLabs.
    try:
        encrypted = encrypt_credentials({"api_key": body.api_key})
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # Validate the key against the live API before storing, so a typo'd key is
    # rejected now rather than failing silently in the poller later.
    try:
        await validate_api_key(body.api_key)
    except ElevenLabsAuthError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid ElevenLabs API key: {exc}")
    except ElevenLabsUnreachable as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not validate the key, ElevenLabs is unreachable: {exc}",
        )

    await upsert_elevenlabs_integration(org_id, encrypted, body.poll_interval_secs)
    status = await get_elevenlabs_integration_status(org_id)
    return _status_response(status)


@router.get(
    "/elevenlabs",
    summary="Get this org's ElevenLabs integration status",
    response_model=ElevenLabsIntegrationStatus,
)
async def get_elevenlabs_integration(
    org_id: str = Depends(require_org),
) -> ElevenLabsIntegrationStatus:
    status = await get_elevenlabs_integration_status(org_id)
    return _status_response(status)


@router.delete(
    "/elevenlabs",
    summary="Remove this org's ElevenLabs integration",
    status_code=204,
)
async def remove_elevenlabs_integration(org_id: str = Depends(require_org)):
    deleted = await delete_elevenlabs_integration(org_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="No ElevenLabs integration configured.")


# ── Correlated generation reads (Phase 5.1) ────────────────────────────────────
#
# ElevenLabs has no reliable per-item dashboard deep link, so instead of
# fabricating one we point at the history page and surface the generation_id for
# the customer to find. This is the honest choice (constraint 5).
ELEVENLABS_HISTORY_URL = "https://elevenlabs.io/app/history"


class ElevenLabsGenerationOut(BaseModel):
    generation_id: str
    voice_id: str | None = None
    voice_name: str | None = None
    model: str | None = None
    character_count: int
    cost_credits: int | None = None
    cost_usd: float  # derived from character_count via voice-pricing.yml's elevenlabs rate
    source: str | None = None
    generated_at: float
    run_id: str | None = None
    agent_id: str | None = None
    correlation_method: str | None = None
    correlation_confidence: float | None = None
    uncertain: bool  # true when the match rests on weak signals (show a "verify" hint)
    dashboard_url: str = ELEVENLABS_HISTORY_URL


class ElevenLabsCallCost(BaseModel):
    generation_count: int
    character_count: int
    cost_credits: int
    cost_usd: float


def _to_generation_out(row: dict, pricing: dict) -> ElevenLabsGenerationOut:
    conf = row.get("correlation_confidence")
    method = row.get("correlation_method")
    uncertain = method == "char_time" or (conf is not None and conf < _CONFIDENT_AT)
    return ElevenLabsGenerationOut(
        generation_id=row["generation_id"],
        voice_id=row.get("voice_id"),
        voice_name=row.get("voice_name"),
        model=row.get("model"),
        character_count=row["character_count"],
        cost_credits=row.get("cost_credits"),
        cost_usd=tts_cost_usd(row["character_count"], "elevenlabs", pricing),
        source=row.get("source"),
        generated_at=row["generated_at"],
        run_id=row.get("run_id"),
        agent_id=row.get("agent_id"),
        correlation_method=method,
        correlation_confidence=conf,
        uncertain=uncertain,
    )


@router.get(
    "/elevenlabs/generations",
    summary="List correlated ElevenLabs generations (by run, or filtered by voice/model/cost)",
    response_model=list[ElevenLabsGenerationOut],
)
async def list_generations(
    run_id: Optional[str] = Query(None, description="Scope to one run's correlated generations."),
    voice_id: Optional[str] = Query(None, description="Filter by ElevenLabs voice id."),
    model: Optional[str] = Query(None, description="Filter by ElevenLabs model id."),
    min_credits: Optional[int] = Query(
        None, ge=0, description="Only generations costing at least this many credits."
    ),
    limit: int = Query(100, ge=1, le=500),
    org_id: str = Depends(require_org),
) -> list[ElevenLabsGenerationOut]:
    pricing = load_pricing()
    if run_id:
        rows = await get_run_elevenlabs_generations(org_id, run_id)
    else:
        rows = await list_elevenlabs_generations(org_id, voice_id, model, min_credits, limit)
    return [_to_generation_out(r, pricing) for r in rows]


@router.get(
    "/elevenlabs/calls/{conversation_id}/cost",
    summary="Aggregate ElevenLabs cost for one call (actual, from ElevenLabs)",
    response_model=ElevenLabsCallCost,
)
async def call_elevenlabs_cost(
    conversation_id: int,
    org_id: str = Depends(require_org),
) -> ElevenLabsCallCost:
    agg = await get_call_elevenlabs_cost(org_id, conversation_id)
    pricing = load_pricing()
    return ElevenLabsCallCost(
        generation_count=agg["generation_count"],
        character_count=agg["character_count"],
        cost_credits=agg["cost_credits"],
        cost_usd=tts_cost_usd(agg["character_count"], "elevenlabs", pricing),
    )


# ── Cross-tool analytics (Phase 6.1) ───────────────────────────────────────────
# Each endpoint returns a plain dict (the shapes are analytic-specific and come
# straight from the pure summarizers, which own the edge-case handling).


@router.get(
    "/elevenlabs/analytics/cost-by-outcome",
    summary="Analytic 1: TTS spend broken down by call outcome (successful vs flagged)",
)
async def analytics_cost_by_outcome_endpoint(
    limit: int = Query(200, ge=1, le=1000),
    org_id: str = Depends(require_org),
) -> dict:
    rows = await analytics_cost_by_outcome(org_id, limit)
    return summarize_cost_by_outcome(rows, load_pricing())


@router.get(
    "/elevenlabs/analytics/voice-impact",
    summary="Analytic 2: per-voice signal rate (does one voice correlate with worse outcomes)",
)
async def analytics_voice_impact_endpoint(org_id: str = Depends(require_org)) -> dict:
    rows = await analytics_voice_impact(org_id)
    return summarize_voice_impact(rows)


@router.get(
    "/elevenlabs/analytics/truncation-impact",
    summary="Analytic 3: frustration rate on truncated vs clean TTS runs",
)
async def analytics_truncation_impact_endpoint(org_id: str = Depends(require_org)) -> dict:
    row = await analytics_truncation_impact(org_id, FRUSTRATION_SIGNALS)
    return summarize_truncation_impact(row)
