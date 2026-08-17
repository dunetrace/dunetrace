"""Generic external signal push (Phase 2.4) — the synchronous counterpart to
Phase 2.1-2.3's pull integrations (Langfuse/LangSmith/Braintrust), for any
evaluation source that doesn't have (or doesn't need) a dedicated poller:
a customer's own eval pipeline, a framework without a pull-friendly API,
one-off backfills, etc.

Deliberately different contract from the pull path, since this is a
synchronous customer-facing call whose response is directly inspected rather
than a background loop nobody's watching:
- An unmatched trace_id is a 404, not a silent no-op.
- Correlation is scoped to the caller's own org_id (fetch_run_by_trace_id_for_org)
  rather than trusting a globally-unique trace_id the way the pull
  integrations do — this endpoint takes a caller-supplied trace_id directly
  over the wire, so it doesn't extend that same trust.
- Dedup reuses the same external_evaluation_processed table and (org_id,
  provider, external_id) key the pull integrations use, but here external_id
  is a caller-supplied idempotency key (maintainer decision) rather than a
  third party's own record id — a naive retry-on-timeout must not create a
  duplicate signal.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator

from api_svc.auth import require_org
from api_svc.db.queries import (
    fetch_run_by_trace_id_for_org,
    has_processed_external,
    mark_processed_external,
    write_pushed_external_signal,
)

logger = logging.getLogger("dunetrace.api.external_signals")
router = APIRouter(prefix="/v1/semantic-signals", tags=["External Signals"])


class ExternalSignalRequest(BaseModel):
    trace_id: str = Field(min_length=1, max_length=255)
    provider: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=64)
    external_id: str = Field(
        min_length=1,
        max_length=255,
        description="Caller-supplied idempotency key — a retry with the same "
        "(provider, external_id) is a no-op, not a duplicate signal.",
    )
    value: Optional[float] = None
    string_value: Optional[str] = None
    comment: Optional[str] = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _requires_a_value(self) -> "ExternalSignalRequest":
        if self.value is None and self.string_value is None:
            raise ValueError("at least one of value or string_value is required")
        return self


class ExternalSignalResponse(BaseModel):
    duplicate: bool
    signal_id: Optional[int] = None
    run_id: Optional[str] = None
    agent_id: Optional[str] = None


@router.post(
    "",
    summary="Push an evaluation result from any external source",
    response_model=ExternalSignalResponse,
)
async def push_external_signal(
    body: ExternalSignalRequest,
    org_id: str = Depends(require_org),
) -> ExternalSignalResponse:
    if await has_processed_external(org_id, body.provider, body.external_id):
        return ExternalSignalResponse(duplicate=True)

    run = await fetch_run_by_trace_id_for_org(org_id, body.trace_id)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail=f"No run found for trace_id={body.trace_id!r} in this org.",
        )

    # Not every source's score is guaranteed to be 0-1 — only trust a numeric
    # value in range as a confidence proxy, same convention worker.py's pull
    # integrations use (services/integrations/integrations_svc/worker.py).
    confidence = body.value if (body.value is not None and 0.0 <= body.value <= 1.0) else 0.5

    signal_id = await write_pushed_external_signal(
        org_id=org_id,
        agent_id=run["agent_id"],
        agent_version=run["agent_version"],
        run_id=run["run_id"],
        provider=body.provider,
        failure_type=f"{body.provider.upper()}_{body.name.upper()}",
        confidence=confidence,
        evidence={
            "raw_value": body.value,
            "string_value": body.string_value,
            "comment": body.comment,
            "source_url": None,
        },
    )
    await mark_processed_external(org_id, body.provider, body.external_id)

    return ExternalSignalResponse(
        duplicate=False,
        signal_id=signal_id,
        run_id=run["run_id"],
        agent_id=run["agent_id"],
    )
