"""services/api/api_svc/routers/signals.py"""
from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from api_svc.auth import require_customer
from api_svc.config import settings
from api_svc.db.queries import list_signals
from api_svc.schemas import SignalDetail, SignalListResponse, Page

router = APIRouter(tags=["Signals"])

_VALID_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
_VALID_FAILURE_TYPES = {
    "TOOL_LOOP", "TOOL_THRASHING", "TOOL_AVOIDANCE", "RETRY_STORM",
    "CASCADING_TOOL_FAILURE", "LLM_TRUNCATION_LOOP", "CONTEXT_BLOAT",
    "EMPTY_LLM_RESPONSE", "GOAL_ABANDONMENT", "REASONING_STALL",
    "RAG_EMPTY_RETRIEVAL", "SLOW_STEP", "FIRST_STEP_FAILURE",
    "STEP_COUNT_INFLATION", "PROMPT_INJECTION_SIGNAL",
}


@router.get(
    "/v1/agents/{agent_id}/signals",
    response_model=SignalListResponse,
    summary="List failure signals for an agent",
)
async def get_signals(
    agent_id:     str,
    offset:       int           = Query(0, ge=0),
    limit:        int           = Query(settings.PAGE_SIZE_DEFAULT, ge=1,
                                        le=settings.PAGE_SIZE_MAX),
    severity:     Optional[str] = Query(None,
        description="Filter by severity: LOW | MEDIUM | HIGH | CRITICAL"),
    failure_type: Optional[str] = Query(None,
        description="Filter by failure type e.g. TOOL_LOOP"),
    _customer:    str           = Depends(require_customer),
) -> SignalListResponse:
    if severity and severity.upper() not in _VALID_SEVERITIES:
        raise HTTPException(status_code=422,
            detail=f"Invalid severity {severity!r}. Valid: {sorted(_VALID_SEVERITIES)}")
    if failure_type and failure_type.upper() not in _VALID_FAILURE_TYPES:
        raise HTTPException(status_code=422,
            detail=f"Invalid failure_type {failure_type!r}. Valid: {sorted(_VALID_FAILURE_TYPES)}")
    rows, total = await list_signals(agent_id, offset, limit, severity, failure_type)

    def _ts(v):
        if v is None:
            return None
        return v.timestamp() if hasattr(v, "timestamp") else float(v)

    signals = [SignalDetail(**{**r, "detected_at": _ts(r["detected_at"])}) for r in rows]
    return SignalListResponse(
        signals=signals,
        page=Page(total=total, offset=offset, limit=limit,
                  has_more=(offset + limit) < total),
    )
