"""Call-level metrics (Phase 2.1). A "call" is a conversation for a voice agent,
read through a voice lens: same conversations/runs tables as the Conversations
view, with voice-specific derived metrics (duration, completion status, silence
percentage, agent-vs-caller talk ratio). No separate storage (see BACKLOG.md V1).
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api_svc.auth import require_org
from api_svc.config import settings
from api_svc.db.queries import get_call_detail, list_calls
from api_svc.schemas import CallDetail, CallListResponse, CallSummary, Page

router = APIRouter(tags=["Calls"])

_COMPLETION_STATUSES = {"natural", "dropped", "escalated"}
_COST_BUCKETS = {"low", "medium", "high"}  # <$0.10 | $0.10-$1 | >$1


@router.get(
    "/v1/calls",
    response_model=CallListResponse,
    summary="List voice calls with call-level metrics",
)
async def list_all_calls(
    agent_id: Optional[str] = Query(None),
    since: Optional[float] = Query(
        None, description="Only calls whose last run is at/after this Unix timestamp."
    ),
    completion_status: Optional[str] = Query(
        None, description="Filter by completion status: natural | dropped | escalated."
    ),
    cost_bucket: Optional[str] = Query(
        None, description="Filter by cost: low (<$0.10) | medium ($0.10-$1) | high (>$1)."
    ),
    offset: int = Query(0, ge=0),
    limit: int = Query(settings.PAGE_SIZE_DEFAULT, ge=1, le=settings.PAGE_SIZE_MAX),
    org_id: str = Depends(require_org),
) -> CallListResponse:
    if completion_status is not None and completion_status not in _COMPLETION_STATUSES:
        raise HTTPException(
            422,
            f"Invalid completion_status {completion_status!r}. Valid: {sorted(_COMPLETION_STATUSES)}",
        )
    if cost_bucket is not None and cost_bucket not in _COST_BUCKETS:
        raise HTTPException(
            422, f"Invalid cost_bucket {cost_bucket!r}. Valid: {sorted(_COST_BUCKETS)}"
        )
    rows, total = await list_calls(
        org_id, agent_id, since, completion_status, cost_bucket, offset, limit
    )
    return CallListResponse(
        calls=[CallSummary(**r) for r in rows],
        page=Page(total=total, offset=offset, limit=limit, has_more=(offset + limit) < total),
    )


@router.get(
    "/v1/calls/{conversation_id}",
    response_model=CallDetail,
    summary="One call's metrics, runs, voice signals, and voice event timeline",
)
async def get_call(
    conversation_id: int,
    org_id: str = Depends(require_org),
) -> CallDetail:
    data = await get_call_detail(org_id, conversation_id)
    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call {conversation_id!r} not found or has no runs",
        )
    return CallDetail(**data)
