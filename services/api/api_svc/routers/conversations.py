"""Conversation detail and cross-conversation search (Phase 3.3) — the
dashboard-facing read layer over Phase 3.1's conversations/runs tables
(detector_svc-owned) and Phase 3.2's ConversationEvaluator signals."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api_svc.auth import require_org
from api_svc.config import settings
from api_svc.db.queries import get_conversation_detail, search_conversations
from api_svc.schemas import (
    ConversationDetail,
    ConversationRun,
    ConversationSearchResponse,
    ConversationSignal,
    ConversationSummary,
    Page,
)

router = APIRouter(tags=["Conversations"])


@router.get(
    "/v1/conversations/search",
    response_model=ConversationSearchResponse,
    summary="Search conversations by agent, user, or frustration-signal presence",
)
async def search(
    agent_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(
        None,
        description="Currently a no-op filter in practice — no SDK parameter "
        "populates conversations.user_id yet.",
    ),
    has_frustration_signal: Optional[bool] = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(settings.PAGE_SIZE_DEFAULT, ge=1, le=settings.PAGE_SIZE_MAX),
    org_id: str = Depends(require_org),
) -> ConversationSearchResponse:
    rows, total = await search_conversations(
        org_id, agent_id, user_id, has_frustration_signal, offset, limit
    )
    return ConversationSearchResponse(
        conversations=[ConversationSummary(**r) for r in rows],
        page=Page(total=total, offset=offset, limit=limit, has_more=(offset + limit) < total),
    )


@router.get(
    "/v1/conversations/{conversation_id}",
    response_model=ConversationDetail,
    summary="Get a conversation's full run list and conversation-level signals",
)
async def get_conversation(
    conversation_id: int,
    org_id: str = Depends(require_org),
) -> ConversationDetail:
    data = await get_conversation_detail(org_id, conversation_id)
    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id!r} not found",
        )

    return ConversationDetail(
        id=data["id"],
        agent_id=data["agent_id"],
        user_id=data["user_id"],
        external_id=data["external_id"],
        first_run_at=data["first_run_at"],
        last_run_at=data["last_run_at"],
        run_count=data["run_count"],
        runs=[ConversationRun(**r) for r in data["runs"]],
        signals=[ConversationSignal(**s) for s in data["signals"]],
    )
