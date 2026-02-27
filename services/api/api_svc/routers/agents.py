"""services/api/api_svc/routers/agents.py"""
from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, Depends, Query
from api_svc.auth import require_customer
from api_svc.config import settings
from api_svc.db.queries import list_agents
from api_svc.schemas import AgentListResponse, AgentSummary, Page

router = APIRouter(prefix="/v1/agents", tags=["Agents"])


@router.get("", response_model=AgentListResponse, summary="List agents")
async def get_agents(
    offset: int = Query(0, ge=0),
    limit:  int = Query(settings.PAGE_SIZE_DEFAULT, ge=1, le=settings.PAGE_SIZE_MAX),
    customer_id: str = Depends(require_customer),
) -> AgentListResponse:
    rows, total = await list_agents(customer_id, offset, limit)

    def ts(v):
        if v is None:
            return None
        return v.timestamp() if hasattr(v, "timestamp") else float(v)

    agents = [
        AgentSummary(
            agent_id=r["agent_id"],
            last_seen=ts(r["last_seen"]),
            run_count=r["run_count"] or 0,
            signal_count=r["signal_count"] or 0,
            critical_count=r["critical_count"] or 0,
            high_count=r["high_count"] or 0,
        )
        for r in rows
    ]
    return AgentListResponse(
        agents=agents,
        page=Page(total=total, offset=offset, limit=limit,
                  has_more=(offset + limit) < total),
    )
