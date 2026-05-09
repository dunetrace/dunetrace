"""Agents endpoints — list agents and their signal summaries."""
from __future__ import annotations
from typing import Optional
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Query
from api_svc.auth import require_customer
from api_svc.config import settings
from api_svc.db.queries import (
    list_agents, agent_signal_sparklines, agent_failure_type_counts,
    get_agent_health_score,
)
from api_svc.schemas import AgentListResponse, AgentSummary, AgentHealthScore, Page

router = APIRouter(prefix="/v1/agents", tags=["Agents"])


@router.get("", response_model=AgentListResponse, summary="List agents")
async def get_agents(
    offset: int = Query(0, ge=0),
    limit:  int = Query(settings.PAGE_SIZE_DEFAULT, ge=1, le=settings.PAGE_SIZE_MAX),
    customer_id: str = Depends(require_customer),
) -> AgentListResponse:
    rows, total = await list_agents(customer_id, offset, limit)
    sparklines, ft_counts = await asyncio.gather(
        agent_signal_sparklines(customer_id),
        agent_failure_type_counts(customer_id),
    )

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
            sparkline=sparklines.get(r["agent_id"], []),
            failure_types=ft_counts.get(r["agent_id"], {}),
        )
        for r in rows
    ]
    return AgentListResponse(
        agents=agents,
        page=Page(total=total, offset=offset, limit=limit,
                  has_more=(offset + limit) < total),
    )


@router.get(
    "/{agent_id}/health-score",
    response_model=AgentHealthScore,
    summary="0–100 health score for an agent",
)
async def get_health_score(
    agent_id:    str,
    customer_id: str = Depends(require_customer),
) -> AgentHealthScore:
    """
    Composite health score derived from the last 30 days of run data.

    Components:
    - **failure_rate** (0–40 pts): proportion of runs with any signal — active from run 1
    - **loop_avoidance** (0–25 pts): proportion of runs free of looping signals — active from run 1
    - **token_efficiency** (0–20 pts): scored relative to agent's own P75 baseline once ≥ 30 runs exist; neutral (15 pts) until then
    - **latency** (0–15 pts): scored relative to agent's own P75 baseline once ≥ 30 runs exist; neutral (10 pts) until then

    Returns `score: null` when fewer than 3 sample runs are available.
    `baseline_ready: false` when fewer than 30 runs — token and latency components are held at neutral.
    """
    result = await get_agent_health_score(agent_id)
    return AgentHealthScore(agent_id=agent_id, **result)
