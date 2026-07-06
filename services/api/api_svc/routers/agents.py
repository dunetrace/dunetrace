"""Agents endpoints — list agents and their signal summaries."""

from __future__ import annotations
from typing import Optional
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Query
from api_svc.auth import require_org
from api_svc.config import settings
from api_svc.db.queries import (
    list_agents,
    agent_signal_sparklines,
    agent_failure_type_counts,
    get_agent_health_score,
    list_agent_fixes,
    agent_token_stats,
)
from api_svc.schemas import (
    AgentListResponse,
    AgentSummary,
    AgentHealthScore,
    AgentTokenStats,
    Page,
    FixListResponse,
    FixRecord,
)

router = APIRouter(prefix="/v1/agents", tags=["Agents"])


@router.get("", response_model=AgentListResponse, summary="List agents")
async def get_agents(
    offset: int = Query(0, ge=0),
    limit: int = Query(settings.PAGE_SIZE_DEFAULT, ge=1, le=settings.PAGE_SIZE_MAX),
    org_id: str = Depends(require_org),
) -> AgentListResponse:
    rows, total = await list_agents(org_id, offset, limit)
    sparklines, ft_counts = await asyncio.gather(
        agent_signal_sparklines(org_id),
        agent_failure_type_counts(org_id),
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
        page=Page(total=total, offset=offset, limit=limit, has_more=(offset + limit) < total),
    )


@router.get(
    "/{agent_id}/fixes",
    response_model=FixListResponse,
    summary="Fixes applied to signals for an agent, with recurrence verdicts",
)
async def get_agent_fixes(
    agent_id: str,
    org_id: str = Depends(require_org),
) -> FixListResponse:
    """
    Returns all fixes applied via the dashboard for this agent — both Langfuse prompt
    applications and clipboard copies — with recurrence status since each fix was applied.

    `verdict` is one of:
    - **verified** — ≥10 runs after fix with zero recurrences
    - **likely_fixed** — ≥5 runs after fix with zero recurrences
    - **still_occurring** — the failure type reappeared after the fix
    - **insufficient_data** — fewer than 5 runs recorded since the fix
    """
    fixes = await list_agent_fixes(org_id, agent_id)
    return FixListResponse(fixes=[FixRecord(**f) for f in fixes])


@router.get(
    "/{agent_id}/token-stats",
    response_model=AgentTokenStats,
    summary="Per-window token usage and waste for an agent (1d / 7d / 30d)",
)
async def get_token_stats(
    agent_id: str,
    org_id: str = Depends(require_org),
) -> AgentTokenStats:
    """
    Token usage broken down by 1-day, 7-day, and 30-day windows.

    Each window returns:
    - **total_tokens** / **prompt_tokens** / **completion_tokens** — tokens consumed
    - **wasted_tokens** — tokens from runs that had at least one live failure signal
    - **total_cost_usd** / **wasted_cost_usd** — estimated API cost
    - **wasted_pct** — fraction of cost on failed runs (0–1)
    - **run_count** / **wasted_run_count** — run counts

    `waste_by_failure_type` (30-day) lists which failure types caused the most wasted spend,
    sorted by wasted_cost_usd descending.
    """
    data = await agent_token_stats(org_id, agent_id)
    return AgentTokenStats(agent_id=agent_id, **data)


@router.get(
    "/{agent_id}/health-score",
    response_model=AgentHealthScore,
    summary="0–100 health score for an agent",
)
async def get_health_score(
    agent_id: str,
    org_id: str = Depends(require_org),
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
    result = await get_agent_health_score(org_id, agent_id)
    return AgentHealthScore(agent_id=agent_id, **result)
