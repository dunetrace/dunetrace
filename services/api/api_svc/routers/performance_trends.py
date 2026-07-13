"""
Phase 4.4 — per-agent performance trends over time: daily structural/semantic
signal rate, cost, and latency; failure-mode rate deltas; and a self-baseline
comparison (never cross-org — see api_svc/performance_trends.py's docstring
for why "industry median" isn't built here).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api_svc.auth import require_org
from api_svc.db.queries import agent_performance_trends, agent_state_analytics
from api_svc.performance_trends import VALID_WINDOWS
from api_svc.state_analytics import VALID_WINDOWS as STATE_WINDOWS
from api_svc.schemas import AgentPerformanceTrends, BaselineComparison, FailureModeDelta, TrendPoint

router = APIRouter(tags=["Performance Trends"])


@router.get(
    "/v1/agents/{agent_id}/performance-trends",
    response_model=AgentPerformanceTrends,
    summary="Per-agent performance trends: signal rate, cost, and latency over time",
)
async def get_performance_trends(
    agent_id: str,
    window: int = 30,
    org_id: str = Depends(require_org),
) -> AgentPerformanceTrends:
    if window not in VALID_WINDOWS:
        raise HTTPException(
            status_code=422,
            detail=f"window must be one of {VALID_WINDOWS}, got {window}.",
        )

    result = await agent_performance_trends(org_id, agent_id, window)
    return AgentPerformanceTrends(
        agent_id=agent_id,
        window_days=window,
        points=[TrendPoint(**p) for p in result["points"]],
        failure_mode_deltas=[FailureModeDelta(**d) for d in result["failure_mode_deltas"]],
        baseline_comparisons=[BaselineComparison(**b) for b in result["baseline_comparisons"]],
    )


@router.get(
    "/v1/agents/{agent_id}/state-analytics",
    summary="Cross-run state analytics: time-per-state averages, trends, and outliers",
)
async def get_state_analytics(
    agent_id: str,
    window: int = 30,
    org_id: str = Depends(require_org),
) -> dict:
    # org_id from require_org(), not the URL. See scripts/check_endpoint_conventions.py.
    if window not in STATE_WINDOWS:
        raise HTTPException(
            status_code=422,
            detail=f"window must be one of {STATE_WINDOWS}, got {window}.",
        )
    result = await agent_state_analytics(org_id, agent_id, window)
    return {"agent_id": agent_id, **result}
