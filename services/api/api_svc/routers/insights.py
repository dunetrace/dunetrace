"""services/api/api_svc/routers/insights.py

GET /v1/agents/{agent_id}/insights

Returns five cross-run analytics metrics in a single call:
  1. input_patterns  — which input hashes consistently trigger which failures
  2. signal_trends   — failure type signal counts by day × version (last 30 d)
  3. versions        — per-version signal rate comparison
  4. time_to_tool    — steps before first tool call (P25/P50/P75 + 14-day trend)
  5. hourly_pattern  — signal rate by UTC hour of day (last 30 d)
"""
from __future__ import annotations

import asyncio
from fastapi import APIRouter, Depends

from api_svc.auth import require_customer
from api_svc.db.queries import (
    agent_input_hash_patterns,
    agent_signal_recurrence,
    agent_version_stats,
    agent_time_to_first_tool,
    agent_hourly_pattern,
)
from api_svc.schemas import (
    AgentInsights,
    InputHashPattern,
    SignalTrendPoint,
    VersionStat,
    TimeToFirstTool,
    HourlyPatternPoint,
)

router = APIRouter(tags=["Insights"])


@router.get(
    "/v1/agents/{agent_id}/insights",
    response_model=AgentInsights,
    summary="Cross-run analytics for an agent",
)
async def get_insights(
    agent_id:  str,
    _customer: str = Depends(require_customer),
) -> AgentInsights:
    (patterns, trends, versions, ttt, hourly) = await asyncio.gather(
        agent_input_hash_patterns(agent_id),
        agent_signal_recurrence(agent_id),
        agent_version_stats(agent_id),
        agent_time_to_first_tool(agent_id),
        agent_hourly_pattern(agent_id),
    )
    return AgentInsights(
        input_patterns=[InputHashPattern(**r) for r in patterns],
        signal_trends=[SignalTrendPoint(**r) for r in trends],
        versions=[VersionStat(**r) for r in versions],
        time_to_tool=TimeToFirstTool(**ttt),
        hourly_pattern=[HourlyPatternPoint(**r) for r in hourly],
    )
