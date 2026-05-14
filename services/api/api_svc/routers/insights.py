"""
Cross-run analytics for a single agent. One endpoint that returns input patterns,
signal trends, version comparisons, time-to-first-tool, hourly signal distribution,
estimated API cost, deploy regressions, and user impact.
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
    agent_failure_rates,
    agent_systemic_patterns,
    agent_deploy_events,
    agent_cost_stats,
    deploy_regression_check,
    agent_user_impact,
)
from api_svc.schemas import (
    AgentInsights,
    InputHashPattern,
    SignalTrendPoint,
    VersionStat,
    TimeToFirstTool,
    HourlyPatternPoint,
    FailureRatePoint,
    SystemicPattern,
    DeployEvent,
    CostStat,
    DeployRegression,
    UserImpact,
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
    (
        patterns, trends, versions, ttt, hourly,
        rates, systemic, deploys,
        cost, regressions, user_impact,
    ) = await asyncio.gather(
        agent_input_hash_patterns(agent_id),
        agent_signal_recurrence(agent_id),
        agent_version_stats(agent_id),
        agent_time_to_first_tool(agent_id),
        agent_hourly_pattern(agent_id),
        agent_failure_rates(agent_id),
        agent_systemic_patterns(agent_id),
        agent_deploy_events(agent_id),
        agent_cost_stats(agent_id),
        deploy_regression_check(agent_id),
        agent_user_impact(agent_id),
    )
    return AgentInsights(
        input_patterns=[InputHashPattern(**r) for r in patterns],
        signal_trends=[SignalTrendPoint(**r) for r in trends],
        versions=[VersionStat(**r) for r in versions],
        time_to_tool=TimeToFirstTool(**ttt),
        hourly_pattern=[HourlyPatternPoint(**r) for r in hourly],
        failure_rates=[FailureRatePoint(**r) for r in rates],
        systemic_patterns=[SystemicPattern(**r) for r in systemic],
        deploy_events=[DeployEvent(**r) for r in deploys],
        cost_stats=CostStat(**cost) if cost.get("total_cost_usd", 0) > 0 else None,
        deploy_regressions=[DeployRegression(**r) for r in regressions],
        user_impact=[UserImpact(**r) for r in user_impact],
    )
