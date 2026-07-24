"""
GET /v1/agents/{agent_id}/failure-patterns/{failure_type}

Cross-run deep-dive for a single failure type — answers "why is this
happening across all my runs?" with evidence aggregates, step distributions,
co-occurring failures, and a 14-day trend.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from api_svc.auth import require_org
from api_svc.db.queries import agent_failure_pattern
from api_svc.schemas import (
    FailurePatternAnalysis,
    FailureOverview,
    StepDistribution,
    DailyTrendPoint,
    CoOccurring,
    TopRun,
)

router = APIRouter(tags=["Insights"])

# Derived from the FailureType enum so it can't go stale as detectors are added.
# A hardcoded list here silently 422'd real, live failure types (e.g.
# TOOL_ARGUMENT_FABRICATION, the top pattern on the overview) when they were
# added to the enum but not to this set. Semantic/custom-detector types
# (TASK_COMPLETION, CUSTOM_*) are additionally allowed so any pattern the
# dashboard links to opens rather than erroring.
from dunetrace.models import FailureType  # noqa: E402

_VALID_FAILURE_TYPES = {t.value for t in FailureType} | {"TASK_COMPLETION"}


@router.get(
    "/v1/agents/{agent_id}/failure-patterns/{failure_type}",
    response_model=FailurePatternAnalysis,
    summary="Cross-run pattern analysis for a specific failure type",
    description=(
        "Returns why a failure type keeps happening across runs: "
        "evidence aggregates (which tools loop, token growth rates, slow step durations, etc.), "
        "step count distribution (where in the run it fires), "
        "co-occurring failures, 14-day trend, and highest-confidence example runs."
    ),
)
async def get_failure_pattern(
    agent_id: str,
    failure_type: str,
    org_id: str = Depends(require_org),
) -> FailurePatternAnalysis:
    # Custom-detector signals are stored under their own CUSTOM_* name; allow them
    # through so a custom pattern the dashboard links to opens too.
    if failure_type not in _VALID_FAILURE_TYPES and not failure_type.startswith("CUSTOM_"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown failure_type '{failure_type}'. Valid: {sorted(_VALID_FAILURE_TYPES)}",
        )

    data = await agent_failure_pattern(org_id, agent_id, failure_type)
    if not data:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="DB unavailable"
        )

    return FailurePatternAnalysis(
        failure_type=data["failure_type"],
        overview=FailureOverview(**data["overview"]),
        step_distribution=StepDistribution(**data["step_distribution"]),
        evidence_aggregates=data["evidence_aggregates"],
        daily_trend=[DailyTrendPoint(**p) for p in data["daily_trend"]],
        co_occurring=[CoOccurring(**c) for c in data["co_occurring"]],
        top_runs=[TopRun(**r) for r in data["top_runs"]],
    )
