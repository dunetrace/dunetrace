"""
GET /v1/patterns

Cross-run signal frequency per agent × detector over the last 7 days.
Powers the Patterns dashboard tab — answers "which detectors are firing
systematically, not just on one run?"
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_svc.auth import require_customer
from api_svc.db.queries import cross_run_patterns
from api_svc.schemas import AgentPatterns, PatternDay, PatternRow, PatternsResponse

router = APIRouter(tags=["Patterns"])


@router.get(
    "/v1/patterns",
    response_model=PatternsResponse,
    summary="Cross-run signal frequency per agent and detector over the last 7 days",
)
async def get_patterns(
    customer: str = Depends(require_customer),
) -> PatternsResponse:
    agents_raw = await cross_run_patterns(customer)
    return PatternsResponse(
        agents=[
            AgentPatterns(
                agent_id=a["agent_id"],
                rows=[
                    PatternRow(
                        failure_type=r["failure_type"],
                        days=[PatternDay(**d) for d in r["days"]],
                        total_occurrences=r["total_occurrences"],
                        total_affected_runs=r["total_affected_runs"],
                        total_runs=r["total_runs"],
                        pct_of_runs=r["pct_of_runs"],
                        trending_up=r["trending_up"],
                    )
                    for r in a["rows"]
                ],
            )
            for a in agents_raw
        ]
    )
