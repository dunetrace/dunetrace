"""
GET /v1/agents/{agent_id}/issues — persistent issue tracking per (agent_id, failure_type).

An issue is opened the first time a failure type fires for an agent, updated on each
subsequent hit, and resolved after CLEAN_RUNS_THRESHOLD (5) consecutive runs with no
signal of that type. If the failure recurs after resolution, the issue is reopened.

Query params:
  status: open | resolved | reopened  (default: all)
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query

from api_svc.auth import require_customer
from api_svc.db.queries import list_issues
from api_svc.schemas import IssueListResponse

router = APIRouter(tags=["Issues"])


@router.get(
    "/v1/agents/{agent_id}/issues",
    response_model=IssueListResponse,
    summary="Persistent issue list for an agent",
)
async def get_issues(
    agent_id: str,
    status: Optional[str] = Query(
        default=None,
        description="Filter by status: open | resolved | reopened",
    ),
    _customer: str = Depends(require_customer),
) -> IssueListResponse:
    issues = await list_issues(agent_id, status=status)
    return IssueListResponse(issues=issues, total=len(issues))
