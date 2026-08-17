"""
GET /v1/agents/{agent_id}/issues — persistent issue tracking per (agent_id, failure_type).

An issue is opened the first time a failure type fires for an agent, updated on each
subsequent hit, and resolved after CLEAN_RUNS_THRESHOLD (5) consecutive runs with no
signal of that type. If the failure recurs after resolution, the issue is reopened.

Query params:
  status: open | resolved | reopened  (default: all)

Phase 4.2 adds three more, for the MCP server's coding-agent-facing tools:
  GET  /v1/issues/search       — cross-agent search (no free-text index; a
                                  plain substring filter over agent_id/
                                  failure_type/resolution_notes)
  GET  /v1/issues/{issue_id}   — single-issue detail: metadata + affected
                                  runs (reuses agent_failure_pattern's
                                  top_runs) + root cause/suggested fix (via
                                  the same native-explain path
                                  POST /v1/signals/{id}/explain already
                                  uses, anchored to the most recent
                                  matching signal) + code_references
                                  (always empty until Phase 4.3's source
                                  mapping exists — see BACKLOG.md)
  POST /v1/issues/{issue_id}/resolve — manual resolve with resolution_notes,
                                  orthogonal to the auto-resolve-after-N-
                                  clean-runs mechanism above (unchanged)
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status as http_status

from api_svc.auth import require_org
from api_svc import llm_provider
from api_svc.config import settings
from api_svc.db.queries import (
    agent_failure_pattern,
    get_issue_by_id,
    get_most_recent_signal_id,
    list_issues,
    resolve_issue_manually,
    search_issues as search_issues_query,
)
from api_svc.schemas import (
    AffectedRun,
    Issue,
    IssueDetail,
    IssueListResponse,
    IssueSearchResponse,
    Page,
    ResolveIssueRequest,
)

logger = logging.getLogger("dunetrace.api.issues")
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
    org_id: str = Depends(require_org),
) -> IssueListResponse:
    issues = await list_issues(org_id, agent_id, status=status)
    return IssueListResponse(issues=issues, total=len(issues))


@router.get(
    "/v1/issues/search",
    response_model=IssueSearchResponse,
    summary="Search issues across all agents",
)
async def search_issues(
    q: str = Query("", description="Plain substring match — not full-text search"),
    status: Optional[str] = Query(None, description="open | resolved | reopened"),
    agent_id: Optional[str] = Query(None),
    failure_type: Optional[str] = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(settings.PAGE_SIZE_DEFAULT, ge=1, le=settings.PAGE_SIZE_MAX),
    org_id: str = Depends(require_org),
) -> IssueSearchResponse:
    rows, total = await search_issues_query(
        org_id, q, status, agent_id, failure_type, offset, limit
    )
    return IssueSearchResponse(
        issues=[Issue(**r) for r in rows],
        page=Page(total=total, offset=offset, limit=limit, has_more=(offset + limit) < total),
    )


@router.get(
    "/v1/issues/{issue_id}",
    response_model=IssueDetail,
    summary="Single issue detail: affected runs, root cause, suggested fix",
)
async def get_issue(issue_id: int, org_id: str = Depends(require_org)) -> IssueDetail:
    issue = await get_issue_by_id(org_id, issue_id)
    if issue is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND, detail=f"Issue {issue_id} not found."
        )

    pattern = await agent_failure_pattern(org_id, issue["agent_id"], issue["failure_type"])
    affected_runs = [
        AffectedRun(
            run_id=r["run_id"],
            detected_at=r.get("detected_at"),
            step_index=r.get("step_index"),
            confidence=r.get("confidence"),
        )
        for r in (pattern.get("top_runs") or [])
    ]

    root_cause = None
    suggested_fix = None
    if llm_provider.llm_configured():
        signal_id = await get_most_recent_signal_id(
            org_id, issue["agent_id"], issue["failure_type"]
        )
        if signal_id is not None:
            try:
                from api_svc.routers.signals import explain_signal

                explain_result = await explain_signal(signal_id, org_id=org_id)
                root_cause = explain_result.get("root_cause")
                suggested_fix = explain_result.get("fix_content") or explain_result.get(
                    "suggested_policy"
                )
            except Exception as exc:
                # get_issue's core value (metadata + affected runs) shouldn't
                # be blocked by a root-cause analysis failure — log and
                # continue with root_cause/suggested_fix left as None.
                logger.warning("get_issue: explain_signal failed for issue=%d: %s", issue_id, exc)

    return IssueDetail(
        id=issue["id"],
        agent_id=issue["agent_id"],
        failure_type=issue["failure_type"],
        status=issue["status"],
        first_seen=issue["first_seen"],
        last_seen=issue["last_seen"],
        resolved_at=issue["resolved_at"],
        affected_runs_count=issue["affected_runs"],
        clean_runs_since=issue["clean_runs_since"],
        resolution_notes=issue["resolution_notes"],
        manually_resolved=issue["manually_resolved"],
        affected_runs=affected_runs,
        root_cause=root_cause,
        suggested_fix=suggested_fix,
        code_references=[],
    )


@router.post(
    "/v1/issues/{issue_id}/resolve",
    summary="Manually resolve an issue with resolution notes",
)
async def resolve_issue(
    issue_id: int,
    body: ResolveIssueRequest,
    org_id: str = Depends(require_org),
) -> dict:
    resolved = await resolve_issue_manually(org_id, issue_id, body.resolution_notes)
    if not resolved:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND, detail=f"Issue {issue_id} not found."
        )
    return {"resolved": True, "issue_id": issue_id}
