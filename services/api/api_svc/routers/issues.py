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
                                  matching signal) — and, for a
                                  `dunetrace_native` fix, the ready-to-submit
                                  policy in `suggested_policy` alongside the
                                  free-text `suggested_fix` + code_references
                                  (always empty until Phase 4.3's source
                                  mapping exists — see BACKLOG.md)
  POST /v1/issues/{issue_id}/resolve — manual resolve with resolution_notes,
                                  orthogonal to the auto-resolve-after-N-
                                  clean-runs mechanism above (unchanged)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

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

_POLICY_OPERATORS = {"gte": ">=", "gt": ">", "lte": "<=", "lt": "<", "eq": "==", "ne": "!="}


def _describe_policy(policy: Any) -> Optional[str]:
    """Render a `dunetrace_native` suggested policy as one line of free text.

    explain_signal returns two different shapes. `customer_code` carries
    `fix_content` — already a string. `dunetrace_native` carries no
    `fix_content` at all; the fix *is* a runtime policy, and it arrives as a
    PolicyCreate-shaped dict. Dropping that dict straight into
    IssueDetail.suggested_fix (typed `Optional[str]`) made this route a
    guaranteed 500 for every TOOL_LOOP, RETRY_STORM, CASCADING_TOOL_FAILURE
    and STEP_COUNT_INFLATION issue the moment an LLM key was configured —
    and the construction sits outside the try/except, so the ValidationError
    escaped unhandled.

    The policy body itself still reaches the caller intact, in
    IssueDetail.suggested_policy. This is its prose sibling, so a client that
    only renders suggested_fix (the MCP server's `get_issue` among them) still
    learns that the fix is a guardrail Dunetrace can enforce itself, and what
    it would do.

    Every read is defensive: a shape this doesn't recognise degrades to the
    generic sentence rather than raising.
    """
    if not isinstance(policy, dict):
        return None

    action = policy.get("action")
    action_type = action.get("type") if isinstance(action, dict) else None

    condition = policy.get("condition")
    trigger = operator = value = None
    if isinstance(condition, dict):
        trigger = condition.get("trigger")
        operator = condition.get("operator")
        value = condition.get("value")

    parts = [
        "Dunetrace-native fix: a runtime policy Dunetrace enforces itself — "
        "no prompt or code change needed."
    ]
    if trigger is not None and value is not None:
        op = _POLICY_OPERATORS.get(str(operator), str(operator))
        verb = {"stop": "Stop the run"}.get(str(action_type), f"Apply `{action_type}`")
        parts.append(f"{verb} when {trigger} {op} {value}.")
    elif action_type is not None:
        parts.append(f"Action: {action_type}.")

    parts.append(
        "The exact request body is in `suggested_policy` — POST it to "
        "/v1/policies unchanged to apply it."
    )
    return " ".join(parts)


def _as_text(value: Any) -> Optional[str]:
    """Coerce an LLM-path field to the `Optional[str]` its model field declares.

    explain_signal's shape is a contract, not a guarantee — this route is the
    one place where breaking it turns into an unhandled 500 rather than a
    visibly wrong string, so it never trusts the type.
    """
    if value is None or isinstance(value, str):
        return value
    return str(value)


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
    suggested_policy = None
    fix_category = None
    if llm_provider.llm_configured():
        signal_id = await get_most_recent_signal_id(
            org_id, issue["agent_id"], issue["failure_type"]
        )
        if signal_id is not None:
            try:
                from api_svc.routers.signals import explain_signal

                explain_result = await explain_signal(signal_id, org_id=org_id)
                root_cause = _as_text(explain_result.get("root_cause"))
                fix_category = _as_text(explain_result.get("fix_category"))

                policy = explain_result.get("suggested_policy")
                if isinstance(policy, dict):
                    # dunetrace_native: the fix is a Policy, and there is no
                    # fix_content key on this branch at all.
                    suggested_policy = policy
                    suggested_fix = _describe_policy(policy)
                else:
                    suggested_fix = _as_text(explain_result.get("fix_content"))
            except Exception as exc:
                # get_issue's core value (metadata + affected runs) shouldn't
                # be blocked by a root-cause analysis failure — log and
                # continue with root_cause/suggested_fix left as None.
                logger.warning("get_issue: explain_signal failed for issue=%d: %s", issue_id, exc)
                root_cause = None
                suggested_fix = None
                suggested_policy = None
                fix_category = None

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
        suggested_policy=suggested_policy,
        fix_category=fix_category,
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
