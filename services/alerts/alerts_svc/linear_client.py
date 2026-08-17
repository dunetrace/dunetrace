"""
Minimal Linear GraphQL client for issue creation (Phase 4.1's alert
delivery path). Deliberately independent of api_svc/linear_client.py (which
only does the team/project picker) — same "read/write independently per
service" convention this codebase already uses elsewhere (detector_svc/
semantic_svc both reading `events` directly rather than sharing a client).

Verified against Linear's own docs (linear.app/developers/graphql) during
Phase 4.1 discovery:
- Endpoint: https://api.linear.app/graphql
- API key auth: `Authorization: <API_KEY>` — no "Bearer" prefix.
- `issueCreate(input: { title, description, teamId })` — confirmed verbatim,
  including that `stateId`/`projectId` are omittable (a missing `stateId`
  lands the issue in the team's default Backlog/Triage state).

NOT verified: whether `projectId` is accepted by the real, current
IssueCreateInput type (see LINEAR_ISSUE_INPUT_FIELDS_ASSUMED below) — grep
for `_ASSUMED` to find every place still needing live confirmation. See
BACKLOG.md's Phase 4.1 exit criteria.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger("dunetrace.alerts.linear_client")

_LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"

_ISSUE_CREATE_MUTATION = """
mutation IssueCreate($input: IssueCreateInput!) {
  issueCreate(input: $input) {
    success
    issue { id }
  }
}
"""


def create_issue(
    api_key: str,
    team_id: str,
    title: str,
    description: str,
    project_id: str | None = None,
) -> str | None:
    """Returns the created Linear issue's id, or None on failure (logged,
    never raised — a Linear delivery failure must not crash the alerts
    worker or block Slack/webhook delivery for the same signal).

    Synchronous, deliberately — deliver() (worker.py) is a sync function
    called via asyncio.to_thread, matching sender.py's send_slack/
    send_webhook; this stays consistent rather than introducing an async
    HTTP client into a function designed to block in its own thread."""
    variables: dict = {"input": {"title": title, "description": description, "teamId": team_id}}
    if project_id:
        # LINEAR_ISSUE_INPUT_FIELDS_ASSUMED — projectId's presence/name on
        # the current IssueCreateInput type is inferred (from a stale
        # third-party SDK schema dump), not confirmed against Linear's live
        # API or current docs. If this field name is wrong, Linear's
        # GraphQL response carries a top-level "errors" array (logged below
        # in full) rather than silently ignoring the field — a real
        # customer's first live use will surface this immediately if so.
        variables["input"]["projectId"] = project_id

    try:
        resp = httpx.post(
            _LINEAR_GRAPHQL_URL,
            json={"query": _ISSUE_CREATE_MUTATION, "variables": variables},
            headers={"Authorization": api_key, "Content-Type": "application/json"},
            timeout=15.0,
        )
        resp.raise_for_status()
        body = resp.json()
    except Exception as exc:
        logger.error("Linear issueCreate request failed: %s", exc)
        return None

    if body.get("errors"):
        logger.error("Linear issueCreate returned errors: %s", body["errors"])
        return None

    result = body.get("data", {}).get("issueCreate", {})
    if not result.get("success"):
        logger.error("Linear issueCreate not successful: %s", body)
        return None

    return result.get("issue", {}).get("id")
