"""
Minimal Linear GraphQL client for the team/project picker (Phase 4.1's
POST /v1/orgs/integrations/linear/preview-teams). Deliberately independent
of alerts_svc/linear_client.py (which does issue creation) — same
"read/write independently per service" convention this codebase already
uses for detector_svc/semantic_svc both reading `events` directly rather
than sharing a client module.

Verified against Linear's own docs (linear.app/developers/graphql,
linear.app/developers/oauth-2-0-authentication) during Phase 4.1 discovery:
- Endpoint: https://api.linear.app/graphql
- Personal/team API key auth: `Authorization: <API_KEY>` — no "Bearer"
  prefix (a different scheme from OAuth tokens, confirmed on that page).
- `query { teams { nodes { id name } } }` — confirmed verbatim.

NOT verified against a real Linear workspace (no test credentials existed
during Phase 4.1's initial build — see BACKLOG.md's exit criteria). Every
assumption below is tagged with an `_ASSUMED` marker — grep for `_ASSUMED`
to find every place still needing live confirmation once real credentials
exist.
"""

from __future__ import annotations

import httpx

_LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"


class LinearApiError(Exception):
    pass


async def _graphql(api_key: str, query: str, variables: dict | None = None) -> dict:
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            _LINEAR_GRAPHQL_URL,
            json={"query": query, "variables": variables or {}},
            headers={"Authorization": api_key, "Content-Type": "application/json"},
        )
    resp.raise_for_status()
    body = resp.json()
    if body.get("errors"):
        raise LinearApiError(str(body["errors"]))
    return body["data"]


async def fetch_teams(api_key: str) -> list[dict]:
    """Returns [{"id": ..., "name": ...}, ...]. Confirmed query shape."""
    data = await _graphql(api_key, "query Teams { teams { nodes { id name } } }")
    return data["teams"]["nodes"]


async def fetch_projects(api_key: str, team_id: str) -> list[dict]:
    """Returns [{"id": ..., "name": ...}, ...] for the given team.

    LINEAR_PROJECTS_QUERY_ASSUMED — no official Linear doc page shows a
    `projects` root-query example (unlike the confirmed `teams` query
    above). This shape is inferred by analogy to `teams`'s confirmed
    Relay-connection pattern, cross-checked against a third-party (Paragon)
    integration doc that uses a similar `projects(first: N) { nodes { ... } }`
    shape — but not confirmed against Linear's own docs or live API. Must
    be verified against a real workspace before this feature ships for
    real customers (see BACKLOG.md's Phase 4.1 exit criteria).
    """
    data = await _graphql(
        api_key,
        """
        query TeamProjects($teamId: String!) {
          team(id: $teamId) {
            projects {
              nodes { id name }
            }
          }
        }
        """,
        {"teamId": team_id},
    )
    return data["team"]["projects"]["nodes"]


async def fetch_workflow_state_type(api_key: str, state_id: str) -> str | None:
    """Returns the WorkflowState's `type` field (expected values per
    LINEAR_WORKFLOW_STATE_ENUM_ASSUMED in routers/linear_webhook.py:
    triage/backlog/unstarted/started/completed/canceled) for a given state
    id. Used by the webhook receiver to authoritatively confirm an issue
    moved to a resolved-type state, rather than trusting any state
    information the webhook payload itself might carry.

    LINEAR_WORKFLOW_STATE_QUERY_ASSUMED — the `workflowState(id: ID!)`
    root-query field name/shape is inferred from Linear's general
    single-entity query pattern (e.g. the confirmed `team(id: ...)` query)
    but not confirmed verbatim against Linear's own docs or a live
    workspace.
    """
    data = await _graphql(
        api_key,
        "query WorkflowState($id: String!) { workflowState(id: $id) { type } }",
        {"id": state_id},
    )
    state = data.get("workflowState")
    return state["type"] if state else None
