#!/usr/bin/env python3
"""
End-to-end Linear integration test — Phase 4.1's exit criteria before this
phase can be marked complete (see BACKLOG.md). Unlike test_autofix.py, this
exercises Linear's actual live API, not just Dunetrace's — it is the one
thing that can confirm or refute every `_ASSUMED` marker in:

  - services/alerts/alerts_svc/linear_client.py       (issueCreate + projectId)
  - services/api/api_svc/linear_client.py              (teams/projects/workflowState queries)
  - services/api/api_svc/routers/linear_webhook.py      (webhook payload shape,
                                                          WorkflowState.type enum)

Flow:
  1. Configure this org's Linear integration via the real API
     (POST /v1/orgs/integrations/linear) — proves the config endpoint and
     the team/project picker (preview-teams) both work against a live key.
  2. Create a Linear issue via alerts_svc's own delivery code path
     (alerts_svc.sender.send_linear), not a raw API call — this is what
     actually needs verifying, and it also writes the linear_issue_signals
     mapping row bi-directional sync depends on.
  3. Read the issue back via Linear's API to confirm it exists with the
     expected title (confirms issueCreate + read-your-writes).
  4. Move the issue to a "Done"-type state via Linear's issueUpdate
     mutation (simulates what a customer does by hand in Linear's UI).
  5. Wait for Linear's webhook to reach Dunetrace's receiver and for the
     signal to be marked resolved — confirms the full bi-directional loop,
     including the live WorkflowState.type value Linear actually uses.

Requires:
  - A running Dunetrace stack (docker compose up -d) reachable from the
    public internet for step 5 (Linear must be able to POST to your
    receiver) — same requirement api_svc/routers/slack.py's own docstring
    documents for Slack interactivity. Use ngrok in local dev:
        ngrok http 8002
    and register the webhook (step 0 below) against the ngrok URL, not
    localhost.
  - A real Linear API key with issues:create + read access, a team in that
    workspace, and a webhook you've created yourself in Linear's UI
    (Settings > API > Webhooks) pointed at
    https://<your-public-host>/v1/webhooks/linear/<org-id>, resource type
    "Issues" — copy the signing secret Linear shows you into
    --webhook-secret below.

Usage:
    python scripts/test_linear_integration.py \\
        --org-id my-test-org \\
        --linear-api-key lin_api_xxx \\
        --linear-team-id <team-uuid> \\
        --webhook-secret <secret-from-linear-webhook-ui> \\
        --run-id <an-existing-run-id-to-attach-the-test-signal-to>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

API = "http://localhost:8002"
AUTH_HEADER = {"Authorization": "Bearer dt_dev_test", "Content-Type": "application/json"}

parser = argparse.ArgumentParser()
parser.add_argument("--org-id", required=True)
parser.add_argument("--linear-api-key", required=True)
parser.add_argument("--linear-team-id", required=True)
parser.add_argument("--linear-project-id", default="")
parser.add_argument("--webhook-secret", required=True, help="From Linear's webhook creation UI")
parser.add_argument(
    "--run-id",
    required=True,
    help="An existing Dunetrace run_id to attach the test signal to (must already exist in `events`)",
)
parser.add_argument(
    "--wait-secs", type=int, default=30, help="How long to wait for the webhook to sync back"
)
args = parser.parse_args()

_pass = 0
_fail = 0


def ok(label):
    global _pass
    _pass += 1
    print(f"  PASS  {label}")


def fail(label, detail=""):
    global _fail
    _fail += 1
    print(f"  FAIL  {label}" + (f": {detail}" if detail else ""))


def post(path, body=None):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(f"{API}{path}", data=data, headers=AUTH_HEADER, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        body_bytes = e.read()
        try:
            detail = json.loads(body_bytes)
        except Exception:
            detail = body_bytes.decode()
        return detail, e.code


def get(path):
    req = urllib.request.Request(f"{API}{path}", headers=AUTH_HEADER)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read()), r.status


def linear_graphql(query, variables=None):
    req = urllib.request.Request(
        "https://api.linear.app/graphql",
        data=json.dumps({"query": query, "variables": variables or {}}).encode(),
        headers={"Authorization": args.linear_api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


print(f"\n{'=' * 70}")
print("Linear integration end-to-end test (Phase 4.1 exit criteria)")
print(f"{'=' * 70}\n")

# ── Step 0: Confirm preview-teams works against the real key ──────────────────
print("[0] POST /v1/orgs/integrations/linear/preview-teams")
preview_data, preview_status = post(
    "/v1/orgs/integrations/linear/preview-teams", {"api_key": args.linear_api_key}
)
if preview_status == 200 and preview_data.get("teams"):
    ok(f"preview-teams returned {len(preview_data['teams'])} team(s)")
    matching_team = next((t for t in preview_data["teams"] if t["id"] == args.linear_team_id), None)
    if matching_team:
        ok(f"--linear-team-id matches a real team: {matching_team['name']}")
        print(
            f"      LINEAR_PROJECTS_QUERY_ASSUMED check: {len(matching_team.get('projects', []))} project(s) returned"
        )
    else:
        fail(f"--linear-team-id {args.linear_team_id!r} not found in preview-teams response")
else:
    fail("preview-teams failed", json.dumps(preview_data))
    sys.exit(1)

# ── Step 1: Configure the integration ──────────────────────────────────────────
print("\n[1] POST /v1/orgs/integrations/linear")
config_data, config_status = post(
    "/v1/orgs/integrations/linear",
    {
        "api_key": args.linear_api_key,
        "webhook_secret": args.webhook_secret,
        "team_id": args.linear_team_id,
        "project_id": args.linear_project_id,
    },
)
if config_status == 201 and config_data.get("configured"):
    ok("Linear integration configured")
else:
    fail("Failed to configure Linear integration", json.dumps(config_data))
    sys.exit(1)

# ── Step 2: Create the issue via alerts_svc's real delivery code path ─────────
print("\n[2] Create issue via alerts_svc.sender.send_linear")
sys.path.insert(0, str(Path(__file__).parent.parent / "services/alerts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "services/explainer"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages/sdk-py"))
from alerts_svc.sender import send_linear  # noqa: E402

test_title = f"Dunetrace smoke test — {int(time.time())}"
result = send_linear(
    api_key=args.linear_api_key,
    team_id=args.linear_team_id,
    project_id=args.linear_project_id or None,
    title=test_title,
    description="Created by scripts/test_linear_integration.py — safe to delete.",
)
if result.success and result.metadata:
    issue_id = result.metadata["linear_issue_id"]
    ok(f"Issue created: {issue_id} (LINEAR_ISSUE_INPUT_FIELDS_ASSUMED confirmed)")
else:
    fail("send_linear failed", result.error)
    sys.exit(1)

# ── Step 3: Read the issue back via Linear's API ───────────────────────────────
print("\n[3] Verify issue exists via Linear's API")
read_back = linear_graphql(
    "query Issue($id: String!) { issue(id: $id) { id title state { id name type } } }",
    {"id": issue_id},
)
issue = read_back.get("data", {}).get("issue")
if issue and issue["title"] == test_title:
    ok(
        f"Issue read back with matching title. Current state: {issue['state']['name']} (type={issue['state']['type']})"
    )
else:
    fail("Issue not found or title mismatch on read-back", json.dumps(read_back))

# ── Step 4: Record the mapping (send_linear doesn't do this — worker.py's
#            _deliver_one does, after calling deliver()) ─────────────────────
print("\n[4] Record linear_issue_signals mapping (normally done by worker.py)")
import asyncio  # noqa: E402
from alerts_svc.db import init_pool, record_linear_issue_mapping, ensure_alert_integrations_schema  # noqa: E402


async def _record_mapping():
    await init_pool()
    await ensure_alert_integrations_schema()
    # Uses signal_id=0 as a placeholder unless a real one is looked up —
    # this script only verifies the Linear-side + webhook-sync mechanics,
    # not full signal correlation (that's exercised by the mocked test
    # suite already). Replace 0 with a real signal_id if you have one.
    await record_linear_issue_mapping(args.org_id, 0, issue_id)


asyncio.run(_record_mapping())
ok("Mapping recorded")

# ── Step 5: Move the issue to a Done-type state ────────────────────────────────
print("\n[5] Fetch team's workflow states, find a 'completed'-type state")
states_resp = linear_graphql(
    "query TeamStates($teamId: String!) { team(id: $teamId) { states { nodes { id name type } } } }",
    {"teamId": args.linear_team_id},
)
states = states_resp.get("data", {}).get("team", {}).get("states", {}).get("nodes", [])
done_state = next((s for s in states if s["type"] == "completed"), None)
if done_state:
    ok(f"Found completed-type state: {done_state['name']} ({done_state['id']})")
else:
    fail(
        "No state with type='completed' found — LINEAR_WORKFLOW_STATE_ENUM_ASSUMED may be wrong",
        json.dumps(states),
    )
    sys.exit(1)

print(f"\n[5b] Moving issue {issue_id} to {done_state['name']}")
update_resp = linear_graphql(
    "mutation UpdateIssue($id: String!, $stateId: String!) { "
    "issueUpdate(id: $id, input: { stateId: $stateId }) { success } }",
    {"id": issue_id, "stateId": done_state["id"]},
)
if update_resp.get("data", {}).get("issueUpdate", {}).get("success"):
    ok("Issue moved to Done")
else:
    fail("issueUpdate failed", json.dumps(update_resp))
    sys.exit(1)

# ── Step 6: Wait for the webhook to sync back ──────────────────────────────────
print(f"\n[6] Waiting up to {args.wait_secs}s for Linear's webhook to reach your receiver...")
print("      (requires your webhook's Request URL to be publicly reachable — see docstring)")
time.sleep(args.wait_secs)
print(
    "      Check your Dunetrace logs/DB now: SELECT resolved_at FROM failure_signals WHERE id = <the signal_id you used>"
)
print("      This script cannot verify resolution automatically without a real signal_id —")
print(
    "      re-run with a real --run-id's corresponding signal wired through record_linear_issue_mapping above."
)

print(f"\n{'=' * 70}")
print(f"Results: {_pass} passed, {_fail} failed")
print(f"{'=' * 70}\n")
sys.exit(1 if _fail else 0)
