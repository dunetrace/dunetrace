"""
Formats a pending approval (Capability 2, Phase 2.3) for delivery to a human.

Slack: a Block Kit message with interactive Approve / Deny buttons. Each button
carries a JSON `value` blob with the approval_id + org_id; the click comes back
to api_svc/routers/slack.py, which verifies the Slack signature and records the
decision (POST-equivalent to /v1/approvals/{id}/decision). action_ids are
"approve_request" / "deny_request" — distinct from the signal-alert buttons so
one callback handler can tell them apart.

Webhook: a signed JSON payload for orgs that want to build their own approval
UI. They render it however they like and call the decision endpoint back.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict

from alerts_svc.formatters.slack import _plain


def _approval_summary(approval: Dict[str, Any]) -> str:
    tool = approval.get("tool_name", "?")
    agent = approval.get("agent_id", "?")
    args = approval.get("tool_args")
    line = f"*Approval needed* — agent `{agent}` wants to call `{tool}`"
    if args:
        # Keep the Slack block readable; args can be large.
        shown = args if len(args) <= 300 else args[:297] + "..."
        line += f"\n```{shown}```"
    return line


def format_slack_approval(approval: Dict[str, Any]) -> dict:
    btn_val = json.dumps(
        {
            "approval_id": approval["id"],
            "org_id": approval.get("org_id", ""),
            "tool_name": approval.get("tool_name", ""),
        }
    )
    blocks = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": _approval_summary(approval)},
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Approve", "emoji": True},
                    "action_id": "approve_request",
                    "value": btn_val,
                    "style": "primary",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Deny", "emoji": True},
                    "action_id": "deny_request",
                    "value": btn_val,
                    "style": "danger",
                },
            ],
        },
    ]
    # Plain text — this is the desktop/mobile notification preview, where
    # mrkdwn is not rendered, so an agent id or tool name containing a
    # backtick/asterisk would leak the marker into the notification. Unlike
    # the signal alert, this payload DOES carry top-level `blocks`, so Slack
    # treats `text` as a pure fallback and it is not shown in the channel.
    return {
        "text": (
            f"Approval needed: agent {_plain(approval.get('agent_id', '?'))} "
            f"wants to call {_plain(approval.get('tool_name', '?'))}"
        ),
        "blocks": blocks,
    }


def format_webhook_approval(approval: Dict[str, Any]) -> dict:
    return {
        "schema_version": "1.0",
        "event": "approval_request",
        "sent_at": time.time(),
        "approval_id": approval["id"],
        "org_id": approval.get("org_id", ""),
        "run_id": approval.get("run_id", ""),
        "agent_id": approval.get("agent_id", ""),
        "tool_name": approval.get("tool_name", ""),
        "tool_args": approval.get("tool_args"),
    }
