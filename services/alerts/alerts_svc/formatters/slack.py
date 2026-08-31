"""Formats an Explanation as a Slack Block Kit payload."""

from __future__ import annotations
import json
import os
import re

from explainer_svc.models import Explanation

_SEVERITY_COLORS = {
    "CRITICAL": "#FF0000",
    "HIGH": "#FF6B00",
    "MEDIUM": "#FFB800",
    "LOW": "#36A64F",
}
_SEVERITY_EMOJI = {
    "CRITICAL": ":red_circle:",
    "HIGH": ":large_orange_circle:",
    "MEDIUM": ":large_yellow_circle:",
    "LOW": ":white_circle:",
}
_DASHBOARD_BASE = os.getenv("DASHBOARD_URL", "https://app.dunetrace.io")


def _fmt_tok(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _fmt_cost(usd: float) -> str:
    if usd < 0.01:
        return "<$0.01"
    if usd < 1000:
        return f"${usd:.2f}"
    return f"${usd:,.0f}"


def _rate_context_text(explanation: Explanation) -> str:
    """Build a one-line rate context string for the Slack message, or empty string if unavailable."""
    rc = getattr(explanation, "rate_context", {})
    if not rc:
        return ""
    total = rc.get("total_runs", 0)
    affected = rc.get("affected_runs", 0)
    rate = rc.get("rate", 0.0)
    systemic = rc.get("is_systemic", False)
    pct = f"{round(rate * 100)}%"

    if systemic:
        return f":warning: *Systemic pattern* — {affected}/{total} runs affected ({pct} of runs in the last 7 days)"
    elif affected == 1:
        return f":information_source: First occurrence of this pattern in the last 7 days"
    else:
        return f":bar_chart: {affected}/{total} runs affected ({pct}) in the last 7 days"


# Underscores are deliberately NOT stripped: tool names and agent ids
# routinely contain them (`web_search`), and removing them corrupts the
# identifier the reader needs.
_MRKDWN_CHARS = re.compile(r"[`*~]")
_NOTIFICATION_MAX = 220


def _plain(text: str) -> str:
    """Strip mrkdwn markers — the notification fallback is rendered verbatim."""
    return _MRKDWN_CHARS.sub("", text).strip()


def notification_text(explanation: Explanation, suppressed_count: int = 0) -> str:
    """One-line plain-text summary for the desktop/mobile notification.

    Slack builds the notification preview from the top-level `blocks` first
    and falls back to the top-level `text` (`message.text`); mobile
    notifications use `message.text` *exclusively*. These payloads carry
    neither top-level blocks nor — before this — a top-level `text`, so both
    platforms had no source of preview text and the client rendered
    "no preview available".

    Kept plain on purpose: a notification does not render mrkdwn, so
    backticks and asterisks would appear literally.

    Note this string is ALSO visible in the channel, as the message body
    above the coloured card: Slack only treats `text` as a pure fallback
    when the payload has top-level `blocks`, and ours are nested inside an
    `attachments` entry (that nesting is what supplies the severity colour
    bar). Moving the blocks up would silence the duplicate line but would
    also break the mobile preview entirely, since mobile reads only
    `message.text` — so the visible line is the deliberate trade.
    https://docs.slack.dev/messaging/formatting-message-text
    """
    parts = [f"{explanation.severity}: {_plain(explanation.title)}"]
    if explanation.agent_id:
        parts.append(f"agent {_plain(explanation.agent_id)}")
    if explanation.run_id:
        parts.append(f"run {_plain(explanation.run_id)}")
    if explanation.total_tokens and explanation.cost_usd:
        parts.append(f"~{_fmt_cost(explanation.cost_usd)} wasted")
    if getattr(explanation, "rate_context", {}).get("is_systemic"):
        rc = explanation.rate_context
        parts.append(f"systemic: {rc.get('affected_runs', 0)}/{rc.get('total_runs', 0)} runs in 7d")
    if suppressed_count > 0:
        parts.append(f"+{suppressed_count} suppressed")

    text = " · ".join(parts)
    if len(text) > _NOTIFICATION_MAX:
        text = text[: _NOTIFICATION_MAX - 1].rstrip() + "…"
    return text


def _fmt_window(seconds: int) -> str:
    if seconds >= 3600 and seconds % 3600 == 0:
        h = seconds // 3600
        return f"{h}h"
    if seconds >= 60:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def format_slack(
    explanation: Explanation,
    suppressed_count: int = 0,
    dedup_window: int = 3600,
    signal_id: int | None = None,
    org_id: str | None = None,
) -> dict:
    """Block Kit payload for Slack Incoming Webhook."""
    severity = explanation.severity
    color = _SEVERITY_COLORS.get(severity, "#CCCCCC")
    emoji = _SEVERITY_EMOJI.get(severity, ":white_circle:")
    conf_pct = explanation.confidence_pct()
    dashboard_url = f"{_DASHBOARD_BASE}/runs/{explanation.run_id}"
    rate_text = _rate_context_text(explanation)

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji}  {explanation.title}",
                "emoji": True,
            },
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"*Agent:* `{explanation.agent_id}`  "
                        f"*Version:* `{explanation.agent_version}`  "
                        f"*Run:* `{explanation.run_id}`  "
                        f"*Step:* {explanation.step_index}  "
                        f"*Confidence:* {conf_pct}"
                    ),
                }
            ],
        },
        *(
            [
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": (
                                f":moneybag: *Tokens:* {_fmt_tok(explanation.total_tokens)} (wasted)  "
                                f"*Cost:* ~{_fmt_cost(explanation.cost_usd)}"
                            ),
                        }
                    ],
                }
            ]
            if explanation.total_tokens and explanation.cost_usd
            else []
        ),
        *(
            [
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": (
                                f":mute: *{suppressed_count} occurrence{'s' if suppressed_count != 1 else ''} "
                                f"suppressed* since the last alert "
                                f"({_fmt_window(dedup_window)} silence window)"
                            ),
                        }
                    ],
                }
            ]
            if suppressed_count > 0
            else []
        ),
        {"type": "divider"},
    ]

    if rate_text:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": rate_text}})

    blocks += [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*What happened*\n{explanation.what}"},
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Why it matters*\n{explanation.why_it_matters}",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Evidence*\n```{explanation.evidence_summary}```",
            },
        },
    ]

    if explanation.suggested_fixes:
        fix = explanation.suggested_fixes[0]
        fix_text = f"*Suggested fix:* {fix.description}"
        if fix.code and len(fix.code) < 800:
            fix_text += f"\n```{fix.code}```"
        blocks.append({"type": "divider"})
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": fix_text}})

    _btn_val = json.dumps(
        {
            "signal_id": signal_id,
            "agent_id": explanation.agent_id,
            "failure_type": explanation.failure_type,
            "org_id": org_id,
        }
    )

    blocks.append({"type": "divider"})
    blocks.append(
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "View Run", "emoji": True},
                    "url": dashboard_url,
                    "style": "primary",
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Mark resolved",
                        "emoji": True,
                    },
                    "action_id": "mark_resolved",
                    "value": _btn_val,
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Not a problem",
                        "emoji": True,
                    },
                    "action_id": "false_positive",
                    "value": _btn_val,
                    "style": "danger",
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Snooze 24h",
                        "emoji": True,
                    },
                    "action_id": "snooze",
                    "value": _btn_val,
                },
            ],
        }
    )

    summary = notification_text(explanation, suppressed_count)
    return {
        # Drives the notification preview; also renders as the message body
        # (no top-level `blocks` here) — see notification_text's docstring.
        "text": summary,
        "attachments": [{"color": color, "fallback": summary, "blocks": blocks}],
    }


def format_slack_simple(explanation: Explanation) -> dict:
    """Compact single-attachment fallback."""
    color = _SEVERITY_COLORS.get(explanation.severity, "#CCCCCC")
    emoji = _SEVERITY_EMOJI.get(explanation.severity, "")
    lines = [
        f"{emoji} *{explanation.title}*",
        f"Agent: `{explanation.agent_id}` | Run: `{explanation.run_id}` | {explanation.confidence_pct()} confidence",
        "",
        explanation.what,
        "",
        f"_{explanation.evidence_summary}_",
    ]
    if explanation.suggested_fixes:
        lines.append(f"\n*Fix:* {explanation.suggested_fixes[0].description}")
    summary = notification_text(explanation)
    return {
        "text": summary,
        "attachments": [
            {
                "color": color,
                "fallback": summary,
                "text": "\n".join(lines),
                "mrkdwn_in": ["text"],
            }
        ],
    }
