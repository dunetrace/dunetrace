"""Formats an Explanation as a Slack Block Kit payload."""
from __future__ import annotations
import os

from explainer_svc.models import Explanation

_SEVERITY_COLORS = {"CRITICAL": "#FF0000", "HIGH": "#FF6B00", "MEDIUM": "#FFB800", "LOW": "#36A64F"}
_SEVERITY_EMOJI  = {"CRITICAL": ":red_circle:", "HIGH": ":large_orange_circle:", "MEDIUM": ":large_yellow_circle:", "LOW": ":white_circle:"}
_DASHBOARD_BASE  = os.getenv("DASHBOARD_URL", "https://app.dunetrace.io")


def _rate_context_text(explanation: Explanation) -> str:
    """Build a one-line rate context string for the Slack message, or empty string if unavailable."""
    rc = getattr(explanation, "rate_context", {})
    if not rc:
        return ""
    total    = rc.get("total_runs", 0)
    affected = rc.get("affected_runs", 0)
    rate     = rc.get("rate", 0.0)
    systemic = rc.get("is_systemic", False)
    pct      = f"{round(rate * 100)}%"

    if systemic:
        return f":warning: *Systemic pattern* — {affected}/{total} runs affected ({pct} of runs in the last 7 days)"
    elif affected == 1:
        return f":information_source: First occurrence of this pattern in the last 7 days"
    else:
        return f":bar_chart: {affected}/{total} runs affected ({pct}) in the last 7 days"


def format_slack(explanation: Explanation) -> dict:
    """Block Kit payload for Slack Incoming Webhook."""
    severity = explanation.severity
    color    = _SEVERITY_COLORS.get(severity, "#CCCCCC")
    emoji    = _SEVERITY_EMOJI.get(severity, ":white_circle:")
    conf_pct = explanation.confidence_pct()
    dashboard_url = f"{_DASHBOARD_BASE}/runs/{explanation.run_id}"
    rate_text = _rate_context_text(explanation)

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{emoji}  {explanation.title}", "emoji": True},
        },
        {
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": (
                    f"*Agent:* `{explanation.agent_id}`  "
                    f"*Version:* `{explanation.agent_version}`  "
                    f"*Run:* `{explanation.run_id}`  "
                    f"*Step:* {explanation.step_index}  "
                    f"*Confidence:* {conf_pct}"
                ),
            }],
        },
        {"type": "divider"},
    ]

    if rate_text:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": rate_text}})

    blocks += [
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*What happened*\n{explanation.what}"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*Why it matters*\n{explanation.why_it_matters}"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*Evidence*\n```{explanation.evidence_summary}```"}},
    ]

    if explanation.suggested_fixes:
        fix      = explanation.suggested_fixes[0]
        fix_text = f"*Suggested fix:* {fix.description}"
        if fix.code and len(fix.code) < 800:
            fix_text += f"\n```{fix.code}```"
        blocks.append({"type": "divider"})
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": fix_text}})

    blocks.append({"type": "divider"})
    blocks.append({
        "type": "actions",
        "elements": [{"type": "button", "text": {"type": "plain_text", "text": "View Run", "emoji": True},
                      "url": dashboard_url, "style": "primary"}],
    })

    return {"attachments": [{"color": color, "blocks": blocks}]}


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
    return {"attachments": [{"color": color, "fallback": explanation.title, "text": "\n".join(lines), "mrkdwn_in": ["text"]}]}
