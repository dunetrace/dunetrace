"""
Weekly failure mode frequency digest.

Sends a Slack summary every DIGEST_DAY at DIGEST_HOUR UTC (default: Monday 9am).
Covers the past 7 days: top failure types, top agents by signal volume,
systemic patterns, and issue open/resolve counts.

Delivery is guarded by digest_log — if a digest was sent in the last 6 days,
it will not send again even if the worker restarts.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

from alerts_svc.config import settings
from alerts_svc.db import (
    was_digest_sent_recently,
    log_digest_sent,
    fetch_weekly_digest_data,
)
from alerts_svc.sender import send_slack

logger = logging.getLogger("dunetrace.alerts.digest")

_SEVERITY_EMOJI = {
    "TOOL_LOOP": ":red_circle:",
    "TOOL_THRASHING": ":red_circle:",
    "LLM_TRUNCATION_LOOP": ":red_circle:",
    "RETRY_STORM": ":red_circle:",
    "EMPTY_LLM_RESPONSE": ":red_circle:",
    "CASCADING_TOOL_FAILURE": ":red_circle:",
    "SLOW_STEP": ":red_circle:",
    "PROMPT_INJECTION_SIGNAL": ":rotating_light:",
    "TOOL_AVOIDANCE": ":large_yellow_circle:",
    "GOAL_ABANDONMENT": ":large_yellow_circle:",
    "RAG_EMPTY_RETRIEVAL": ":large_yellow_circle:",
    "CONTEXT_BLOAT": ":large_yellow_circle:",
    "STEP_COUNT_INFLATION": ":large_yellow_circle:",
    "FIRST_STEP_FAILURE": ":large_yellow_circle:",
    "REASONING_STALL": ":large_yellow_circle:",
}


def should_send_digest() -> bool:
    """True if today is DIGEST_DAY and we're in the DIGEST_HOUR window."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.weekday() == settings.DIGEST_DAY and now.hour == settings.DIGEST_HOUR


def _pct(rate: Any) -> str:
    try:
        return f"{round(float(rate) * 100)}%"
    except (TypeError, ValueError):
        return "—"


def format_digest_slack(data: dict[str, Any]) -> dict:
    """Build the weekly digest Slack Block Kit payload."""
    now = datetime.datetime.now(datetime.timezone.utc)
    week_start = (now - datetime.timedelta(days=7)).strftime("%-d %b")
    week_end = now.strftime("%-d %b %Y")

    total_runs = data.get("total_runs", 0)
    total_agents = data.get("total_agents", 0)
    total_signals = data.get("total_signals", 0)
    top_failures = data.get("top_failures", [])
    top_agents = data.get("top_agents", [])
    systemic = data.get("systemic", [])
    opened = data.get("issues_opened", 0)
    resolved = data.get("issues_resolved", 0)

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f":bar_chart: Weekly Agent Health Digest — {week_start}–{week_end}",
                "emoji": True,
            },
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"*{total_agents}* agent(s) · *{total_runs}* runs · "
                        f"*{total_signals}* signal(s) in the last 7 days"
                    ),
                }
            ],
        },
        {"type": "divider"},
    ]

    # Top failure types
    if top_failures:
        lines = []
        for f in top_failures:
            emoji = _SEVERITY_EMOJI.get(f["failure_type"], ":white_circle:")
            pct = _pct(f.get("rate", 0))
            lines.append(
                f"{emoji} *{f['failure_type']}* — {f['affected_runs']} run(s) affected ({pct})"
            )
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Top failure types*\n" + "\n".join(lines),
                },
            }
        )
    else:
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": ":white_check_mark: No failures detected this week.",
                },
            }
        )

    # Top agents by signal volume
    if top_agents:
        lines = []
        for a in top_agents:
            dominant = a.get("dominant_failure", "—")
            lines.append(
                f"`{a['agent_id']}` — {a['signal_count']} signal(s) "
                f"across {a['run_count']} runs  _(dominant: {dominant})_"
            )
        blocks.append({"type": "divider"})
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Agents by signal volume*\n" + "\n".join(lines),
                },
            }
        )

    # Systemic patterns
    if systemic:
        lines = []
        for s in systemic:
            lines.append(
                f":warning: *{s['failure_type']}* on `{s['agent_id']}` — "
                f"{s['affected_runs']}/{s['total_runs']} runs ({_pct(s['rate'])})"
            )
        blocks.append({"type": "divider"})
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Systemic patterns* _(≥10% of runs affected)_\n" + "\n".join(lines),
                },
            }
        )

    # Issue summary
    blocks.append({"type": "divider"})
    blocks.append(
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f":ticket: *Issues* — *{opened}* opened this week · *{resolved}* resolved"
                ),
            },
        }
    )

    # Dashboard link
    blocks.append(
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Open Dashboard", "emoji": True},
                    "url": settings.DASHBOARD_URL,
                    "style": "primary",
                }
            ],
        }
    )

    return {"attachments": [{"color": "#4a9ede", "blocks": blocks}]}


async def send_weekly_digest() -> bool:
    """
    Check if it's time, fetch data, format, and send the digest.
    Returns True if a digest was sent, False otherwise.
    """
    if not settings.digest_enabled:
        return False

    if not should_send_digest():
        return False

    if await was_digest_sent_recently(within_days=6):
        logger.debug("Digest already sent this week — skipping")
        return False

    try:
        data = await fetch_weekly_digest_data()
    except Exception as exc:
        logger.error("Failed to fetch weekly digest data: %s", exc)
        return False

    if data.get("total_runs", 0) == 0:
        logger.info("No runs in the last 7 days — skipping digest")
        await log_digest_sent()  # don't retry until next week
        return False

    payload = format_digest_slack(data)

    try:
        result = send_slack(payload)
    except Exception as exc:
        logger.error("Failed to send weekly digest: %s", exc)
        return False

    if result.success:
        await log_digest_sent()
        logger.info(
            "Weekly digest sent. runs=%d agents=%d signals=%d",
            data["total_runs"],
            data["total_agents"],
            data["total_signals"],
        )
        return True
    else:
        logger.error("Digest Slack delivery failed: %s", result.error)
        return False
