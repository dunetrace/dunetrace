"""Formats an Explanation as a Linear issue title + Markdown description
(Phase 4.1). Same content Slack alerts already show (root cause via the
free deterministic explainer's why_it_matters/evidence_summary, not the
paid LLM-based native_explain — see BACKLOG.md's Phase 4.1 entry for why
that substitution wasn't made), reformatted for Linear's Markdown
description field instead of Slack's Block Kit."""

from __future__ import annotations

import os

from explainer_svc.models import Explanation

_DASHBOARD_BASE = os.getenv("DASHBOARD_URL", "https://app.dunetrace.io")


def format_linear_issue(explanation: Explanation) -> tuple[str, str]:
    """Returns (title, description_markdown)."""
    dashboard_url = f"{_DASHBOARD_BASE}/runs/{explanation.run_id}"

    lines = [
        f"**Agent:** `{explanation.agent_id}`  **Version:** `{explanation.agent_version}`  "
        f"**Run:** `{explanation.run_id}`  **Confidence:** {explanation.confidence_pct()}",
        "",
        f"**What happened**\n{explanation.what}",
        "",
        f"**Why it matters**\n{explanation.why_it_matters}",
        "",
        f"**Evidence**\n```\n{explanation.evidence_summary}\n```",
    ]

    if explanation.suggested_fixes:
        fix = explanation.suggested_fixes[0]
        lines.append("")
        fix_text = f"**Suggested fix:** {fix.description}"
        if fix.code and len(fix.code) < 800:
            fix_text += f"\n```{fix.language}\n{fix.code}\n```"
        lines.append(fix_text)

    lines.append("")
    lines.append(f"[View run in Dunetrace]({dashboard_url})")

    return explanation.title, "\n".join(lines)
