"""Translates plain-English detector descriptions to structured config via LLM."""

from __future__ import annotations

import json
import logging

import httpx

from api_svc import llm_provider
from api_svc.config import settings

logger = logging.getLogger("dunetrace.api.custom_detector_translator")

# Metadata metrics available to custom detectors — counts, ratios, durations.
SUPPORTED_METRICS: dict[str, str] = {
    "step_count": "total number of steps in the run",
    "tool_call_count": "total number of tool calls made",
    "llm_call_count": "total number of LLM calls made",
    "consecutive_identical_tool_calls": "longest consecutive run of identical tool+args calls",
    "consecutive_tool_failures": "longest consecutive run of failed tool calls",
    "token_growth_ratio": "ratio of last/first prompt token count (context growth factor)",
    "total_latency_ms": "total run wall-clock time in milliseconds",
    "steps_since_last_tool": "steps elapsed since the last tool call",
    "finish_reason_length_count": "number of LLM calls that hit the token limit (finish_reason=length)",
    "tool_failure_rate": "fraction of tool calls that failed (0.0 to 1.0)",
    "avg_llm_latency_ms": "average LLM call latency in milliseconds",
    "max_step_latency_ms": "maximum single step duration in milliseconds",
}

# Text fields available to content conditions. Must match _CONTENT_FIELDS in
# services/detector/detector_svc/custom_detector.py exactly — the two are
# defined independently (api_svc and detector_svc are separately deployed
# services with no shared import), same pattern as SUPPORTED_METRICS above
# and _compute_metric() in that same file.
CONTENT_FIELDS: dict[str, str] = {
    "tool_args": "arguments passed to tool calls",
    "tool_error": "error messages from failed tool calls",
    "llm_output": "text of the agent's LLM responses",
    "input_text": "the run's initial input/prompt text",
}

# Must match _CONTENT_OPERATORS in detector_svc/custom_detector.py.
CONTENT_OPERATORS = {
    "contains",
    "starts_with",
    "ends_with",
    "equals",
    "length_gt",
    "length_lt",
    "regex_matches",
}

_SYSTEM_PROMPT = """\
You translate plain-English descriptions of agent failure modes into structured detector configs.

You have access to two kinds of conditions:

1. Metadata metrics — numeric, computed from run-level aggregates:
{metrics_list}

2. Content conditions — text inspection against these fields:
{content_fields_list}

Return ONLY valid JSON in one of two forms:

Form 1 — Detector config (the vast majority of descriptions fit this):
{{
  "detector_name": "CUSTOM_ALL_CAPS_SNAKE_CASE",
  "conditions": [
    {{"metric": "<metric_name>", "operator": ">=", "threshold": <number>}},
    {{"field": "<field_name>", "operator": "contains", "value": "<text>", "case_sensitive": true}}
  ],
  "severity": "HIGH",
  "evidence_template": "Brief description of what happened (≤80 chars)",
  "fix_template": "One-sentence suggested fix (≤120 chars)",
  "requires_content": false
}}

A condition is either metric-shaped ("metric"/"operator"/"threshold") or
content-shaped ("field"/"operator"/"value"/"case_sensitive"). Mix freely.
Multiple conditions (of either shape) are ANDed together. A content condition
matches if ANY occurrence of that field within the run satisfies it (e.g. ANY
tool call's args, not all of them).

Metric operators: >=, <=, >, <, ==, !=
Content operators: contains, starts_with, ends_with, equals, length_gt, length_lt, regex_matches
  - length_gt / length_lt compare the field's text length against "value" (a number)
  - regex_matches "value" is a regular expression (evaluated with a timeout — keep
    patterns simple; avoid nested quantifiers like (a+)+ which some engines can
    hang on regardless of timeout)
  - case_sensitive defaults to true if omitted
Severity values: CRITICAL, HIGH, MEDIUM, LOW

Form 2 — Declined (only when the description needs something genuinely outside
both lists above — e.g. semantic/fuzzy judgment ("sounds frustrated"), fields not
listed, or cross-run history):
{{
  "requires_content": true,
  "reason": "One sentence explaining why this can't be expressed with the available metrics/fields"
}}

Rules:
- detector_name must start with CUSTOM_ and use ALL_CAPS_SNAKE_CASE
- threshold must be a number (not a string); length_gt/length_lt's value must be a number too
- Output only JSON, no explanation text
"""

_USER_PROMPT = "Translate this detector description:\n\n{description}"


async def translate_description(description: str) -> dict:
    """Call LLM to convert plain-English description to structured detector config.

    Returns either a config dict (requires_content=false) or
    {requires_content: true, reason: "..."} when content access is needed.
    Raises ValueError if no LLM key is configured, or httpx errors on failure.
    """
    metrics_list = "\n".join(f"  - {k}: {v}" for k, v in SUPPORTED_METRICS.items())
    content_fields_list = "\n".join(f"  - {k}: {v}" for k, v in CONTENT_FIELDS.items())
    system = _SYSTEM_PROMPT.format(
        metrics_list=metrics_list, content_fields_list=content_fields_list
    )
    user = _USER_PROMPT.format(description=description.strip())

    if not llm_provider.llm_configured():
        raise ValueError(llm_provider.missing_key_message())
    text = await llm_provider.complete(system, user, max_tokens=512)
    return _parse_json((text or "").strip())


def _parse_json(text: str) -> dict:
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(ln for ln in lines if not ln.startswith("```"))
    return json.loads(text.strip())
