"""Translates plain-English detector descriptions to structured config via LLM."""

from __future__ import annotations

import json
import logging

import httpx

from api_svc.config import settings

logger = logging.getLogger("dunetrace.api.custom_detector_translator")

# Metrics available to custom detectors — metadata only, no content access.
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

_SYSTEM_PROMPT = """\
You translate plain-English descriptions of agent failure modes into structured detector configs.

You only have access to these metrics (metadata only — no prompt text or LLM output content):
{metrics_list}

Return ONLY valid JSON in one of two forms:

Form 1 — Detector config (when the description can be detected from metadata above):
{{
  "detector_name": "CUSTOM_ALL_CAPS_SNAKE_CASE",
  "conditions": [
    {{"metric": "<metric_name>", "operator": ">=", "threshold": <number>}}
  ],
  "severity": "HIGH",
  "evidence_template": "Brief description of what happened (≤80 chars)",
  "fix_template": "One-sentence suggested fix (≤120 chars)",
  "requires_content": false
}}

Valid operators: >=, <=, >, <, ==, !=
Severity values: CRITICAL, HIGH, MEDIUM, LOW
Multiple conditions are ANDed together.

Form 2 — Declined (when the description requires reading prompt text or LLM response content):
{{
  "requires_content": true,
  "reason": "One sentence explaining why content access is needed"
}}

Rules:
- detector_name must start with CUSTOM_ and use ALL_CAPS_SNAKE_CASE
- threshold must be a number (not a string)
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
    system = _SYSTEM_PROMPT.format(metrics_list=metrics_list)
    user = _USER_PROMPT.format(description=description.strip())

    if settings.ANTHROPIC_API_KEY:
        return await _call_anthropic(system, user)
    if settings.OPENAI_API_KEY:
        return await _call_openai(system, user)
    raise ValueError("No LLM API key configured (set ANTHROPIC_API_KEY or OPENAI_API_KEY)")


async def _call_anthropic(system: str, user: str) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": settings.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 512,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
    resp.raise_for_status()
    text = resp.json()["content"][0]["text"].strip()
    return _parse_json(text)


async def _call_openai(system: str, user: str) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "content-type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "max_tokens": 512,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    return _parse_json(text)


def _parse_json(text: str) -> dict:
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(ln for ln in lines if not ln.startswith("```"))
    return json.loads(text.strip())
