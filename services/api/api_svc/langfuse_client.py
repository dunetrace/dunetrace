"""
Langfuse trace fetcher — standalone, no side effects.

Fetches a single trace by ID from the Langfuse API and returns the raw dict.
Never writes to Postgres. Raises on credential errors or missing traces.
"""
from __future__ import annotations

import base64
import logging
from typing import Any, Dict

logger = logging.getLogger("dunetrace.api.langfuse")


def _basic_auth(public_key: str, secret_key: str) -> str:
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return f"Basic {token}"


async def fetch_langfuse_trace(trace_id: str) -> Dict[str, Any]:
    """
    Fetch a Langfuse trace by ID.

    Returns the full trace dict including `observations`.
    Raises ValueError on missing credentials, httpx.HTTPStatusError on API errors.
    """
    try:
        import httpx
    except ImportError as exc:
        raise ImportError("httpx is required for Langfuse integration: pip install httpx") from exc

    from api_svc.config import settings

    if not settings.langfuse_configured:
        raise ValueError(
            "Langfuse credentials not configured. "
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in your .env."
        )

    # Langfuse v4 uses OTel-style 32-char hex IDs. Dunetrace stores UUID with
    # dashes — strip them so both formats resolve to the same Langfuse trace.
    lf_trace_id = trace_id.replace("-", "")
    url = f"{settings.LANGFUSE_HOST.rstrip('/')}/api/public/traces/{lf_trace_id}"
    headers = {
        "Authorization": _basic_auth(settings.LANGFUSE_PUBLIC_KEY, settings.LANGFUSE_SECRET_KEY),
        "Accept": "application/json",
    }

    import asyncio

    async with httpx.AsyncClient(timeout=15.0) as client:
        # Langfuse ingests asynchronously — retry a few times on 404
        # before giving up, to handle the common case of a freshly flushed trace.
        for attempt in range(4):
            resp = await client.get(url, headers=headers)
            if resp.status_code != 404:
                break
            if attempt < 3:
                await asyncio.sleep(3 * (attempt + 1))

    if resp.status_code == 404:
        raise LookupError(
            f"Trace {lf_trace_id!r} not found in Langfuse. "
            "The run may predate your Langfuse connection."
        )
    resp.raise_for_status()
    return resp.json()


def _format_observations(observations: list, first_step: int, last_step: int) -> str:
    """
    Extract and format the observations relevant to the loop step range.

    Langfuse observations don't have step numbers, but we can use their order
    and timestamps to approximate the right window. We take all observations
    and include them if they look like tool calls (type SPAN or GENERATION)
    within a reasonable window around the loop.
    """
    if not observations:
        return "(no observations recorded)"

    lines = []
    for i, obs in enumerate(observations):
        obs_type  = obs.get("type", "")
        obs_name  = obs.get("name", "")
        obs_input = obs.get("input")
        obs_out   = obs.get("output")
        level     = obs.get("level", "DEFAULT")

        level_tag = f" [{level}]" if level not in ("DEFAULT", "") else ""
        inp_str   = str(obs_input)[:200] if obs_input else ""
        out_str   = str(obs_out)[:200]   if obs_out   else ""

        lines.append(f"  [{i}] {obs_type} {obs_name!r}{level_tag}")
        if inp_str:
            lines.append(f"      in:  {inp_str}")
        if out_str:
            lines.append(f"      out: {out_str}")

    return "\n".join(lines) if lines else "(no observations)"


async def build_explain_prompt(signal_dict: dict, trace: Dict[str, Any]) -> str:
    """
    Compose the analysis prompt from a Dunetrace signal dict and a Langfuse trace.
    """
    evidence   = signal_dict.get("evidence", {})
    first_step = evidence.get("first_step", 0)
    last_step  = evidence.get("last_step", "?")
    obs_text   = _format_observations(
        trace.get("observations", []), first_step, last_step
    )

    trace_input = trace.get("input") or trace.get("name") or ""
    if isinstance(trace_input, dict):
        trace_input = str(trace_input)[:400]
    else:
        trace_input = str(trace_input)[:400]

    prompt = f"""BEHAVIORAL SIGNAL: {signal_dict.get("failure_type")}
Confidence: {int(signal_dict.get("confidence", 0) * 100)}%
{signal_dict.get("evidence_summary", "")}

WHAT DUNETRACE DETECTED:
{signal_dict.get("what", "")}

LANGFUSE TRACE (run_id: {signal_dict.get("run_id")}):
User input: {trace_input}

Observations (steps {first_step}–{last_step}):
{obs_text}

Why did this specific failure occur, and what is the single most important fix?"""

    return prompt
