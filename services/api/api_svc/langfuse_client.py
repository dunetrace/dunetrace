"""
Langfuse trace fetcher — standalone, no side effects.

Fetches a single trace by ID from the Langfuse API and returns the raw dict.
Never writes to Postgres. Raises on credential errors or missing traces.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any, Dict, List, Optional

# Re-exported for backward compatibility — moved to explain_common.py so
# native_explain.py (the non-Langfuse root-cause path) can use the same
# step-range logic without importing "native" code from a module named
# "langfuse". Existing code/tests importing these names from here still work.
from api_svc.explain_common import _get_step_range, _STEP_RANGE_FIELDS  # noqa: F401

logger = logging.getLogger("dunetrace.api.langfuse")

_SKIP_OBS_TYPES = {"EVENT", "SCORE"}
_SKIP_OBS_NAMES = {"langchain_on_chain_start", "langchain_on_chain_end"}

_INPUT_LIMIT_FOCUS = 600
_INPUT_LIMIT_OTHER = 150
_SYSTEM_PROMPT_LIMIT = 800
_TRACE_INPUT_LIMIT = 400


def _basic_auth(public_key: str, secret_key: str) -> str:
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return f"Basic {token}"


def _extract_system_prompt(obs_input: Any) -> Optional[str]:
    """Pull the system message out of a GENERATION input, if present.

    Handles two shapes:
      - dict with a "messages" key  (OpenAI SDK / most integrations)
      - list of message dicts       (LangChain Langfuse callback)
    Each message may use "role" or "type" for the role field, and
    "content" or "text" for the content field.
    """
    if isinstance(obs_input, dict):
        messages = obs_input.get("messages", [])
    elif isinstance(obs_input, list):
        messages = obs_input
    else:
        return None

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or msg.get("type", "")
        if role == "system":
            content = msg.get("content") or msg.get("text") or ""
            return str(content)[:_SYSTEM_PROMPT_LIMIT]
    return None


async def fetch_langfuse_trace(trace_id: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    Fetch a Langfuse trace by ID, including all observations.

    max_retries controls how many times to retry on 404 before giving up.
    Set to 0 for signals older than ~5 minutes — the trace is either there or not.
    Set to 3 (default) for freshly completed runs where Langfuse may still be ingesting.

    Returns the full trace dict. Raises ValueError on missing credentials,
    LookupError if the trace doesn't exist, httpx.HTTPStatusError on other errors.
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
    base_url = settings.LANGFUSE_HOST.rstrip("/")
    headers = {
        "Authorization": _basic_auth(settings.LANGFUSE_PUBLIC_KEY, settings.LANGFUSE_SECRET_KEY),
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        # Retry on 404 only for fresh traces — Langfuse ingests asynchronously
        # and a just-flushed trace may not be queryable immediately.
        # For older signals max_retries=0, so we do exactly one attempt.
        for attempt in range(max_retries + 1):
            resp = await client.get(
                f"{base_url}/api/public/traces/{lf_trace_id}",
                headers=headers,
            )
            if resp.status_code != 404:
                break
            if attempt < max_retries:
                await asyncio.sleep(3 * (attempt + 1))

        if resp.status_code == 404:
            raise LookupError(
                f"Trace {lf_trace_id!r} not found in Langfuse. "
                "The run may predate your Langfuse connection."
            )
        resp.raise_for_status()
        trace = resp.json()

        # The trace endpoint embeds observations but paginates at ~10 items.
        # Fetch the full list separately for any trace that looks paginated so
        # looping agents (which produce the most observations) are fully covered.
        if len(trace.get("observations", [])) >= 10:
            obs_resp = await client.get(
                f"{base_url}/api/public/observations",
                headers=headers,
                params={"traceId": lf_trace_id, "limit": 50},
            )
            obs_resp.raise_for_status()
            obs_data = obs_resp.json()
            trace["observations"] = obs_data.get("data", trace["observations"])

    return trace


def _format_observations(
    observations: list,
    first_step: int,
    last_step: int,
    signal_steps: Optional[List[int]] = None,
) -> tuple[str, Optional[str]]:
    """
    Format observations for the explain prompt.

    Returns (obs_text, system_prompt). system_prompt is the first system prompt
    found in any GENERATION observation; None if not present.
    """
    if not observations:
        return "(no observations recorded)", None

    focus_steps = set(signal_steps) if signal_steps else set()
    found_system_prompt: Optional[str] = None
    lines: list[str] = []

    for i, obs in enumerate(observations):
        obs_type = obs.get("type", "")
        obs_name = obs.get("name", "")

        if obs_type in _SKIP_OBS_TYPES or obs_name in _SKIP_OBS_NAMES:
            continue

        obs_input = obs.get("input")
        obs_output = obs.get("output")
        level = obs.get("level", "DEFAULT")

        # Use Langfuse metadata step number if present; fall back to list order.
        meta = obs.get("metadata") or {}
        step_num = meta.get("step") if isinstance(meta, dict) else None
        if step_num is None:
            step_num = i

        is_focus = bool(focus_steps) and step_num in focus_steps
        limit = _INPUT_LIMIT_FOCUS if is_focus else _INPUT_LIMIT_OTHER

        level_tag = f" [{level}]" if level not in ("DEFAULT", "") else ""
        inp_str = str(obs_input)[:limit] if obs_input else ""
        out_str = str(obs_output)[:limit] if obs_output else ""

        lines.append(f"  [{i}] {obs_type} {obs_name!r}{level_tag}")
        if inp_str:
            lines.append(f"      in:  {inp_str}")
        if out_str:
            lines.append(f"      out: {out_str}")

        if obs_type == "GENERATION" and found_system_prompt is None and obs_input:
            found_system_prompt = _extract_system_prompt(obs_input)

    obs_text = "\n".join(lines) if lines else "(no observations)"
    return obs_text, found_system_prompt


def _detect_langfuse_prompt(observations: list) -> tuple[Optional[str], Optional[int]]:
    """
    Scan GENERATION observations for a Langfuse-managed prompt name and version.

    Supports Langfuse SDK v2+ metadata keys (`lf.prompt.name`) as well as older
    patterns (`prompt_name`, nested `langfuse_prompt` dict).
    Returns (prompt_name, prompt_version) or (None, None).
    """
    for obs in observations:
        if obs.get("type") not in ("GENERATION", "SPAN"):
            continue
        meta = obs.get("metadata") or {}
        if not isinstance(meta, dict):
            continue

        name = (
            meta.get("lf.prompt.name")
            or meta.get("prompt_name")
            or (meta.get("langfuse_prompt") or {}).get("name")
        )
        version = (
            meta.get("lf.prompt.version")
            or meta.get("prompt_version")
            or (meta.get("langfuse_prompt") or {}).get("version")
        )

        if name:
            try:
                v = int(version) if version is not None else None
            except (ValueError, TypeError):
                v = None
            return str(name), v

    return None, None


async def apply_langfuse_fix(prompt_name: str, fix_content: str) -> Dict[str, Any]:
    """
    Append fix_content to the current Langfuse prompt and publish a new version.

    Returns {"new_version": n, "prompt_url": "...", "old_text": "...", "new_text": "..."}.
    Raises ValueError on missing credentials, httpx.HTTPStatusError on API errors.
    """
    try:
        import httpx
    except ImportError as exc:
        raise ImportError("httpx is required for Langfuse integration: pip install httpx") from exc

    from api_svc.config import settings

    if not settings.langfuse_configured:
        raise ValueError("Langfuse credentials not configured.")

    base_url = settings.LANGFUSE_HOST.rstrip("/")
    headers = {
        "Authorization": _basic_auth(settings.LANGFUSE_PUBLIC_KEY, settings.LANGFUSE_SECRET_KEY),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        get_resp = await client.get(
            f"{base_url}/api/public/prompts/{prompt_name}",
            headers=headers,
        )
        if get_resp.status_code == 404:
            raise LookupError(f"Prompt {prompt_name!r} not found in Langfuse.")
        get_resp.raise_for_status()
        current = get_resp.json()

        old_text = current.get("prompt", "")
        new_text = old_text.rstrip() + "\n\n" + fix_content

        post_resp = await client.post(
            f"{base_url}/api/public/prompts",
            headers=headers,
            json={
                "name": prompt_name,
                "prompt": new_text,
                "type": current.get("type", "text"),
            },
        )
        post_resp.raise_for_status()
        created = post_resp.json()

    return {
        "new_version": created.get("version"),
        "prompt_url": f"{base_url}/prompts/{prompt_name}",
        "old_text": old_text,
        "new_text": new_text,
    }


async def build_explain_prompt(signal_dict: dict, trace: Dict[str, Any]) -> str:
    """
    Compose the analysis prompt from a Dunetrace signal dict and a Langfuse trace.
    """
    evidence = signal_dict.get("evidence", {})
    failure_type = signal_dict.get("failure_type", "UNKNOWN")
    fallback = signal_dict.get("step_index", 0) or 0

    first_step, last_step = _get_step_range(evidence, failure_type, fallback)

    try:
        signal_steps = list(range(int(first_step), int(last_step) + 1))
    except (TypeError, ValueError):
        signal_steps = []

    obs_text, system_prompt = _format_observations(
        trace.get("observations", []),
        first_step,
        last_step,
        signal_steps=signal_steps,
    )

    trace_input = trace.get("input") or trace.get("name") or ""
    if isinstance(trace_input, dict):
        trace_input = str(trace_input)[:_TRACE_INPUT_LIMIT]
    else:
        trace_input = str(trace_input)[:_TRACE_INPUT_LIMIT]

    confidence_pct = int(signal_dict.get("confidence", 0) * 100)
    step_range_str = f"{first_step}–{last_step}" if first_step != last_step else str(first_step)

    system_prompt_section = (
        f"System prompt in use:\n{system_prompt}"
        if system_prompt
        else "System prompt: (not found in trace)"
    )

    return f"""Dunetrace detected: {failure_type}
Confidence: {confidence_pct}%
Severity: {signal_dict.get("severity", "")}
Evidence: {signal_dict.get("evidence_summary", "")}
Failing steps: {step_range_str}

{signal_dict.get("what", "")}

Your job: explain WHY this happened using the trace below.
Be specific — quote the actual system prompt text or tool output that caused it.
Suggest one precise fix (a sentence to add to the system prompt, or a code change).

Trace input: {trace_input}

{system_prompt_section}

Observations (steps {first_step}–{last_step}):
{obs_text}"""
