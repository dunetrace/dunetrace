"""
Native root-cause context builder.

Builds the "what happened" prompt section from Dunetrace's own events table
(the same data the dashboard's Event Log tab renders — see
db/queries.py::get_run_detail). This is the only root-cause path — it works
for every signal, with no external tracing system involved.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from api_svc.explain_common import _get_step_range

_INPUT_LIMIT_FOCUS = 600
_INPUT_LIMIT_OTHER = 150
_SYSTEM_PROMPT_LIMIT = 800
_TRACE_INPUT_LIMIT = 400


def _extract_system_prompt(events: List[dict]) -> Optional[str]:
    """Pull the system prompt out of the run.started event's payload.

    dunetrace.client.Dunetrace.run's system_prompt param is sent as-is in
    run.started's payload (still also folded into agent_version's hash for
    grouping — see dunetrace.models.agent_version). This is instrumentation-
    dependent, not guaranteed: callers of dt.run() that omit system_prompt
    (or SDK versions predating this field) still produce a run.started event
    with no system_prompt key, so this returns None for those — same as
    before for events ingested prior to this change.
    """
    for e in events:
        if e.get("event_type") == "run.started":
            payload = e.get("payload") or {}
            system_prompt = payload.get("system_prompt")
            if system_prompt:
                return str(system_prompt)[:_SYSTEM_PROMPT_LIMIT]
            return None
    return None


def _format_events(
    events: List[dict],
    first_step: int,
    last_step: int,
    signal_steps: Optional[List[int]] = None,
) -> str:
    """Format native events for the explain prompt."""
    if not events:
        return "(no events recorded)"

    focus_steps = set(signal_steps) if signal_steps else set()
    lines: List[str] = []

    for e in events:
        event_type = e.get("event_type", "")
        step = e.get("step_index", 0)
        payload = e.get("payload") or {}

        is_focus = bool(focus_steps) and step in focus_steps
        limit = _INPUT_LIMIT_FOCUS if is_focus else _INPUT_LIMIT_OTHER

        lines.append(f"  [step {step}] {event_type}")
        for key in ("tool_name", "model", "input_text", "args", "output", "error"):
            value = payload.get(key)
            if value is None:
                continue
            value_str = str(value)[:limit]
            lines.append(f"      {key}: {value_str}")

    return "\n".join(lines) if lines else "(no events)"


async def build_native_explain_prompt(signal_dict: dict, events: List[dict]) -> str:
    """
    Compose the analysis prompt from a Dunetrace signal dict and this run's
    own native events. async even though this is purely deterministic string
    work — callers await it either way, so _call_llm() (which builds on this)
    needs no special-casing.
    """
    evidence = signal_dict.get("evidence", {})
    failure_type = signal_dict.get("failure_type", "UNKNOWN")
    fallback = signal_dict.get("step_index", 0) or 0

    first_step, last_step = _get_step_range(evidence, failure_type, fallback)

    try:
        signal_steps = list(range(int(first_step), int(last_step) + 1))
    except (TypeError, ValueError):
        signal_steps = []

    events_text = _format_events(events, first_step, last_step, signal_steps=signal_steps)
    system_prompt = _extract_system_prompt(events)

    run_input = ""
    for e in events:
        if e.get("event_type") == "run.started":
            run_input = str((e.get("payload") or {}).get("input_text", ""))[:_TRACE_INPUT_LIMIT]
            break

    confidence_pct = int(signal_dict.get("confidence", 0) * 100)
    step_range_str = f"{first_step}–{last_step}" if first_step != last_step else str(first_step)

    system_prompt_section = (
        f"System prompt in use:\n{system_prompt}"
        if system_prompt
        else "System prompt: (not found in events)"
    )

    return f"""Dunetrace detected: {failure_type}
Confidence: {confidence_pct}%
Severity: {signal_dict.get("severity", "")}
Evidence: {signal_dict.get("evidence_summary", "")}
Failing steps: {step_range_str}

{signal_dict.get("what", "")}

Your job: explain WHY this happened using the run events below.
Be specific — quote the actual system prompt text or tool output that caused it.
Suggest one precise fix (a sentence to add to the system prompt, or a code change).

Run input: {run_input}

{system_prompt_section}

Events (steps {first_step}–{last_step}):
{events_text}"""
