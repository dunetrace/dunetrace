"""
Shared helpers for building a root-cause explain prompt from Dunetrace's own
stored events — see native_explain.py.
"""

from __future__ import annotations

from typing import Dict

# Maps each failure type to the (first_step_key, last_step_key) it stores in evidence.
# Detectors not listed here have no step range — fall back to signal.step_index for both.
_STEP_RANGE_FIELDS: Dict[str, tuple] = {
    # Whole-run breadth pattern: without a range the UI highlights only the
    # last tool call of something that spans the run.
    "SCATTERSHOT_TOOL_USE": ("first_step", "last_step"),
    "TOOL_LOOP": ("first_step", "last_step"),
    "EMPTY_LLM_RESPONSE": ("first_step", "first_step"),
    "LLM_TRUNCATION_LOOP": ("first_truncation_step", "last_truncation_step"),
    "CONTEXT_BLOAT": ("first_call_step", "last_call_step"),
    "RETRY_STORM": ("first_fail_step", "first_fail_step"),
    "CASCADING_TOOL_FAILURE": ("first_fail_step", "first_fail_step"),
    "GOAL_ABANDONMENT": ("last_tool_step", "current_step"),
    "FIRST_STEP_FAILURE": ("failed_step", "failed_step"),
    "SLOW_STEP": ("step_index", "step_index"),
}


def _get_step_range(
    evidence: dict,
    failure_type: str,
    fallback: int,
) -> tuple[int, int]:
    """Return (first_step, last_step) for a signal, using the correct evidence keys."""
    fields = _STEP_RANGE_FIELDS.get(failure_type)
    if fields:
        first_key, last_key = fields
        raw_first = evidence.get(first_key, fallback)
        raw_last = evidence.get(last_key, fallback)
    else:
        raw_first = raw_last = fallback
    try:
        return int(raw_first), int(raw_last)
    except (TypeError, ValueError):
        return fallback, fallback
