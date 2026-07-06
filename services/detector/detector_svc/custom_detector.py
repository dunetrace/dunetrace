"""Runtime evaluation of user-defined custom detectors against RunState."""

from __future__ import annotations

import logging
from typing import Optional

from dunetrace.models import RunState

logger = logging.getLogger("dunetrace.detector.custom")

_OPERATORS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def _compute_metric(state: RunState, metric: str) -> float:
    tc = state.tool_calls
    lc = state.llm_calls

    if metric == "step_count":
        return float(state.current_step)

    if metric == "tool_call_count":
        return float(len(tc))

    if metric == "llm_call_count":
        return float(len(lc))

    if metric == "consecutive_identical_tool_calls":
        if not tc:
            return 0.0
        max_run = curr = 1
        for i in range(1, len(tc)):
            if tc[i].tool_name == tc[i - 1].tool_name and tc[i].args == tc[i - 1].args:
                curr += 1
                max_run = max(max_run, curr)
            else:
                curr = 1
        return float(max_run)

    if metric == "consecutive_tool_failures":
        if not tc:
            return 0.0
        max_run = curr = 0
        for t in tc:
            if t.success is False:
                curr += 1
                max_run = max(max_run, curr)
            else:
                curr = 0
        return float(max_run)

    if metric == "token_growth_ratio":
        tokens = [c.prompt_tokens for c in lc if c.prompt_tokens and c.prompt_tokens > 0]
        if len(tokens) < 2:
            return 1.0
        return float(tokens[-1]) / float(tokens[0])

    if metric == "total_latency_ms":
        return float(sum(state.step_durations_ms.values()))

    if metric == "steps_since_last_tool":
        if not tc:
            return float(state.current_step)
        last_tool_step = max(t.step_index for t in tc)
        return float(state.current_step - last_tool_step)

    if metric == "finish_reason_length_count":
        return float(sum(1 for c in lc if c.finish_reason == "length"))

    if metric == "tool_failure_rate":
        if not tc:
            return 0.0
        return float(sum(1 for t in tc if t.success is False)) / float(len(tc))

    if metric == "avg_llm_latency_ms":
        lats = [c.latency_ms for c in lc if c.latency_ms is not None]
        return float(sum(lats) / len(lats)) if lats else 0.0

    if metric == "max_step_latency_ms":
        if not state.step_durations_ms:
            return 0.0
        return float(max(state.step_durations_ms.values()))

    logger.warning("Unknown custom detector metric: %s", metric)
    return 0.0


def evaluate_custom_detector(config: dict, state: RunState) -> Optional[dict]:
    """Evaluate a custom detector config against RunState.

    Returns a signal dict if all conditions are met, None otherwise.
    The caller handles writing to DB and recording shadow results.
    """
    for condition in config.get("conditions", []):
        metric = condition["metric"]
        operator = condition["operator"]
        threshold = float(condition["threshold"])
        value = _compute_metric(state, metric)
        op_fn = _OPERATORS.get(operator)
        if op_fn is None:
            logger.warning(
                "Unknown operator %r in custom detector %s", operator, config.get("detector_name")
            )
            return None
        if not op_fn(value, threshold):
            return None

    # All conditions met — build evidence dict
    evidence = {
        "description": config.get("evidence_template", "Custom detector fired"),
        "fix_suggestion": config.get("fix_template", ""),
        "detector_name": config.get("detector_name"),
        "conditions": config.get("conditions", []),
    }
    return {
        "failure_type": config["detector_name"],
        "severity": config.get("severity", "HIGH"),
        "step_index": state.current_step,
        "confidence": 0.75,
        "evidence": evidence,
    }
