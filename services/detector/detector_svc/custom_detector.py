"""Runtime evaluation of user-defined custom detectors against RunState."""

from __future__ import annotations

import logging
import time
from typing import Optional

import regex

from dunetrace.models import EventType, RunState

logger = logging.getLogger("dunetrace.detector.custom")

_OPERATORS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}

_DEFAULT_EVALUATION_BUDGET_MS = 10.0
_DEFAULT_REGEX_TIMEOUT_MS = 5.0
_WARNING_RATE_LIMIT_S = 60.0

# Text fields a content condition can inspect. Deliberately not "any field on any
# object" — a fixed, documented set (see docs/detectors.md) evaluated the same
# way regardless of which detector class or event shape produced the text.
_CONTENT_FIELDS = {"tool_args", "tool_error", "llm_output", "input_text"}

# Last time this process logged a budget-exceeded warning, keyed by detector
# name. Rate-limited the same way as the SDK's built-in detector cost-budget
# warning (packages/sdk-py/dunetrace/detectors.py) — None, never 0.0, as the
# "never warned" sentinel; time.monotonic()'s epoch is unspecified.
_last_budget_warning: dict[str, float] = {}


def _extract_field_values(state: RunState, field: str) -> list[str]:
    """Return every occurrence of `field`'s text within this run, in step order."""
    if field == "tool_args":
        return [t.args for t in state.tool_calls if t.args]
    if field == "tool_error":
        return [t.error for t in state.tool_calls if t.error]
    if field == "llm_output":
        return [
            e.payload["output"]
            for e in state.events
            if e.event_type == EventType.LLM_RESPONDED and e.payload.get("output")
        ]
    if field == "input_text":
        return [state.input_text] if state.input_text else []
    logger.warning("Unknown custom detector content field: %s", field)
    return []


def _content_op_contains(text: str, target: str, case_sensitive: bool, **_: object) -> bool:
    return (target in text) if case_sensitive else (target.lower() in text.lower())


def _content_op_starts_with(text: str, target: str, case_sensitive: bool, **_: object) -> bool:
    return text.startswith(target) if case_sensitive else text.lower().startswith(target.lower())


def _content_op_ends_with(text: str, target: str, case_sensitive: bool, **_: object) -> bool:
    return text.endswith(target) if case_sensitive else text.lower().endswith(target.lower())


def _content_op_equals(text: str, target: str, case_sensitive: bool, **_: object) -> bool:
    return (text == target) if case_sensitive else (text.lower() == target.lower())


def _content_op_length_gt(text: str, target: str, case_sensitive: bool, **_: object) -> bool:
    return len(text) > float(target)


def _content_op_length_lt(text: str, target: str, case_sensitive: bool, **_: object) -> bool:
    return len(text) < float(target)


def _content_op_regex_matches(
    text: str,
    target: str,
    case_sensitive: bool,
    regex_timeout_ms: float = _DEFAULT_REGEX_TIMEOUT_MS,
) -> bool:
    flags = 0 if case_sensitive else regex.IGNORECASE
    try:
        return (
            regex.search(target, text, flags=flags, timeout=regex_timeout_ms / 1000.0) is not None
        )
    except TimeoutError:
        logger.warning(
            "Custom detector regex_matches condition timed out after %sms — pattern %r "
            "treated as non-matching for this occurrence (ReDoS protection).",
            regex_timeout_ms,
            target,
        )
        return False
    except regex.error as exc:
        logger.warning("Custom detector regex_matches has an invalid pattern %r: %s", target, exc)
        return False


_CONTENT_OPERATORS = {
    "contains": _content_op_contains,
    "starts_with": _content_op_starts_with,
    "ends_with": _content_op_ends_with,
    "equals": _content_op_equals,
    "length_gt": _content_op_length_gt,
    "length_lt": _content_op_length_lt,
    "regex_matches": _content_op_regex_matches,
}


def _warn_budget_exceeded(detector_name: str, elapsed_ms: float, budget_ms: float) -> None:
    now = time.monotonic()
    last = _last_budget_warning.get(detector_name)
    if last is not None and now - last < _WARNING_RATE_LIMIT_S:
        return
    _last_budget_warning[detector_name] = now
    logger.warning(
        "Custom detector %s exceeded its evaluation budget: %.2fms > %.2fms — "
        "aborting evaluation for this run (treated as not-fired).",
        detector_name,
        elapsed_ms,
        budget_ms,
    )


def _evaluate_content_condition(state: RunState, condition: dict, regex_timeout_ms: float) -> bool:
    """True if ANY occurrence of condition['field'] in this run satisfies the operator."""
    field = condition["field"]
    operator = condition["operator"]
    target = str(condition["value"])
    case_sensitive = condition.get("case_sensitive", True)

    op_fn = _CONTENT_OPERATORS.get(operator)
    if op_fn is None:
        logger.warning("Unknown custom detector content operator: %s", operator)
        return False
    if field not in _CONTENT_FIELDS:
        logger.warning("Unknown custom detector content field: %s", field)
        return False

    return any(
        op_fn(text, target, case_sensitive, regex_timeout_ms=regex_timeout_ms)
        for text in _extract_field_values(state, field)
    )


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


def evaluate_custom_detector(
    config: dict,
    state: RunState,
    evaluation_budget_ms: float = _DEFAULT_EVALUATION_BUDGET_MS,
    regex_timeout_ms: float = _DEFAULT_REGEX_TIMEOUT_MS,
) -> Optional[dict]:
    """Evaluate a custom detector config against RunState.

    Returns a signal dict if all conditions are met, None otherwise.
    The caller handles writing to DB and recording shadow results.

    Each condition is either metric-based ({"metric", "operator", "threshold"} —
    numeric, computed from RunState aggregates) or content-based ({"field",
    "operator", "value", "case_sensitive"} — text inspection, see
    docs/detectors.md). All conditions are ANDed, same as before.

    evaluation_budget_ms bounds the wall-clock cost of evaluating this one
    detector against this one run (all conditions combined) — exceeding it
    aborts evaluation and treats the detector as not-fired for this run, rather
    than letting one expensive detector (e.g. a slow regex_matches, or a run
    with an unusually large number of tool calls) stall the shared detector
    worker loop for every other org/agent on this shard. Rate-limited to one
    warning per detector name per minute.
    """
    detector_name = config.get("detector_name", "?")
    t0 = time.perf_counter()

    for condition in config.get("conditions", []):
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms > evaluation_budget_ms:
            _warn_budget_exceeded(detector_name, elapsed_ms, evaluation_budget_ms)
            return None

        if "field" in condition:
            if not _evaluate_content_condition(state, condition, regex_timeout_ms):
                return None
            continue

        metric = condition["metric"]
        operator = condition["operator"]
        threshold = float(condition["threshold"])
        value = _compute_metric(state, metric)
        op_fn = _OPERATORS.get(operator)
        if op_fn is None:
            logger.warning("Unknown operator %r in custom detector %s", operator, detector_name)
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
