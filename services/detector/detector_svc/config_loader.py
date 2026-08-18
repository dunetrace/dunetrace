"""
Reads detector thresholds from detectors.yml and returns kwargs for each detector class.
Falls back to SDK defaults if the file is missing or a detector isn't listed.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dunetrace.models import Severity

logger = logging.getLogger("dunetrace.config_loader")

# "severity" and "max_cost_ns" are accepted for every detector, independent of
# _PARAM_MAP below — both are cross-cutting BaseDetector attributes (see
# packages/sdk-py/dunetrace/detectors.py), not detector-specific tunables like
# THRESHOLD/WINDOW.
_SEVERITY_KEY = "severity"
_MAX_COST_NS_KEY = "max_cost_ns"

# Top-level section that is NOT a per-agent-category: the global custom-detector
# budget, parsed by load_custom_detector_config() instead.
_CUSTOM_DETECTORS_KEY = "custom_detectors"

# Keys valid inside any detector section that aren't threshold tunables, so the
# unknown-key check must not flag them. `alert_policy` and `destinations` are
# read by alerts_svc (see its config.py), not by this loader — they live in the
# same file but belong to a different consumer.
_RESERVED_DETECTOR_KEYS = frozenset(
    {_SEVERITY_KEY, _MAX_COST_NS_KEY, "alert_policy", "destinations"}
)

# Maps YAML section key -> detector constructor kwarg names (all uppercase).
# Only detectors with tunable params need an entry here.
_PARAM_MAP: dict[str, dict[str, str]] = {
    "oversized_tool_arguments": {"max_arg_length": "MAX_ARG_LENGTH"},
    "tool_loop": {"threshold": "THRESHOLD", "window": "WINDOW"},
    "tool_thrashing": {"window": "WINDOW"},
    "tool_avoidance": {"min_llm_calls": "MIN_LLM_CALLS"},
    "goal_abandonment": {"stall_steps": "STALL_STEPS"},
    "rag_empty_retrieval": {"min_score": "MIN_SCORE", "min_results": "MIN_RESULTS"},
    "excessive_retrieval": {"max_retrievals": "MAX_RETRIEVALS"},
    "llm_truncation_loop": {"threshold": "THRESHOLD"},
    "silent_truncation": {
        "min_output_length": "MIN_OUTPUT_LENGTH",
        "loop_threshold": "LOOP_THRESHOLD",
    },
    "model_fallback_drift": {"model_tiers": "MODEL_TIERS"},
    "context_bloat": {
        "growth_factor": "GROWTH_FACTOR",
        "min_calls": "MIN_CALLS",
        "min_last_tokens": "MIN_LAST_TOKENS",
        "inflation_factor": "INFLATION_FACTOR",
    },
    "slow_step": {"inflation_factor": "INFLATION_FACTOR"},
    "retry_storm": {"threshold": "THRESHOLD"},
    "step_count_inflation": {"inflation_factor": "INFLATION_FACTOR"},
    "cascading_tool_failure": {"threshold": "THRESHOLD"},
    "first_step_failure": {"max_step": "MAX_STEP"},
    "reasoning_stall": {
        "ratio_threshold": "RATIO_THRESHOLD",
        "min_llm_calls": "MIN_LLM_CALLS",
        "inflation_factor": "INFLATION_FACTOR",
    },
    "cost_spike": {
        "inflation_factor": "INFLATION_FACTOR",
        "static_threshold_tokens": "STATIC_THRESHOLD_TOKENS",
        "min_llm_calls": "MIN_LLM_CALLS",
    },
    "session_latency": {
        "inflation_factor": "INFLATION_FACTOR",
        "static_threshold_secs": "STATIC_THRESHOLD_SECS",
        "min_events": "MIN_EVENTS",
    },
    "premature_termination": {
        "completion_terms": "COMPLETION_TERMS",
        "error_acknowledgment_terms": "ERROR_ACKNOWLEDGMENT_TERMS",
        "error_markers": "ERROR_MARKERS",
        "case_sensitive": "CASE_SENSITIVE",
        "min_message_length": "MIN_MESSAGE_LENGTH",
    },
    "unread_tool_error": {
        "error_acknowledgment_terms": "ERROR_ACKNOWLEDGMENT_TERMS",
        "error_markers": "ERROR_MARKERS",
        "case_sensitive": "CASE_SENSITIVE",
    },
    "tool_argument_fabrication": {
        "allowlist": "ALLOWLIST",
        "destructive_tool_patterns": "DESTRUCTIVE_TOOL_PATTERNS",
        "small_int_min": "SMALL_INT_MIN",
        "small_int_max": "SMALL_INT_MAX",
        "case_sensitive": "CASE_SENSITIVE",
    },
    "retrieved_content_injection": {
        "injection_phrases": "INJECTION_PHRASES",
        "case_sensitive": "CASE_SENSITIVE",
        "detect_behavior_deviation": "DETECT_BEHAVIOR_DEVIATION",
    },
    "agent_handoff_failure": {
        "min_output_length": "MIN_OUTPUT_LENGTH",
        "handoff_patterns": "HANDOFF_PATTERNS",
        "excluded_tool_names": "EXCLUDED_TOOL_NAMES",
    },
    "handoff_context_loss": {
        "size_drop_threshold": "SIZE_DROP_THRESHOLD",
        "entity_loss_threshold": "ENTITY_LOSS_THRESHOLD",
    },
    "runaway_iteration": {
        "step_threshold": "STEP_THRESHOLD",
        "cost_threshold_usd": "COST_THRESHOLD_USD",
        "completion_patterns": "COMPLETION_PATTERNS",
        "lookback_messages": "LOOKBACK_MESSAGES",
        "case_sensitive": "CASE_SENSITIVE",
    },
    "memory_poisoning": {
        "poison_phrases": "POISON_PHRASES",
        "case_sensitive": "CASE_SENSITIVE",
        "require_untrusted_source": "REQUIRE_UNTRUSTED_SOURCE",
    },
    "delegation_loop": {
        "min_loop_runs": "MIN_LOOP_RUNS",
        "critical_loop_runs": "CRITICAL_LOOP_RUNS",
    },
    "ungrounded_destination": {
        "candidate_types": "CANDIDATE_TYPES",
        "allowlisted_domains": "ALLOWLISTED_DOMAINS",
        "mode": "MODE",
        "min_baseline_runs": "MIN_BASELINE_RUNS",
        "max_candidates_per_run": "MAX_CANDIDATES_PER_RUN",
        "max_depth": "MAX_DEPTH",
        "max_nodes": "MAX_NODES",
        "max_scan_ns": "MAX_SCAN_NS",
        "max_surface_chars": "MAX_SURFACE_CHARS",
        "max_args_chars": "MAX_ARGS_CHARS",
        "tool_name_scope": "TOOL_NAME_SCOPE",
        "send_tool_patterns": "SEND_TOOL_PATTERNS",
        "demotion_phrases": "DEMOTION_PHRASES",
        "case_sensitive": "CASE_SENSITIVE",
    },
}


_DEFAULT_CUSTOM_DETECTOR_BUDGET_MS = 10.0
_DEFAULT_CUSTOM_DETECTOR_REGEX_TIMEOUT_MS = 5.0


def load_custom_detector_budget(config_path: str | None = None) -> dict[str, float]:
    """Parse detectors.yml's top-level custom_detectors: section (global, not
    per-agent-category — see the comment above `default:` in detectors.yml).

    Returns {"evaluation_budget_ms": ..., "regex_timeout_ms": ...}, falling back
    to defaults for a missing file, missing section, or invalid values.
    """
    path = config_path or os.environ.get("DETECTOR_CONFIG", "/app/detectors.yml")
    defaults = {
        "evaluation_budget_ms": _DEFAULT_CUSTOM_DETECTOR_BUDGET_MS,
        "regex_timeout_ms": _DEFAULT_CUSTOM_DETECTOR_REGEX_TIMEOUT_MS,
    }

    try:
        import yaml  # type: ignore[import]
    except ImportError:
        return defaults

    if not os.path.exists(path):
        return defaults

    try:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("Failed to parse detectors.yml for custom_detectors config: %s", exc)
        return defaults

    section = raw.get("custom_detectors")
    if not isinstance(section, dict):
        return defaults

    result = dict(defaults)
    for key in defaults:
        if key not in section:
            continue
        try:
            value = float(section[key])
            if value <= 0:
                raise ValueError("must be positive")
            result[key] = value
        except (TypeError, ValueError) as exc:
            logger.warning(
                "detectors.yml: custom_detectors.%s=%r is invalid (%s) — using default %s",
                key,
                section[key],
                exc,
                defaults[key],
            )

    if result["regex_timeout_ms"] > result["evaluation_budget_ms"]:
        logger.warning(
            "detectors.yml: custom_detectors.regex_timeout_ms (%s) exceeds "
            "evaluation_budget_ms (%s) — capping to evaluation_budget_ms.",
            result["regex_timeout_ms"],
            result["evaluation_budget_ms"],
        )
        result["regex_timeout_ms"] = result["evaluation_budget_ms"]

    return result


def load_detector_kwargs(
    config_path: str | None = None,
    known_detectors: set[str] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Parse detectors.yml into a nested dict of detector kwargs keyed by category, then detector name:

        {
            "default": {"tool_loop": {"THRESHOLD": 2, "WINDOW": 5}, ...},
            "web-research": {"tool_loop": {"THRESHOLD": 5}},
        }

    Returns an empty dict if the file is missing — SDK defaults apply.

    `known_detectors` is the set of detector keys the worker will actually run
    (`_DETECTOR_CLASSES`). Passing it turns a typo'd section name into a startup
    WARNING instead of a silent no-op: threshold lookup is `category_cfg.get(key,
    {})`, so a misspelled key just falls back to the hardcoded class defaults and
    the operator's tuning never applies. Callers that don't have the class list
    (tests, tooling) can omit it to skip that check; unknown *parameters* inside
    a recognized section are always reported, since _PARAM_MAP alone is enough
    to judge those.
    """
    path = config_path or os.environ.get("DETECTOR_CONFIG", "/app/detectors.yml")

    try:
        import yaml  # type: ignore[import]
    except ImportError:
        logger.warning("PyYAML not installed — using SDK defaults for all detectors.")
        return {}

    if not os.path.exists(path):
        logger.info("No detectors.yml found at %s — using SDK defaults.", path)
        return {}

    try:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("Failed to parse detectors.yml: %s — using SDK defaults.", exc)
        return {}

    result: dict[str, dict[str, dict[str, Any]]] = {}

    for category, detectors in raw.items():
        if not isinstance(detectors, dict):
            continue
        if category == _CUSTOM_DETECTORS_KEY:
            # Global custom-detector budget section, not a per-agent category —
            # parsed separately by load_custom_detector_config().
            continue
        result[category] = {}
        for det_key, params in detectors.items():
            if not isinstance(params, dict):
                continue
            if known_detectors is not None and det_key not in known_detectors:
                logger.warning(
                    "detectors.yml: %s.%s is not a known detector — every key "
                    "under it is being ignored. Valid detectors: %s",
                    category,
                    det_key,
                    ", ".join(sorted(known_detectors)),
                )
                continue
            param_map = _PARAM_MAP.get(det_key, {})
            unknown_params = set(params) - set(param_map) - _RESERVED_DETECTOR_KEYS
            if unknown_params:
                logger.warning(
                    "detectors.yml: %s.%s has unrecognized key(s) %s — ignored. "
                    "Tunables for this detector: %s",
                    category,
                    det_key,
                    ", ".join(sorted(unknown_params)),
                    ", ".join(sorted(param_map)) or "(none)",
                )
            kwargs = {param_map[k]: v for k, v in params.items() if k in param_map}

            if _SEVERITY_KEY in params:
                raw_severity = str(params[_SEVERITY_KEY]).upper()
                try:
                    kwargs["SEVERITY"] = Severity(raw_severity)
                except ValueError:
                    logger.warning(
                        "detectors.yml: %s.%s has invalid severity %r — valid: %s. Ignoring override.",
                        category,
                        det_key,
                        params[_SEVERITY_KEY],
                        [s.value for s in Severity],
                    )

            if _MAX_COST_NS_KEY in params:
                try:
                    max_cost_ns = int(params[_MAX_COST_NS_KEY])
                    if max_cost_ns <= 0:
                        raise ValueError("must be positive")
                    kwargs["MAX_COST_NS"] = max_cost_ns
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "detectors.yml: %s.%s has invalid max_cost_ns %r (%s). Ignoring override.",
                        category,
                        det_key,
                        params[_MAX_COST_NS_KEY],
                        exc,
                    )

            if kwargs:
                result[category][det_key] = kwargs

    logger.info("Loaded detector config from %s — categories: %s", path, list(result))
    return result
