"""
detectors.yml parser — the single implementation shared by the detector worker
and the ingest service.

The detector worker overlays this file's thresholds onto detector class
defaults (``detector_svc/detectors.py``); ingest serves the same merged kwargs
to the SDK at ``GET /v1/detector-config`` so the client-side pass stops
drifting from the server. Ingest has no SDK on its PYTHONPATH, which is why the
parser lives here rather than in ``detector_svc`` — this module depends on
nothing but the stdlib and ``dunetrace_schemas.enums``. PyYAML is optional and
import-guarded: without it every loader returns its "no config" value and logs
a WARNING, exactly as a missing file does.

``services/detector/detector_svc/config_loader.py`` is a thin adapter over
this module that converts ``SEVERITY`` to the SDK's own ``Severity`` enum.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from dunetrace_schemas.enums import Severity

logger = logging.getLogger("dunetrace.detector_config")

DEFAULT_CONFIG_PATH = "/app/detectors.yml"

# The section every agent inherits from; a per-agent-category section (named
# after the agent_id) overrides it key by key.
DEFAULT_CATEGORY = "default"

# "severity" and "max_cost_ns" are accepted for every detector, independent of
# _PARAM_MAP below — both are cross-cutting BaseDetector attributes (see
# packages/sdk-py/dunetrace/detectors.py), not detector-specific tunables like
# THRESHOLD/WINDOW.
_SEVERITY_KEY = "severity"
_MAX_COST_NS_KEY = "max_cost_ns"

# Top-level section that is NOT a per-agent-category: the global custom-detector
# budget, parsed by load_custom_detector_budget() instead.
_CUSTOM_DETECTORS_KEY = "custom_detectors"

# Keys valid inside any detector section that aren't threshold tunables, so the
# unknown-key check must not flag them. `alert_policy` and `destinations` are
# read by alerts_svc (see its config.py), not by this loader — they live in the
# same file but belong to a different consumer.
_RESERVED_DETECTOR_KEYS = frozenset(
    {_SEVERITY_KEY, _MAX_COST_NS_KEY, "alert_policy", "destinations"}
)

# Every built-in detector section detectors.yml may carry — one per entry in
# detector_svc's _DETECTOR_CLASSES. Consumers without the class list (ingest)
# pass this as `known_detectors` so a typo'd section is reported rather than
# silently served; services/detector/tests asserts the two sets stay equal.
BUILTIN_DETECTOR_KEYS: frozenset[str] = frozenset(
    {
        "instrumentation_degraded",
        "oversized_tool_arguments",
        "tool_loop",
        "tool_thrashing",
        "scattershot_tool_use",
        "tool_avoidance",
        "goal_abandonment",
        "prompt_injection_signal",
        "rag_empty_retrieval",
        "excessive_retrieval",
        "llm_truncation_loop",
        "silent_truncation",
        "context_bloat",
        "slow_step",
        "retry_storm",
        "empty_llm_response",
        "step_count_inflation",
        "cascading_tool_failure",
        "first_step_failure",
        "reasoning_stall",
        "cost_spike",
        "session_latency",
        "premature_termination",
        "unread_tool_error",
        "tool_argument_fabrication",
        "retrieved_content_injection",
        "agent_handoff_failure",
        "handoff_context_loss",
        "runaway_iteration",
        "model_fallback_drift",
        "memory_poisoning",
        "delegation_loop",
        "ungrounded_destination",
        "unresolved_ambiguity",
    }
)

# Maps YAML section key -> detector constructor kwarg names (all uppercase).
# Only detectors with tunable params need an entry here.
_PARAM_MAP: dict[str, dict[str, str]] = {
    "instrumentation_degraded": {"min_calls": "MIN_CALLS"},
    "oversized_tool_arguments": {"max_arg_length": "MAX_ARG_LENGTH"},
    "tool_loop": {"threshold": "THRESHOLD", "window": "WINDOW"},
    "tool_thrashing": {"window": "WINDOW"},
    # MIN_, not MAX_: the class renamed these when the comparison (>=) was
    # found to contradict the MAX_ name, and detectors.yml was updated to
    # match — but this map was not, so every scattershot tunable in the
    # shipped file was silently dropped as an unknown key. scripts/
    # check_detector_defaults.py now fails the build on that class of drift.
    "scattershot_tool_use": {
        "min_distinct_tools": "MIN_DISTINCT_TOOLS",
        "min_total_calls": "MIN_TOTAL_CALLS",
        "min_repeat_ratio": "MIN_REPEAT_RATIO",
        "scan_limit": "SCAN_LIMIT",
    },
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
    "unresolved_ambiguity": {
        "min_candidates": "MIN_CANDIDATES",
        "irreversible_tools": "IRREVERSIBLE_TOOLS",
        "warrant_surfaces": "WARRANT_SURFACES",
        "min_token_len": "MIN_TOKEN_LEN",
        "max_candidates_per_run": "MAX_CANDIDATES_PER_RUN",
        "max_scan_ns": "MAX_SCAN_NS",
        "max_output_chars": "MAX_OUTPUT_CHARS",
        "max_depth": "MAX_DEPTH",
        "user_turn_signals": "USER_TURN_SIGNALS",
        "user_turn_text_keys": "USER_TURN_TEXT_KEYS",
    },
}


_DEFAULT_CUSTOM_DETECTOR_BUDGET_MS = 10.0
_DEFAULT_CUSTOM_DETECTOR_REGEX_TIMEOUT_MS = 5.0


def resolve_config_path(config_path: Optional[str] = None) -> str:
    """The detectors.yml path a loader reads: explicit argument, then the
    DETECTOR_CONFIG env var, then DEFAULT_CONFIG_PATH."""
    return config_path or os.environ.get("DETECTOR_CONFIG", DEFAULT_CONFIG_PATH)


def _read_yaml(path: str, purpose: str) -> Optional[dict]:
    """Parse `path` with PyYAML. None means "no config" — the file is missing,
    PyYAML is absent, or the file failed to parse — and every caller falls back
    to its defaults. Each cause is logged, the two unexpected ones at WARNING."""
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        logger.warning("PyYAML not installed — using SDK defaults for %s.", purpose)
        return None

    if not os.path.exists(path):
        logger.info("No detectors.yml found at %s — using SDK defaults for %s.", path, purpose)
        return None

    try:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning(
            "Failed to parse detectors.yml for %s: %s — using SDK defaults.", purpose, exc
        )
        return None

    if not isinstance(raw, dict):
        logger.warning(
            "detectors.yml at %s is not a mapping (got %s) — using SDK defaults for %s.",
            path,
            type(raw).__name__,
            purpose,
        )
        return None
    return raw


def load_custom_detector_budget(config_path: Optional[str] = None) -> dict[str, float]:
    """Parse detectors.yml's top-level custom_detectors: section (global, not
    per-agent-category — see the comment above `default:` in detectors.yml).

    Returns {"evaluation_budget_ms": ..., "regex_timeout_ms": ...}, falling back
    to defaults for a missing file, missing section, or invalid values.
    """
    path = resolve_config_path(config_path)
    defaults = {
        "evaluation_budget_ms": _DEFAULT_CUSTOM_DETECTOR_BUDGET_MS,
        "regex_timeout_ms": _DEFAULT_CUSTOM_DETECTOR_REGEX_TIMEOUT_MS,
    }

    raw = _read_yaml(path, "custom_detectors config")
    if raw is None:
        return defaults

    section = raw.get(_CUSTOM_DETECTORS_KEY)
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
    config_path: Optional[str] = None,
    known_detectors: Optional[set[str]] = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Parse detectors.yml into a nested dict of detector kwargs keyed by category, then detector name:

        {
            "default": {"tool_loop": {"THRESHOLD": 2, "WINDOW": 5}, ...},
            "web-research": {"tool_loop": {"THRESHOLD": 5}},
        }

    Returns an empty dict if the file is missing — SDK defaults apply.

    `SEVERITY` values are ``dunetrace_schemas.enums.Severity`` members (a str
    enum, so they JSON-encode as their name); ``detector_svc.config_loader``
    converts them to the SDK's enum before they reach a detector constructor.

    `known_detectors` is the set of detector keys the worker will actually run
    (`_DETECTOR_CLASSES`, or BUILTIN_DETECTOR_KEYS for a consumer without the
    class list). Passing it turns a typo'd section name into a startup WARNING
    instead of a silent no-op: threshold lookup is `category_cfg.get(key, {})`,
    so a misspelled key just falls back to the hardcoded class defaults and the
    operator's tuning never applies. Callers that don't have the class list
    (tests, tooling) can omit it to skip that check; unknown *parameters* inside
    a recognized section are always reported, since _PARAM_MAP alone is enough
    to judge those.
    """
    path = resolve_config_path(config_path)

    raw = _read_yaml(path, "all detectors")
    if raw is None:
        return {}

    result: dict[str, dict[str, dict[str, Any]]] = {}

    for category, detectors in raw.items():
        if not isinstance(detectors, dict):
            continue
        if category == _CUSTOM_DETECTORS_KEY:
            # Global custom-detector budget section, not a per-agent category —
            # parsed separately by load_custom_detector_budget().
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


def effective_detector_kwargs(
    config: dict[str, dict[str, dict[str, Any]]],
    category: str,
) -> dict[str, dict[str, Any]]:
    """The kwargs the detector worker instantiates each built-in detector with
    for `category` (an agent_id): the `default` section with the category's
    own section merged over it, key by key. Mirrors the merge in
    ``detector_svc.detectors.get_detectors`` — a category with no section of
    its own gets `default` unchanged. Detectors with no override are absent
    (the worker passes them no kwargs at all).

    Values are returned as-is — call json_safe_kwargs() before serialising.
    """
    default_cfg = config.get(DEFAULT_CATEGORY, {})
    if category == DEFAULT_CATEGORY or category not in config:
        return {det: dict(kwargs) for det, kwargs in default_cfg.items()}
    category_cfg = config[category]
    merged: dict[str, dict[str, Any]] = {}
    for det in set(default_cfg) | set(category_cfg):
        kwargs = {**default_cfg.get(det, {}), **category_cfg.get(det, {})}
        if kwargs:
            merged[det] = kwargs
    return merged


def json_safe_kwargs(kwargs_by_detector: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Copy of an effective_detector_kwargs() result with enum values replaced
    by their string value, so the mapping round-trips through JSON unchanged.
    Every other value is already a YAML scalar/list/mapping."""
    out: dict[str, dict[str, Any]] = {}
    for det, kwargs in kwargs_by_detector.items():
        out[det] = {k: (v.value if isinstance(v, Severity) else v) for k, v in kwargs.items()}
    return out
