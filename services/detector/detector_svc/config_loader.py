"""
services/detector/detector_svc/config_loader.py

Loads detector thresholds from detectors.yml and returns constructor kwargs
for each detector class.

Falls back to empty dicts (SDK defaults) if the file is missing or a
detector is not mentioned.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("dunetrace.config_loader")

# Maps YAML section key -> detector constructor kwarg names (all uppercase).
# Only detectors with tunable params need an entry here.
_PARAM_MAP: dict[str, dict[str, str]] = {
    "tool_loop":               {"threshold": "THRESHOLD", "window": "WINDOW"},
    "tool_thrashing":          {"window": "WINDOW"},
    "tool_avoidance":          {"min_llm_calls": "MIN_LLM_CALLS"},
    "goal_abandonment":        {"stall_steps": "STALL_STEPS"},
    "rag_empty_retrieval":     {"min_score": "MIN_SCORE", "min_results": "MIN_RESULTS"},
    "llm_truncation_loop":     {"threshold": "THRESHOLD"},
    "context_bloat":           {
        "growth_factor":    "GROWTH_FACTOR",
        "min_calls":        "MIN_CALLS",
        "min_last_tokens":  "MIN_LAST_TOKENS",
    },
    "retry_storm":             {"threshold": "THRESHOLD"},
    "step_count_inflation":    {"inflation_factor": "INFLATION_FACTOR"},
    "cascading_tool_failure":  {"threshold": "THRESHOLD"},
    "first_step_failure":      {"max_step": "MAX_STEP"},
}


def load_detector_kwargs(config_path: str | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Parse detectors.yml and return a nested dict:

        {
            "default": {
                "tool_loop": {"THRESHOLD": 2, "WINDOW": 5},
                ...
            },
            "web-research": {
                "tool_loop": {"THRESHOLD": 5},
            },
        }

    Returns an empty dict if the file is missing (SDK defaults apply).
    """
    path = config_path or os.environ.get(
        "DETECTOR_CONFIG", "/app/detectors.yml"
    )

    try:
        import yaml  # type: ignore[import]
    except ImportError:
        logger.warning("PyYAML not installed i.e. using SDK defaults for all detectors.")
        return {}

    if not os.path.exists(path):
        logger.info("No detectors.yml found at %s i.e. using SDK defaults.", path)
        return {}

    try:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("Failed to parse detectors.yml: %s i.e. using SDK defaults.", exc)
        return {}

    result: dict[str, dict[str, dict[str, Any]]] = {}

    for category, detectors in raw.items():
        if not isinstance(detectors, dict):
            continue
        result[category] = {}
        for det_key, params in detectors.items():
            if not isinstance(params, dict):
                continue
            param_map = _PARAM_MAP.get(det_key, {})
            kwargs = {
                param_map[k]: v
                for k, v in params.items()
                if k in param_map
            }
            if kwargs:
                result[category][det_key] = kwargs

    logger.info("Loaded detector config from %s i.e. categories: %s", path, list(result))
    return result
