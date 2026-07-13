"""
Reads adaptive sampling rate defaults from docs/config/semantic-sampling.yml,
and per-evaluator second-opinion config from docs/config/semantic-evaluators.yml
(the same file alerts_svc reads for alert_confidence_floor — different keys,
read independently by each service). Mirrors detector_svc/config_loader.py's
load_custom_detector_budget: falls back to sane defaults for a missing file,
missing key, or invalid value, and never raises.
"""

from __future__ import annotations

import logging
import os

from semantic_svc.sampling import SamplingRates, with_overrides

logger = logging.getLogger("dunetrace.semantic.config_loader")

_RATE_KEYS = (
    "structural_signal_rate",
    "semantic_critical_rate",
    "retrieval_rate",
    "baseline_rate",
    "conversation_evaluation_rate",
)


def load_sampling_rates(config_path: str | None = None) -> SamplingRates:
    path = config_path or os.environ.get(
        "SEMANTIC_SAMPLING_CONFIG", "/app/docs/config/semantic-sampling.yml"
    )
    defaults = SamplingRates()

    try:
        import yaml  # type: ignore[import]
    except ImportError:
        logger.warning("PyYAML not installed — using default sampling rates.")
        return defaults

    if not os.path.exists(path):
        logger.info("No semantic-sampling.yml found at %s — using default sampling rates.", path)
        return defaults

    try:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("Failed to parse %s: %s — using default sampling rates.", path, exc)
        return defaults

    overrides: dict[str, float] = {}
    for key in _RATE_KEYS:
        if key not in raw:
            continue
        try:
            value = float(raw[key])
            if not (0.0 <= value <= 1.0):
                raise ValueError("must be between 0.0 and 1.0")
            overrides[key] = value
        except (TypeError, ValueError) as exc:
            logger.warning(
                "%s: %s=%r is invalid (%s) — using default %s",
                path,
                key,
                raw[key],
                exc,
                getattr(defaults, key),
            )

    return with_overrides(defaults, **overrides)


# ── Per-evaluator second-opinion config (Phase 1.4.4) ─────────────────────────

_DEFAULT_EVALUATOR_ENTRY = {
    "require_second_opinion": False,
    "second_opinion_provider": None,
    "second_opinion_model": None,
}


def load_evaluator_config(config_path: str | None = None) -> dict[str, dict]:
    """Parses docs/config/semantic-evaluators.yml's require_second_opinion /
    second_opinion_provider / second_opinion_model per evaluator name. Falls
    back to {} (meaning every evaluator defaults to no second opinion) for a
    missing file, missing PyYAML, or any parse error. Provider/model default
    resolution ("the other provider" when unset) happens in worker.py at
    evaluator-build time, not here — this is a pure parse, no runtime settings.
    """
    path = config_path or os.environ.get(
        "SEMANTIC_EVALUATORS_YML", "/app/docs/config/semantic-evaluators.yml"
    )
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        logger.warning("PyYAML not installed — no evaluator requires a second opinion.")
        return {}

    if not os.path.exists(path):
        logger.info("No semantic-evaluators.yml found at %s — using defaults.", path)
        return {}

    try:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("Failed to parse %s: %s — using defaults.", path, exc)
        return {}

    result: dict[str, dict] = {}
    for evaluator_key, cfg in raw.items():
        if not isinstance(cfg, dict):
            continue
        entry = dict(_DEFAULT_EVALUATOR_ENTRY)
        if "require_second_opinion" in cfg:
            entry["require_second_opinion"] = bool(cfg["require_second_opinion"])
        entry["second_opinion_provider"] = cfg.get("second_opinion_provider")
        entry["second_opinion_model"] = cfg.get("second_opinion_model")
        result[evaluator_key.upper()] = entry

    return result
