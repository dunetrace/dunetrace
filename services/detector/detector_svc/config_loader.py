"""
Reads detector thresholds from detectors.yml and returns kwargs for each detector class.
Falls back to SDK defaults if the file is missing or a detector isn't listed.

Thin adapter: the parser and _PARAM_MAP live in
``dunetrace_schemas.detector_config`` so the ingest service (no SDK on its
PYTHONPATH) can serve the same merged kwargs at ``GET /v1/detector-config``.
This module's one job is to hand the worker SDK-typed values — the shared
parser emits ``dunetrace_schemas.enums.Severity`` and every detector expects
``dunetrace.models.Severity``.
"""

from __future__ import annotations

from typing import Any

from dunetrace.models import Severity
from dunetrace_schemas.detector_config import (
    _CUSTOM_DETECTORS_KEY,
    _MAX_COST_NS_KEY,
    _PARAM_MAP,
    _RESERVED_DETECTOR_KEYS,
    _SEVERITY_KEY,
    BUILTIN_DETECTOR_KEYS,
    load_custom_detector_budget,
)
from dunetrace_schemas.detector_config import load_detector_kwargs as _load_detector_kwargs

__all__ = [
    "load_detector_kwargs",
    "load_custom_detector_budget",
    "BUILTIN_DETECTOR_KEYS",
    "_PARAM_MAP",
]


def _to_sdk_types(kwargs: dict[str, Any]) -> dict[str, Any]:
    if "SEVERITY" in kwargs:
        # Both enums are str-valued with identical members (test_sdk_parity.py),
        # so this is a rename, never a lossy conversion.
        kwargs = {**kwargs, "SEVERITY": Severity(kwargs["SEVERITY"].value)}
    return kwargs


def load_detector_kwargs(
    config_path: str | None = None,
    known_detectors: set[str] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """See ``dunetrace_schemas.detector_config.load_detector_kwargs``. Identical
    contract; ``SEVERITY`` values are ``dunetrace.models.Severity`` members."""
    parsed = _load_detector_kwargs(config_path, known_detectors)
    return {
        category: {det: _to_sdk_types(kwargs) for det, kwargs in detectors.items()}
        for category, detectors in parsed.items()
    }
