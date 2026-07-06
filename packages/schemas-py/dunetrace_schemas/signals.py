"""
Canonical wire-format model for a detected failure signal — the shape written
to `failure_signals` and returned by the customer API's signal endpoints.

Field-for-field equivalent to the SDK's `dunetrace.models.FailureSignal`
dataclass, plus the two fields that only exist once a signal has been
persisted (`shadow`, `alerted`). The SDK's dataclass stays a dataclass for
hot-path construction cost; this model is for the validated wire/storage
boundary, not the in-process detector loop.

`failure_type` is `str`, not the `FailureType` enum — custom detectors store
arbitrary `CUSTOM_*` failure type names as raw TEXT (see
`detector_svc.custom_detector.write_custom_signal`), so this field must accept
values outside the built-in enum. Use `FailureType` for validating/documenting
the built-in set where that distinction matters, not on this general model.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from pydantic import BaseModel, Field

from dunetrace_schemas.enums import Severity


class FailureSignalSchema(BaseModel):
    failure_type: str = Field(min_length=1)
    severity: Severity
    run_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    agent_version: str = Field(min_length=1)
    step_index: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    detected_at: float = Field(default_factory=time.time)
    co_signal_count: int = 0
    shadow: bool = True
    alerted: bool = False
