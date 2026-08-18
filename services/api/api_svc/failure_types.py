"""
Validation for the open set of `failure_signals.failure_type` values.

The column is TEXT, and the set of values is genuinely open — six independent
producers write into it:

  * the 31 built-in structural detectors     (FailureType enum)
  * detector packs                           (VOICE_*, from PACK_REGISTRY)
  * JSON-config custom detectors             (CUSTOM_*)
  * Python-class custom detector plugins     (arbitrary, registered at runtime)
  * semantic evaluators                      (TASK_COMPLETION, HALLUCINATION, …)
  * operational markers                      (SEMANTIC_QUOTA_EXCEEDED, …)

Any hand-maintained whitelist therefore drifts behind the code, and rejects rows
that demonstrably exist in the database. That already happened twice: once in
failure_patterns.py (fixed), and once in signals.py, where a frozen 17-entry
literal 422'd `TOOL_ARGUMENT_FABRICATION`, `UNREAD_TOOL_ERROR`,
`PREMATURE_TERMINATION` and `RETRIEVED_CONTENT_INJECTION` — including the single
most common failure type in the data, making it unqueryable.

So this derives what it can from the code and accepts the rest by shape. The
point of validating at all is to turn a typo into a clear error instead of a
silently-empty result; it is not a security boundary — the value is passed as a
bound query parameter either way.
"""

from __future__ import annotations

import re

from dunetrace.models import FailureType

# Written by semantic_svc and integrations_svc rather than by a detector class,
# so they appear in no registry this service can import.
_OPERATIONAL_TYPES = {
    "SEMANTIC_QUOTA_EXCEEDED",
    "EXTERNAL_INTEGRATION_DOWN",
}

# The seven DeepEval-backed evaluators in semantic_svc/worker.py. Listed rather
# than imported: api_svc must not pull in the semantic service's dependencies.
_SEMANTIC_TYPES = {
    "HALLUCINATION",
    "TASK_COMPLETION",
    "TASK_UNDERSTANDING_FAILURE",
    "OFF_TOPIC_DRIFT",
    "USER_FRUSTRATION",
    "CONFUSION_LOOP",
    "SYCOPHANCY_SIGNAL",
}

# Prefixes owned by producers whose names aren't knowable ahead of time.
_OPEN_PREFIXES = ("CUSTOM_", "VOICE_")

# An UPPER_SNAKE identifier. Anything shaped like a failure type from a
# Python-class plugin or a future pack is accepted rather than 422'd — a query
# for something that doesn't exist returns no rows, which is the honest answer.
_SHAPE = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$")


def known_failure_types() -> set[str]:
    """Every failure type this service can enumerate from code.

    Used for error messages and docs, not as the accept/reject test — see
    is_valid_failure_type, which also accepts the open-ended producers.
    """
    names = {t.value for t in FailureType} | _OPERATIONAL_TYPES | _SEMANTIC_TYPES
    try:
        from dunetrace.packs import PACK_REGISTRY

        for pack in PACK_REGISTRY.values():
            names |= {d.name for d in pack.detectors}
    except Exception:
        # Packs are optional; their absence must not narrow validation.
        pass
    return names


def is_valid_failure_type(value: str) -> bool:
    """True if `value` could name a failure type written by any producer."""
    upper = value.upper()
    if upper in known_failure_types():
        return True
    if upper.startswith(_OPEN_PREFIXES):
        return True
    return bool(_SHAPE.match(upper))


def invalid_failure_type_detail(value: str) -> str:
    """Error body for a rejected value: only malformed input reaches this."""
    return (
        f"Invalid failure_type {value!r} — expected an UPPER_SNAKE_CASE name. "
        f"Known types: {', '.join(sorted(known_failure_types()))}"
    )
