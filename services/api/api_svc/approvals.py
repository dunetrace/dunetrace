"""
Approval state model (Capability 2, Phase 2.1). Pure logic — no I/O, no DB.
The DB layer (db/queries.py) and the API/SDK flow (later phases) both build on
these definitions so the legal-transition rule lives in exactly one place.

An approval moves through:

    pending ──► granted     (a human approved the tool call)
            ├─► denied      (a human rejected it)
            └─► timeout     (no human decided before the deadline; the SDK
                             treats this as fail-closed / deny)

granted / denied / timeout are terminal — once an approval is decided it never
changes again. A decision arriving for an already-terminal approval (a late
Slack click after the SDK already timed out, a double-click) is rejected by
is_valid_transition() rather than silently overwriting the recorded outcome.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Set


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    GRANTED = "granted"
    DENIED = "denied"
    TIMEOUT = "timeout"


# The three terminal outcomes a pending approval can resolve to.
DECISION_STATUSES: Set[ApprovalStatus] = {
    ApprovalStatus.GRANTED,
    ApprovalStatus.DENIED,
    ApprovalStatus.TIMEOUT,
}

TERMINAL_STATUSES: Set[ApprovalStatus] = set(DECISION_STATUSES)

# Only pending has any outgoing transitions; terminal states have none.
_LEGAL_TRANSITIONS: Dict[ApprovalStatus, Set[ApprovalStatus]] = {
    ApprovalStatus.PENDING: set(DECISION_STATUSES),
    ApprovalStatus.GRANTED: set(),
    ApprovalStatus.DENIED: set(),
    ApprovalStatus.TIMEOUT: set(),
}


def is_terminal(status: ApprovalStatus) -> bool:
    return status in TERMINAL_STATUSES


def is_valid_transition(current: ApprovalStatus, target: ApprovalStatus) -> bool:
    """True iff an approval in `current` may legally move to `target`."""
    return target in _LEGAL_TRANSITIONS.get(current, set())


def coerce_status(value: str) -> ApprovalStatus:
    """Parse a status string, raising ValueError on anything unknown rather
    than letting a bad value flow into the DB as a raw string."""
    return ApprovalStatus(value)
