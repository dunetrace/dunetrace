"""
API-key scopes — what a credential is allowed to do, not just who it is.

Authorization used to be a single flat capability: one key submitted events,
minted further keys, created `stop` policies that terminate live agent runs,
opened GitHub PRs, and granted its own approval requests.

That last one inverted human-in-the-loop approval entirely. Approval exists to
gate "sending a customer email, deleting data, wiring money" — but the agent
process being gated held the credential that could approve it, so the control
defended against everything except its actual threat model.

Scopes are deliberately coarse. Fine-grained permissions nobody configures are
worse than three that everyone understands.
"""

from __future__ import annotations

from typing import Iterable, Sequence

#: Submit events and read your org's own data. What an SDK/agent key gets.
INGEST = "ingest"
#: Decide pending approval requests. Deliberately NOT granted to agent keys —
#: this is the credential a human (dashboard, Slack) presents.
APPROVE = "approve"
#: Manage keys, policies, integrations. Operator credential.
ADMIN = "admin"

ALL_SCOPES: tuple[str, ...] = (INGEST, APPROVE, ADMIN)

#: What POST /v1/keys hands out unless the caller asks for more. An agent needs
#: exactly this, so the safe default is also the common case.
DEFAULT_SCOPES: tuple[str, ...] = (INGEST,)


def normalise(scopes: Iterable[str] | None) -> tuple[str, ...]:
    """Lower-cased, de-duplicated, ordered, unknown values dropped."""
    if not scopes:
        return DEFAULT_SCOPES
    seen = {s.strip().lower() for s in scopes if isinstance(s, str)}
    kept = tuple(s for s in ALL_SCOPES if s in seen)
    return kept or DEFAULT_SCOPES


def has_scope(granted: Sequence[str] | None, required: str) -> bool:
    """ADMIN implies every other scope; nothing else implies anything.

    An absent scope list means a key predating scopes, which is treated as
    ingest-only rather than as unrestricted — failing closed is the whole point.
    """
    held = set(granted or DEFAULT_SCOPES)
    return required in held or ADMIN in held
