"""
Groups semantic signals by a crude, deliberately unsophisticated hash of their
free-text reasoning — no embeddings, no semantic clustering, matching the
"no proprietary ML in v1" constraint. Same spirit as the existing
agent_input_hash_patterns analytics feature (services/api/api_svc/db/queries.py):
plain hash after light normalization, nothing fancier.

Known, disclosed limitation (see BACKLOG.md): two reasoning strings that
describe the same underlying problem but are worded differently, or whose
specifics (dates, values, tool args) appear before the text diverges, will
NOT reliably group together — this only catches near-identical openings.
Revisit with real customer data once false-positive rates from Phase 1.4's
other pieces (confidence floor, feedback) are known; embedding-based
clustering is explicitly out of scope for v1.
"""

from __future__ import annotations

import hashlib
import re

_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RE = re.compile(r"\s+")

# How much of the normalized reasoning contributes to the grouping key. Short
# enough to catch "same stated verdict, different specifics" (LLM reasoning
# conventionally states its verdict before drilling into per-run details);
# long enough that two unrelated findings rarely share an 80-char prefix.
_GROUPING_PREFIX_CHARS = 80


def normalize_root_cause(reasoning: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Not semantic —
    see module docstring."""
    lowered = (reasoning or "").lower()
    stripped = _NON_ALNUM_RE.sub(" ", lowered)
    return _WHITESPACE_RE.sub(" ", stripped).strip()


def root_cause_hash(reasoning: str) -> str:
    """Stable grouping key for a signal's reasoning text. Two signals with the
    same (agent_id, evaluator, root_cause_hash) are considered the same
    recurring pattern — see db.record_signal_group_membership."""
    normalized = normalize_root_cause(reasoning)[:_GROUPING_PREFIX_CHARS]
    # Non-cryptographic use: a stable grouping key for near-identical root-cause
    # strings, not a security digest. usedforsecurity=False satisfies scanners.
    return hashlib.md5(normalized.encode("utf-8"), usedforsecurity=False).hexdigest()
