"""
Dunetrace-native interface every external evaluation provider (Langfuse now;
LangSmith/Braintrust in 2.2/2.3) implements — the same "wrap the third-party
shape behind our own" seam Phase 1.3 used for DeepEval evaluators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ExternalEvaluation:
    """One evaluation result pulled from an external provider, already
    reduced to the fields worker.py needs — nothing provider-specific leaks
    past this point."""

    external_id: str  # for dedup (external_evaluation_processed)
    trace_id: str  # correlation key back to a Dunetrace run
    name: str  # the provider's own score/metric name, e.g. "hallucination"
    value: float | None  # numeric score, if the provider's dataType is numeric
    string_value: str | None  # categorical/boolean value, if not numeric
    comment: str | None
    timestamp: float
    # Click-through URL into the provider's own dashboard. None when a
    # provider can't construct one it's actually confident is correct (e.g.
    # LangSmith's feedback-list endpoint doesn't return enough to build a
    # verified private-dashboard URL — see providers/langsmith.py) — better
    # to omit than to fabricate a link that's likely wrong.
    source_url: str | None


class EvaluationProvider(Protocol):
    name: str

    async def fetch_new_evaluations(self, since: float) -> list[ExternalEvaluation]:
        """Returns evaluations with a timestamp >= since (epoch seconds).
        Callers should pass `since` with some backward overlap to tolerate
        the provider's own indexing lag — see worker.py's _OVERLAP_SECS —
        and rely on external_evaluation_processed for dedup, not on this
        method returning an exact non-overlapping set."""
        ...
