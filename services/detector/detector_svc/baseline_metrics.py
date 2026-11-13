"""
Per-run scalars the P75 baselines aggregate.

Why this exists
---------------
The six P75 baselines used to re-derive their metric from raw ``events`` on
every run. Two problems followed from that:

1. **Retention capped them.** ``events`` is pruned at EVENT_RETENTION_DAYS, so
   an agent doing fewer than the minimum number of completed runs inside the
   retention window could never mature a baseline and sat on static thresholds
   forever — precisely the low-traffic agent an adaptive threshold helps most.

2. **The SQL drifted from the detector.** CONTEXT_BLOAT's baseline computed
   ``MAX/MIN`` prompt tokens while the detector computed positional
   ``last/first``, and read only ``llm.called`` while the detector honoured
   ``llm.responded``'s override. The effective threshold was therefore not the
   configured one, and nothing caught it because the two implementations had no
   reason to agree.

Both are fixed by computing the metrics **once, from the same RunState the
detectors consume**, and storing the scalars. A stored metric cannot drift from
the metric it is compared against, because there is only one implementation.

Every function returns ``None`` rather than a value when the run does not
qualify, mirroring the guard conditions in the corresponding detector — a run
that the detector would skip must not contribute to the baseline it is judged
against.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from dunetrace.models import RunState

# Mirrors ContextBloatDetector.MIN_CALLS / MIN_LAST_TOKENS and the first_tokens
# floor. Duplicated as module constants rather than imported from the detector
# instance because detectors.yml can retune a *live* detector, and a baseline
# whose population shifts with a config change is not a baseline.
_GROWTH_MIN_CALLS = 3
_GROWTH_MIN_FIRST_TOKENS = 10
_GROWTH_MIN_LAST_TOKENS = 2000

# Mirrors ReasoningSpinDetector.MIN_LLM_CALLS, same rationale.
_RATIO_MIN_LLM_CALLS = 5


def _p75(values: List[float]) -> Optional[float]:
    """Linear-interpolated 75th percentile — matches Postgres PERCENTILE_CONT,
    which is what the cross-run aggregate uses, so a run's own contribution is
    computed the same way as the aggregate over runs."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = 0.75 * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))


def _gaps_after(state: RunState, event_type: str) -> List[float]:
    """Wall-clock ms from each event of ``event_type`` to whatever came next.

    Mirrors run_builder's step_durations_ms: the cost of a step is the time
    until the next event, not a self-reported duration.
    """
    ordered = sorted(state.events, key=lambda e: (e.step_index, e.timestamp))
    gaps: List[float] = []
    for i, ev in enumerate(ordered[:-1]):
        if ev.event_type.value != event_type:
            continue
        gap_ms = (ordered[i + 1].timestamp - ev.timestamp) * 1000.0
        if gap_ms >= 0:
            gaps.append(gap_ms)
    return gaps


def _token_growth(state: RunState) -> Optional[float]:
    """Positional last/first prompt tokens — the exact quantity
    ContextBloatDetector compares against this baseline."""
    calls = [c for c in state.llm_calls if c.prompt_tokens is not None and c.prompt_tokens > 0]
    if len(calls) < _GROWTH_MIN_CALLS:
        return None
    first = calls[0].prompt_tokens or 0
    last = calls[-1].prompt_tokens or 0
    if first < _GROWTH_MIN_FIRST_TOKENS or last < _GROWTH_MIN_LAST_TOKENS:
        return None
    return last / first


def build_baseline_metrics(state: RunState) -> Dict[str, Optional[float]]:
    """The per-run scalars stored for later P75 aggregation.

    A ``None`` means "this run does not qualify for that baseline" and is stored
    as SQL NULL, which the aggregate skips — so a run can contribute to the
    token baseline while being excluded from the growth baseline.
    """
    total_tokens = sum(
        (c.prompt_tokens or 0) + (c.completion_tokens or 0) + (c.reasoning_tokens or 0)
        for c in state.llm_calls
    )

    timestamps = [e.timestamp for e in state.events]
    duration_s = (max(timestamps) - min(timestamps)) if len(timestamps) >= 2 else None

    llm_tool_ratio = (
        len(state.llm_calls) / max(len(state.tool_calls), 1)
        if len(state.llm_calls) >= _RATIO_MIN_LLM_CALLS
        else None
    )

    return {
        "step_count": float(max((e.step_index for e in state.events), default=0)),
        "gap_p75_tool_ms": _p75(_gaps_after(state, "tool.called")),
        "gap_p75_llm_ms": _p75(_gaps_after(state, "llm.called")),
        "token_growth": _token_growth(state),
        "llm_tool_ratio": llm_tool_ratio,
        "total_tokens": float(total_tokens) if total_tokens > 0 else None,
        "duration_s": duration_s if (duration_s or 0) > 0 else None,
    }
