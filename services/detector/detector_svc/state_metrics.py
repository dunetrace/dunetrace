"""
Per-run state metrics (Capability 3, Phase 3.3). Pure logic — no I/O.

The detector worker precomputes, for each completed run, how long the agent
spent in each state (thinking / acting / retrieving / waiting_approval) and
writes it to run_state_metrics. api_svc then aggregates those rows into
cross-run analytics (averages, trends, outliers) without re-reading raw events
for every query.

This reconstructs states with the same open/close event model as
api_svc/run_states.py::reconstruct_states — that module produces the per-run
timeline for the dashboard; this one produces per-state totals for analytics.
Kept intentionally in sync (see test_state_metrics.py, which mirrors that
module's cases). They're duplicated rather than shared because the two services
don't share application code — the same deliberate isolation the packs DDL and
schema-parity test already live with.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

_OPENERS = {
    "llm.called": "thinking",
    "tool.called": "acting",
    "retrieval.called": "retrieving",
    "approval.requested": "waiting_approval",
}
_CLOSERS = {
    "llm.responded": "thinking",
    "tool.responded": "acting",
    "retrieval.responded": "retrieving",
    "approval.granted": "waiting_approval",
    "approval.denied": "waiting_approval",
    "approval.timeout": "waiting_approval",
}
_TERMINALS = {"run.completed", "run.errored"}


def summarize_states(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return {"run_started_ts": float|None, "states": {state: {total_ms, count}}}.

    Same forgiving reconstruction as the per-run timeline: a missing response
    still contributes its span (up to the next opener or run end); orphan
    closers and point events are ignored; events are sorted defensively;
    negative spans (clock skew) are clamped to zero.
    """
    ordered = sorted(events, key=lambda e: (e.get("timestamp", 0.0), e.get("step_index", 0)))

    states: Dict[str, Dict[str, int]] = {}
    open_state: Optional[str] = None
    open_start: float = 0.0

    def _accumulate(end_ts: float) -> None:
        nonlocal open_state
        dur = end_ts - open_start
        if dur < 0:
            dur = 0.0
        bucket = states.setdefault(open_state, {"total_ms": 0, "count": 0})
        bucket["total_ms"] += round(dur * 1000)
        bucket["count"] += 1
        open_state = None

    for e in ordered:
        et = e.get("event_type", "")
        ts = e.get("timestamp", 0.0)
        if et in _OPENERS:
            if open_state is not None:
                _accumulate(ts)
            open_state = _OPENERS[et]
            open_start = ts
        elif et in _CLOSERS:
            if open_state is not None and open_state == _CLOSERS[et]:
                _accumulate(ts)
        elif et in _TERMINALS:
            if open_state is not None:
                _accumulate(ts)

    if open_state is not None:
        last_ts = ordered[-1].get("timestamp", open_start) if ordered else open_start
        _accumulate(last_ts)

    run_started_ts = None
    for e in ordered:
        if e.get("event_type") == "run.started":
            run_started_ts = e.get("timestamp")
            break
    if run_started_ts is None and ordered:
        run_started_ts = ordered[0].get("timestamp")

    return {"run_started_ts": run_started_ts, "states": states}
