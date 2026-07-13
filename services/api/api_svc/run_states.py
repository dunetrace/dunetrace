"""
State-machine reconstruction (Capability 3, Phase 3.1). Pure logic — no I/O.
Turns a run's raw event stream into an ordered list of the states the agent
occupied, each with a duration, for the dashboard's state timeline.

Agents don't emit "states" — they emit paired call/response events. A state is
the span between an opening event and its matching closing event:

    thinking          llm.called      → llm.responded
    acting            tool.called     → tool.responded
    retrieving        retrieval.called → retrieval.responded
    waiting_approval  approval.requested → approval.granted|denied|timeout

Point events (voice, external signals, policy triggers) aren't states and are
ignored here. Reconstruction is deliberately forgiving: a missing response (run
crashed mid-tool) still yields a segment, marked closed=False, running to the
run's end (or the next opening event) rather than being dropped.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# opening event_type → (state, payload key holding the human-readable label)
_OPENERS: Dict[str, tuple] = {
    "llm.called": ("thinking", "model"),
    "tool.called": ("acting", "tool_name"),
    "retrieval.called": ("retrieving", "index_name"),
    "approval.requested": ("waiting_approval", "tool_name"),
}

# closing event_type → the state it closes
_CLOSERS: Dict[str, str] = {
    "llm.responded": "thinking",
    "tool.responded": "acting",
    "retrieval.responded": "retrieving",
    "approval.granted": "waiting_approval",
    "approval.denied": "waiting_approval",
    "approval.timeout": "waiting_approval",
}

_TERMINALS = {"run.completed", "run.errored"}


def _label(payload: Dict[str, Any], key: str) -> str:
    v = payload.get(key)
    return str(v) if v else ""


def reconstruct_states(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return {"segments": [...], "summary": {...}} for a run's events.

    Each segment: {state, label, step_index, start_ts, end_ts, duration_ms,
    closed}. Segments are in start order. `closed` is False when the state
    never got its matching close event (crash / missing response) — the segment
    still gets a best-effort end time so the timeline stays contiguous.
    """
    # Sort defensively by (timestamp, step_index) — events usually arrive
    # ordered, but never assume it for a viz that reads durations off the order.
    ordered = sorted(events, key=lambda e: (e.get("timestamp", 0.0), e.get("step_index", 0)))

    segments: List[Dict[str, Any]] = []
    open_seg: Optional[Dict[str, Any]] = None

    def _close(end_ts: float, closed: bool) -> None:
        nonlocal open_seg
        assert open_seg is not None
        start = open_seg["start_ts"]
        # Guard clock skew — never emit a negative duration.
        open_seg["end_ts"] = end_ts if end_ts >= start else start
        # round, not int — float math like (1.2 - 1.0)*1000 = 199.99… would
        # otherwise truncate to 199ms.
        open_seg["duration_ms"] = round((open_seg["end_ts"] - start) * 1000)
        open_seg["closed"] = closed
        segments.append(open_seg)
        open_seg = None

    for e in ordered:
        et = e.get("event_type", "")
        ts = e.get("timestamp", 0.0)
        payload = e.get("payload") or {}

        if et in _OPENERS:
            if open_seg is not None:
                # Previous state never closed (missing response) — end it here.
                _close(ts, closed=False)
            state, label_key = _OPENERS[et]
            open_seg = {
                "state": state,
                "label": _label(payload, label_key),
                "step_index": e.get("step_index", 0),
                "start_ts": ts,
            }
        elif et in _CLOSERS:
            if open_seg is not None and open_seg["state"] == _CLOSERS[et]:
                _close(ts, closed=True)
            # a closer with no matching open state is ignored
        elif et in _TERMINALS:
            if open_seg is not None:
                _close(ts, closed=False)

    if open_seg is not None:
        # Ran off the end with a state still open and no terminal event.
        last_ts = (
            ordered[-1].get("timestamp", open_seg["start_ts"]) if ordered else open_seg["start_ts"]
        )
        _close(last_ts, closed=False)

    # Summary: per-state time totals + overall span.
    by_state: Dict[str, int] = {}
    for seg in segments:
        by_state[seg["state"]] = by_state.get(seg["state"], 0) + seg["duration_ms"]

    total_ms = 0
    if ordered:
        span = ordered[-1].get("timestamp", 0.0) - ordered[0].get("timestamp", 0.0)
        total_ms = round(span * 1000) if span > 0 else 0

    return {
        "segments": segments,
        "summary": {
            "total_ms": total_ms,
            "by_state": by_state,
            "segment_count": len(segments),
        },
    }
