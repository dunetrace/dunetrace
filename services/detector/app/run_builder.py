"""
services/detector/app/run_builder.py

Reconstructs a RunState from raw event rows fetched from the DB.
This is the bridge between flat DB records and the typed RunState
that detectors operate on.

Event rows come from the events table written by the ingest service.
The SDK models live in packages/sdk-py — this module imports from there.
"""
from __future__ import annotations

import sys
import os
from typing import Any

# Allow importing from the SDK package without installing it
_SDK_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../packages/sdk-py")
)
if _SDK_PATH not in sys.path:
    sys.path.insert(0, _SDK_PATH)

from dunetrace.models import (
    AgentEvent, EventType, RunState, ToolCall, LlmCall, RetrievalResult
)


def build_run_state(events: list[dict]) -> RunState:
    """
    Reconstruct a RunState from a list of raw event dicts.

    Handles missing/partial data gracefully — a partial RunState
    is still worth running detectors against.
    """
    if not events:
        raise ValueError("Cannot build RunState from empty event list")

    # Grab identity fields from first event (all events share run_id/agent_id)
    first = events[0]
    run_id        = first["run_id"]
    agent_id      = first["agent_id"]
    agent_version = first["agent_version"]

    state = RunState(
        run_id=run_id,
        agent_id=agent_id,
        agent_version=agent_version,
    )

    # Track pending llm.called data so we can merge finish_reason when llm.responded arrives
    _pending_llm: dict[int, dict] = {}

    for raw in events:
        event_type = raw["event_type"]
        payload    = raw.get("payload") or {}
        step_index = raw.get("step_index", 0)

        # ── run.started — extract available tools and input hash ──────────────
        if event_type == "run.started":
            state.available_tools  = payload.get("tools", [])
            state.input_text_hash  = payload.get("input_hash")

        # ── run.completed — record exit reason ────────────────────────────────
        elif event_type == "run.completed":
            state.exit_reason = payload.get("exit_reason", "completed")

        # ── run.errored ───────────────────────────────────────────────────────
        elif event_type == "run.errored":
            state.exit_reason = "error"

        # ── llm.called — store pending call keyed by step_index ──────────────
        elif event_type == "llm.called":
            _pending_llm[step_index] = {
                "model":         payload.get("model", "unknown"),
                "prompt_tokens": payload.get("prompt_tokens"),
                "step_index":    step_index,
                "timestamp":     raw.get("timestamp", 0.0),
            }

        # ── llm.responded — merge with pending call, append LlmCall ──────────
        elif event_type == "llm.responded":
            pending = _pending_llm.pop(step_index, {})
            state.llm_calls.append(
                LlmCall(
                    model=pending.get("model", payload.get("model", "unknown")),
                    prompt_tokens=pending.get("prompt_tokens"),
                    finish_reason=payload.get("finish_reason"),
                    latency_ms=payload.get("latency_ms"),
                    step_index=step_index,
                    timestamp=pending.get("timestamp", raw.get("timestamp", 0.0)),
                )
            )

        # ── tool.called — append to tool_calls list ───────────────────────────
        elif event_type == "tool.called":
            tool_name = payload.get("tool_name", "unknown")
            state.tool_calls.append(
                ToolCall(
                    tool_name=tool_name,
                    args_hash=payload.get("args_hash", ""),
                    step_index=step_index,
                    timestamp=raw.get("timestamp", 0.0),
                )
            )

        # ── retrieval.responded — append to retrievals list ───────────────────
        elif event_type == "retrieval.responded":
            state.retrievals.append(
                RetrievalResult(
                    index_name=payload.get("index_name", "unknown"),
                    result_count=payload.get("result_count", 0),
                    top_score=payload.get("top_score"),
                    step_index=step_index,
                )
            )

        # ── Reconstruct AgentEvent for event list ─────────────────────────────
        try:
            et = EventType(event_type)
        except ValueError:
            continue  # unknown type — skip silently

        state.events.append(
            AgentEvent(
                event_type=et,
                run_id=run_id,
                agent_id=agent_id,
                agent_version=agent_version,
                step_index=step_index,
                timestamp=raw.get("timestamp", 0.0),
                payload=payload,
                parent_run_id=raw.get("parent_run_id"),
            )
        )

    # current_step = highest step_index seen
    if state.events:
        state.current_step = max(e.step_index for e in state.events)

    # step_durations_ms: gap (ms) from event[i].timestamp to event[i+1].timestamp
    # Keyed by event[i].step_index. Gives wall-clock cost of each step transition.
    for i in range(len(state.events) - 1):
        gap_ms = int((state.events[i + 1].timestamp - state.events[i].timestamp) * 1000)
        if gap_ms >= 0:  # guard against clock skew
            state.step_durations_ms[state.events[i].step_index] = gap_ms

    return state