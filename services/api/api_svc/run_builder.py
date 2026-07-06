"""
Rebuilds a RunState from raw event rows. Bridges the flat DB records
with the typed RunState that detectors expect.

Copied from services/detector/detector_svc/run_builder.py — only depends on
dunetrace.models, so it works in the API service without adding detector to PYTHONPATH.
"""

from __future__ import annotations

from typing import Any

from dunetrace.models import (
    AgentEvent,
    EventType,
    RunState,
    ToolCall,
    LlmCall,
    RetrievalResult,
)


def build_run_state(events: list[dict]) -> RunState:
    """Reconstruct a RunState from raw event dicts. Handles missing/partial data — a partial RunState is still worth running detectors against."""
    if not events:
        raise ValueError("Cannot build RunState from empty event list")

    # Grab identity fields from first event (all events share run_id/agent_id)
    first = events[0]
    run_id = first["run_id"]
    agent_id = first["agent_id"]
    agent_version = first["agent_version"]

    state = RunState(
        run_id=run_id,
        agent_id=agent_id,
        agent_version=agent_version,
    )

    # Track pending llm.called data so we can merge with llm.responded.
    _pending_llm: list[dict] = []

    for raw in events:
        event_type = raw["event_type"]
        payload = raw.get("payload") or {}
        step_index = raw.get("step_index", 0)

        if event_type == "run.started":
            state.available_tools = payload.get("tools", [])
            state.input_text = payload.get("input_text")

        elif event_type == "run.completed":
            state.exit_reason = payload.get("exit_reason", "completed")

        elif event_type == "run.errored":
            state.exit_reason = "error"

        elif event_type == "llm.called":
            _pending_llm.append(
                {
                    "model": payload.get("model", "unknown"),
                    "prompt_tokens": payload.get("prompt_tokens"),
                    "step_index": step_index,
                    "timestamp": raw.get("timestamp", 0.0),
                }
            )

        elif event_type == "llm.responded":
            pending = _pending_llm.pop() if _pending_llm else {}
            prompt_tokens = pending.get("prompt_tokens") or payload.get("prompt_tokens")
            state.llm_calls.append(
                LlmCall(
                    model=pending.get("model", payload.get("model", "unknown")),
                    prompt_tokens=prompt_tokens,
                    finish_reason=payload.get("finish_reason"),
                    latency_ms=payload.get("latency_ms"),
                    output_length=payload.get("output_length"),
                    completion_tokens=payload.get("completion_tokens"),
                    step_index=pending.get("step_index", step_index),
                    timestamp=pending.get("timestamp", raw.get("timestamp", 0.0)),
                )
            )

        elif event_type == "tool.called":
            tool_name = payload.get("tool_name", "unknown")
            state.tool_calls.append(
                ToolCall(
                    tool_name=tool_name,
                    args=payload.get("args", ""),
                    step_index=step_index,
                    timestamp=raw.get("timestamp", 0.0),
                )
            )

        elif event_type == "tool.responded":
            success = payload.get("success")
            tool_name = payload.get("tool_name", "")
            if success is not None:
                for tc in reversed(state.tool_calls):
                    if tc.tool_name == tool_name and tc.success is None:
                        tc.success = bool(success)
                        tc.error = payload.get("error")
                        break

        elif event_type == "retrieval.responded":
            state.retrievals.append(
                RetrievalResult(
                    index_name=payload.get("index_name", "unknown"),
                    result_count=payload.get("result_count", 0),
                    top_score=payload.get("top_score"),
                    step_index=step_index,
                )
            )

        try:
            et = EventType(event_type)
        except ValueError:
            continue

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

    if state.events:
        state.current_step = max(e.step_index for e in state.events)

    for i in range(len(state.events) - 1):
        gap_ms = int((state.events[i + 1].timestamp - state.events[i].timestamp) * 1000)
        step_idx = state.events[i].step_index
        if gap_ms >= 0 and step_idx not in state.step_durations_ms:
            state.step_durations_ms[step_idx] = gap_ms

    return state
