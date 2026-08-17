"""
Rebuilds a RunState from raw event rows — the inverse of what the SDK emits.

Lives beside the models it reconstructs, and depends on nothing but
``dunetrace.models``, so every service that already has the SDK on its path
shares ONE implementation. It previously existed as two hand-copied forks
(detector_svc and api_svc), and they had drifted: the API copy had silently lost
memory events, the system prompt, tool output, retrieval content and LLM output
text. Replay ran on that copy, so five detectors that read those fields reported
"resolved" for any modification — including a no-op — which made the product's
counterfactual feature quietly wrong.

Both services now re-export from here. Add a field to RunState and there is one
place that has to learn to rebuild it.
"""

from __future__ import annotations

from typing import Any, Optional

from dunetrace.models import (
    AgentEvent,
    EventType,
    RunState,
    ToolCall,
    LlmCall,
    MemoryEvent,
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

    # Correlating llm.called with its llm.responded.
    #
    # The call is appended to state.llm_calls as soon as llm.called arrives and
    # the response is back-filled onto it later, which reconstructs exactly what
    # the SDK's own RunState holds: calls in call order, and a call that never
    # got a response still present with empty response fields. A streamed call
    # the caller abandoned therefore stops disappearing from the run.
    #
    # Pairing is by call_id (payload field, the call's index within the run).
    # Ordering cannot do it: a streamed llm.responded lands whenever the caller
    # drains the stream, so two overlapping streams emit
    # called(A), called(B), responded(A), responded(B) and popping the most
    # recent pending call would give B's response to A — swapping model against
    # tokens, latency and cost. Events without call_id (manual callers, older
    # SDKs) fall back to that pop, which is correct for them because their
    # called/responded pairs are always adjacent.
    _pending_by_id: dict[int, int] = {}  # call_id -> index into state.llm_calls
    _pending_order: list[int] = []  # indices awaiting a response, oldest first

    for raw in events:
        event_type = raw["event_type"]
        payload = raw.get("payload") or {}
        step_index = raw.get("step_index", 0)

        # run.started - extract available tools and input text
        if event_type == "run.started":
            state.available_tools = payload.get("tools", [])
            state.input_text = payload.get("input_text")
            state.system_prompt = payload.get("system_prompt") or None

        # run.completed - record exit reason
        elif event_type == "run.completed":
            state.exit_reason = payload.get("exit_reason", "completed")

        # run.errored
        elif event_type == "run.errored":
            state.exit_reason = "error"

        # llm.called - record the call, response fields filled in later
        elif event_type == "llm.called":
            call_id = payload.get("call_id")
            new_index = len(state.llm_calls)
            state.llm_calls.append(
                LlmCall(
                    model=payload.get("model", "unknown"),
                    prompt_tokens=payload.get("prompt_tokens"),
                    finish_reason=None,
                    latency_ms=None,
                    step_index=step_index,
                    timestamp=raw.get("timestamp", 0.0),
                    # Only the auto-instrumentation patchers set this. Stays
                    # None for manual callers and for events recorded before
                    # the field existed.
                    provider=payload.get("provider"),
                    call_id=call_id,
                )
            )
            _pending_order.append(new_index)
            if call_id is not None:
                _pending_by_id[call_id] = new_index

        # llm.responded - back-fill the call it belongs to
        elif event_type == "llm.responded":
            call_id = payload.get("call_id")
            index: Optional[int] = (
                _pending_by_id.pop(call_id) if call_id in _pending_by_id else None
            )
            if index is None and _pending_order:
                # No id to match on: the adjacent-pair case, so the most recent
                # outstanding call is the right one.
                index = _pending_order[-1]
            if index is not None and index in _pending_order:
                _pending_order.remove(index)

            if index is None:
                # A response with no call at all — a custom SDK that only emits
                # llm.responded. Record what it carries rather than dropping it.
                state.llm_calls.append(
                    LlmCall(
                        model=payload.get("model", "unknown"),
                        prompt_tokens=payload.get("prompt_tokens"),
                        finish_reason=payload.get("finish_reason"),
                        latency_ms=payload.get("latency_ms"),
                        output_length=payload.get("output_length"),
                        completion_tokens=payload.get("completion_tokens"),
                        reasoning_tokens=payload.get("reasoning_tokens"),
                        output_text=payload.get("output") or None,
                        step_index=step_index,
                        timestamp=raw.get("timestamp", 0.0),
                    )
                )
            else:
                call = state.llm_calls[index]
                call.finish_reason = payload.get("finish_reason")
                call.latency_ms = payload.get("latency_ms")
                call.output_length = payload.get("output_length")
                call.completion_tokens = payload.get("completion_tokens")
                # Emitted by the SDK and carried on LlmCall, but previously
                # never rebuilt here — so COST_SPIKE summed 0 reasoning
                # tokens for every reasoning model.
                call.reasoning_tokens = payload.get("reasoning_tokens")
                call.output_text = payload.get("output") or None
                # prompt_tokens come from llm.called in our SDK. Fall back to
                # llm.responded in case a custom SDK puts them there — and the
                # streaming patchers deliberately send the exact count here,
                # overriding llm_called's estimate.
                if payload.get("prompt_tokens"):
                    call.prompt_tokens = payload.get("prompt_tokens")

        # tool.called - append to tool_calls list
        elif event_type == "tool.called":
            tool_name = payload.get("tool_name", "unknown")
            state.tool_calls.append(
                ToolCall(
                    tool_name=tool_name,
                    args=payload.get("args", ""),
                    step_index=step_index,
                    timestamp=raw.get("timestamp", 0.0),
                    # Set only by the OTLP path, which truncates `args`.
                    args_length=payload.get("args_length"),
                )
            )

        # tool.responded - backfill success onto the most recent unmatched ToolCall
        # matching by tool_name (not step_index) because tool.called and tool.responded
        # have consecutive step indices in the SDK, not the same step_index.
        elif event_type == "tool.responded":
            success = payload.get("success")
            tool_name = payload.get("tool_name", "")
            if success is not None:
                for tc in reversed(state.tool_calls):
                    if tc.tool_name == tool_name and tc.success is None:
                        tc.success = bool(success)
                        tc.error = payload.get("error")
                        tc.output = payload.get("output") or None
                        # Prefer the reported length; fall back to the reconstructed
                        # output only when it wasn't reported. `is not None` (not
                        # truthiness) so a genuinely-reported 0 isn't recomputed.
                        output_length = payload.get("output_length")
                        if output_length is not None:
                            tc.output_length = output_length
                        else:
                            tc.output_length = len(tc.output) if tc.output else 0
                        break

        # retrieval.responded - append to retrievals list
        elif event_type == "retrieval.responded":
            state.retrievals.append(
                RetrievalResult(
                    index_name=payload.get("index_name", "unknown"),
                    result_count=payload.get("result_count", 0),
                    top_score=payload.get("top_score"),
                    step_index=step_index,
                    content=payload.get("content") or None,
                )
            )

        # memory.written / read / cleared - reconstruct the typed memory view the
        # MEMORY_POISONING detector consumes. Content lives on the wire once (in
        # the event payload) and is referenced into memory_events here.
        elif event_type == "memory.written":
            state.memory_events.append(
                MemoryEvent(
                    op="written",
                    key=payload.get("key"),
                    value=payload.get("value"),
                    source=payload.get("source"),
                    step_index=step_index,
                    timestamp=raw.get("timestamp", 0.0),
                )
            )
        elif event_type == "memory.read":
            state.memory_events.append(
                MemoryEvent(
                    op="read",
                    key=payload.get("key"),
                    step_index=step_index,
                    timestamp=raw.get("timestamp", 0.0),
                )
            )
        elif event_type == "memory.cleared":
            state.memory_events.append(
                MemoryEvent(
                    op="cleared",
                    key=payload.get("key"),
                    step_index=step_index,
                    timestamp=raw.get("timestamp", 0.0),
                )
            )

        # Reconstruct AgentEvent for event list
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

    # step_durations_ms: wall-clock cost of the first gap at each step_index.
    # First-write-wins so that the initiating event (tool.called, llm.called)
    # captures the actual operation latency rather than being overwritten by the
    # near-zero gap between the responding event and the next step's first event.
    for i in range(len(state.events) - 1):
        gap_ms = int((state.events[i + 1].timestamp - state.events[i].timestamp) * 1000)
        step_idx = state.events[i].step_index
        if (
            gap_ms >= 0 and step_idx not in state.step_durations_ms
        ):  # guard clock skew + first-write
            state.step_durations_ms[step_idx] = gap_ms

    return state
