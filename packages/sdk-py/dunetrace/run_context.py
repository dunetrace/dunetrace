"""
The object returned by `dt.run(...)`. Provides emit helpers like tool_called
and llm_called, and builds up a RunState for local detection.
"""
from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional

from dunetrace.models import (
    AgentEvent,
    EventType,
    ExternalSignal,
    RetrievalResult,
    RunState,
    ToolCall,
    hash_content,
)

if TYPE_CHECKING:
    from dunetrace.client import Dunetrace


class RunContext:
    """Thin wrapper around a single agent run."""

    def __init__(
        self,
        client:          "Dunetrace",
        agent_id:        str,
        agent_version:   str,
        available_tools: list,
        input_text_hash: str,
        parent_run_id:   Optional[str] = None,
    ) -> None:
        self._client       = client
        self.run_id        = str(uuid.uuid4())
        self.agent_id      = agent_id
        self.agent_version = agent_version
        self.step          = 0
        self.exit_reason:  Optional[str] = None
        self._parent_run_id = parent_run_id

        self.state = RunState(
            run_id=self.run_id,
            agent_id=agent_id,
            agent_version=agent_version,
            available_tools=available_tools,
            input_text_hash=input_text_hash,
        )

    # ── LLM hooks ─────────────────────────────────────────────────────────────

    def llm_called(self, model: str, prompt_tokens: int = 0) -> None:
        self._emit(EventType.LLM_CALLED, {
            "model":         model,
            "prompt_tokens": prompt_tokens,
        })

    def llm_responded(
        self,
        completion_tokens: int = 0,
        latency_ms:        int = 0,
        finish_reason:     str = "stop",
        output_hash:       str = "",
        output_length:     int = 0,
    ) -> None:
        self._emit(EventType.LLM_RESPONDED, {
            "completion_tokens": completion_tokens,
            "latency_ms":        latency_ms,
            "finish_reason":     finish_reason,
            "output_hash":       output_hash,
            "output_length":     output_length,
        })

    # ── Tool hooks ────────────────────────────────────────────────────────────

    def tool_called(self, tool_name: str, args: Optional[Dict[str, Any]] = None) -> None:
        args_hash = hash_content(str(args or {}))
        self.state.tool_calls.append(ToolCall(
            tool_name=tool_name,
            args_hash=args_hash,
            step_index=self.step,
            timestamp=time.time(),
        ))
        self._emit(EventType.TOOL_CALLED, {
            "tool_name": tool_name,
            "args_hash": args_hash,
        })

    def tool_responded(
        self,
        tool_name:     str,
        success:       bool = True,
        output_length: int  = 0,
        latency_ms:    int  = 0,
        error:         Optional[str] = None,
    ) -> None:
        error_hash = hash_content(error) if (not success and error) else None
        # Back-fill success and error_hash on the most recent matching ToolCall
        for tc in reversed(self.state.tool_calls):
            if tc.tool_name == tool_name and tc.success is None:
                tc.success    = success
                tc.error_hash = error_hash
                break
        payload: dict = {
            "tool_name":     tool_name,
            "success":       success,
            "output_length": output_length,
            "latency_ms":    latency_ms,
        }
        if error_hash:
            payload["error_hash"] = error_hash
        self._emit(EventType.TOOL_RESPONDED, payload)

    # ── Retrieval hooks (RAG) ─────────────────────────────────────────────────

    def retrieval_called(self, index_name: str, query_hash: str = "") -> None:
        self._emit(EventType.RETRIEVAL_CALLED, {
            "index_name": index_name,
            "query_hash": query_hash,
        })

    def retrieval_responded(
        self,
        index_name:   str,
        result_count: int,
        top_score:    Optional[float] = None,
        latency_ms:   int = 0,
    ) -> None:
        self.state.retrievals.append(RetrievalResult(
            index_name=index_name,
            result_count=result_count,
            top_score=top_score,
            step_index=self.step,
        ))
        self._emit(EventType.RETRIEVAL_RESPONDED, {
            "index_name":   index_name,
            "result_count": result_count,
            "top_score":    top_score,
            "latency_ms":   latency_ms,
        })

    # ── External signal hooks ─────────────────────────────────────────────────

    def external_signal(self, signal_name: str, source: str = "", **meta: Any) -> None:
        """
        Emit an infrastructure context event at the current agent step.

        Does not advance the step counter, the signal annotates whatever
        agent step is currently in progress, not a new one.

        Usage::

            run.external_signal("rate_limit", source="openai")
            run.external_signal("cache_miss", source="redis", key_prefix="emb:")
            run.external_signal("upstream_error", source="serp_api", http_status=503)

        Detectors (e.g. SLOW_STEP) correlate these signals with failures to
        provide richer evidence: "tool took 100s i.e. coincided with rate_limit
        from openai" rather than just "tool took 100s".
        """
        ts = time.time()
        self.state.external_signals.append(ExternalSignal(
            signal_name=signal_name,
            step_index=self.step,
            timestamp=ts,
            source=source,
            meta=dict(meta),
        ))
        payload: dict = {"signal_name": signal_name}
        if source:
            payload["source"] = source
        if meta:
            payload["meta"] = dict(meta)
        # Emit directly — bypass _emit() so step counter does not advance.
        event = AgentEvent(
            event_type=EventType.EXTERNAL_SIGNAL,
            run_id=self.run_id,
            agent_id=self.agent_id,
            agent_version=self.agent_version,
            step_index=self.step,
            timestamp=ts,
            payload=payload,
            parent_run_id=self._parent_run_id,
        )
        self.state.events.append(event)
        self._client._emit(event)

    def final_answer(self) -> None:
        """Call when the agent produces its final answer."""
        self.exit_reason       = "final_answer"
        self.state.exit_reason = "final_answer"

    # ── Internal ──────────────────────────────────────────────────────────────

    def _emit(self, event_type: EventType, payload: dict) -> None:
        self.step += 1
        event = AgentEvent(
            event_type=event_type,
            run_id=self.run_id,
            agent_id=self.agent_id,
            agent_version=self.agent_version,
            step_index=self.step,
            payload=payload,
            parent_run_id=self._parent_run_id,
        )
        self.state.events.append(event)
        self._client._emit(event)
