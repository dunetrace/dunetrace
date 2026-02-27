"""
dunetrace/run_context.py

RunContext is what the developer holds during a run.
It provides the emit helpers (tool_called, llm_called, etc.)
and accumulates RunState for the local sidecar detectors.
"""
from __future__ import annotations

import uuid
import time
from typing import TYPE_CHECKING, Optional, Any, Dict

from dunetrace.models import (
    AgentEvent, EventType, RunState,
    ToolCall, RetrievalResult, hash_content,
)

if TYPE_CHECKING:
    from dunetrace.client import Dunetrace


class RunContext:
    """
    Thin wrapper around a single agent run.
    Developer calls methods on this; all events go through _emit.
    """

    def __init__(
        self,
        client:          "Dunetrace",
        agent_id:        str,
        agent_version:   str,
        available_tools: list,
        input_text_hash: str,
        parent_run_id:   Optional[str] = None,
    ):
        self._client       = client
        self.run_id        = str(uuid.uuid4())
        self.agent_id      = agent_id
        self.agent_version = agent_version
        self.step          = 0
        self.exit_reason: Optional[str] = None

        self.state = RunState(
            run_id=self.run_id,
            agent_id=agent_id,
            agent_version=agent_version,
            available_tools=available_tools,
            input_text_hash=input_text_hash,
        )
        self._parent_run_id = parent_run_id

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
    ) -> None:
        self._emit(EventType.LLM_RESPONDED, {
            "completion_tokens": completion_tokens,
            "latency_ms":        latency_ms,
            "finish_reason":     finish_reason,
            "output_hash":       output_hash,
        })

    # ── Tool hooks ─────────────────────────────────────────────────────────────

    def tool_called(self, tool_name: str, args: Dict[str, Any] = None) -> None:
        args = args or {}
        args_hash = hash_content(str(args))
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
    ) -> None:
        self._emit(EventType.TOOL_RESPONDED, {
            "tool_name":     tool_name,
            "success":       success,
            "output_length": output_length,
            "latency_ms":    latency_ms,
        })

    # ── Retrieval hooks (RAG) ──────────────────────────────────────────────────

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

    def final_answer(self) -> None:
        """Call when the agent produces its final answer."""
        self.exit_reason = "final_answer"
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
