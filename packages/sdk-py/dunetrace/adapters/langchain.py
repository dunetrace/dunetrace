"""
dunetrace/adapters/langchain.py

Drop-in LangChain callback. Zero code changes for LangChain users:

    from dunetrace.adapters.langchain import DunetraceCallback
    from dunetrace import Dunetrace

    vig = Dunetrace(api_key="...", agent_id="my-agent")

    agent = initialize_agent(
        tools, llm,
        callbacks=[DunetraceCallback(vig)]  # ← one line
    )
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

try:
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.schema import AgentAction, AgentFinish, LLMResult
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    BaseCallbackHandler = object  # type: ignore

from dunetrace.models import hash_content, agent_version as calc_version

if TYPE_CHECKING:
    from dunetrace.client import Dunetrace


class DunetraceCallback(BaseCallbackHandler):  # type: ignore[misc]
    """
    LangChain callback handler for Dunetrace.
    Translates LangChain events to the Dunetrace canonical event schema.

    on_chain_start fires for EVERY chain in LangChain (AgentExecutor,
    LLMChain, tool chains, etc). We guard against this by only treating
    the FIRST on_chain_start per invoke() as the real run start, using
    the LangChain-provided run_id from kwargs to identify it.
    """

    def __init__(
        self,
        client: "Dunetrace",
        system_prompt: str = "",
        model:         str = "unknown",
        tools:         Optional[List[str]] = None,
    ):
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain is not installed. "
                "Run: pip install dunetrace-sdk[langchain]"
            )
        super().__init__()
        self._client  = client
        self._version = calc_version(system_prompt, model, tools or [])
        self._run_id: Optional[str] = None      # our Dunetrace run_id
        self._root_lc_run_id: Optional[str] = None  # LangChain's root run id
        self._step    = 0
        self._tools   = tools or []
        self._llm_start_time: Optional[float] = None

    # ── LangChain Hooks ───────────────────────────────────────────────────────

    def on_chain_start(self, serialized: Dict, inputs: Dict, **kwargs: Any) -> None:
        # LangChain passes its own run_id in kwargs — use it to identify
        # the root chain (AgentExecutor). All sub-chains are ignored.
        lc_run_id = str(kwargs.get("run_id", ""))

        # If we already have a root run in flight, ignore sub-chain starts
        if self._root_lc_run_id is not None:
            return

        # This is the root chain — start a new Dunetrace run
        self._root_lc_run_id = lc_run_id
        self._run_id = str(uuid.uuid4())
        self._step   = 0
        user_input   = str(inputs.get("input", ""))

        from dunetrace.models import AgentEvent, EventType
        self._client._emit(AgentEvent(
            event_type=EventType.RUN_STARTED,
            run_id=self._run_id,
            agent_id=self._client.agent_id,
            agent_version=self._version,
            step_index=0,
            payload={
                "input_hash": hash_content(user_input),
                "tools":      self._tools,
            },
        ))

    def on_chain_end(self, outputs: Dict, **kwargs: Any) -> None:
        # Only emit run.completed when the ROOT chain ends
        lc_run_id = str(kwargs.get("run_id", ""))
        if lc_run_id != self._root_lc_run_id:
            return

        from dunetrace.models import AgentEvent, EventType
        self._client._emit(AgentEvent(
            event_type=EventType.RUN_COMPLETED,
            run_id=self._run_id,
            agent_id=self._client.agent_id,
            agent_version=self._version,
            step_index=self._step,
            payload={
                "exit_reason": "final_answer",
                "total_steps": self._step,
            },
        ))
        # Reset so next invoke() starts a fresh run
        self._root_lc_run_id = None
        self._run_id = None
        self._step   = 0

    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        # Only emit run.errored when the ROOT chain errors
        lc_run_id = str(kwargs.get("run_id", ""))
        if lc_run_id != self._root_lc_run_id:
            return

        from dunetrace.models import AgentEvent, EventType
        self._client._emit(AgentEvent(
            event_type=EventType.RUN_ERRORED,
            run_id=self._run_id,
            agent_id=self._client.agent_id,
            agent_version=self._version,
            step_index=self._step,
            payload={
                "error_type": type(error).__name__,
                "error_hash": hash_content(str(error)),
            },
        ))
        # Reset for next run
        self._root_lc_run_id = None
        self._run_id = None
        self._step   = 0

    def on_llm_start(self, serialized: Dict, prompts: List[str], **kwargs: Any) -> None:
        if not self._run_id:
            return
        self._llm_start_time = time.time()
        self._step += 1

        from dunetrace.models import AgentEvent, EventType
        self._client._emit(AgentEvent(
            event_type=EventType.LLM_CALLED,
            run_id=self._run_id,
            agent_id=self._client.agent_id,
            agent_version=self._version,
            step_index=self._step,
            payload={
                "model": serialized.get("name", "unknown"),
            },
        ))

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        if not self._run_id:
            return
        latency_ms = 0
        if self._llm_start_time:
            latency_ms = int((time.time() - self._llm_start_time) * 1000)

        gen = response.generations[0][0] if response.generations else None
        output_text = gen.text if gen else ""

        from dunetrace.models import AgentEvent, EventType
        self._client._emit(AgentEvent(
            event_type=EventType.LLM_RESPONDED,
            run_id=self._run_id,
            agent_id=self._client.agent_id,
            agent_version=self._version,
            step_index=self._step,
            payload={
                "finish_reason":  gen.generation_info.get("finish_reason", "stop") if gen and gen.generation_info else "stop",
                "output_hash":    hash_content(output_text),
                "output_length":  len(output_text),
                "latency_ms":     latency_ms,
            },
        ))

    def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        if not self._run_id:
            return
        self._step += 1
        from dunetrace.models import AgentEvent, EventType
        self._client._emit(AgentEvent(
            event_type=EventType.TOOL_CALLED,
            run_id=self._run_id,
            agent_id=self._client.agent_id,
            agent_version=self._version,
            step_index=self._step,
            payload={
                "tool_name": action.tool,
                "args_hash": hash_content(str(action.tool_input)),
            },
        ))

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        if not self._run_id:
            return
        from dunetrace.models import AgentEvent, EventType
        self._client._emit(AgentEvent(
            event_type=EventType.TOOL_RESPONDED,
            run_id=self._run_id,
            agent_id=self._client.agent_id,
            agent_version=self._version,
            step_index=self._step,
            payload={
                "success":       True,
                "output_length": len(str(output)),
            },
        ))

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        if not self._run_id:
            return
        from dunetrace.models import AgentEvent, EventType
        self._client._emit(AgentEvent(
            event_type=EventType.TOOL_RESPONDED,
            run_id=self._run_id,
            agent_id=self._client.agent_id,
            agent_version=self._version,
            step_index=self._step,
            payload={
                "success":    False,
                "error_type": type(error).__name__,
            },
        ))