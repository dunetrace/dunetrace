"""
LangChain callback handler. Plug it into any agent and it auto-instruments everything:

    from dunetrace import Dunetrace
    from dunetrace.integrations.langchain import DunetraceCallbackHandler

    dt = Dunetrace()
    callback = DunetraceCallbackHandler(dt, agent_id="my-agent")

    agent = create_agent(llm, tools, system_prompt="...")
    result = agent.invoke(
        {"messages": [("human", user_input)]},
        config={"callbacks": [callback]},
    )

No changes to agent code needed. Works with LangChain 1.x and LangGraph.
For older AgentExecutor setups, pass the handler to AgentExecutor(callbacks=[...]) instead.
"""
from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

try:
    try:
        from langchain_core.callbacks.base import BaseCallbackHandler  # langchain >= 0.2
    except ImportError:
        from langchain.callbacks.base import BaseCallbackHandler  # langchain < 0.2
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    BaseCallbackHandler = object  # type: ignore[assignment,misc]

from dunetrace.models import hash_content, agent_version as calc_version

if TYPE_CHECKING:
    from dunetrace.client import Dunetrace


class DunetraceCallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
    """
    LangChain callback handler for Dunetrace.

    Translates LangChain events into the Dunetrace canonical event schema.
    ``on_chain_start`` fires for every chain in LangChain (AgentExecutor,
    LLMChain, tool chains, etc.). We guard against this by only treating the
    FIRST ``on_chain_start`` per ``invoke()`` as the real run start, using
    LangChain's run_id from kwargs to identify it.
    """

    def __init__(
        self,
        client:        "Dunetrace",
        agent_id:      str,
        system_prompt: str = "",
        model:         str = "unknown",
        tools:         Optional[List[str]] = None,
    ) -> None:
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain is not installed. "
                "Run: pip install 'dunetrace[langchain]'"
            )
        super().__init__()
        self._client   = client
        self._agent_id = agent_id
        self._version  = calc_version(system_prompt, model, tools or [])
        self._tools    = tools or []

        self._run_id:         Optional[str]   = None
        self._root_lc_run_id: Optional[str]   = None
        self._step:           int             = 0
        self._llm_start_time: Optional[float] = None

    # ── LangChain hooks ───────────────────────────────────────────────────────

    def on_chain_start(self, serialized: Dict, inputs: Dict, **kwargs: Any) -> None:
        lc_run_id = str(kwargs.get("run_id", ""))
        if self._root_lc_run_id is not None:
            return  # sub-chain start — ignore

        self._root_lc_run_id = lc_run_id
        self._run_id = str(uuid.uuid4())
        self._step   = 0

        # AgentExecutor passes {"input": "..."}, LangGraph passes {"messages": [...]}
        user_input = str(inputs.get("input", ""))
        if not user_input and "messages" in inputs:
            msgs = inputs["messages"]
            last = msgs[-1] if msgs else None
            if isinstance(last, (list, tuple)):
                user_input = str(last[1]) if len(last) > 1 else ""
            elif hasattr(last, "content"):
                user_input = str(last.content)

        from dunetrace.models import AgentEvent, EventType
        self._client._emit(AgentEvent(
            event_type=EventType.RUN_STARTED,
            run_id=self._run_id,
            agent_id=self._agent_id,
            agent_version=self._version,
            step_index=0,
            payload={
                "input_hash": hash_content(user_input),
                "tools":      self._tools,
            },
        ))

    def on_chain_end(self, outputs: Dict, **kwargs: Any) -> None:
        if str(kwargs.get("run_id", "")) != self._root_lc_run_id:
            return

        from dunetrace.models import AgentEvent, EventType
        self._client._emit(AgentEvent(
            event_type=EventType.RUN_COMPLETED,
            run_id=self._run_id,
            agent_id=self._agent_id,
            agent_version=self._version,
            step_index=self._step,
            payload={
                "exit_reason": "final_answer",
                "total_steps": self._step,
            },
        ))
        self._reset()

    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        if str(kwargs.get("run_id", "")) != self._root_lc_run_id:
            return

        from dunetrace.models import AgentEvent, EventType
        self._client._emit(AgentEvent(
            event_type=EventType.RUN_ERRORED,
            run_id=self._run_id,
            agent_id=self._agent_id,
            agent_version=self._version,
            step_index=self._step,
            payload={
                "error_type": type(error).__name__,
                "error_hash": hash_content(str(error)),
            },
        ))
        self._reset()

    def on_llm_start(self, serialized: Dict, prompts: List[str], **kwargs: Any) -> None:
        """Fires for text-completion LLMs. Chat models use on_chat_model_start."""
        if not self._run_id:
            return
        self._llm_start_time = time.time()
        self._step += 1

        from dunetrace.models import AgentEvent, EventType
        self._client._emit(AgentEvent(
            event_type=EventType.LLM_CALLED,
            run_id=self._run_id,
            agent_id=self._agent_id,
            agent_version=self._version,
            step_index=self._step,
            payload={"model": serialized.get("name", "unknown")},
        ))

    def on_chat_model_start(
        self, serialized: Dict, messages: List[List[Any]], **kwargs: Any
    ) -> None:
        """Fires for chat models. on_llm_start does NOT fire for these."""
        if not self._run_id:
            return
        self._llm_start_time = time.time()
        self._step += 1

        from dunetrace.models import AgentEvent, EventType
        self._client._emit(AgentEvent(
            event_type=EventType.LLM_CALLED,
            run_id=self._run_id,
            agent_id=self._agent_id,
            agent_version=self._version,
            step_index=self._step,
            payload={"model": serialized.get("name", "unknown")},
        ))

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        if not self._run_id:
            return

        latency_ms = 0
        if self._llm_start_time:
            latency_ms = int((time.time() - self._llm_start_time) * 1000)

        gen         = response.generations[0][0] if response.generations else None
        output_text = gen.text if gen else ""

        # Extract token usage.
        # LangChain ≤ 0.1.x: llm_output["token_usage"] with prompt_tokens / completion_tokens
        # LangChain 0.3.x LCEL: llm_output is None; usage in gen.message.usage_metadata
        prompt_tokens     = None
        completion_tokens = None
        if hasattr(response, "llm_output") and response.llm_output:
            usage             = response.llm_output.get("token_usage", {})
            prompt_tokens     = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
        if prompt_tokens is None and gen:
            msg  = getattr(gen, "message", None)
            meta = getattr(msg, "usage_metadata", None) if msg else None
            if meta:
                prompt_tokens     = meta.get("input_tokens")
                completion_tokens = meta.get("output_tokens")

        payload: Dict[str, Any] = {
            "finish_reason": (
                gen.generation_info.get("finish_reason", "stop")
                if gen and gen.generation_info else "stop"
            ),
            "output_hash":   hash_content(output_text),
            "output_length": len(output_text),
            "latency_ms":    latency_ms,
        }
        if prompt_tokens is not None:
            payload["prompt_tokens"]     = prompt_tokens
        if completion_tokens is not None:
            payload["completion_tokens"] = completion_tokens

        from dunetrace.models import AgentEvent, EventType
        self._client._emit(AgentEvent(
            event_type=EventType.LLM_RESPONDED,
            run_id=self._run_id,
            agent_id=self._agent_id,
            agent_version=self._version,
            step_index=self._step,
            payload=payload,
        ))

    def on_tool_start(self, serialized: Dict, input_str: str, **kwargs: Any) -> None:
        """Fires in LangGraph (prebuilt react agent). on_agent_action does NOT fire there."""
        if not self._run_id:
            return
        self._step += 1
        tool_name = serialized.get("name", kwargs.get("name", "unknown"))
        from dunetrace.models import AgentEvent, EventType
        self._client._emit(AgentEvent(
            event_type=EventType.TOOL_CALLED,
            run_id=self._run_id,
            agent_id=self._agent_id,
            agent_version=self._version,
            step_index=self._step,
            payload={
                "tool_name": tool_name,
                "args_hash": hash_content(input_str),
            },
        ))

    def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        """Fires in AgentExecutor (LangChain < 1.x). on_tool_start does NOT fire there."""
        if not self._run_id:
            return
        self._step += 1
        from dunetrace.models import AgentEvent, EventType
        self._client._emit(AgentEvent(
            event_type=EventType.TOOL_CALLED,
            run_id=self._run_id,
            agent_id=self._agent_id,
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
            agent_id=self._agent_id,
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
            agent_id=self._agent_id,
            agent_version=self._version,
            step_index=self._step,
            payload={
                "success":    False,
                "error_type": type(error).__name__,
            },
        ))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _reset(self) -> None:
        self._root_lc_run_id = None
        self._run_id         = None
        self._step           = 0
