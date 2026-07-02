"""
LangChain callback handler. Plug it into any agent and it auto-instruments everything:

    from dunetrace import Dunetrace
    from dunetrace.integrations.langchain import DunetraceCallbackHandler

    dt = Dunetrace()
    callback = DunetraceCallbackHandler(dt, agent_id="my-agent")

    agent = create_react_agent(llm, tools, prompt="...")
    result = agent.invoke(
        {"messages": [("human", user_input)]},
        config={"callbacks": [callback]},
    )

No changes to agent code needed. Works with LangChain 1.x, LangGraph, and ainvoke().
For older AgentExecutor setups, pass the handler to AgentExecutor(callbacks=[...]) instead.

Runtime policies registered via ``dt.add_policy(...)`` are evaluated on every
LLM/tool call routed through this handler, exactly as they are for the manual
``dt.run(...)`` context manager — a ``stop`` policy raises ``PolicyViolation``,
which propagates out of the callback and halts the LangChain/LangGraph run
before the tool/LLM call it would have guarded actually executes.

Thread-safe: a single handler instance can be shared across concurrent invoke() calls.
Each invocation is tracked by LangChain's root run_id, so parallel calls don't collide.

Note: LangChain's callback manager runs sync handlers in a thread-pool executor for
ainvoke(), so no async variants are needed here.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

try:
    try:
        from langchain_core.callbacks.base import (
            BaseCallbackHandler,
        )  # langchain >= 0.2
    except ImportError:
        from langchain.callbacks.base import BaseCallbackHandler  # langchain < 0.2
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    BaseCallbackHandler = object  # type: ignore[assignment,misc]

from dunetrace.context import _current_run
from dunetrace.models import hash_content, agent_version as calc_version
from dunetrace.policies import PolicyViolation

if TYPE_CHECKING:
    from dunetrace.client import Dunetrace

logger = logging.getLogger("dunetrace.langchain")


_STALE_RUN_SECS = 1800  # runs not completed after 30 min are pruned


@dataclass
class _RunCtx:
    """State for one active agent invocation.

    ``dt_ctx`` is a real ``RunContext`` (the same object ``dt.run(...)`` yields) —
    step/tool_call_count are exposed as properties over it rather than tracked
    separately, so LLM/tool events routed through it are policy-checked exactly
    like the manual instrumentation path.
    """

    run_id: str
    dt_ctx: Any  # RunContext — typed Any to avoid a hard import cycle at class-definition time
    llm_start_time: Optional[float] = None
    model: str = ""
    children: Set[str] = field(default_factory=set)  # child lc_run_ids
    start_time: float = field(default_factory=time.time)
    child_steps: Dict[str, int] = field(default_factory=dict)  # lc_run_id → step_index at call time
    child_tool_names: Dict[str, str] = field(
        default_factory=dict
    )  # lc_run_id → tool_name at call time
    ctx_token: Any = field(default=None, repr=False)  # _current_run reset token

    @property
    def step(self) -> int:
        return self.dt_ctx.step

    @property
    def tool_call_count(self) -> int:
        return len(self.dt_ctx.state.tool_calls)


class DunetraceCallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
    """
    LangChain callback handler that translates LangChain events into Dunetrace events.

    ``on_chain_start`` fires for every chain node (AgentExecutor, LLMChain, tool chains, etc.).
    Only the outermost chain per invocation is treated as the Dunetrace run boundary, identified
    by LangChain's root run_id. Sub-chain events are attributed back to the root automatically.

    Thread-safe: one instance may be shared across concurrent invocations. Each LangChain root
    run_id maps to an independent _RunCtx, so concurrent calls don't collide.
    """

    # LangChain's callback manager logs-and-swallows exceptions raised inside a
    # handler unless raise_error is True (see callbacks/manager.py::handle_event).
    # Our hooks already swallow anything unexpected internally (see _safe_call) —
    # the only exception ever allowed to escape a hook is PolicyViolation, which
    # is a deliberate "stop this run" signal from a 'stop' policy and must reach
    # the caller of agent.invoke()/.stream().
    raise_error = True

    def __init__(
        self,
        client: "Dunetrace",
        agent_id: str,
        system_prompt: str = "",
        model: str = "unknown",
        tools: Optional[List[str]] = None,
    ) -> None:
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError("langchain is not installed. Run: pip install 'dunetrace[langchain]'")
        super().__init__()
        self._client = client
        self._agent_id = agent_id
        self._model = model
        self._version = calc_version(system_prompt, model, tools or [])
        self._tools = tools or []

        # Per-invocation state, keyed by LangChain root run_id.
        self._lock: Lock = Lock()
        self._runs: Dict[str, _RunCtx] = {}  # root_lc_run_id → ctx
        self._lc_parent: Dict[str, str] = {}  # any_lc_run_id  → root_lc_run_id
        self._last_run_id: Optional[str] = None  # Dunetrace run_id of the last completed root run

    @property
    def last_run_id(self) -> Optional[str]:
        """Dunetrace run_id of the most recently completed root invocation."""
        return self._last_run_id

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ctx(self, kwargs: dict) -> Optional[_RunCtx]:
        """Look up the RunCtx for this callback by walking parent_run_id → root."""
        with self._lock:
            for key in ("parent_run_id", "run_id"):
                lc_id = str(kwargs.get(key) or "")
                if not lc_id:
                    continue
                root = self._lc_parent.get(lc_id) or (lc_id if lc_id in self._runs else None)
                if root and root in self._runs:
                    return self._runs[root]
        return None

    def _register_child(self, child_lc_id: str, parent_lc_id: str) -> None:
        """Map a sub-chain/tool run_id back to its root so _ctx() can find it later."""
        with self._lock:
            root = self._lc_parent.get(parent_lc_id, parent_lc_id)
            if root in self._runs:
                self._lc_parent[child_lc_id] = root
                self._runs[root].children.add(child_lc_id)

    def _safe_emit(
        self, event_type: Any, ctx: _RunCtx, payload: dict, step: Optional[int] = None
    ) -> None:
        """Emit one run-lifecycle/retrieval event, swallowing any exception so the
        agent is never broken. Only used for events with no policy semantics
        (RUN_STARTED/COMPLETED/ERRORED). LLM/tool events go through _safe_call
        instead so policies registered via add_policy() actually get evaluated.
        """
        try:
            from dunetrace.models import AgentEvent

            self._client._emit(
                AgentEvent(
                    event_type=event_type,
                    run_id=ctx.run_id,
                    agent_id=self._agent_id,
                    agent_version=self._version,
                    step_index=step if step is not None else ctx.step,
                    payload=payload,
                )
            )
        except Exception as exc:
            logger.warning("Dunetrace: failed to emit %s: %s", event_type, exc)

    def _safe_call(self, fn, *args: Any, **kwargs: Any) -> None:
        """Call a RunContext hook (tool_called/llm_called/tool_responded/...).

        These are the calls that make policies registered via add_policy() take
        effect — RunContext evaluates policies internally after each one and
        raises PolicyViolation for a 'stop' action. That exception is a
        deliberate control-flow signal and must propagate. Any other exception
        is swallowed (with a warning) so a bug in Dunetrace itself can never
        break the wrapped agent.
        """
        try:
            fn(*args, **kwargs)
        except PolicyViolation:
            raise
        except Exception as exc:
            logger.warning("Dunetrace: %s failed: %s", getattr(fn, "__name__", fn), exc)

    # ── LangChain hooks ───────────────────────────────────────────────────────

    def on_chain_start(self, serialized: Dict, inputs: Dict, **kwargs: Any) -> None:
        try:
            lc_run_id = str(kwargs.get("run_id") or "")
            parent_lc_id = str(kwargs.get("parent_run_id") or "")

            with self._lock:
                is_root = not parent_lc_id or (
                    parent_lc_id not in self._runs and parent_lc_id not in self._lc_parent
                )

            if not is_root:
                self._register_child(lc_run_id, parent_lc_id)
                return

            # Prune stale runs before adding a new one.
            self._sweep_stale()

            user_input = str(inputs.get("input", ""))
            if not user_input and "messages" in inputs:
                msgs = inputs["messages"]
                last = msgs[-1] if msgs else None
                if isinstance(last, (list, tuple)):
                    user_input = str(last[1]) if len(last) > 1 else ""
                elif last is not None and hasattr(last, "content"):
                    user_input = str(last.content)

            # A real RunContext — the same object dt.run(...) yields. Routing
            # LLM/tool events through it below (instead of emitting AgentEvents
            # directly) is what makes add_policy()'d policies actually evaluate.
            # Use the LangChain run_id as the Dunetrace run_id so it matches the
            # Langfuse trace_id for the same run.
            from dunetrace.run_context import RunContext

            dt_ctx = RunContext(
                client=self._client,
                agent_id=self._agent_id,
                agent_version=self._version,
                available_tools=self._tools,
                input_text_hash=hash_content(user_input) if user_input else "",
                run_id=lc_run_id or str(uuid.uuid4()),
            )

            ctx = _RunCtx(run_id=dt_ctx.run_id, dt_ctx=dt_ctx)
            with self._lock:
                self._runs[lc_run_id] = ctx
                self._lc_parent[lc_run_id] = lc_run_id

            # Expose the active run via get_current_run() for tool callbacks.
            ctx.ctx_token = _current_run.set(dt_ctx)

            from dunetrace.models import EventType

            self._safe_emit(
                EventType.RUN_STARTED,
                ctx,
                {
                    "input_hash": hash_content(user_input),
                    "tools": self._tools,
                    "model": self._model,
                },
            )
        except Exception as exc:
            logger.warning("Dunetrace: on_chain_start failed: %s", exc)

    def on_chain_end(self, outputs: Dict, **kwargs: Any) -> None:
        try:
            lc_run_id = str(kwargs.get("run_id") or "")
            with self._lock:
                ctx = self._runs.get(lc_run_id)  # only fires for roots
            if not ctx:
                return

            from dunetrace.models import EventType

            self._safe_emit(
                EventType.RUN_COMPLETED,
                ctx,
                {
                    "exit_reason": "final_answer",
                    "total_steps": ctx.step,
                    "tool_call_count": ctx.tool_call_count,
                },
            )
            self._last_run_id = ctx.run_id
            self._cleanup(lc_run_id)
            self._client.flush()
        except Exception as exc:
            logger.warning("Dunetrace: on_chain_end failed: %s", exc)

    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        try:
            lc_run_id = str(kwargs.get("run_id") or "")
            with self._lock:
                ctx = self._runs.get(lc_run_id)
            if not ctx:
                return

            from dunetrace.models import EventType

            payload: Dict[str, Any] = {
                "error_type": type(error).__name__,
                "error_hash": hash_content(str(error)),
            }
            if isinstance(error, PolicyViolation):
                # Mirror dt.run()'s own RUN_ERRORED payload shape for a stopped run.
                payload["exit_reason"] = "policy_violation"
                payload["policy_name"] = error.policy_name

            self._safe_emit(EventType.RUN_ERRORED, ctx, payload)
            self._cleanup(lc_run_id)
            self._client.flush()
        except Exception as exc:
            logger.warning("Dunetrace: on_chain_error failed: %s", exc)

    @staticmethod
    def _model_name(serialized: Dict, kwargs: Dict) -> str:
        """Extract the actual model string from LangChain's serialized dict.
        serialized['name'] is the class name (e.g. 'ChatOpenAI'), not the model.
        The real model lives in serialized['kwargs']['model_name'] or
        invocation_params['model_name'], or as a kwarg metadata field."""
        kw = serialized.get("kwargs") or {}
        model = (
            kw.get("model_name")
            or kw.get("model")
            or (kwargs.get("invocation_params") or {}).get("model_name")
            or serialized.get("name", "unknown")
        )
        return str(model)

    def on_llm_start(self, serialized: Dict, prompts: List[str], **kwargs: Any) -> None:
        """Fires for text-completion LLMs. Chat models use on_chat_model_start."""
        try:
            ctx = self._ctx(kwargs)
            if not ctx:
                return
            model = self._model_name(serialized, kwargs)
            if not ctx.model:
                ctx.model = model
            ctx.llm_start_time = time.time()
            self._safe_call(ctx.dt_ctx.llm_called, model)
        except PolicyViolation:
            raise
        except Exception as exc:
            logger.warning("Dunetrace: on_llm_start failed: %s", exc)

    def on_chat_model_start(
        self, serialized: Dict, messages: List[List[Any]], **kwargs: Any
    ) -> None:
        """Fires for chat models. on_llm_start does NOT fire for these."""
        try:
            ctx = self._ctx(kwargs)
            if not ctx:
                return
            model = self._model_name(serialized, kwargs)
            if not ctx.model:
                ctx.model = model
            ctx.llm_start_time = time.time()
            self._safe_call(ctx.dt_ctx.llm_called, model)
        except PolicyViolation:
            raise
        except Exception as exc:
            logger.warning("Dunetrace: on_chat_model_start failed: %s", exc)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            ctx = self._ctx(kwargs)
            if not ctx:
                return

            latency_ms = int((time.time() - ctx.llm_start_time) * 1000) if ctx.llm_start_time else 0
            ctx.llm_start_time = None

            gen = response.generations[0][0] if response.generations else None
            output_text = gen.text if gen else ""

            # Token counts: try llm_output (LangChain ≤0.1), then usage_metadata (0.3+).
            # Both may be absent for streaming runs depending on provider.
            prompt_tokens = completion_tokens = reasoning_tokens = None
            if hasattr(response, "llm_output") and response.llm_output:
                usage = response.llm_output.get("token_usage", {})
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                details = usage.get("completion_tokens_details") or {}
                reasoning_tokens = details.get("reasoning_tokens")
            if prompt_tokens is None and gen:
                msg = getattr(gen, "message", None)
                meta = getattr(msg, "usage_metadata", None) if msg else None
                if meta:
                    prompt_tokens = meta.get("input_tokens")
                    completion_tokens = meta.get("output_tokens")
                    details = meta.get("output_token_details") or {}
                    reasoning_tokens = details.get("reasoning")

            finish_reason = (
                gen.generation_info.get("finish_reason", "stop")
                if gen and gen.generation_info
                else "stop"
            )

            self._safe_call(
                ctx.dt_ctx.llm_responded,
                completion_tokens=completion_tokens or 0,
                reasoning_tokens=reasoning_tokens or 0,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
                output_hash=hash_content(output_text),
                output_length=len(output_text),
                prompt_tokens=prompt_tokens or 0,
            )
            self._client.flush()
        except PolicyViolation:
            raise
        except Exception as exc:
            logger.warning("Dunetrace: on_llm_end failed: %s", exc)

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        try:
            ctx = self._ctx(kwargs)
            if not ctx:
                return
            ctx.llm_start_time = None
            self._safe_call(
                ctx.dt_ctx.llm_responded,
                finish_reason="error",
                output_hash="",
                output_length=0,
                latency_ms=0,
                error=str(error),
            )
        except PolicyViolation:
            raise
        except Exception as exc:
            logger.warning("Dunetrace: on_llm_error failed: %s", exc)

    def on_tool_start(self, serialized: Dict, input_str: str, **kwargs: Any) -> None:
        """Fires in LangGraph (prebuilt react agent). on_agent_action does NOT fire there."""
        try:
            ctx = self._ctx(kwargs)
            if not ctx:
                return
            lc_run_id = str(kwargs.get("run_id") or "")
            if lc_run_id:
                self._register_child(lc_run_id, str(kwargs.get("parent_run_id") or ""))
            tool_name = serialized.get("name", kwargs.get("name", "unknown"))
            self._safe_call(ctx.dt_ctx.tool_called, tool_name, {"input": input_str})
            if lc_run_id:
                ctx.child_steps[lc_run_id] = ctx.step
                ctx.child_tool_names[lc_run_id] = tool_name
        except PolicyViolation:
            raise
        except Exception as exc:
            logger.warning("Dunetrace: on_tool_start failed: %s", exc)

    def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        """Fires in AgentExecutor (LangChain < 1.x). on_tool_start does NOT fire there."""
        try:
            ctx = self._ctx(kwargs)
            if not ctx:
                return
            self._safe_call(ctx.dt_ctx.tool_called, action.tool, {"input": action.tool_input})
        except PolicyViolation:
            raise
        except Exception as exc:
            logger.warning("Dunetrace: on_agent_action failed: %s", exc)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        try:
            ctx = self._ctx(kwargs)
            if not ctx:
                return
            lc_run_id = str(kwargs.get("run_id") or "")
            ctx.child_steps.pop(lc_run_id, None)
            tool_name = kwargs.get("name") or ctx.child_tool_names.pop(lc_run_id, "")

            # LangChain calls on_tool_end instead of on_tool_error when
            # handle_tool_error=True (AgentExecutor) or handle_tool_errors=True
            # (LangGraph ToolNode, the default). In that case the error is passed
            # as kwargs["error"] so we can still report success=False.
            error = kwargs.get("error")
            if error is not None:
                self._safe_call(
                    ctx.dt_ctx.tool_responded,
                    tool_name,
                    success=False,
                    error=str(error),
                )
            else:
                self._safe_call(
                    ctx.dt_ctx.tool_responded,
                    tool_name,
                    success=True,
                    output_length=len(str(output)),
                )
            self._client.flush()
        except PolicyViolation:
            raise
        except Exception as exc:
            logger.warning("Dunetrace: on_tool_end failed: %s", exc)

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        try:
            ctx = self._ctx(kwargs)
            if not ctx:
                return
            lc_run_id = str(kwargs.get("run_id") or "")
            ctx.child_steps.pop(lc_run_id, None)
            tool_name = kwargs.get("name") or ctx.child_tool_names.pop(lc_run_id, "")

            self._safe_call(
                ctx.dt_ctx.tool_responded,
                tool_name,
                success=False,
                error=str(error),
            )
            self._client.flush()
        except PolicyViolation:
            raise
        except Exception as exc:
            logger.warning("Dunetrace: on_tool_error failed: %s", exc)

    def on_retriever_start(self, serialized: Dict, query: str, **kwargs: Any) -> None:
        try:
            ctx = self._ctx(kwargs)
            if not ctx:
                return
            lc_run_id = str(kwargs.get("run_id") or "")
            if lc_run_id:
                self._register_child(lc_run_id, str(kwargs.get("parent_run_id") or ""))
            # Retrieval isn't a policy-evaluated trigger (see RunContext._emit's
            # advance-and-check list), so this stays on _safe_emit rather than
            # delegating to a dt_ctx method — but step now lives on the real
            # RunContext, so it's bumped there directly rather than on a local
            # counter (ctx.step has no setter; it's a read-through property).
            ctx.dt_ctx.step += 1
            if lc_run_id:
                ctx.child_steps[lc_run_id] = ctx.step
            index_name = serialized.get(
                "name",
                (
                    serialized.get("id", ["unknown"])[-1]
                    if isinstance(serialized.get("id"), list)
                    else "unknown"
                ),
            )
            from dunetrace.models import EventType

            self._safe_emit(
                EventType.RETRIEVAL_CALLED,
                ctx,
                {
                    "index_name": index_name,
                    "query_hash": hash_content(query),
                },
            )
        except Exception as exc:
            logger.warning("Dunetrace: on_retriever_start failed: %s", exc)

    def on_retriever_end(self, documents: Any, **kwargs: Any) -> None:
        try:
            ctx = self._ctx(kwargs)
            if not ctx:
                return
            lc_run_id = str(kwargs.get("run_id") or "")
            step = ctx.child_steps.pop(lc_run_id, None)
            docs = list(documents) if documents else []
            result_count = len(docs)

            # Extract best similarity score from document metadata if available.
            top_score: Optional[float] = None
            for doc in docs:
                meta = getattr(doc, "metadata", {}) or {}
                score = meta.get("score") or meta.get("relevance_score") or meta.get("similarity")
                if score is not None:
                    top_score = max(top_score or 0.0, float(score))

            from dunetrace.models import EventType

            self._safe_emit(
                EventType.RETRIEVAL_RESPONDED,
                ctx,
                {
                    "result_count": result_count,
                    "top_score": top_score,
                },
                step=step,
            )
        except Exception as exc:
            logger.warning("Dunetrace: on_retriever_end failed: %s", exc)

    def on_retriever_error(self, error: Exception, **kwargs: Any) -> None:
        try:
            ctx = self._ctx(kwargs)
            if not ctx:
                return
            lc_run_id = str(kwargs.get("run_id") or "")
            step = ctx.child_steps.pop(lc_run_id, None)
            from dunetrace.models import EventType

            self._safe_emit(
                EventType.RETRIEVAL_RESPONDED,
                ctx,
                {
                    "result_count": 0,
                    "top_score": None,
                    "error_hash": hash_content(str(error)),
                },
                step=step,
            )
        except Exception as exc:
            logger.warning("Dunetrace: on_retriever_error failed: %s", exc)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _cleanup(self, root_lc_run_id: str) -> None:
        """Remove all state for a completed or errored invocation. O(k) where k = child count."""
        with self._lock:
            ctx = self._runs.pop(root_lc_run_id, None)
            self._lc_parent.pop(root_lc_run_id, None)
            if ctx:
                for child_id in ctx.children:
                    self._lc_parent.pop(child_id, None)
        if ctx and ctx.ctx_token is not None:
            _current_run.reset(ctx.ctx_token)

    def _sweep_stale(self) -> None:
        """Remove runs that started more than _STALE_RUN_SECS ago and never completed.
        Called on each new on_chain_start so no background thread is needed."""
        cutoff = time.time() - _STALE_RUN_SECS
        with self._lock:
            stale = [k for k, ctx in self._runs.items() if ctx.start_time < cutoff]
        for root_id in stale:
            logger.warning(
                "Dunetrace: pruning stale run %s (started >%ds ago)",
                root_id,
                _STALE_RUN_SECS,
            )
            self._cleanup(root_id)

    def _reset(self) -> None:
        """Clear all state (e.g. for testing). Not needed in normal usage."""
        with self._lock:
            self._runs.clear()
            self._lc_parent.clear()
