"""
Haystack 2.x tracer integration.

Register once before running any pipeline::

    import haystack.tracing
    from dunetrace import Dunetrace
    from dunetrace.integrations.haystack import DunetraceHaystackTracer

    dt = Dunetrace()
    haystack.tracing.enable_tracing(DunetraceHaystackTracer(dt, agent_id="my-pipeline"))

    pipeline = Pipeline()
    pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini"))
    result = pipeline.run({"llm": {"messages": [ChatMessage.from_user("Hello")]}})

    dt.shutdown()

Works with sync Pipeline, AsyncPipeline, and the high-level Agent component.
Compatible with haystack-ai >= 2.0.

Content tracing is enabled (so Haystack passes component I/O to us), but all text
is SHA-256 hashed before any network transmission — raw content never leaves the
process.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, List, Optional

try:
    import haystack.tracing as _hs_tracing  # noqa: F401

    _HAYSTACK_AVAILABLE = True
except ImportError:
    _HAYSTACK_AVAILABLE = False

from dunetrace.models import hash_content, agent_version as calc_version

if TYPE_CHECKING:
    from dunetrace.client import Dunetrace

logger = logging.getLogger("dunetrace.haystack")

_STALE_RUN_SECS = 1800

# ContextVar tracks the active span so current_span() works for nested component traces.
_CURRENT_SPAN: ContextVar[Optional["_DunetraceSpan"]] = ContextVar(
    "_dunetrace_hs_span", default=None
)


# ── component classification ─────────────────────────────────────────────────

_LLM_KEYWORDS = ("Generator", "LLM", "ChatModel")
_RETRIEVAL_KEYWORDS = ("Retriever",)
_TOOL_KEYWORDS = ("ToolInvoker", "ComponentTool", "SuperComponent")
# Components that wrap other components — don't emit events for them directly.
_WRAPPER_KEYWORDS = (
    "Agent",
    "Pipeline",
    "Builder",
    "Converter",
    "Adapter",
    "Joiner",
    "Writer",
    "Extractor",
    "Router",
    "Classifier",
)


def _classify_component(comp_type: str) -> str:
    """Return 'llm', 'retrieval', 'tool', or 'other'."""
    for kw in _WRAPPER_KEYWORDS:
        if kw in comp_type:
            return "other"
    for kw in _LLM_KEYWORDS:
        if kw in comp_type:
            return "llm"
    for kw in _RETRIEVAL_KEYWORDS:
        if kw in comp_type:
            return "retrieval"
    for kw in _TOOL_KEYWORDS:
        if kw in comp_type:
            return "tool"
    return "other"


def _model_hint_from_type(comp_type: str) -> str:
    """Best-effort model family from component class name."""
    t = comp_type.lower()
    if "openai" in t:
        return "gpt-4o"
    if "anthropic" in t:
        return "claude-3-5-sonnet"
    if "amazon" in t or "bedrock" in t:
        return "bedrock"
    if "cohere" in t:
        return "command-r"
    if "gemini" in t or "google" in t:
        return "gemini"
    if "mistral" in t:
        return "mistral"
    if "ollama" in t or "local" in t or "hugging" in t:
        return "local"
    return "unknown"


def _extract_tool_calls(inp: Any, output: Any, exc_type: Any) -> List[tuple]:
    """Return list of (tool_name, args_hash, success) from ToolInvoker input/output.

    Prefers output tool_messages (has per-tool error info). Falls back to input
    messages tool_calls when output is absent (e.g. ToolInvoker raised an exception).
    Returns [] when nothing can be extracted.
    """
    results: List[tuple] = []

    # Primary: output tool_messages carry ToolCallResult with origin tool name + args
    tool_messages = []
    if isinstance(output, dict):
        tool_messages = output.get("tool_messages") or []
    for msg in tool_messages:
        for tcr in getattr(msg, "tool_call_results", None) or []:
            origin = getattr(tcr, "origin", None)
            if origin:
                name = getattr(origin, "tool_name", "unknown")
                args = getattr(origin, "arguments", {})
                ok = not getattr(tcr, "error", True)
                results.append((name, hash_content(str(args)), ok))

    if results:
        return results

    # Fallback: use input messages when output is unavailable
    inp_messages = []
    if isinstance(inp, dict):
        inp_messages = inp.get("messages") or []
    for msg in inp_messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            name = getattr(tc, "tool_name", "unknown")
            args = getattr(tc, "arguments", {})
            results.append((name, hash_content(str(args)), exc_type is None))

    return results


def _extract_tokens(output: Any) -> tuple[Optional[int], Optional[int], str, str]:
    """Return (prompt_tokens, completion_tokens, finish_reason, model) from component output."""
    if not isinstance(output, dict):
        return None, None, "stop", ""
    replies = output.get("replies", [])
    if not replies:
        return None, None, "stop", ""
    reply = replies[0]
    meta: Dict[str, Any] = {}
    if hasattr(reply, "meta"):
        meta = reply.meta or {}
    elif isinstance(reply, dict):
        meta = reply.get("meta", {}) or {}

    usage = meta.get("usage") or meta.get("token_usage") or {}
    pt = usage.get("prompt_tokens") or usage.get("input_tokens") or usage.get("prompt_token_count")
    ct = (
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("generated_token_count")
    )
    finish = meta.get("finish_reason", "stop") or "stop"
    model = str(meta.get("model", "")) if meta.get("model") else ""
    return pt, ct, finish, model


def _extract_reply_text(output: Any) -> str:
    """Pull the first reply text out of a ChatGenerator output dict."""
    if not isinstance(output, dict):
        return ""
    replies = output.get("replies", [])
    if not replies:
        return ""
    reply = replies[0]
    if hasattr(reply, "text") and reply.text:
        return str(reply.text)
    if hasattr(reply, "content") and reply.content:
        return str(reply.content)
    if isinstance(reply, dict):
        return str(reply.get("text") or reply.get("content") or "")
    return ""


def _flatten_input(data: Any, depth: int = 0) -> str:
    """Best-effort extraction of user-visible text from pipeline input."""
    if depth > 5:
        return ""
    if isinstance(data, str):
        return data
    if hasattr(data, "text") and data.text:
        return str(data.text)
    if hasattr(data, "content") and data.content:
        return str(data.content)
    if isinstance(data, dict):
        for v in data.values():
            t = _flatten_input(v, depth + 1)
            if t:
                return t
    if isinstance(data, (list, tuple)):
        for item in data:
            t = _flatten_input(item, depth + 1)
            if t:
                return t
    return ""


# ── span + run state ──────────────────────────────────────────────────────────


@dataclass
class _RunState:
    run_id: str
    step: int = 0
    start_time: float = field(default_factory=time.time)


@dataclass
class _ProviderTokens:
    """Token data collected from provider-level child spans (e.g. haystack.openai.*)."""

    model: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: str = "stop"


class _DunetraceSpan:
    """Implements Haystack's Span protocol and emits Dunetrace events on __exit__."""

    def __init__(
        self,
        tracer: "DunetraceHaystackTracer",
        operation_name: str,
        tags: Dict[str, Any],
        parent_span: Optional["_DunetraceSpan"],
    ) -> None:
        self._tracer = tracer
        self._operation = operation_name
        self._tags: Dict[str, Any] = dict(tags)
        self._ctags: Dict[str, Any] = {}  # content tags (locally only, never transmitted)
        self._parent = parent_span
        self._start = time.time()
        # Set by root pipeline spans:
        self._run_state: Optional[_RunState] = None
        # Set by component.run spans on __enter__:
        self._comp_kind: str = "other"
        self._comp_name: str = ""
        self._comp_type: str = ""
        self._comp_step: int = 0
        # Filled in by provider child spans (haystack.openai.*, etc.):
        self._provider_tok: _ProviderTokens = _ProviderTokens()
        # ContextVar token for restoring on __exit__:
        self._cv_token: Any = None

    # ── Span protocol ────────────────────────────────────────────────────────

    def set_tag(self, key: str, value: Any) -> None:
        self._tags[key] = value
        self._maybe_propagate_provider_tags(key, value)

    def set_content_tag(self, key: str, value: Any) -> None:
        self._ctags[key] = value

    def raw_span(self) -> Any:
        return self._tags

    def get_correlation_data_for_logs(self) -> Dict[str, Any]:
        run_id = self._active_run_id()
        return {"dunetrace_run_id": run_id} if run_id else {}

    # ── context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "_DunetraceSpan":
        self._cv_token = _CURRENT_SPAN.set(self)
        try:
            self._on_enter()
        except Exception as exc:
            logger.warning("Dunetrace haystack: __enter__ error: %s", exc)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            self._on_exit(exc_type, exc_val)
        except Exception as exc:
            logger.warning("Dunetrace haystack: __exit__ error: %s", exc)
        finally:
            if self._cv_token is not None:
                _CURRENT_SPAN.reset(self._cv_token)

    # ── internal helpers ─────────────────────────────────────────────────────

    def _active_run_id(self) -> Optional[str]:
        if self._run_state:
            return self._run_state.run_id
        if self._parent:
            return self._parent._active_run_id()
        return None

    def _root_run_state(self) -> Optional[_RunState]:
        if self._run_state:
            return self._run_state
        if self._parent:
            return self._parent._root_run_state()
        return None

    def _maybe_propagate_provider_tags(self, key: str, value: Any) -> None:
        """Copy provider token/model tags up to the nearest component.run parent."""
        op = self._operation
        if not (
            op.startswith("haystack.openai.")
            or op.startswith("haystack.anthropic.")
            or op.startswith("haystack.amazon.")
            or op.startswith("haystack.cohere.")
            or op.startswith("haystack.google.")
            or op.startswith("haystack.mistral.")
            or op.startswith("haystack.ollama.")
        ):
            return
        parent = self._parent
        while parent:
            if parent._operation == "haystack.component.run":
                tok = parent._provider_tok
                if "response.usage.prompt_tokens" in key or "usage.prompt_tokens" in key:
                    tok.prompt_tokens = int(value) if value is not None else None
                elif "response.usage.completion_tokens" in key or "usage.completion_tokens" in key:
                    tok.completion_tokens = int(value) if value is not None else None
                elif "response.model" in key or ".model" in key:
                    if not tok.model:
                        tok.model = str(value)
                elif "finish_reason" in key:
                    tok.finish_reason = str(value) or "stop"
                break
            parent = parent._parent

    def _on_enter(self) -> None:
        t = self._tracer
        op = self._operation

        if op == "haystack.pipeline.run":
            if self._root_run_state() is not None:
                # Nested pipeline (e.g. inside Agent) — reuse the outer run.
                return

            t._sweep_stale()
            state = _RunState(run_id=str(uuid.uuid4()))
            self._run_state = state
            with t._lock:
                t._active_runs[id(self)] = state

            input_data = self._tags.get("haystack.pipeline.input_data") or {}
            user_input = _flatten_input(input_data)

            from dunetrace.models import AgentEvent, EventType

            t._client._emit(
                AgentEvent(
                    event_type=EventType.RUN_STARTED,
                    run_id=state.run_id,
                    agent_id=t._agent_id,
                    agent_version=t._version,
                    step_index=state.step,
                    payload={
                        "input_hash": hash_content(user_input),
                        "tools": t._tools,
                        "model": t._model,
                    },
                )
            )
            state.step += 1

        elif op == "haystack.component.run":
            state = self._root_run_state()
            if not state:
                return

            comp_type = self._tags.get("haystack.component.type", "")
            comp_name = self._tags.get("haystack.component.name", comp_type)
            kind = _classify_component(comp_type)

            self._comp_kind = kind
            self._comp_name = comp_name
            self._comp_type = comp_type
            self._comp_step = state.step
            state.step += 1

            from dunetrace.models import AgentEvent, EventType

            if kind == "llm":
                t._client._emit(
                    AgentEvent(
                        event_type=EventType.LLM_CALLED,
                        run_id=state.run_id,
                        agent_id=t._agent_id,
                        agent_version=t._version,
                        step_index=self._comp_step,
                        payload={"model": _model_hint_from_type(comp_type)},
                    )
                )
            elif kind == "retrieval":
                inp = self._tags.get("haystack.component.input") or {}
                query = _flatten_input(inp)
                t._client._emit(
                    AgentEvent(
                        event_type=EventType.RETRIEVAL_CALLED,
                        run_id=state.run_id,
                        agent_id=t._agent_id,
                        agent_version=t._version,
                        step_index=self._comp_step,
                        payload={
                            "index_name": comp_name,
                            "query_hash": hash_content(query),
                        },
                    )
                )
            elif kind == "tool":
                pass  # TOOL_CALLED deferred to _on_exit where actual tool names are known

    def _on_exit(self, exc_type: Any, exc_val: Any) -> None:
        t = self._tracer
        op = self._operation

        if op == "haystack.pipeline.run":
            state = self._run_state
            if not state:
                return  # nested pipeline — outer run handles it

            latency_ms = int((time.time() - state.start_time) * 1000)
            from dunetrace.models import AgentEvent, EventType

            if exc_type is not None:
                t._client._emit(
                    AgentEvent(
                        event_type=EventType.RUN_ERRORED,
                        run_id=state.run_id,
                        agent_id=t._agent_id,
                        agent_version=t._version,
                        step_index=state.step,
                        payload={
                            "error_type": exc_type.__name__ if exc_type else "unknown",
                            "error_hash": hash_content(str(exc_val)),
                        },
                    )
                )
            else:
                t._client._emit(
                    AgentEvent(
                        event_type=EventType.RUN_COMPLETED,
                        run_id=state.run_id,
                        agent_id=t._agent_id,
                        agent_version=t._version,
                        step_index=state.step,
                        payload={
                            "exit_reason": "final_answer",
                            "total_steps": state.step,
                            "latency_ms": latency_ms,
                        },
                    )
                )

            with t._lock:
                t._active_runs.pop(id(self), None)
            t._client.flush()

        elif op == "haystack.component.run":
            state = self._root_run_state()
            if not state or self._comp_kind == "other":
                return

            kind = self._comp_kind
            step = self._comp_step
            comp_name = self._comp_name
            latency_ms = int((time.time() - self._start) * 1000)
            tok = self._provider_tok
            from dunetrace.models import AgentEvent, EventType

            if kind == "llm":
                # Content tags are set by the pipeline after component.run() returns.
                output = self._ctags.get("haystack.component.output") or {}
                pt, ct, finish, model_from_reply = _extract_tokens(output)

                # Provider child spans (set_tag propagation) take precedence.
                pt = tok.prompt_tokens if tok.prompt_tokens is not None else pt
                ct = tok.completion_tokens if tok.completion_tokens is not None else ct
                finish = tok.finish_reason if tok.finish_reason != "stop" else finish
                model = tok.model or model_from_reply

                reply_text = _extract_reply_text(output)

                payload: Dict[str, Any] = {
                    "finish_reason": "error" if exc_type else (finish or "stop"),
                    "output_hash": hash_content(reply_text),
                    "output_length": len(reply_text),
                    "latency_ms": latency_ms,
                }
                if model:
                    payload["model"] = model
                if pt is not None:
                    payload["prompt_tokens"] = pt
                if ct is not None:
                    payload["completion_tokens"] = ct
                if exc_type is not None:
                    payload["error_hash"] = hash_content(str(exc_val))

                t._client._emit(
                    AgentEvent(
                        event_type=EventType.LLM_RESPONDED,
                        run_id=state.run_id,
                        agent_id=t._agent_id,
                        agent_version=t._version,
                        step_index=step,
                        payload=payload,
                    )
                )

            elif kind == "retrieval":
                output = self._ctags.get("haystack.component.output") or {}
                docs = output.get("documents", []) if isinstance(output, dict) else []
                result_count = len(docs) if docs else 0

                top_score: Optional[float] = None
                for doc in docs or []:
                    score = (
                        getattr(doc, "score", None)
                        if hasattr(doc, "score")
                        else (doc.get("score") if isinstance(doc, dict) else None)
                    )
                    if score is not None:
                        top_score = max(top_score or 0.0, float(score))

                t._client._emit(
                    AgentEvent(
                        event_type=EventType.RETRIEVAL_RESPONDED,
                        run_id=state.run_id,
                        agent_id=t._agent_id,
                        agent_version=t._version,
                        step_index=step,
                        payload={
                            "result_count": 0 if exc_type else result_count,
                            "top_score": None if exc_type else top_score,
                            **({"error_hash": hash_content(str(exc_val))} if exc_type else {}),
                        },
                    )
                )

            elif kind == "tool":
                output = self._ctags.get("haystack.component.output") or {}
                inp_data = self._ctags.get("haystack.component.input") or {}
                calls = _extract_tool_calls(inp_data, output, exc_type)

                if not calls:
                    # Fallback when ToolInvoker has no messages (direct tool component)
                    calls = [(comp_name, "", exc_type is None)]

                for tool_name, args_hash, success in calls:
                    t._client._emit(
                        AgentEvent(
                            event_type=EventType.TOOL_CALLED,
                            run_id=state.run_id,
                            agent_id=t._agent_id,
                            agent_version=t._version,
                            step_index=step,
                            payload={"tool_name": tool_name, "args_hash": args_hash},
                        )
                    )
                    t._client._emit(
                        AgentEvent(
                            event_type=EventType.TOOL_RESPONDED,
                            run_id=state.run_id,
                            agent_id=t._agent_id,
                            agent_version=t._version,
                            step_index=step,
                            payload={
                                "tool_name": tool_name,
                                "success": success,
                                "latency_ms": latency_ms,
                                **({"error_hash": hash_content(str(exc_val))} if exc_type else {}),
                            },
                        )
                    )

            t._client.flush()


# ── tracer ────────────────────────────────────────────────────────────────────


class DunetraceHaystackTracer:
    """
    Haystack 2.x tracer that emits Dunetrace events for pipeline runs, LLM calls,
    retrievals, and tool invocations.

    Register once before running any pipeline::

        import haystack.tracing
        haystack.tracing.enable_tracing(
            DunetraceHaystackTracer(dt, agent_id="my-pipeline")
        )

    All ``Pipeline.run()`` and ``AsyncPipeline.run()`` calls are automatically
    tracked from that point on. Works with the high-level ``Agent`` component too.
    """

    # Tell Haystack to call set_content_tag so we can see component outputs.
    # All text is hashed before any network transmission.
    is_content_tracing_enabled: bool = True

    def __init__(
        self,
        client: "Dunetrace",
        agent_id: str = "haystack-pipeline",
        system_prompt: str = "",
        model: str = "unknown",
        tools: Optional[List[str]] = None,
    ) -> None:
        if not _HAYSTACK_AVAILABLE:
            raise ImportError(
                "haystack-ai is not installed. Run: pip install 'dunetrace[haystack]'"
            )
        self._client = client
        self._agent_id = agent_id
        self._model = model
        self._tools = tools or []
        self._version = calc_version(system_prompt, model, self._tools)

        self._lock: Lock = Lock()
        self._active_runs: Dict[int, _RunState] = {}  # span id(obj) → _RunState

    # ── Tracer protocol ──────────────────────────────────────────────────────

    def trace(
        self,
        operation_name: str,
        tags: Optional[Dict[str, Any]] = None,
        parent_span: Optional[_DunetraceSpan] = None,
    ) -> _DunetraceSpan:
        """Return a span context manager for the given operation."""
        # Haystack's Pipeline.run() never passes parent_span even when called from
        # inside a component (e.g. Agent's internal pipeline). Fall back to the
        # ContextVar so nested pipelines correctly inherit the outer run state.
        if parent_span is None:
            parent_span = _CURRENT_SPAN.get()
        return _DunetraceSpan(self, operation_name, tags or {}, parent_span)

    def current_span(self) -> Optional[_DunetraceSpan]:
        """Return the span active in the current async/thread context, or None."""
        return _CURRENT_SPAN.get()

    # ── internal ─────────────────────────────────────────────────────────────

    def _sweep_stale(self) -> None:
        cutoff = time.time() - _STALE_RUN_SECS
        with self._lock:
            stale = [k for k, s in self._active_runs.items() if s.start_time < cutoff]
            for k in stale:
                logger.warning(
                    "Dunetrace haystack: pruning stale run %s",
                    self._active_runs[k].run_id,
                )
                del self._active_runs[k]
