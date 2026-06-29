"""
Dunetrace integration for the OpenAI Agents SDK (``openai-agents``).

The Agents SDK ships a first-class tracing interface (``add_trace_processor``),
so instrumentation is a drop-in trace processor — no monkey-patching and no
changes to your agent code. Each SDK *trace* becomes one Dunetrace run; the
spans inside it (generation/response, function tool, handoff) are translated
into ``llm.called`` / ``llm.responded`` / ``tool.called`` / ``tool.responded``
events. Handoffs map to ``tool.called`` / ``tool.responded`` with
``tool_name`` ``handoff:<to_agent>``.

Usage::

    from agents import Agent, Runner
    from dunetrace import Dunetrace
    from dunetrace.integrations.openai_agents import add_dunetrace_processor

    dt = Dunetrace()
    add_dunetrace_processor(dt, agent_id="my-agent", model="gpt-4o-mini")

    agent = Agent(name="Assistant", instructions="You are helpful.")
    result = Runner.run_sync(agent, "What is the capital of France?")

``add_dunetrace_processor`` registers the processor alongside any existing ones
(e.g. the default OpenAI exporter), so it composes with Langfuse, Logfire, etc.
To run it as the *only* processor, build a ``DunetraceTracingProcessor`` and
pass it to ``agents.set_trace_processors([...])`` yourself.

Thread/async-safe: a single processor instance handles concurrent runs. Each
SDK ``trace_id`` maps to an independent run context, so parallel runs and
sub-agent handoffs (which share a trace) don't collide.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

try:
    from agents.tracing import TracingProcessor, add_trace_processor

    _AGENTS_AVAILABLE = True
except ImportError:
    _AGENTS_AVAILABLE = False
    TracingProcessor = object  # type: ignore[assignment,misc]

from dunetrace.context import _current_run
from dunetrace.models import hash_content, agent_version as calc_version

if TYPE_CHECKING:
    from dunetrace.client import Dunetrace

logger = logging.getLogger("dunetrace.openai_agents")

_registered: Dict[tuple[int, str], "DunetraceTracingProcessor"] = {}


_STALE_RUN_SECS = 1800  # traces not finished after 30 min are pruned

# span_data.type strings emitted by the Agents SDK.
# NB: "mcp_tools" is deliberately *not* a tool type — that span is
# MCPListToolsSpanData (the agent listing a server's available tools), not a
# tool invocation. Real MCP calls arrive as "function" spans. Counting it
# would emit a phantom tool.called/responded and inflate tool_call_count.
_TOOL_SPAN_TYPES = frozenset({"function"})
_HANDOFF_SPAN_TYPE = "handoff"
_TRACE_INPUT_KEYS = ("input", "user_input", "query", "prompt", "message")
_TOOL_CALL_TYPES = frozenset({"function_call", "tool_call", "tool_calls", "function_tool_call"})


@dataclass
class _RunCtx:
    """State for one active SDK trace (== one Dunetrace run)."""

    run_id: str
    step: int = 0
    tool_call_count: int = 0
    start_time: float = field(default_factory=time.time)
    last_activity_time: float = field(default_factory=time.time)
    input_text: str = ""
    run_started: bool = False
    had_error: bool = False
    last_error: str = ""
    # Set when a generation span ends; the immediately following response span
    # (same SDK model call) is skipped. Cleared when any other span type starts.
    skip_next_response_span: bool = False
    open_spans: int = 0  # in-flight spans; stale prune skips traces with open spans
    # span_id → wall-clock start time, for latency.
    span_starts: Dict[str, float] = field(default_factory=dict)
    # span_id → step index assigned at span start, so the *responded*
    # event carries the same step as its *called* event.
    span_steps: Dict[str, int] = field(default_factory=dict)
    # span_id → "llm" | "tool" | "handoff" for end handling.
    span_kinds: Dict[str, str] = field(default_factory=dict)
    ctx_token: Any = field(default=None, repr=False)  # _current_run reset token


class DunetraceTracingProcessor(TracingProcessor):  # type: ignore[misc]
    """
    OpenAI Agents SDK trace processor that translates SDK traces/spans into
    Dunetrace events.

    One processor instance may be shared across concurrent runs: each SDK
    ``trace_id`` maps to an independent :class:`_RunCtx`.
    """

    def __init__(
        self,
        client: "Dunetrace",
        agent_id: str,
        system_prompt: str = "",
        model: str = "unknown",
        tools: Optional[List[str]] = None,
    ) -> None:
        if not _AGENTS_AVAILABLE:
            raise ImportError(
                "openai-agents is not installed. Run: pip install 'dunetrace[openai-agents]'"
            )
        self._client = client
        self._agent_id = agent_id
        self._model = model
        self._version = calc_version(system_prompt, model, tools or [])
        self._tools = tools or []

        self._lock: Lock = Lock()
        self._runs: Dict[str, _RunCtx] = {}  # trace_id → ctx
        self._last_run_id: Optional[str] = None
        # trace_id → run_id for traces evicted by the stale sweep (memory only —
        # no terminal event is emitted). If such a trace resumes we resurrect it
        # under the same run_id rather than dropping its remaining events.
        self._stale_pruned: Dict[str, str] = {}

    @property
    def last_run_id(self) -> Optional[str]:
        """Dunetrace run_id of the most recently finished trace."""
        return self._last_run_id

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _safe_emit(
        self, event_type: Any, ctx: _RunCtx, payload: dict, step: Optional[int] = None
    ) -> None:
        """Emit one event, swallowing any exception so the agent is never broken."""
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

    def _ensure_run_started(self, ctx: _RunCtx) -> None:
        """Emit ``run.started`` once we know the user input (or at trace end)."""
        if ctx.run_started:
            return
        ctx.run_started = True

        from dunetrace.models import EventType

        input_hash = hash_content(ctx.input_text)
        # run.started is always the run's first event. On the lazy path it is
        # emitted from on_span_start *after* ctx.step was incremented for that
        # span, so pin step=0 explicitly rather than defaulting to ctx.step.
        self._safe_emit(
            EventType.RUN_STARTED,
            ctx,
            {
                "input_hash": input_hash,
                "tools": self._tools,
                "model": self._model,
            },
            step=0,
        )

        # Only stamp the hash onto the *current* run if it is this run. When two
        # traces share a contextvar context (e.g. interleaved in one task), the
        # active RunContext may belong to a different trace — writing to it would
        # corrupt that run's input hash.
        run_ctx = _current_run.get(None)
        if run_ctx is not None and run_ctx.run_id == ctx.run_id:
            run_ctx.state.input_text_hash = input_hash

    def _finish_run(self, ctx: _RunCtx, trace_id: str) -> None:
        from dunetrace.models import EventType

        self._ensure_run_started(ctx)

        if ctx.had_error:
            error_hash = hash_content(ctx.last_error or "span_error")
            self._safe_emit(
                EventType.RUN_ERRORED,
                ctx,
                {
                    "error_type": "AgentRunError",
                    "error_hash": error_hash,
                },
            )
        else:
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
        self._cleanup(trace_id)
        self._client.flush()

    # ── TracingProcessor hooks ─────────────────────────────────────────────────

    def on_trace_start(self, trace: Any) -> None:
        try:
            trace_id = getattr(trace, "trace_id", None)
            if not trace_id:
                return

            self._sweep_stale()

            ctx = _RunCtx(
                run_id=str(trace_id),
                input_text=_trace_input_text(trace),
            )
            with self._lock:
                self._runs[str(trace_id)] = ctx

            from dunetrace.run_context import RunContext

            run_ctx = RunContext(
                client=self._client,
                agent_id=self._agent_id,
                agent_version=self._version,
                available_tools=self._tools,
                input_text_hash=hash_content(ctx.input_text) if ctx.input_text else "",
                run_id=ctx.run_id,
            )
            existing = _current_run.get(None)
            if existing is None or existing.run_id == ctx.run_id:
                ctx.ctx_token = _current_run.set(run_ctx)

            if ctx.input_text:
                self._ensure_run_started(ctx)
        except Exception as exc:
            logger.warning("Dunetrace: on_trace_start failed: %s", exc)

    def on_trace_end(self, trace: Any) -> None:
        try:
            trace_id = str(getattr(trace, "trace_id", "") or "")
            with self._lock:
                ctx = self._runs.get(trace_id)
                if ctx is None:
                    ctx = self._resurrect_locked(trace_id)
            if not ctx:
                return

            self._finish_run(ctx, trace_id)
        except Exception as exc:
            logger.warning("Dunetrace: on_trace_end failed: %s", exc)

    def on_span_start(self, span: Any) -> None:
        try:
            trace_id = str(getattr(span, "trace_id", "") or "")
            span_data = getattr(span, "span_data", None)
            span_type = getattr(span_data, "type", "") if span_data else ""
            span_id = str(getattr(span, "span_id", "") or "")

            ctx: Optional[_RunCtx] = None
            emit_llm: Optional[Tuple[str, int]] = None
            emit_tool: Optional[Tuple[str, str, int]] = None
            emit_handoff: Optional[Tuple[str, str, int]] = None

            with self._lock:
                ctx = self._runs.get(trace_id)
                if ctx is None:
                    ctx = self._resurrect_locked(trace_id)
                if not ctx:
                    return

                ctx.last_activity_time = time.time()

                if not ctx.input_text and span_data is not None:
                    span_input = _span_input_text(span_data)
                    if span_input:
                        ctx.input_text = span_input

                if span_type != "response":
                    ctx.skip_next_response_span = False

                if span_type == "generation":
                    model = self._span_model(span_data)
                    ctx.step += 1
                    step = ctx.step
                    if span_id:
                        ctx.span_starts[span_id] = time.time()
                        ctx.span_steps[span_id] = step
                        ctx.span_kinds[span_id] = "llm"
                        ctx.open_spans += 1
                    emit_llm = (model, step)

                elif span_type == "response":
                    # Skip the response half of a generation+response pair for the
                    # same SDK model call. Any intervening span clears the flag.
                    if ctx.skip_next_response_span:
                        ctx.skip_next_response_span = False
                    else:
                        model = self._span_model(span_data)
                        ctx.step += 1
                        step = ctx.step
                        if span_id:
                            ctx.span_starts[span_id] = time.time()
                            ctx.span_steps[span_id] = step
                            ctx.span_kinds[span_id] = "llm"
                            ctx.open_spans += 1
                        emit_llm = (model, step)

                elif span_type in _TOOL_SPAN_TYPES:
                    ctx.step += 1
                    ctx.tool_call_count += 1
                    step = ctx.step
                    if span_id:
                        ctx.span_starts[span_id] = time.time()
                        ctx.span_steps[span_id] = step
                        ctx.span_kinds[span_id] = "tool"
                        ctx.open_spans += 1
                    tool_name = self._tool_name(span_data)
                    args_hash = hash_content(str(getattr(span_data, "input", "") or ""))
                    emit_tool = (tool_name, args_hash, step)

                elif span_type == _HANDOFF_SPAN_TYPE:
                    ctx.step += 1
                    step = ctx.step
                    if span_id:
                        ctx.span_starts[span_id] = time.time()
                        ctx.span_steps[span_id] = step
                        ctx.span_kinds[span_id] = "handoff"
                        ctx.open_spans += 1
                    from_agent = str(getattr(span_data, "from_agent", None) or "unknown")
                    to_agent = str(getattr(span_data, "to_agent", None) or "unknown")
                    emit_handoff = (from_agent, to_agent, step)

            self._ensure_run_started(ctx)

            from dunetrace.models import EventType

            if emit_llm is not None:
                model, step = emit_llm
                self._safe_emit(EventType.LLM_CALLED, ctx, {"model": model}, step=step)
            elif emit_tool is not None:
                tool_name, args_hash, step = emit_tool
                self._safe_emit(
                    EventType.TOOL_CALLED,
                    ctx,
                    {"tool_name": tool_name, "args_hash": args_hash},
                    step=step,
                )
            elif emit_handoff is not None:
                from_agent, to_agent, step = emit_handoff
                self._safe_emit(
                    EventType.TOOL_CALLED,
                    ctx,
                    {
                        "tool_name": f"handoff:{to_agent}",
                        "args_hash": hash_content(from_agent),
                    },
                    step=step,
                )
        except Exception as exc:
            logger.warning("Dunetrace: on_span_start failed: %s", exc)

    def on_span_end(self, span: Any) -> None:
        try:
            trace_id = str(getattr(span, "trace_id", "") or "")
            span_data = getattr(span, "span_data", None)
            span_type = getattr(span_data, "type", "") if span_data else ""
            span_id = str(getattr(span, "span_id", "") or "")
            error = getattr(span, "error", None)

            ctx: Optional[_RunCtx] = None
            latency_ms = 0
            step: Optional[int] = None
            emit_llm = False
            emit_tool_ok: Optional[Tuple[str, int]] = None
            emit_tool_err: Optional[Tuple[str, str]] = None
            emit_handoff_ok: Optional[Tuple[str, int]] = None
            emit_handoff_err: Optional[Tuple[str, str]] = None

            with self._lock:
                ctx = self._runs.get(trace_id)
                if ctx is None:
                    ctx = self._resurrect_locked(trace_id)
                if not ctx:
                    return

                ctx.last_activity_time = time.time()

                if error is not None:
                    err_str = self._error_str(error)
                    ctx.had_error = True
                    ctx.last_error = err_str

                start = ctx.span_starts.pop(span_id, None)
                step = ctx.span_steps.pop(span_id, None)
                span_kind = ctx.span_kinds.pop(span_id, "")
                if span_kind:
                    ctx.open_spans = max(0, ctx.open_spans - 1)
                if start is not None:
                    latency_ms = int((time.time() - start) * 1000)

                if span_kind == "llm":
                    if span_type == "generation":
                        ctx.skip_next_response_span = True
                    emit_llm = True
                elif span_kind == "tool":
                    tool_name = self._tool_name(span_data)
                    if error is not None:
                        emit_tool_err = (tool_name, self._error_str(error))
                    else:
                        output = getattr(span_data, "output", "") or ""
                        emit_tool_ok = (tool_name, len(str(output)))
                elif span_kind == "handoff":
                    to_agent = str(getattr(span_data, "to_agent", None) or "unknown")
                    tool_name = f"handoff:{to_agent}"
                    if error is not None:
                        emit_handoff_err = (tool_name, self._error_str(error))
                    else:
                        emit_handoff_ok = (tool_name, 0)

            from dunetrace.models import EventType

            if emit_llm:
                if step is None:
                    logger.warning(
                        "Dunetrace: skipping llm.responded for span %s (no matching start)",
                        span_id,
                    )
                else:
                    self._emit_llm_responded(ctx, span_data, error, latency_ms, step)
            elif emit_tool_err is not None:
                if step is None:
                    logger.warning(
                        "Dunetrace: skipping tool.responded for span %s (no matching start)",
                        span_id,
                    )
                else:
                    tool_name, err_str = emit_tool_err
                    self._safe_emit(
                        EventType.TOOL_RESPONDED,
                        ctx,
                        {
                            "tool_name": tool_name,
                            "success": False,
                            "latency_ms": latency_ms,
                            "error_hash": hash_content(err_str),
                        },
                        step=step,
                    )
                    self._client.flush()
            elif emit_tool_ok is not None:
                if step is None:
                    logger.warning(
                        "Dunetrace: skipping tool.responded for span %s (no matching start)",
                        span_id,
                    )
                else:
                    tool_name, output_len = emit_tool_ok
                    self._safe_emit(
                        EventType.TOOL_RESPONDED,
                        ctx,
                        {
                            "tool_name": tool_name,
                            "success": True,
                            "output_length": output_len,
                            "latency_ms": latency_ms,
                        },
                        step=step,
                    )
                    self._client.flush()
            elif emit_handoff_err is not None:
                if step is not None:
                    tool_name, err_str = emit_handoff_err
                    self._safe_emit(
                        EventType.TOOL_RESPONDED,
                        ctx,
                        {
                            "tool_name": tool_name,
                            "success": False,
                            "latency_ms": latency_ms,
                            "error_hash": hash_content(err_str),
                        },
                        step=step,
                    )
                    self._client.flush()
            elif emit_handoff_ok is not None:
                if step is not None:
                    tool_name, _ = emit_handoff_ok
                    self._safe_emit(
                        EventType.TOOL_RESPONDED,
                        ctx,
                        {
                            "tool_name": tool_name,
                            "success": True,
                            "output_length": 0,
                            "latency_ms": latency_ms,
                        },
                        step=step,
                    )
                    self._client.flush()
        except Exception as exc:
            logger.warning("Dunetrace: on_span_end failed: %s", exc)

    def shutdown(self) -> None:
        """Flush any buffered events when the application stops."""
        try:
            self._client.flush()
        except Exception as exc:
            logger.warning("Dunetrace: shutdown flush failed: %s", exc)

    def force_flush(self) -> None:
        try:
            self._client.flush()
        except Exception as exc:
            logger.warning("Dunetrace: force_flush failed: %s", exc)

    # ── span_data extraction ────────────────────────────────────────────────────

    def _emit_llm_responded(
        self,
        ctx: _RunCtx,
        span_data: Any,
        error: Any,
        latency_ms: int,
        step: Optional[int],
    ) -> None:
        from dunetrace.models import EventType

        if error is not None:
            self._safe_emit(
                EventType.LLM_RESPONDED,
                ctx,
                {
                    "finish_reason": "error",
                    "output_hash": "",
                    "output_length": 0,
                    "latency_ms": latency_ms,
                    "error_hash": hash_content(self._error_str(error)),
                },
                step=step,
            )
            return

        output_text = _span_output_text(span_data)
        finish_reason = _span_finish_reason(span_data, output_text)
        if finish_reason == "error":
            ctx.had_error = True
            if not ctx.last_error:
                ctx.last_error = "llm_response_failed"
        prompt_tokens, completion_tokens, reasoning_tokens = self._span_usage(span_data)

        payload: Dict[str, Any] = {
            "finish_reason": finish_reason,
            "output_hash": hash_content(output_text),
            "output_length": len(output_text),
            "latency_ms": latency_ms,
        }
        if prompt_tokens is not None:
            payload["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            payload["completion_tokens"] = completion_tokens
        if reasoning_tokens is not None:
            payload["reasoning_tokens"] = reasoning_tokens

        self._safe_emit(EventType.LLM_RESPONDED, ctx, payload, step=step)
        self._client.flush()

    def _span_model(self, span_data: Any) -> str:
        """generation spans expose .model directly; response spans carry the
        model on the nested OpenAI Response object."""
        model = getattr(span_data, "model", None)
        if not model:
            response = getattr(span_data, "response", None)
            model = getattr(response, "model", None)
        return str(model) if model else self._model

    def _span_usage(self, span_data: Any) -> "tuple[Optional[int], Optional[int], Optional[int]]":
        """Pull token counts from a generation span's usage dict or a response
        span's nested usage object. Returns (prompt, completion, reasoning)."""
        usage = getattr(span_data, "usage", None)
        # response spans keep usage on the Response object, not span_data.
        if not usage:
            response = getattr(span_data, "response", None)
            usage = getattr(response, "usage", None)
        if not usage:
            return None, None, None

        def _get(obj: Any, *keys: str) -> Optional[int]:
            for key in keys:
                if isinstance(obj, dict):
                    val = obj.get(key)
                else:
                    val = getattr(obj, key, None)
                if val is not None:
                    return val
            return None

        prompt = _get(usage, "input_tokens", "prompt_tokens")
        completion = _get(usage, "output_tokens", "completion_tokens")

        reasoning = None
        details = (
            usage.get("output_tokens_details")
            if isinstance(usage, dict)
            else getattr(usage, "output_tokens_details", None)
        )
        if details is not None:
            reasoning = _get(details, "reasoning_tokens", "reasoning")

        return prompt, completion, reasoning

    @staticmethod
    def _tool_name(span_data: Any) -> str:
        return str(
            getattr(span_data, "name", None) or getattr(span_data, "server", None) or "unknown"
        )

    @staticmethod
    def _error_str(error: Any) -> str:
        """SDK span errors are SpanError objects (or dicts) with a message."""
        if isinstance(error, dict):
            return str(error.get("message") or error)
        return str(getattr(error, "message", error))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _reset_ctx_token(self, ctx: _RunCtx) -> None:
        if ctx.ctx_token is not None:
            try:
                current = _current_run.get(None)
                if current is not None and current.run_id == ctx.run_id:
                    _current_run.reset(ctx.ctx_token)
            except ValueError:
                # Token was created in a different context — expected when the
                # stale sweep evicts a trace from another task/thread. The
                # ContextVar is task-local, so the leftover value is harmless.
                pass

    def _cleanup(self, trace_id: str) -> None:
        with self._lock:
            ctx = self._runs.pop(trace_id, None)
        if ctx:
            self._reset_ctx_token(ctx)

    def _sweep_stale(self) -> None:
        """Evict traces idle for more than _STALE_RUN_SECS from in-memory state.

        Uses last span activity, not trace start time, so long-running agents
        are not cut off. Traces with in-flight spans are never evicted. No
        terminal Dunetrace event is emitted — idle runs remain open in the
        backend until ``on_trace_end`` or the stall detector fires.
        """
        cutoff = time.time() - _STALE_RUN_SECS
        evicted: List[Tuple[str, _RunCtx]] = []
        with self._lock:
            for trace_id in list(self._runs):
                ctx = self._runs.get(trace_id)
                if ctx is None:
                    continue
                if ctx.open_spans > 0 or ctx.last_activity_time >= cutoff:
                    continue
                logger.warning(
                    "Dunetrace: pruning stale trace %s (idle >%ds)",
                    trace_id,
                    _STALE_RUN_SECS,
                )
                evicted.append((trace_id, self._runs.pop(trace_id)))
                self._stale_pruned[trace_id] = ctx.run_id
                if len(self._stale_pruned) > 1024:
                    self._stale_pruned.pop(next(iter(self._stale_pruned)))
        for _trace_id, ctx in evicted:
            self._reset_ctx_token(ctx)

    def _resurrect_locked(self, trace_id: str) -> Optional[_RunCtx]:
        """Rebuild a context for a trace that resumed after being stale-pruned.

        Caller must hold ``self._lock``. The earlier (already-emitted) steps are
        gone, but new activity is captured under the original run_id. ``run_started``
        is preset so we don't emit a duplicate ``run.started``.
        """
        run_id = self._stale_pruned.pop(trace_id, None)
        if run_id is None:
            return None
        logger.warning(
            "Dunetrace: trace %s resumed after stale-prune; resurrecting run %s",
            trace_id,
            run_id,
        )
        ctx = _RunCtx(run_id=run_id, run_started=True)
        self._runs[trace_id] = ctx
        return ctx

    def _reset(self) -> None:
        """Clear all state (e.g. for testing). Not needed in normal usage."""
        with self._lock:
            self._runs.clear()
            self._stale_pruned.clear()


# ── Module-level span/trace text helpers ──────────────────────────────────────


def _trace_input_text(trace: Any) -> str:
    """Best-effort user input from trace metadata."""
    metadata = getattr(trace, "metadata", None)
    if not isinstance(metadata, dict):
        return ""
    for key in _TRACE_INPUT_KEYS:
        val = metadata.get(key)
        if val is not None and str(val).strip():
            return str(val)
    return ""


def _span_input_text(span_data: Any) -> str:
    """Extract hashable user input from an LLM span's input field."""
    raw = getattr(span_data, "input", None)
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        # Response API input items or chat message list — take the last user turn.
        for item in reversed(raw):
            text = _message_text(item)
            if text:
                return text
        return ""
    return str(raw)


def _span_output_text(span_data: Any) -> str:
    span_type = getattr(span_data, "type", "")
    if span_type == "response":
        return _response_output_text(getattr(span_data, "response", None))
    if span_type == "generation":
        return _generation_output_text(getattr(span_data, "output", None))
    output = getattr(span_data, "output", None)
    return "" if output is None else str(output)


def _span_finish_reason(span_data: Any, output_text: str) -> str:
    span_type = getattr(span_data, "type", "")
    if span_type == "response":
        return _response_finish_reason(getattr(span_data, "response", None))
    if span_type == "generation":
        return _generation_finish_reason(getattr(span_data, "output", None))
    return "stop"


def _generation_output_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, (list, tuple)):
        parts: List[str] = []
        for item in output:
            text = _message_text(item)
            if text:
                parts.append(text)
        if parts:
            return "".join(parts)
    return str(output)


def _response_output_text(response: Any) -> str:
    if response is None:
        return ""
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)
    output = getattr(response, "output", None)
    if not output:
        return ""
    parts: List[str] = []
    for item in output:
        item_type = _item_type(item)
        if item_type in _TOOL_CALL_TYPES:
            continue
        text = _message_text(item)
        if text:
            parts.append(text)
    return "".join(parts)


def _response_finish_reason(response: Any) -> str:
    if response is None:
        return "stop"
    status = getattr(response, "status", None)
    if status == "failed":
        return "error"
    if status == "incomplete":
        return "length"
    output = getattr(response, "output", None) or []
    if _output_has_tool_calls(output):
        return "tool_calls"
    return "stop"


def _generation_finish_reason(output: Any) -> str:
    if _output_has_tool_calls(output):
        return "tool_calls"
    return "stop"


def _output_has_tool_calls(output: Any) -> bool:
    if not output:
        return False
    if not isinstance(output, (list, tuple)):
        return False
    for item in output:
        item_type = _item_type(item)
        if item_type in _TOOL_CALL_TYPES:
            return True
        if isinstance(item, dict) and item.get("tool_calls"):
            return True
        tool_calls = getattr(item, "tool_calls", None)
        if tool_calls:
            return True
    return False


def _item_type(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("type") or "")
    return str(getattr(item, "type", "") or "")


def _message_text(item: Any) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        content = item.get("content")
        if content is not None:
            return _content_text(content)
        text = item.get("text")
        if text is not None:
            return str(text)
        return ""
    content = getattr(item, "content", None)
    if content is not None:
        return _content_text(content)
    text = getattr(item, "text", None)
    if text is not None:
        return str(text)
    return ""


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "output_text" and block.get("text"):
                    parts.append(str(block["text"]))
                elif block.get("text"):
                    parts.append(str(block["text"]))
            elif getattr(block, "text", None):
                parts.append(str(block.text))
        return "".join(parts)
    return str(content)


def add_dunetrace_processor(
    client: "Dunetrace",
    agent_id: str,
    system_prompt: str = "",
    model: str = "unknown",
    tools: Optional[List[str]] = None,
) -> DunetraceTracingProcessor:
    """
    Build a :class:`DunetraceTracingProcessor` and register it with the Agents
    SDK's global trace provider via ``add_trace_processor``. Returns the
    processor so callers can read ``.last_run_id``.

    Registers *alongside* existing processors, so the default OpenAI exporter
    (and any others) keep working.
    """
    if not _AGENTS_AVAILABLE:
        raise ImportError(
            "openai-agents is not installed. Run: pip install 'dunetrace[openai-agents]'"
        )
    key = (id(client), agent_id)
    existing = _registered.get(key)
    if existing is not None:
        logger.warning(
            "Dunetrace: add_dunetrace_processor already registered for agent_id=%r",
            agent_id,
        )
        return existing
    if _registered:
        # The Agents SDK trace provider is process-global: every registered
        # processor receives *every* trace. A second Dunetrace processor would
        # re-emit each run under a second agent_id (duplicate, mis-attributed
        # events). Refuse to register it and reuse the first one instead.
        owner = next(iter(_registered.values()))
        logger.warning(
            "Dunetrace: a Dunetrace processor for agent_id=%r is already "
            "registered; Agents SDK tracing is process-global, so it handles "
            "all traces. Ignoring registration for agent_id=%r — use a single "
            "processor per process.",
            owner._agent_id,
            agent_id,
        )
        return owner
    processor = DunetraceTracingProcessor(
        client, agent_id=agent_id, system_prompt=system_prompt, model=model, tools=tools
    )
    _registered[key] = processor
    add_trace_processor(processor)
    return processor
