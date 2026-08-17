"""
OTel span exporter for Dunetrace. Translates Dunetrace AgentEvents into
OpenTelemetry spans so agent runs show up in whatever OTel backend a customer
already runs (Datadog, Grafana Tempo, Honeycomb, Signoz, Jaeger).

This runs alongside Dunetrace's own ingest, not instead of it: the same events
still ship to Dunetrace. OTel export is additive and opt-in.

Wiring is env-driven (see dunetrace/otel.py). When DUNETRACE_OTEL_ENABLED is set
the Dunetrace client builds one of these on the shared tracer automatically. You
can also construct it directly against your own TracerProvider:

    from opentelemetry.sdk.trace import TracerProvider
    from dunetrace.integrations.otel import DunetraceOTelExporter

    dt = Dunetrace(otel_exporter=DunetraceOTelExporter(tracer_provider=provider))

Span model (run + LLM + tool + retrieval + voice; signals/policies in Phase 4):

    Trace (trace_id derived from run_id, so a Dunetrace run is one trace)
    └── span "dunetrace.run"              [dunetrace.run.id, dunetrace.run.agent_id, ...]
        ├── span "chat {model}"           [gen_ai.provider.name, gen_ai.request.model, ...]
        ├── span "dunetrace.tool.{name}"  [dunetrace.tool.*, url.full/server.address if HTTP]
        ├── span "dunetrace.retrieval"    [dunetrace.retrieval.vector_store, .document_count]
        ├── span "dunetrace.voice.transcription" / "dunetrace.voice.tts"
        └── event "dunetrace.voice.{vad}" [silence/speech_start/...] on the run span

Semantic conventions:
  - LLM spans follow the OpenTelemetry GenAI conventions (gen_ai.*). Provider is
    emitted as gen_ai.provider.name, not the deprecated gen_ai.system.
  - HTTP-shaped tool calls use the current stable HTTP conventions
    (url.full, server.address, http.request.method, http.response.status_code),
    not the deprecated http.url / http.method / http.status_code.
  - Agent runs, tools, retrievals, and voice events have no standard OTel
    operation type, so those fields are namespaced under dunetrace.*.

PII: content-bearing attributes (tool args, request URL, retrieval query) are
emitted only when capture_content is True (the default; disable with
DUNETRACE_OTEL_CAPTURE_CONTENT=false). Voice spans never carry raw transcript or
TTS text, only character counts and metadata.

Correlation:
  - trace_id is uuid(run_id).int and the root span_id is its lower 64 bits, both
    deterministic. trace_id_hex()/root_span_id_hex() expose them so the client
    can stamp them on the run object and a backend can deep-link to the run.
  - A child run (parent_run_id set) gets its own trace and a span Link back to
    the parent run's root span, which is the OTel-correct way to express
    causally-related spans that live in different traces.
"""

from __future__ import annotations

import ast
import logging
import time
import uuid
from dataclasses import dataclass
from threading import Lock
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger("dunetrace.otel")

try:
    from opentelemetry import trace
    from opentelemetry.trace import (
        Link,
        NonRecordingSpan,
        SpanContext,
        SpanKind,
        StatusCode,
        TraceFlags,
    )

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

from dunetrace.models import AgentEvent, EventType

# ── ID derivation ───────────────────────────────────────────────────────────────


def _ns(ts: float) -> int:
    """Float Unix seconds to OTel nanoseconds."""
    return int(ts * 1_000_000_000)


def _trace_id(run_id: str) -> int:
    """Run UUID to a 128-bit OTel trace ID. Deterministic, so a run maps to a
    stable trace a backend and the Dunetrace dashboard can both address."""
    return uuid.UUID(run_id).int


def _root_span_id(run_id: str) -> int:
    """Stable 64-bit root span ID from the run UUID (its lower 64 bits)."""
    return uuid.UUID(run_id).int & 0xFFFF_FFFF_FFFF_FFFF


def trace_id_hex(run_id: str) -> str:
    """32-hex-char trace ID for run_id (W3C traceparent format)."""
    return format(_trace_id(run_id), "032x")


def root_span_id_hex(run_id: str) -> str:
    """16-hex-char root span ID for run_id."""
    return format(_root_span_id(run_id), "016x")


def _span_context(run_id: str, *, is_remote: bool) -> "SpanContext":
    """Deterministic SpanContext for a run's root span. Used to seed a run's own
    trace, to link a child run to its parent, and (server-side) to parent signal
    and policy spans onto a run whose root span the SDK already closed."""
    return SpanContext(
        trace_id=_trace_id(run_id),
        span_id=_root_span_id(run_id),
        is_remote=is_remote,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )


def _parent_ctx_for_run(run_id: str) -> Any:
    """Context whose current span is the run's (deterministic) root span, as a
    non-recording remote span. Parents children onto the root without needing the
    live root span object."""
    return trace.set_span_in_context(NonRecordingSpan(_span_context(run_id, is_remote=True)))


# ── Provider inference (gen_ai.provider.name) ────────────────────────────────────
#
# The GenAI conventions want a low-cardinality provider name. We infer it from
# the model string. Prefix match, longest-intent first; unknown models get no
# provider attribute rather than a wrong guess.

_PROVIDER_PREFIXES = (
    ("claude", "anthropic"),
    ("gpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("chatgpt", "openai"),
    ("gemini", "gcp.gemini"),
    ("mistral", "mistral_ai"),
    ("mixtral", "mistral_ai"),
    ("command", "cohere"),
    ("deepseek", "deepseek"),
    ("grok", "x_ai"),
)


def _provider_from_model(model: str) -> str:
    m = (model or "").lower()
    for prefix, provider in _PROVIDER_PREFIXES:
        if m.startswith(prefix):
            return provider
    return ""


def _http_info_from_args(args_repr: str) -> Optional[Dict[str, str]]:
    """Best-effort HTTP shape from a tool call's serialized args. Returns
    {"url": ...} (and "method" when present) if the args look like an HTTP call,
    else None. Uses ast.literal_eval, so only genuine dict-repr args parse and
    anything else is treated as a non-HTTP tool. The auto-instrumented httpx and
    requests patches pass {"url": ...}, which is what this recognizes."""
    if "url" not in args_repr:
        return None
    try:
        parsed = ast.literal_eval(args_repr)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    url = parsed.get("url")
    if not isinstance(url, str) or not url:
        return None
    info: Dict[str, str] = {"url": url}
    method = parsed.get("method")
    if isinstance(method, str) and method:
        info["method"] = method.upper()
    return info


def _hostname(url: str) -> str:
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""


def _llm_cost_usd(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """Best-effort USD cost for one LLM call, reusing the SDK's pricing table so
    the span figure matches Dunetrace's own cost accounting. Returns None if the
    cost helper is unavailable or errors (cost is a nice-to-have, never fatal)."""
    try:
        from dunetrace.policies import compute_run_cost

        call = SimpleNamespace(
            model=model, prompt_tokens=input_tokens, completion_tokens=output_tokens
        )
        return compute_run_cost([call])
    except Exception as exc:
        logger.debug("Dunetrace OTel: cost computation skipped: %s", exc)
        return None


# ── Per-run span state ───────────────────────────────────────────────────────────


@dataclass
class _RunSpans:
    root_span: Any  # opentelemetry Span
    start_ts: float
    child_span: Any = None  # currently open child span, or None
    child_kind: str = ""  # "llm" | "tool" | "retrieval"
    llm_model: str = ""
    llm_input_tokens: int = 0
    tool_is_http: bool = False  # HTTP-shaped tool: maps error -> http status code


# ── Exporter ─────────────────────────────────────────────────────────────────────


class DunetraceOTelExporter:
    """
    Translates Dunetrace AgentEvents into OpenTelemetry spans. Thread-safe: the
    per-run span map is lock-protected. Span open/close is sequential per run
    since every event for a run is emitted from that run's own thread.

    :param tracer:          A pre-built OTel Tracer (what the env-driven wiring
                            passes, from dunetrace.otel.get_tracer()).
    :param tracer_provider: A TracerProvider to get a tracer from, when you wire
                            your own pipeline instead of the env config.
    :param tracer_name:     Instrumentation scope name. Default "dunetrace".
    :param capture_content: When False, drop content-bearing attributes (tool
                            args, request URL, retrieval query) from spans. The
                            env-driven wiring passes the resolved config value.
    """

    def __init__(
        self,
        tracer: Any = None,
        *,
        tracer_provider: Any = None,
        tracer_name: str = "dunetrace",
        capture_content: bool = True,
    ) -> None:
        if not _OTEL_AVAILABLE:
            raise ImportError("opentelemetry is not installed. Run: pip install 'dunetrace[otel]'")
        if tracer is not None:
            self._tracer = tracer
        else:
            tp = tracer_provider or trace.get_tracer_provider()
            self._tracer = tp.get_tracer(tracer_name)
        self._capture_content = capture_content
        self._runs: Dict[str, _RunSpans] = {}
        self._lock = Lock()

    # ── Client entry point ─────────────────────────────────────────────────────

    def handle(self, event: AgentEvent) -> None:
        """Route one AgentEvent to its span operation. Never raises: an export
        failure must not touch the agent's own run."""
        fn = self._DISPATCH.get(event.event_type)
        if fn is None:
            return
        try:
            fn(self, event)
        except Exception as exc:
            logger.warning(
                "Dunetrace OTel: failed to handle %s for run %s: %s",
                event.event_type,
                event.run_id,
                exc,
            )

    # ── Run span ───────────────────────────────────────────────────────────────

    def _on_run_started(self, event: AgentEvent) -> None:
        # Seed the trace with a deterministic trace ID derived from run_id. The
        # parent is a synthetic NonRecordingSpan carrying only the trace ID, so
        # it doesn't show up as a real span in the backend.
        parent_ctx = _parent_ctx_for_run(event.run_id)

        attrs: Dict[str, Any] = {
            "dunetrace.run.id": event.run_id,
            "dunetrace.run.agent_id": event.agent_id,
            "dunetrace.run.agent_version": event.agent_version,
            "dunetrace.run.status": "running",
        }
        if event.conversation_id:
            attrs["dunetrace.run.conversation_id"] = event.conversation_id
        if event.parent_run_id:
            attrs["dunetrace.run.parent_run_id"] = event.parent_run_id
        model = event.payload.get("model")
        if model:
            attrs["dunetrace.run.model"] = model
        tools = event.payload.get("tools")
        if tools:
            attrs["dunetrace.run.tools"] = ",".join(tools)

        # A child run lives in its own trace; link it back to the parent run's
        # root span so a backend can follow the causal edge across traces.
        links: List["Link"] = []
        if event.parent_run_id:
            try:
                links.append(Link(_span_context(event.parent_run_id, is_remote=True)))
            except Exception as exc:
                logger.debug("Dunetrace OTel: parent link skipped: %s", exc)

        root = self._tracer.start_span(
            "dunetrace.run",
            context=parent_ctx,
            kind=SpanKind.INTERNAL,
            start_time=_ns(event.timestamp),
            attributes=attrs,
            links=links or None,
        )
        with self._lock:
            self._runs[event.run_id] = _RunSpans(root_span=root, start_ts=event.timestamp)

    def _on_run_ended(self, event: AgentEvent) -> None:
        with self._lock:
            rs = self._runs.pop(event.run_id, None)
        if rs is None:
            return

        # Close any child span left open (e.g. an LLM call with no response
        # event). If we're here mid-step, something went wrong: mark it ERROR.
        if rs.child_span is not None:
            rs.child_span.set_status(StatusCode.ERROR, "span ended without a response event")
            rs.child_span.end(end_time=_ns(event.timestamp))

        root = rs.root_span
        failed = event.event_type is EventType.RUN_ERRORED
        root.set_attribute("dunetrace.run.status", "failed" if failed else "completed")
        root.set_attribute("dunetrace.run.duration_ms", int((event.timestamp - rs.start_ts) * 1000))
        root.set_attribute("dunetrace.run.total_steps", event.payload.get("total_steps", 0))
        root.set_attribute("dunetrace.run.exit_reason", event.payload.get("exit_reason", ""))
        root.set_attribute("dunetrace.run.tool_call_count", event.payload.get("tool_call_count", 0))
        if failed:
            error_type = event.payload.get("error_type", "run errored")
            root.set_attribute("dunetrace.run.error_type", error_type)
            root.set_status(StatusCode.ERROR, error_type)

        root.end(end_time=_ns(event.timestamp))

    # ── LLM span (GenAI conventions) ───────────────────────────────────────────

    def _on_llm_called(self, event: AgentEvent) -> None:
        with self._lock:
            rs = self._runs.get(event.run_id)
        if rs is None:
            return

        self._close_orphan_child(rs, event.timestamp)

        model = event.payload.get("model", "")
        input_tokens = int(event.payload.get("prompt_tokens") or 0)
        name = f"chat {model}".strip() if model else "chat"

        attrs: Dict[str, Any] = {
            "gen_ai.operation.name": "chat",
            "dunetrace.run.id": event.run_id,
            "dunetrace.step_index": event.step_index,
        }
        if model:
            attrs["gen_ai.request.model"] = model
        provider = _provider_from_model(model)
        if provider:
            attrs["gen_ai.provider.name"] = provider
        if input_tokens:
            attrs["gen_ai.usage.input_tokens"] = input_tokens

        parent_ctx = trace.set_span_in_context(rs.root_span)
        rs.child_span = self._tracer.start_span(
            name,
            context=parent_ctx,
            kind=SpanKind.CLIENT,
            start_time=_ns(event.timestamp),
            attributes=attrs,
        )
        rs.child_kind = "llm"
        rs.llm_model = model
        rs.llm_input_tokens = input_tokens

    def _on_llm_responded(self, event: AgentEvent) -> None:
        with self._lock:
            rs = self._runs.get(event.run_id)
        if rs is None or rs.child_span is None or rs.child_kind != "llm":
            return

        p, span = event.payload, rs.child_span
        finish_reason = p.get("finish_reason", "")
        completion = int(p.get("completion_tokens") or 0)
        reasoning = int(p.get("reasoning_tokens") or 0)
        output_tokens = completion + reasoning
        # prompt_tokens may be back-filled here if it was only known post-call.
        input_tokens = int(p.get("prompt_tokens") or 0) or rs.llm_input_tokens
        if input_tokens and not rs.llm_input_tokens:
            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)

        if finish_reason:
            span.set_attribute("gen_ai.response.finish_reasons", [finish_reason])
            span.set_attribute("dunetrace.llm.output_truncated", finish_reason == "length")
        if output_tokens:
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        if reasoning:
            span.set_attribute("dunetrace.llm.reasoning_tokens", reasoning)
        if p.get("latency_ms"):
            span.set_attribute("dunetrace.llm.latency_ms", p["latency_ms"])
        if p.get("output_length"):
            span.set_attribute("dunetrace.llm.output_length", p["output_length"])

        cost = _llm_cost_usd(rs.llm_model, input_tokens, completion)
        if cost:
            span.set_attribute("dunetrace.llm.cost_usd", cost)

        if p.get("error"):
            span.set_status(StatusCode.ERROR, str(p["error"]))

        span.end(end_time=_ns(event.timestamp))
        rs.child_span = None
        rs.child_kind = ""

    # ── Tool span ──────────────────────────────────────────────────────────────

    def _on_tool_called(self, event: AgentEvent) -> None:
        with self._lock:
            rs = self._runs.get(event.run_id)
        if rs is None:
            return
        self._close_orphan_child(rs, event.timestamp)

        tool_name = event.payload.get("tool_name", "")
        args_repr = event.payload.get("args", "")
        http = _http_info_from_args(args_repr)

        attrs: Dict[str, Any] = {
            "dunetrace.tool.name": tool_name,
            "dunetrace.run.id": event.run_id,
            "dunetrace.step_index": event.step_index,
        }
        if self._capture_content and args_repr:
            attrs["dunetrace.tool.args"] = args_repr
        if http:
            host = _hostname(http["url"])
            if host:
                attrs["server.address"] = host
            if http.get("method"):
                attrs["http.request.method"] = http["method"]
            if self._capture_content:
                attrs["url.full"] = http["url"]

        parent_ctx = trace.set_span_in_context(rs.root_span)
        rs.child_span = self._tracer.start_span(
            f"dunetrace.tool.{tool_name}" if tool_name else "dunetrace.tool",
            context=parent_ctx,
            kind=SpanKind.CLIENT if http else SpanKind.INTERNAL,
            start_time=_ns(event.timestamp),
            attributes=attrs,
        )
        rs.child_kind = "tool"
        rs.tool_is_http = bool(http)

    def _on_tool_responded(self, event: AgentEvent) -> None:
        with self._lock:
            rs = self._runs.get(event.run_id)
        if rs is None or rs.child_span is None or rs.child_kind != "tool":
            return

        p, span = event.payload, rs.child_span
        success = p.get("success", True)
        span.set_attribute("dunetrace.tool.result_status", "success" if success else "error")
        if p.get("output_length"):
            span.set_attribute("dunetrace.tool.output_length", p["output_length"])
        if p.get("latency_ms"):
            span.set_attribute("dunetrace.tool.latency_ms", p["latency_ms"])

        error = p.get("error")
        if error is not None and error != "":
            # The HTTP auto-instrumentation puts the status code in `error` on a
            # failed request; surface it as http.response.status_code when it's a
            # bare number. Otherwise it's a free-text message (gated as content).
            if rs.tool_is_http and str(error).isdigit():
                span.set_attribute("http.response.status_code", int(error))
            elif self._capture_content:
                span.set_attribute("dunetrace.tool.error_message", str(error))
        if not success:
            span.set_status(StatusCode.ERROR, "tool call failed")

        span.end(end_time=_ns(event.timestamp))
        rs.child_span = None
        rs.child_kind = ""
        rs.tool_is_http = False

    # ── Retrieval span ─────────────────────────────────────────────────────────

    def _on_retrieval_called(self, event: AgentEvent) -> None:
        with self._lock:
            rs = self._runs.get(event.run_id)
        if rs is None:
            return
        self._close_orphan_child(rs, event.timestamp)

        attrs: Dict[str, Any] = {
            "dunetrace.run.id": event.run_id,
            "dunetrace.step_index": event.step_index,
        }
        index_name = event.payload.get("index_name")
        if index_name:
            attrs["dunetrace.retrieval.vector_store"] = index_name
        query = event.payload.get("query")
        if query and self._capture_content:
            attrs["dunetrace.retrieval.query"] = query

        parent_ctx = trace.set_span_in_context(rs.root_span)
        rs.child_span = self._tracer.start_span(
            "dunetrace.retrieval",
            context=parent_ctx,
            kind=SpanKind.CLIENT,
            start_time=_ns(event.timestamp),
            attributes=attrs,
        )
        rs.child_kind = "retrieval"

    def _on_retrieval_responded(self, event: AgentEvent) -> None:
        with self._lock:
            rs = self._runs.get(event.run_id)
        if rs is None or rs.child_span is None or rs.child_kind != "retrieval":
            return

        p, span = event.payload, rs.child_span
        span.set_attribute("dunetrace.retrieval.document_count", p.get("result_count", 0))
        if p.get("top_score") is not None:
            span.set_attribute("dunetrace.retrieval.top_score", p["top_score"])
        if p.get("latency_ms"):
            span.set_attribute("dunetrace.retrieval.latency_ms", p["latency_ms"])
        # An empty retrieval is a soft signal (RAG_EMPTY_RETRIEVAL), not a failed
        # operation, so the span stays OK. Phase 4 surfaces it as a signal.

        span.end(end_time=_ns(event.timestamp))
        rs.child_span = None
        rs.child_kind = ""

    # ── Voice spans (only fire when the voice hooks are used) ───────────────────

    def _on_voice_transcription(self, event: AgentEvent) -> None:
        with self._lock:
            rs = self._runs.get(event.run_id)
        if rs is None:
            return
        p = event.payload
        latency_ms = int(p.get("latency_ms") or 0)
        attrs: Dict[str, Any] = {
            "dunetrace.run.id": event.run_id,
            "dunetrace.voice.confidence": float(p.get("confidence", 0.0)),
            "dunetrace.voice.char_count": len(p.get("text", "") or ""),
        }
        if latency_ms:
            attrs["dunetrace.voice.latency_ms"] = latency_ms
        if p.get("audio_seconds"):
            attrs["dunetrace.voice.audio_seconds"] = p["audio_seconds"]
        self._emit_point_span(
            rs, "dunetrace.voice.transcription", event.timestamp, latency_ms, attrs
        )

    def _on_voice_tts(self, event: AgentEvent) -> None:
        with self._lock:
            rs = self._runs.get(event.run_id)
        if rs is None:
            return
        p = event.payload
        latency_ms = int(p.get("latency_ms") or 0)
        attrs: Dict[str, Any] = {
            "dunetrace.run.id": event.run_id,
            "dunetrace.voice.char_count": len(p.get("text", "") or ""),
            "dunetrace.voice.truncated": bool(p.get("truncated", False)),
        }
        if p.get("model"):
            attrs["dunetrace.voice.model"] = p["model"]
        if p.get("voice_id"):
            attrs["dunetrace.voice.voice_id"] = p["voice_id"]
        if p.get("provider"):
            attrs["dunetrace.voice.provider"] = p["provider"]
        if latency_ms:
            attrs["dunetrace.voice.latency_ms"] = latency_ms
        if p.get("audio_seconds"):
            attrs["dunetrace.voice.audio_seconds"] = p["audio_seconds"]
        self._emit_point_span(rs, "dunetrace.voice.tts", event.timestamp, latency_ms, attrs)

    def _on_voice_activity(self, event: AgentEvent) -> None:
        # VAD transitions are high-frequency; a span per audio frame would blow up
        # span counts, so they're span events on the run span (per the brief).
        with self._lock:
            rs = self._runs.get(event.run_id)
        if rs is None:
            return
        vad_type = event.payload.get("type", "vad")
        attrs: Dict[str, Any] = {}
        if event.payload.get("duration_ms"):
            attrs["dunetrace.voice.duration_ms"] = event.payload["duration_ms"]
        rs.root_span.add_event(
            f"dunetrace.voice.{vad_type}",
            attributes=attrs,
            timestamp=_ns(event.timestamp),
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _emit_point_span(
        self, rs: _RunSpans, name: str, ts: float, latency_ms: int, attrs: Dict[str, Any]
    ) -> None:
        """Open-and-close a standalone child span for a single-event operation
        (transcription, TTS). Duration reflects latency when known. Doesn't touch
        rs.child_span, so it can't disturb an open called/responded pair."""
        start = ts - (latency_ms / 1000.0) if latency_ms else ts
        parent_ctx = trace.set_span_in_context(rs.root_span)
        span = self._tracer.start_span(
            name,
            context=parent_ctx,
            kind=SpanKind.INTERNAL,
            start_time=_ns(start),
            attributes=attrs,
        )
        span.end(end_time=_ns(ts))

    def _close_orphan_child(self, rs: _RunSpans, ts: float) -> None:
        """Close a still-open child (mismatched called/responded pair) as ERROR
        so a dropped response event can't leak an unclosed span."""
        if rs.child_span is not None:
            rs.child_span.set_status(StatusCode.ERROR, "span ended without a response event")
            rs.child_span.end(end_time=_ns(ts))
            rs.child_span = None
            rs.child_kind = ""

    _DISPATCH = {
        EventType.RUN_STARTED: _on_run_started,
        EventType.RUN_COMPLETED: _on_run_ended,
        EventType.RUN_ERRORED: _on_run_ended,
        EventType.LLM_CALLED: _on_llm_called,
        EventType.LLM_RESPONDED: _on_llm_responded,
        EventType.TOOL_CALLED: _on_tool_called,
        EventType.TOOL_RESPONDED: _on_tool_responded,
        EventType.RETRIEVAL_CALLED: _on_retrieval_called,
        EventType.RETRIEVAL_RESPONDED: _on_retrieval_responded,
        EventType.TRANSCRIPTION_RECEIVED: _on_voice_transcription,
        EventType.TTS_GENERATED: _on_voice_tts,
        EventType.VOICE_ACTIVITY_DETECTED: _on_voice_activity,
    }


# ── Server-side findings (signals + policies) ────────────────────────────────────
#
# Signals and policies are decided server-side (detector_svc), after the SDK has
# already closed the run's root span. An OTel span event can't be added to a
# closed span, so these become child spans in the run's trace, parented onto the
# run's deterministic root span context. A backend renders them nested under the
# run even though they were exported from a different process at a later time.


def emit_signal_span(
    tracer: Any,
    run_id: str,
    *,
    failure_type: str,
    severity: str,
    confidence: float = 0.0,
    detector_name: str = "",
    signal_id: Optional[str] = None,
    evidence: Optional[Dict[str, Any]] = None,
    step_index: int = 0,
    org_id: str = "",
    timestamp: Optional[float] = None,
    capture_content: bool = True,
) -> None:
    """Emit a ``dunetrace.signal.{failure_type}`` span into ``run_id``'s trace.

    signal_id defaults to ``{run_id}:{failure_type}`` — the (run, type) pair is
    a signal's unique key in Dunetrace, so a backend can deep-link back to it.
    HIGH/CRITICAL signals also mark the span ERROR so they surface in error views.
    """
    ts = timestamp or time.time()
    sig_id = signal_id if signal_id is not None else f"{run_id}:{failure_type}"
    attrs: Dict[str, Any] = {
        "dunetrace.run.id": run_id,
        "dunetrace.signal.type": failure_type,
        "dunetrace.signal.severity": severity,
        "dunetrace.signal.confidence": confidence,
        "dunetrace.signal.step_index": step_index,
        "dunetrace.signal.id": sig_id,
    }
    if detector_name:
        attrs["dunetrace.signal.detector"] = detector_name
    if org_id:
        attrs["dunetrace.run.org_id"] = org_id
    for key, value in (evidence or {}).items():
        # Numbers/bools are always safe; string evidence may echo content, so
        # it's gated like the other content attributes.
        if isinstance(value, bool) or isinstance(value, (int, float)):
            attrs[f"dunetrace.signal.evidence.{key}"] = value
        elif isinstance(value, str) and capture_content:
            attrs[f"dunetrace.signal.evidence.{key}"] = value

    span = tracer.start_span(
        f"dunetrace.signal.{failure_type}",
        context=_parent_ctx_for_run(run_id),
        kind=SpanKind.INTERNAL,
        start_time=_ns(ts),
        attributes=attrs,
    )
    if severity in ("HIGH", "CRITICAL"):
        span.set_status(StatusCode.ERROR, f"{failure_type} [{severity}]")
    span.end(end_time=_ns(ts))


def emit_policy_span(
    tracer: Any,
    run_id: str,
    *,
    action: str,
    policy_name: str = "",
    trigger: str = "",
    trigger_value: Any = None,
    org_id: str = "",
    timestamp: Optional[float] = None,
) -> None:
    """Emit a ``dunetrace.policy.{action}`` span into ``run_id``'s trace, for a
    policy intervention (stop, switch_model, require_approval, ...)."""
    ts = timestamp or time.time()
    attrs: Dict[str, Any] = {
        "dunetrace.run.id": run_id,
        "dunetrace.policy.action": action,
    }
    if policy_name:
        attrs["dunetrace.policy.name"] = policy_name
    if trigger:
        attrs["dunetrace.policy.trigger"] = trigger
    if trigger_value is not None:
        attrs["dunetrace.policy.trigger_value"] = str(trigger_value)
    if org_id:
        attrs["dunetrace.run.org_id"] = org_id

    span = tracer.start_span(
        f"dunetrace.policy.{action}" if action else "dunetrace.policy",
        context=_parent_ctx_for_run(run_id),
        kind=SpanKind.INTERNAL,
        start_time=_ns(ts),
        attributes=attrs,
    )
    span.end(end_time=_ns(ts))


def emit_run_findings(
    tracer: Any,
    run_id: str,
    *,
    signals: Any = (),
    policy_events: Any = (),
    org_id: str = "",
    capture_content: bool = True,
) -> None:
    """Emit signal and policy child spans for one processed run.

    ``signals`` are FailureSignal-like objects (``.failure_type``, ``.severity``,
    ``.confidence``, ``.evidence``, ``.step_index``, ``.detected_at``).
    ``policy_events`` are ``policy.triggered`` event dicts. Each span is emitted
    independently and best-effort: one bad finding never blocks the rest, and
    nothing raises into the caller (detection must not break on telemetry)."""
    for sig in signals:
        try:
            emit_signal_span(
                tracer,
                run_id,
                failure_type=_enum_value(sig.failure_type),
                severity=_enum_value(sig.severity),
                confidence=getattr(sig, "confidence", 0.0),
                detector_name=(getattr(sig, "evidence", None) or {}).get("detector_name", ""),
                signal_id=getattr(sig, "signal_id", None),
                evidence=getattr(sig, "evidence", None),
                step_index=getattr(sig, "step_index", 0),
                org_id=org_id,
                timestamp=getattr(sig, "detected_at", None),
                capture_content=capture_content,
            )
        except Exception as exc:
            logger.warning("Dunetrace OTel: signal span failed for run %s: %s", run_id, exc)

    for ev in policy_events:
        try:
            payload = ev.get("payload", {}) if isinstance(ev, dict) else {}
            emit_policy_span(
                tracer,
                run_id,
                action=payload.get("action_type", "") or payload.get("action", ""),
                policy_name=payload.get("policy_name", ""),
                trigger=payload.get("trigger", ""),
                trigger_value=payload.get("value"),
                org_id=org_id,
                timestamp=ev.get("timestamp") if isinstance(ev, dict) else None,
            )
        except Exception as exc:
            logger.warning("Dunetrace OTel: policy span failed for run %s: %s", run_id, exc)


def _enum_value(v: Any) -> str:
    return v.value if hasattr(v, "value") else str(v)
