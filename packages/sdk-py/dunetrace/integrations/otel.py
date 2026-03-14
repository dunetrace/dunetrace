"""
OTel span exporter for Dunetrace. Converts agent events into spans so you can
correlate failure signals with infra metrics in Grafana Tempo, Honeycomb, Datadog, or Jaeger.

Install:
    pip install 'dunetrace[otel]'

Usage:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from dunetrace.integrations.otel import DunetraceOTelExporter

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

    dt = Dunetrace(otel_exporter=DunetraceOTelExporter(provider))

    # No other changes — existing run/tool_called/llm_called code is unchanged.

Span hierarchy:
    Trace (trace_id = run_id as 128-bit int — stable, correlatable)
    └── Span: "agent_run"                 [dunetrace.agent_id, dunetrace.run_id, ...]
        ├── Span: "llm_call"              [gen_ai.request.model, gen_ai.usage.*, ...]
        ├── Span: "tool_call"             [dunetrace.tool_name, dunetrace.success, ...]
        │   └── SpanEvent: "rate_limit"   [from run.external_signal("rate_limit", source="openai")]
        └── Span: "retrieval"             [dunetrace.index_name, dunetrace.result_count]

Failure signals detected at run end annotate the root span:
    dunetrace.signal.0.failure_type = "SLOW_STEP"
    dunetrace.signal.0.severity     = "HIGH"
    dunetrace.signal.0.confidence   = 0.92
    dunetrace.signal.0.evidence.*   = ...  (scalar fields only)
    span.status                     = ERROR  (for HIGH / CRITICAL signals)

parent_run_id is recorded as an attribute. Full W3C trace-context propagation
for nested agent calls requires the caller to pass the parent span context
explicitly via the standard OTel API, see CONTEXT PROPAGATION below.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Optional

try:
    from opentelemetry import trace
    from opentelemetry.trace import (
        NonRecordingSpan,
        SpanContext,
        SpanKind,
        StatusCode,
        TraceFlags,
    )
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

from dunetrace.models import AgentEvent, EventType, RunState, Severity


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ns(ts: float) -> int:
    """Float Unix seconds → OTel nanoseconds."""
    return int(ts * 1_000_000_000)


def _trace_id(run_id: str) -> int:
    """UUID → 128-bit OTel trace ID int."""
    return uuid.UUID(run_id).int


def _root_span_id(run_id: str) -> int:
    """Derive a stable 64-bit span ID from the run UUID (lower 64 bits)."""
    return uuid.UUID(run_id).int & 0xFFFF_FFFF_FFFF_FFFF


# ── Per-run span state ─────────────────────────────────────────────────────────

@dataclass
class _RunSpans:
    root_span:  Any               # opentelemetry Span
    child_span: Any       = None  # currently open child span, or None
    run_state:  Optional[RunState] = None  # set via notify_run_state before run end


# ── Exporter ──────────────────────────────────────────────────────────────────

class DunetraceOTelExporter:
    """
    Translates DuneTrace AgentEvents into OpenTelemetry spans.

    Thread-safe: the internal run-state dict is protected by a lock.
    All span operations are called from the emitting thread (typically
    the agent thread), so span open/close is always sequential per run.
    """

    def __init__(
        self,
        tracer_provider: Any = None,   # opentelemetry TracerProvider
        tracer_name: str = "dunetrace",
    ) -> None:
        if not _OTEL_AVAILABLE:
            raise ImportError(
                "opentelemetry-sdk is not installed. "
                "Run: pip install 'dunetrace[otel]'"
            )
        tp = tracer_provider or trace.get_tracer_provider()
        self._tracer = tp.get_tracer(tracer_name)
        self._runs:  Dict[str, _RunSpans] = {}
        self._lock = Lock()

    # ── Called by Dunetrace client ─────────────────────────────────────────────

    def notify_run_state(self, run_id: str, state: RunState) -> None:
        """
        Called by the client just before run.completed / run.errored is emitted.
        Stores the completed RunState so _on_run_ended can run detectors on it.
        """
        with self._lock:
            rs = self._runs.get(run_id)
        if rs:
            rs.run_state = state

    def handle(self, event: AgentEvent) -> None:
        """Route one AgentEvent to the appropriate span operation."""
        _DISPATCH = {
            EventType.RUN_STARTED:         self._on_run_started,
            EventType.RUN_COMPLETED:       self._on_run_ended,
            EventType.RUN_ERRORED:         self._on_run_ended,
            EventType.LLM_CALLED:          self._on_child_start,
            EventType.LLM_RESPONDED:       self._on_llm_responded,
            EventType.TOOL_CALLED:         self._on_child_start,
            EventType.TOOL_RESPONDED:      self._on_tool_responded,
            EventType.RETRIEVAL_CALLED:    self._on_child_start,
            EventType.RETRIEVAL_RESPONDED: self._on_retrieval_responded,
            EventType.EXTERNAL_SIGNAL:     self._on_external_signal,
        }
        fn = _DISPATCH.get(event.event_type)
        if fn:
            fn(event)

    # ── Span handlers ─────────────────────────────────────────────────────────

    def _on_run_started(self, event: AgentEvent) -> None:
        # Seed the trace with a deterministic ID derived from run_id.
        # The parent is a synthetic NonRecordingSpan that carries the trace ID
        # without appearing as a real span in the backend.
        parent_ctx = trace.set_span_in_context(
            NonRecordingSpan(
                SpanContext(
                    trace_id=_trace_id(event.run_id),
                    span_id=_root_span_id(event.run_id),
                    is_remote=True,
                    trace_flags=TraceFlags(TraceFlags.SAMPLED),
                )
            )
        )
        attrs: Dict[str, Any] = {
            "dunetrace.agent_id":      event.agent_id,
            "dunetrace.run_id":        event.run_id,
            "dunetrace.agent_version": event.agent_version,
            "dunetrace.input_hash":    event.payload.get("input_hash", ""),
            "dunetrace.model":         event.payload.get("model", ""),
            "dunetrace.tools":         ",".join(event.payload.get("tools", [])),
        }
        if event.parent_run_id:
            attrs["dunetrace.parent_run_id"] = event.parent_run_id

        root = self._tracer.start_span(
            "agent_run",
            context=parent_ctx,
            kind=SpanKind.INTERNAL,
            start_time=_ns(event.timestamp),
            attributes=attrs,
        )
        with self._lock:
            self._runs[event.run_id] = _RunSpans(root_span=root)

    def _on_child_start(self, event: AgentEvent) -> None:
        with self._lock:
            rs = self._runs.get(event.run_id)
        if not rs:
            return

        # Guard: close any orphaned child (mismatched called/responded pairs).
        if rs.child_span:
            rs.child_span.end(end_time=_ns(event.timestamp))

        parent_ctx = trace.set_span_in_context(rs.root_span)

        if event.event_type is EventType.LLM_CALLED:
            name = "llm_call"
            attrs: Dict[str, Any] = {
                "gen_ai.operation.name":  "chat",
                "gen_ai.request.model":   event.payload.get("model", ""),
                "dunetrace.step_index":   event.step_index,
            }
            if event.payload.get("prompt_tokens"):
                attrs["gen_ai.usage.input_tokens"] = event.payload["prompt_tokens"]

        elif event.event_type is EventType.TOOL_CALLED:
            name = "tool_call"
            attrs = {
                "dunetrace.tool_name":  event.payload.get("tool_name", ""),
                "dunetrace.args_hash":  event.payload.get("args_hash", ""),
                "dunetrace.step_index": event.step_index,
            }

        else:  # RETRIEVAL_CALLED
            name = "retrieval"
            attrs = {
                "dunetrace.index_name":  event.payload.get("index_name", ""),
                "dunetrace.query_hash":  event.payload.get("query_hash", ""),
                "dunetrace.step_index":  event.step_index,
            }

        rs.child_span = self._tracer.start_span(
            name,
            context=parent_ctx,
            kind=SpanKind.INTERNAL,
            start_time=_ns(event.timestamp),
            attributes=attrs,
        )

    def _on_llm_responded(self, event: AgentEvent) -> None:
        with self._lock:
            rs = self._runs.get(event.run_id)
        if not rs or not rs.child_span:
            return

        p, span = event.payload, rs.child_span
        span.set_attribute("gen_ai.response.finish_reason", p.get("finish_reason", ""))
        if p.get("completion_tokens"):
            span.set_attribute("gen_ai.usage.output_tokens", p["completion_tokens"])
        if p.get("latency_ms"):
            span.set_attribute("dunetrace.latency_ms", p["latency_ms"])
        if p.get("output_length"):
            span.set_attribute("dunetrace.output_length", p["output_length"])
        if p.get("finish_reason") == "length":
            span.set_status(StatusCode.ERROR, "LLM output truncated (finish_reason=length)")

        span.end(end_time=_ns(event.timestamp))
        rs.child_span = None

    def _on_tool_responded(self, event: AgentEvent) -> None:
        with self._lock:
            rs = self._runs.get(event.run_id)
        if not rs or not rs.child_span:
            return

        p, span = event.payload, rs.child_span
        success = p.get("success", True)
        span.set_attribute("dunetrace.success", success)
        if p.get("output_length"):
            span.set_attribute("dunetrace.output_length", p["output_length"])
        if p.get("latency_ms"):
            span.set_attribute("dunetrace.latency_ms", p["latency_ms"])
        if p.get("error_hash"):
            span.set_attribute("dunetrace.error_hash", p["error_hash"])
        if not success:
            span.set_status(StatusCode.ERROR, "tool call failed")

        span.end(end_time=_ns(event.timestamp))
        rs.child_span = None

    def _on_retrieval_responded(self, event: AgentEvent) -> None:
        with self._lock:
            rs = self._runs.get(event.run_id)
        if not rs or not rs.child_span:
            return

        p, span = event.payload, rs.child_span
        span.set_attribute("dunetrace.result_count", p.get("result_count", 0))
        if p.get("top_score") is not None:
            span.set_attribute("dunetrace.top_score", p["top_score"])
        if p.get("latency_ms"):
            span.set_attribute("dunetrace.latency_ms", p["latency_ms"])
        if p.get("result_count", 0) == 0:
            span.set_status(StatusCode.ERROR, "retrieval returned 0 results")

        span.end(end_time=_ns(event.timestamp))
        rs.child_span = None

    def _on_external_signal(self, event: AgentEvent) -> None:
        """
        Attach the signal as a SpanEvent on the currently-open child span
        (so it appears at the right time within the tool/LLM call), or on
        the root span if no child is open.
        """
        with self._lock:
            rs = self._runs.get(event.run_id)
        if not rs:
            return

        target = rs.child_span if rs.child_span is not None else rs.root_span
        attrs: Dict[str, Any] = {
            "dunetrace.signal_name": event.payload.get("signal_name", ""),
        }
        if event.payload.get("source"):
            attrs["dunetrace.source"] = event.payload["source"]
        for k, v in event.payload.get("meta", {}).items():
            if isinstance(v, (str, int, float, bool)):
                attrs[f"dunetrace.meta.{k}"] = v

        target.add_event(
            event.payload.get("signal_name", "external_signal"),
            attributes=attrs,
            timestamp=_ns(event.timestamp),
        )

    def _on_run_ended(self, event: AgentEvent) -> None:
        with self._lock:
            rs = self._runs.pop(event.run_id, None)
        if not rs:
            return

        # Close any orphaned child span (e.g. tool that never got a response).
        if rs.child_span:
            rs.child_span.end(end_time=_ns(event.timestamp))

        root = rs.root_span

        # Run Tier 1 detectors on the completed RunState and annotate the
        # root span. Each signal becomes a set of indexed attributes so any
        # OTel backend can display them without custom parsing.
        if rs.run_state is not None:
            from dunetrace.detectors import run_detectors
            signals = run_detectors(rs.run_state)
            for i, sig in enumerate(signals):
                pfx = f"dunetrace.signal.{i}"
                root.set_attribute(f"{pfx}.failure_type", sig.failure_type.value)
                root.set_attribute(f"{pfx}.severity",     sig.severity.value)
                root.set_attribute(f"{pfx}.confidence",   sig.confidence)
                root.set_attribute(f"{pfx}.step_index",   sig.step_index)
                for k, v in sig.evidence.items():
                    if isinstance(v, (str, int, float, bool)):
                        root.set_attribute(f"{pfx}.evidence.{k}", v)

            # Set span ERROR status for any HIGH or CRITICAL signal.
            severe = [s for s in signals if s.severity in (Severity.HIGH, Severity.CRITICAL)]
            if severe:
                worst = severe[0]
                root.set_status(
                    StatusCode.ERROR,
                    f"{worst.failure_type.value} [{worst.severity.value}]",
                )

        root.set_attribute("dunetrace.total_steps",     event.payload.get("total_steps", 0))
        root.set_attribute("dunetrace.exit_reason",     event.payload.get("exit_reason", ""))
        root.set_attribute("dunetrace.tool_call_count", event.payload.get("tool_call_count", 0))

        if event.event_type is EventType.RUN_ERRORED:
            root.set_attribute("dunetrace.error_type", event.payload.get("error_type", ""))
            root.set_status(
                StatusCode.ERROR,
                event.payload.get("error_type", "run errored"),
            )

        root.end(end_time=_ns(event.timestamp))
