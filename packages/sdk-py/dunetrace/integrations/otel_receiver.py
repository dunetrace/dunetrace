"""
OTel span receiver for Dunetrace.

Translates incoming OpenTelemetry spans (gen_ai.* semantic conventions) into
Dunetrace AgentEvents and runs the full behavioral detector suite on them.

Privacy model — Path 2 (self-hosted):
    Raw span content (gen_ai.prompt, gen_ai.completion, tool arguments) is
    SHA-256 hashed HERE, at the receiver boundary, before any AgentEvent is
    created. Nothing downstream — the Dunetrace ingest API, the database, the
    detector — ever sees plaintext. The guarantee is identical to the native
    SDK (Path 1); the only difference is where hashing occurs: in-process for
    Path 1, at the self-hosted receiver boundary for Path 2.

    Path 3 (managed cloud) requires either the native SDK or a customer-side
    hash proxy placed before the cloud endpoint.

Instrumentation paths:

    Path 1 — Native SDK (strongest, recommended for new agents):
        Hash in-process. No raw content leaves the agent process.
        Use: dt.init() / @dt.agent() / middleware.

    Path 2 — OTel receiver, self-hosted:
        Raw spans travel agent → self-hosted receiver on internal network.
        Hashed at receiver boundary before persistence.
        Use: already instrumented with OpenLLMetry, self-hosted Dunetrace.

    Path 3 — OTel receiver, managed cloud (future):
        Raw spans would travel to an external service.
        Requirement: use Path 1 (native SDK), or deploy a customer-side
        hash proxy before cloud ingestion.

Usage with OpenLLMetry (Path 2)::

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from dunetrace import Dunetrace
    from dunetrace.integrations.otel_receiver import DunetraceOTelReceiver

    dt = Dunetrace(api_key="dt_live_...")

    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))   # existing pipeline unchanged
    DunetraceOTelReceiver.attach(provider, dt, agent_id="my-agent")       # add Dunetrace as second exporter

    from traceloop.sdk import Traceloop
    Traceloop.init(app_name="my-agent", tracer_provider=provider)

gen_ai.* attributes read and their privacy handling:

    gen_ai.request.model           → model name (not sensitive, passed as-is)
    gen_ai.usage.prompt_tokens     → integer count (not sensitive)
    gen_ai.usage.completion_tokens → integer count (not sensitive)
    gen_ai.completion.0.finish_reason → short string like "stop" (not sensitive)
    gen_ai.tool.name               → tool name (not sensitive)
    gen_ai.prompt                  → SHA-256 hashed at receiver boundary
    gen_ai.completion              → SHA-256 hashed at receiver boundary
    gen_ai.prompt.0.content        → SHA-256 hashed at receiver boundary
    gen_ai.completion.0.content    → SHA-256 hashed at receiver boundary
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import TYPE_CHECKING, Optional, Sequence

from dunetrace.models import hash_content

if TYPE_CHECKING:
    from dunetrace.client import Dunetrace

logger = logging.getLogger("dunetrace.otel_receiver")

# gen_ai finish-reason attribute — OpenLLMetry uses different keys across versions
_FINISH_REASON_KEYS = (
    "gen_ai.completion.0.finish_reason",
    "gen_ai.response.finish_reasons.0",
    "llm.response.finish_reason",
)


class DunetraceOTelReceiver:
    """
    OTel SpanExporter that translates ``gen_ai.*`` spans into Dunetrace
    behavioral events and runs the detector suite on each completed trace.

    Add it as a ``SimpleSpanProcessor`` exporter alongside your existing
    OTel pipeline — no changes to agent code required.

    :param dt:       Dunetrace client instance.
    :param agent_id: Label for runs. Defaults to the root span name.
    """

    def __init__(self, dt: "Dunetrace", agent_id: str = "") -> None:
        self._dt       = dt
        self._agent_id = agent_id
        # Accumulate spans per trace until the root span arrives.
        self._pending: dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()

    # ── SpanExporter interface ────────────────────────────────────────────────

    def export(self, spans: Sequence) -> int:
        """Called by the OTel SDK with a batch of completed spans."""
        by_trace: dict[str, list] = defaultdict(list)
        for span in spans:
            tid = _trace_id(span)
            by_trace[tid].append(span)

        with self._lock:
            for tid, batch in by_trace.items():
                self._pending[tid].extend(batch)

            # Process any trace that now has its root span
            completed = [
                tid for tid, spans in self._pending.items()
                if _has_root(spans)
            ]
            for tid in completed:
                self._process_trace(self._pending.pop(tid))

        try:
            from opentelemetry.sdk.trace.export import SpanExportResult
            return SpanExportResult.SUCCESS
        except ImportError:
            return 0

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

    # ── Internal ──────────────────────────────────────────────────────────────

    def _process_trace(self, spans: list) -> None:
        spans = sorted(spans, key=lambda s: s.start_time)
        root  = _root_span(spans)

        agent_id = self._agent_id or (root.name if root else "agent")
        model    = _first_attr(spans, "gen_ai.request.model") or "unknown"

        try:
            with self._dt.run(agent_id, model=model) as run:
                for span in spans:
                    _emit_span(run, span)
                run.final_answer()
        except Exception as exc:
            logger.debug("OTel receiver: error processing trace: %s", exc)

    # ── Class-level constructor ───────────────────────────────────────────────

    @classmethod
    def attach(
        cls,
        provider,
        dt: "Dunetrace",
        agent_id: str = "",
        *,
        batch: bool = False,
    ) -> "DunetraceOTelReceiver":
        """
        Convenience method: create a receiver and attach it to *provider*.

        :param provider: ``TracerProvider`` instance.
        :param dt:       Dunetrace client instance.
        :param agent_id: Run label.
        :param batch:    Use ``BatchSpanProcessor`` instead of ``SimpleSpanProcessor``.

        Usage::

            DunetraceOTelReceiver.attach(provider, dt, agent_id="my-agent")
        """
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            SimpleSpanProcessor,
        )
        receiver  = cls(dt, agent_id=agent_id)
        processor = BatchSpanProcessor(receiver) if batch else SimpleSpanProcessor(receiver)
        provider.add_span_processor(processor)
        return receiver


# ── Span helpers ──────────────────────────────────────────────────────────────

def _trace_id(span) -> str:
    return format(span.context.trace_id, "032x")


def _has_root(spans: list) -> bool:
    return any(_is_root(s) for s in spans)


def _root_span(spans: list):
    return next((s for s in spans if _is_root(s)), spans[0] if spans else None)


def _is_root(span) -> bool:
    return span.parent is None or not hasattr(span.parent, "span_id")


def _first_attr(spans: list, key: str):
    for span in spans:
        val = (span.attributes or {}).get(key)
        if val:
            return val
    return None


def _emit_span(run, span) -> None:
    """Emit llm_called/llm_responded or tool_called/tool_responded for one span."""
    attrs      = dict(span.attributes or {})
    latency_ms = max(0, int((span.end_time - span.start_time) / 1_000_000))
    is_error   = _span_is_error(span)

    if "gen_ai.request.model" in attrs:
        _emit_llm(run, attrs, latency_ms, is_error)
    elif "gen_ai.tool.name" in attrs:
        _emit_tool(run, attrs, latency_ms, is_error, span.name)
    # Other span kinds (chains, retrievers, etc.) are silently skipped —
    # only LLM and tool calls feed the behavioral detectors.


def _emit_llm(run, attrs: dict, latency_ms: int, is_error: bool) -> None:
    model         = attrs.get("gen_ai.request.model", "unknown")
    prompt_toks   = int(attrs.get("gen_ai.usage.prompt_tokens", 0) or 0)
    comp_toks     = int(attrs.get("gen_ai.usage.completion_tokens", 0) or 0)
    finish_reason = "error" if is_error else _finish_reason(attrs)

    # Path 2 privacy boundary: hash raw content fields here, at the receiver,
    # before any AgentEvent is constructed.  Non-sensitive metadata (model name,
    # token counts, latency, finish_reason) is passed as-is.
    raw_output = (
        attrs.get("gen_ai.completion")
        or attrs.get("gen_ai.completion.0.content")
        or ""
    )
    output_hash   = hash_content(str(raw_output)) if raw_output else ""
    output_length = len(str(raw_output)) if raw_output else 0

    run.llm_called(model, prompt_tokens=prompt_toks)
    run.llm_responded(
        completion_tokens=comp_toks,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
        output_hash=output_hash,
        output_length=output_length,
    )


def _emit_tool(run, attrs: dict, latency_ms: int, is_error: bool, span_name: str) -> None:
    tool_name = attrs.get("gen_ai.tool.name") or span_name or "tool"
    run.tool_called(tool_name)
    run.tool_responded(tool_name, success=not is_error, latency_ms=latency_ms)


def _finish_reason(attrs: dict) -> str:
    for key in _FINISH_REASON_KEYS:
        val = attrs.get(key)
        if val:
            return str(val)
    return "stop"


def _span_is_error(span) -> bool:
    try:
        from opentelemetry.trace import StatusCode
        return span.status.status_code == StatusCode.ERROR
    except Exception:
        return False
