"""
OTel span receiver for Dunetrace.

Translates incoming OpenTelemetry spans (gen_ai.* semantic conventions) into
Dunetrace AgentEvents and runs the full structural detector suite on them.
Span content (gen_ai.prompt, gen_ai.completion, tool arguments) is carried
through as-is, same as the native SDK — nothing is hashed or stripped at the
receiver boundary.

Use this when an agent is already instrumented with an OTel-based tracer
(e.g. OpenLLMetry/Traceloop) and you want Dunetrace's detectors without adding
manual dt.run() / @dt.agent() instrumentation. Attach it as a second span
processor alongside whatever exporter you already have.

Usage::

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

Attributes read (Gen AI semconv first, then the OpenLLMetry / vector-store keys
real emitters also use):

  LLM span:
    gen_ai.request.model / gen_ai.response.model / llm.request.model  -> model
    gen_ai.usage.input_tokens  (or gen_ai.usage.prompt_tokens,
                                llm.usage.prompt_tokens)               -> prompt tokens
    gen_ai.usage.output_tokens (or gen_ai.usage.completion_tokens,
                                llm.usage.completion_tokens)           -> completion tokens
    gen_ai.usage.reasoning_tokens                                     -> reasoning tokens
    gen_ai.completion / .0.content / traceloop.entity.output          -> output text
    gen_ai.completion.0.finish_reason and variants                    -> finish reason

  Tool span:
    gen_ai.tool.name / tool.name                                     -> tool name
    gen_ai.tool.call.arguments / tool.arguments /
      traceloop.entity.input                                         -> args
    gen_ai.tool.call.result / tool.result / traceloop.entity.output  -> output

  Retrieval span:
    retrieval.index_name / vector_db.collection_name / db.name       -> index
    retrieval.result_count / db.result_count                         -> document count
    retrieval.top_score                                              -> top score
    retrieval.documents / traceloop.entity.output                    -> content

Voice events have no OTel convention and are best sent via the SDK directly.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from typing import TYPE_CHECKING, Optional, Sequence


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
    structural events and runs the detector suite on each completed trace.

    Add it as a ``SimpleSpanProcessor`` exporter alongside your existing
    OTel pipeline — no changes to agent code required.

    :param dt:       Dunetrace client instance.
    :param agent_id: Label for runs. Defaults to the root span name.
    """

    def __init__(self, dt: "Dunetrace", agent_id: str = "") -> None:
        self._dt = dt
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
            completed = [tid for tid, spans in self._pending.items() if _has_root(spans)]
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
        root = _root_span(spans)

        agent_id = self._agent_id or (root.name if root else "agent")
        model = _first_attr(spans, "gen_ai.request.model") or "unknown"

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

        receiver = cls(dt, agent_id=agent_id)
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


# Attribute keys that mark each span kind. Gen AI semconv first, then the
# OpenLLMetry / vector-store keys real emitters (OpenLIT, Traceloop) also use.
_LLM_KEYS = (
    "gen_ai.request.model",
    "gen_ai.response.model",
    "gen_ai.system",
    "gen_ai.provider.name",
    "llm.request.model",
)
_TOOL_KEYS = ("gen_ai.tool.name", "tool.name")
_RETRIEVAL_KEYS = (
    "retrieval.index_name",
    "vector_db.collection_name",
    "vector_db.vendor",
    "db.name",
)


def _has_any(attrs: dict, keys: tuple) -> bool:
    return any(k in attrs for k in keys)


def _attr_text(attrs: dict, keys: tuple) -> str:
    """First present attribute among keys, as a string. Non-string values
    (a dict of tool args, a list of documents) are JSON-serialized."""
    for key in keys:
        val = attrs.get(key)
        if val:
            return val if isinstance(val, str) else json.dumps(val, default=str)
    return ""


def _messages_content(value) -> str:
    """Extract text from the gen_ai.output.messages structure (current GenAI
    convention, emitted by Traceloop): a list of messages each with `parts`
    [{type, content}]. Accepts a JSON string or a parsed list."""
    try:
        data = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return ""
    if not isinstance(data, list):
        return ""
    texts = []
    for msg in data:
        if not isinstance(msg, dict):
            continue
        parts = msg.get("parts")
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, dict) and part.get("content"):
                    texts.append(str(part["content"]))
        elif msg.get("content"):
            texts.append(str(msg["content"]))
    return " ".join(t for t in texts if t)


def _llm_output(attrs: dict) -> str:
    """Assistant output text across conventions: the plain-string keys, then the
    structured gen_ai.output.messages form."""
    text = _attr_text(
        attrs, ("gen_ai.completion", "gen_ai.completion.0.content", "traceloop.entity.output")
    )
    if text:
        return text
    messages = attrs.get("gen_ai.output.messages")
    return _messages_content(messages) if messages else ""


def _emit_span(run, span) -> None:
    """Emit the Dunetrace call(s) for one span. LLM and tool spans feed the
    structural detectors most heavily; retrieval spans feed the RAG detectors.
    Chains, agents, and other lifecycle spans have no distinct Dunetrace event
    and are skipped."""
    attrs = dict(span.attributes or {})
    latency_ms = max(0, int((span.end_time - span.start_time) / 1_000_000))
    is_error = _span_is_error(span)

    if _has_any(attrs, _LLM_KEYS):
        _emit_llm(run, attrs, latency_ms, is_error)
    elif _has_any(attrs, _TOOL_KEYS):
        _emit_tool(run, attrs, latency_ms, is_error, span.name)
    elif _has_any(attrs, _RETRIEVAL_KEYS):
        _emit_retrieval(run, attrs, latency_ms, span.name)


def _emit_llm(run, attrs: dict, latency_ms: int, is_error: bool) -> None:
    model = (
        attrs.get("gen_ai.request.model")
        or attrs.get("gen_ai.response.model")
        or attrs.get("llm.request.model")
        or "unknown"
    )
    # Current Gen AI semconv is input_tokens/output_tokens; the older
    # OpenLLMetry naming (prompt/completion) is accepted as a fallback so both
    # modern and legacy emitters populate token counts.
    prompt_toks = int(
        attrs.get("gen_ai.usage.input_tokens")
        or attrs.get("gen_ai.usage.prompt_tokens")
        or attrs.get("llm.usage.prompt_tokens")
        or 0
    )
    comp_toks = int(
        attrs.get("gen_ai.usage.output_tokens")
        or attrs.get("gen_ai.usage.completion_tokens")
        or attrs.get("llm.usage.completion_tokens")
        or 0
    )
    reason_toks = int(attrs.get("gen_ai.usage.reasoning_tokens", 0) or 0)
    finish_reason = "error" if is_error else _finish_reason(attrs)

    output_text = _llm_output(attrs)

    run.llm_called(model, prompt_tokens=prompt_toks)
    run.llm_responded(
        completion_tokens=comp_toks,
        reasoning_tokens=reason_toks,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
        output=output_text,
        output_length=len(output_text),
    )


def _emit_tool(run, attrs: dict, latency_ms: int, is_error: bool, span_name: str) -> None:
    tool_name = attrs.get("gen_ai.tool.name") or attrs.get("tool.name") or span_name or "tool"
    args = _attr_text(
        attrs,
        ("gen_ai.tool.call.arguments", "tool.arguments", "traceloop.entity.input"),
    )
    output = _attr_text(
        attrs, ("gen_ai.tool.call.result", "tool.result", "traceloop.entity.output")
    )
    run.tool_called(tool_name, args or None)
    run.tool_responded(tool_name, success=not is_error, latency_ms=latency_ms, output=output)


def _emit_retrieval(run, attrs: dict, latency_ms: int, span_name: str) -> None:
    index = (
        attrs.get("retrieval.index_name")
        or attrs.get("vector_db.collection_name")
        or attrs.get("db.name")
        or span_name
        or "retrieval"
    )
    result_count = int(attrs.get("retrieval.result_count") or attrs.get("db.result_count") or 0)
    top_score = attrs.get("retrieval.top_score")
    content = _attr_text(attrs, ("retrieval.documents", "traceloop.entity.output"))
    run.retrieval_called(index)
    run.retrieval_responded(
        index,
        result_count=result_count,
        top_score=top_score,
        latency_ms=latency_ms,
        content=content,
    )


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
