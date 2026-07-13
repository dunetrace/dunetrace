"""
OTLP/HTTP trace → Dunetrace event mapper.

Converts OpenTelemetry spans into Dunetrace IngestEvent dicts that can be
written directly to the events table via the existing insert_events() path.

Supported instrumentation libraries:
  - opentelemetry-instrumentation-openai   (OpenAI SDK)
  - opentelemetry-instrumentation-anthropic
  - opentelemetry-instrumentation-langchain (LangChain / LangGraph)
  - traceloop-sdk (OpenLLMetry — Gen AI semantic conventions)
  - Any OTel SDK instrumentation following Gen AI semconv

Span → event mapping:
  Root span (no parent)     → run.started + run.completed / run.errored
  LLM spans                 → llm.called + llm.responded
  Tool spans                → tool.called + tool.responded
  Retrieval / vector spans  → retrieval.called + retrieval.responded
  Other / lifecycle spans   → skipped (covered by root span)

Each OTLP request is treated as a self-contained batch. Spans are sorted by
startTimeUnixNano and assigned sequential step_index values.
"""

from __future__ import annotations

import base64
import logging
import time
import uuid
from typing import Any

logger = logging.getLogger("dunetrace.ingest.otel")

# ── Protobuf decoding ──────────────────────────────────────────────────────────
#
# google.protobuf.json_format.MessageToDict() already produces the exact
# proto3 JSON mapping for an ExportTraceServiceRequest — which IS the OTLP/
# HTTP JSON wire format (resourceSpans/scopeSpans/spans, startTimeUnixNano as
# a string, attributes as {key, value: {stringValue: ...}} — the same shape
# otlp_to_events() below already expects from a JSON request body). So no
# separate protobuf-specific mapper is needed; only one wrinkle needs fixing
# up afterward: MessageToDict base64-encodes `bytes` fields (traceId/spanId/
# parentSpanId are `bytes` in the .proto schema), but the OTLP spec's own
# JSON convention — and everything in this module — represents them as plain
# hex strings, not base64. _fix_ids_to_hex() corrects that in place.


def protobuf_to_resource_spans(raw: bytes) -> list[dict]:
    """Parse an application/x-protobuf ExportTraceServiceRequest body into
    the same resourceSpans list[dict] shape otlp_to_events() expects from
    JSON. Raises on malformed protobuf — callers should catch and 400."""
    from google.protobuf.json_format import MessageToDict
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    req = ExportTraceServiceRequest()
    req.ParseFromString(raw)
    body = MessageToDict(req)
    resource_spans = body.get("resourceSpans", [])
    _fix_ids_to_hex(resource_spans)
    return resource_spans


def _fix_ids_to_hex(resource_spans: list[dict]) -> None:
    for rs in resource_spans:
        for scope_span in rs.get("scopeSpans", []):
            for span in scope_span.get("spans", []):
                for key in ("traceId", "spanId", "parentSpanId"):
                    value = span.get(key)
                    if value:
                        span[key] = base64.b64decode(value).hex()


# ── Attribute helpers ──────────────────────────────────────────────────────────


def _val(v: dict) -> Any:
    """Extract a Python value from an OTLP AnyValue dict."""
    if "stringValue" in v:
        return v["stringValue"]
    if "intValue" in v:
        return int(v["intValue"])  # intValue is often a string in JSON
    if "doubleValue" in v:
        return float(v["doubleValue"])
    if "boolValue" in v:
        return bool(v["boolValue"])
    if "arrayValue" in v:
        return [_val(x) for x in v["arrayValue"].get("values", [])]
    if "kvlistValue" in v:
        return {kv["key"]: _val(kv["value"]) for kv in v["kvlistValue"].get("values", [])}
    return None


def _attr(attrs: list[dict], key: str, default: Any = None) -> Any:
    """Look up a single attribute by key from an OTLP attribute list."""
    for a in attrs:
        if a.get("key") == key:
            return _val(a.get("value", {}))
    return default


def _attrs(attrs: list[dict]) -> dict[str, Any]:
    """Flatten an OTLP attribute list to a plain dict."""
    return {a["key"]: _val(a.get("value", {})) for a in attrs if "key" in a}


# ── Span classification ────────────────────────────────────────────────────────

# Signals that identify LLM provider spans in the span name
_LLM_NAME_TOKENS = {
    "openai",
    "anthropic",
    "gemini",
    "claude",
    "gpt",
    "cohere",
    "mistral",
    "bedrock",
    "ollama",
    "llm",
    "chat",
    "completion",
    "gen_ai",
    "langfuse",
    "together",
    "groq",
}
_TOOL_NAME_TOKENS = {"tool", "function_call", "function"}
_RETRIEVAL_TOKENS = {
    "retrieval",
    "retrieve",
    "vector",
    "search",
    "embed",
    "rag",
    "pinecone",
    "weaviate",
    "chroma",
    "qdrant",
    "milvus",
    "faiss",
}
_LIFECYCLE_TOKENS = {
    "agent",
    "chain",
    "executor",
    "graph",
    "workflow",
    "pipeline",
    "run",
    "invoke",
    "arun",
    "ainvoke",
}


def _classify(name: str, ad: dict[str, Any]) -> str:
    """Classify a span as 'llm' | 'tool' | 'retrieval' | 'lifecycle' | 'skip'.

    Attribute-first: Gen AI / OpenLLMetry semantic conventions are definitive.
    Falls back to span name token matching.
    """
    # ── Attribute-based (definitive) ──────────────────────────────────────────
    if (
        ad.get("gen_ai.system")
        or ad.get("gen_ai.request.model")
        or ad.get("llm.request.model")
        or ad.get("llm.vendor")
    ):
        return "llm"

    if ad.get("tool.name") or ad.get("gen_ai.tool.name"):
        return "tool"

    if (
        ad.get("retrieval.index_name")
        or ad.get("vector_db.vendor")
        or ad.get("vector_db.collection_name")
    ):
        return "retrieval"

    # ── Name-based (fallback) ─────────────────────────────────────────────────
    low = name.lower()
    tokens = set(low.replace(".", " ").replace("-", " ").replace("_", " ").split())

    if tokens & _LLM_NAME_TOKENS or any(t in low for t in _LLM_NAME_TOKENS):
        return "llm"
    if tokens & _TOOL_NAME_TOKENS or any(t in low for t in _TOOL_NAME_TOKENS):
        return "tool"
    if any(t in low for t in _RETRIEVAL_TOKENS):
        return "retrieval"
    if tokens & _LIFECYCLE_TOKENS or any(t in low for t in _LIFECYCLE_TOKENS):
        return "lifecycle"

    return "skip"


# ── Time helpers ───────────────────────────────────────────────────────────────


def _ns(ns: str | int | None) -> float:
    """Convert nanosecond timestamp (possibly a string) to Unix float seconds."""
    if not ns:
        return time.time()
    return int(ns) / 1_000_000_000


def _latency_ms(start_ns: Any, end_ns: Any) -> int | None:
    s, e = int(start_ns or 0), int(end_ns or 0)
    return round((e - s) / 1_000_000) if e > s else None


def _is_root(span: dict) -> bool:
    pid = span.get("parentSpanId", "")
    return not pid or set(pid) <= {"0"}


def _trace_to_uuid(trace_id: str) -> str:
    """Map a 32-hex OTel traceId to UUID string format."""
    t = trace_id.replace("-", "").zfill(32)[:32]
    return f"{t[0:8]}-{t[8:12]}-{t[12:16]}-{t[16:20]}-{t[20:32]}"


# ── Core converter ─────────────────────────────────────────────────────────────


def otlp_to_events(
    resource_spans: list[dict],
    agent_id_override: str | None = None,
    agent_version_override: str | None = None,
    batch_id: str | None = None,
) -> list[dict]:
    """
    Convert an OTLP resourceSpans payload into Dunetrace event dicts.

    Returns a flat list of event dicts compatible with insert_events().
    One OTLP request can yield events for multiple traces / runs.

    Each resourceSpan and each trace is processed in isolation (own
    try/except) — a single malformed resource or trace is logged and
    skipped rather than raising and losing every other valid trace in the
    same batch. A real OTel Collector batches spans from many concurrent
    runs into one export; one bad one shouldn't cost the rest.
    """
    batch_id = batch_id or str(uuid.uuid4())

    # Collect spans per traceId, preserving resource metadata
    traces: dict[str, dict] = {}

    for rs in resource_spans:
        try:
            res_attrs = _attrs(rs.get("resource", {}).get("attributes", []))
            agent_id = agent_id_override or res_attrs.get("service.name", "unknown-agent")
            version = agent_version_override or res_attrs.get("service.version", "unknown")

            for scope_span in rs.get("scopeSpans", []):
                for span in scope_span.get("spans", []):
                    tid = span.get("traceId", "") or str(uuid.uuid4()).replace("-", "")
                    if tid not in traces:
                        traces[tid] = {
                            "agent_id": agent_id,
                            "agent_version": version,
                            "spans": [],
                        }
                    traces[tid]["spans"].append(span)
        except Exception as exc:
            logger.warning(
                "OTLP: skipping malformed resourceSpan. batch_id=%s error=%s", batch_id, exc
            )

    events: list[dict] = []

    for trace_id, trace in traces.items():
        try:
            events.extend(_events_for_trace(trace_id, trace))
        except Exception as exc:
            logger.warning(
                "OTLP: skipping malformed trace %s. batch_id=%s error=%s", trace_id, batch_id, exc
            )

    return events


def _events_for_trace(trace_id: str, trace: dict) -> list[dict]:
    """Build the run.started/.../run.completed|errored event sequence for
    one trace's spans. Raises on malformed span data — otlp_to_events()
    catches it per-trace so one bad trace doesn't affect the others."""
    events: list[dict] = []
    run_id = _trace_to_uuid(trace_id)
    agent_id = trace["agent_id"]
    version = trace["agent_version"]

    # Sort chronologically
    spans = sorted(trace["spans"], key=lambda s: int(s.get("startTimeUnixNano") or 0))

    root = next((s for s in spans if _is_root(s)), spans[0] if spans else None)
    if not root:
        return events

    root_ad = _attrs(root.get("attributes", []))
    started_ts = _ns(root.get("startTimeUnixNano"))
    ended_ts = _ns(root.get("endTimeUnixNano")) or time.time()
    errored = root.get("status", {}).get("code") == 2  # STATUS_CODE_ERROR

    # Infer model from root span if present (some frameworks attach it there)
    root_model = root_ad.get("gen_ai.request.model") or root_ad.get("llm.request.model") or ""

    def _ev(event_type: str, step: int, ts: float, payload: dict) -> dict:
        return {
            "event_type": event_type,
            "run_id": run_id,
            "agent_id": agent_id,
            "agent_version": version,
            "step_index": step,
            "timestamp": ts,
            "payload": payload,
            # Raw OTel trace_id, not the lossy _trace_to_uuid() conversion
            # used for run_id — external evaluation integrations (Langfuse/
            # LangSmith/Braintrust) correlate against the real trace_id.
            "trace_id": trace_id,
        }

    events.append(
        _ev(
            "run.started",
            0,
            started_ts,
            {
                "model": root_model,
                "tools": [],
                "source": "otlp",
            },
        )
    )

    step = 1
    for span in spans:
        if span is root:
            continue

        ad = _attrs(span.get("attributes", []))
        name = span.get("name", "")
        kind = _classify(name, ad)
        start_ts = _ns(span.get("startTimeUnixNano"))
        end_ts = _ns(span.get("endTimeUnixNano")) or time.time()
        lat_ms = _latency_ms(span.get("startTimeUnixNano"), span.get("endTimeUnixNano"))
        span_ok = span.get("status", {}).get("code") != 2

        if kind == "llm":
            model = (
                ad.get("gen_ai.request.model")
                or ad.get("llm.request.model")
                or ad.get("gen_ai.system")
                or root_model
                or ""
            )
            # Accept both Gen AI semconv (input/output) and OpenLLMetry (prompt/completion)
            prompt_tok = int(
                ad.get("gen_ai.usage.input_tokens") or ad.get("llm.usage.prompt_tokens") or 0
            )
            completion_tok = int(
                ad.get("gen_ai.usage.output_tokens") or ad.get("llm.usage.completion_tokens") or 0
            )
            finish_reasons = ad.get("gen_ai.response.finish_reasons") or []
            finish_reason = (
                finish_reasons[0]
                if isinstance(finish_reasons, list) and finish_reasons
                else ad.get("llm.response.finish_reason") or "stop"
            )

            events.append(
                _ev(
                    "llm.called",
                    step,
                    start_ts,
                    {
                        "model": model,
                        "prompt_tokens": prompt_tok,
                    },
                )
            )
            events.append(
                _ev(
                    "llm.responded",
                    step,
                    end_ts,
                    {
                        "model": model,
                        "prompt_tokens": prompt_tok,
                        "completion_tokens": completion_tok,
                        "finish_reason": finish_reason,
                        "latency_ms": lat_ms,
                        "output_length": completion_tok,
                    },
                )
            )

        elif kind == "tool":
            tool_name = (
                ad.get("tool.name")
                or ad.get("gen_ai.tool.name")
                or ad.get("traceloop.entity.name")
                or name
            )
            events.append(
                _ev(
                    "tool.called",
                    step,
                    start_ts,
                    {
                        "tool_name": tool_name,
                    },
                )
            )
            events.append(
                _ev(
                    "tool.responded",
                    step,
                    end_ts,
                    {
                        "tool_name": tool_name,
                        "success": span_ok,
                        "latency_ms": lat_ms,
                        "output_length": 0,
                    },
                )
            )

        elif kind == "retrieval":
            index = (
                ad.get("retrieval.index_name")
                or ad.get("vector_db.collection_name")
                or ad.get("db.name")
                or name
            )
            result_count = int(ad.get("retrieval.result_count") or ad.get("db.result_count") or 0)
            top_score = ad.get("retrieval.top_score")

            events.append(
                _ev(
                    "retrieval.called",
                    step,
                    start_ts,
                    {
                        "index_name": index,
                    },
                )
            )
            events.append(
                _ev(
                    "retrieval.responded",
                    step,
                    end_ts,
                    {
                        "index_name": index,
                        "result_count": result_count,
                        "top_score": top_score,
                        "latency_ms": lat_ms,
                    },
                )
            )

        else:
            # lifecycle / skip — not a distinct Dunetrace event type
            step -= 1  # don't consume a step slot

        step += 1

    events.append(
        _ev(
            "run.errored" if errored else "run.completed",
            step,
            ended_ts,
            {"exit_reason": "error" if errored else "completed"},
        )
    )

    return events

    return events
