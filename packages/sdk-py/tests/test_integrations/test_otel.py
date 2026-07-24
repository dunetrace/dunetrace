"""
Phase 2 tests for DunetraceOTelExporter — run spans and LLM spans with correct
OpenTelemetry conventions (dunetrace.run.* for the run, gen_ai.* for the LLM
call). Tool/retrieval/voice spans arrive in Phase 3, signals/policies in Phase 4.

Everything runs against InMemorySpanExporter — no real OTLP endpoint needed.
"""

from __future__ import annotations

import time
import unittest
import uuid
from typing import Optional

try:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from opentelemetry.trace import SpanKind, StatusCode

    from dunetrace.integrations.otel import (
        DunetraceOTelExporter,
        _provider_from_model,
        _root_span_id,
        _trace_id,
        emit_policy_span,
        emit_run_findings,
        emit_signal_span,
        root_span_id_hex,
        trace_id_hex,
    )
    from dunetrace.models import AgentEvent, EventType, FailureSignal, FailureType, Severity

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

if not _OTEL_AVAILABLE:
    raise unittest.SkipTest("opentelemetry not installed — skipping OTel exporter tests")


# ── Helpers ──────────────────────────────────────────────────────────────────────


def _make_exporter():
    """Return (DunetraceOTelExporter, InMemorySpanExporter)."""
    mem = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(mem))
    return DunetraceOTelExporter(tracer_provider=provider), mem


def _event(
    event_type: EventType,
    run_id: str,
    *,
    agent_id: str = "test-agent",
    step_index: int = 0,
    payload: Optional[dict] = None,
    parent_run_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    ts: Optional[float] = None,
) -> AgentEvent:
    return AgentEvent(
        event_type=event_type,
        run_id=run_id,
        agent_id=agent_id,
        agent_version="v1",
        step_index=step_index,
        timestamp=ts if ts is not None else time.time(),
        payload=payload or {},
        parent_run_id=parent_run_id,
        conversation_id=conversation_id,
    )


def _run_completed(run_id: str, **payload) -> AgentEvent:
    base = {"total_steps": 0, "exit_reason": "completed", "tool_call_count": 0}
    base.update(payload)
    return _event(EventType.RUN_COMPLETED, run_id, payload=base)


def _named(spans, name):
    return next(s for s in spans if s.name == name)


# ── ID derivation ─────────────────────────────────────────────────────────────────


class TestIdDerivation:
    def test_trace_id_deterministic(self):
        run_id = str(uuid.uuid4())
        assert trace_id_hex(run_id) == trace_id_hex(run_id)

    def test_trace_id_matches_uuid_int(self):
        run_id = str(uuid.uuid4())
        assert trace_id_hex(run_id) == format(uuid.UUID(run_id).int, "032x")

    def test_trace_id_differs_per_run(self):
        assert trace_id_hex(str(uuid.uuid4())) != trace_id_hex(str(uuid.uuid4()))

    def test_root_span_id_is_lower_64_bits(self):
        run_id = str(uuid.uuid4())
        expected = uuid.UUID(run_id).int & 0xFFFF_FFFF_FFFF_FFFF
        assert root_span_id_hex(run_id) == format(expected, "016x")


# ── Provider inference ────────────────────────────────────────────────────────────


class TestProviderInference:
    def test_openai_models(self):
        assert _provider_from_model("gpt-4o") == "openai"
        assert _provider_from_model("o3-mini") == "openai"

    def test_anthropic_models(self):
        assert _provider_from_model("claude-3-5-sonnet") == "anthropic"

    def test_google_models(self):
        assert _provider_from_model("gemini-1.5-pro") == "gcp.gemini"

    def test_unknown_model_returns_empty(self):
        assert _provider_from_model("llama-3-70b") == ""
        assert _provider_from_model("") == ""


# ── Run span ───────────────────────────────────────────────────────────────────────


class TestRunSpan:
    def test_run_creates_dunetrace_run_span(self):
        dt_otel, mem = _make_exporter()
        run_id = str(uuid.uuid4())
        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={"model": "gpt-4o"}))
        dt_otel.handle(_run_completed(run_id))

        root = _named(mem.get_finished_spans(), "dunetrace.run")
        assert root.kind == SpanKind.INTERNAL
        assert root.attributes["dunetrace.run.id"] == run_id
        assert root.attributes["dunetrace.run.agent_id"] == "test-agent"
        assert root.attributes["dunetrace.run.agent_version"] == "v1"

    def test_run_trace_id_derived_from_run_id(self):
        dt_otel, mem = _make_exporter()
        run_id = str(uuid.uuid4())
        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(_run_completed(run_id))

        root = _named(mem.get_finished_spans(), "dunetrace.run")
        assert root.context.trace_id == _trace_id(run_id)

    def test_run_records_model_and_tools(self):
        dt_otel, mem = _make_exporter()
        run_id = str(uuid.uuid4())
        dt_otel.handle(
            _event(
                EventType.RUN_STARTED,
                run_id,
                payload={"model": "gpt-4o", "tools": ["search", "calc"]},
            )
        )
        dt_otel.handle(_run_completed(run_id))

        root = _named(mem.get_finished_spans(), "dunetrace.run")
        assert root.attributes["dunetrace.run.model"] == "gpt-4o"
        assert root.attributes["dunetrace.run.tools"] == "search,calc"

    def test_run_conversation_id_set_when_present(self):
        dt_otel, mem = _make_exporter()
        run_id = str(uuid.uuid4())
        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}, conversation_id="conv-9"))
        dt_otel.handle(_run_completed(run_id))

        root = _named(mem.get_finished_spans(), "dunetrace.run")
        assert root.attributes["dunetrace.run.conversation_id"] == "conv-9"

    def test_run_completed_sets_status_and_duration(self):
        dt_otel, mem = _make_exporter()
        run_id = str(uuid.uuid4())
        t0 = time.time()
        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}, ts=t0))
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 3, "exit_reason": "final_answer", "tool_call_count": 1},
                ts=t0 + 2.0,
            )
        )

        root = _named(mem.get_finished_spans(), "dunetrace.run")
        assert root.attributes["dunetrace.run.status"] == "completed"
        assert root.attributes["dunetrace.run.duration_ms"] == 2000
        assert root.attributes["dunetrace.run.total_steps"] == 3
        assert root.attributes["dunetrace.run.exit_reason"] == "final_answer"
        assert root.attributes["dunetrace.run.tool_call_count"] == 1
        assert root.status.status_code != StatusCode.ERROR

    def test_run_errored_sets_failed_status_and_error_type(self):
        dt_otel, mem = _make_exporter()
        run_id = str(uuid.uuid4())
        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.RUN_ERRORED,
                run_id,
                payload={"error_type": "TimeoutError", "total_steps": 1},
            )
        )

        root = _named(mem.get_finished_spans(), "dunetrace.run")
        assert root.attributes["dunetrace.run.status"] == "failed"
        assert root.attributes["dunetrace.run.error_type"] == "TimeoutError"
        assert root.status.status_code == StatusCode.ERROR


# ── LLM span ─────────────────────────────────────────────────────────────────────


class TestLlmSpan:
    def _run_with_llm(self, called_payload, responded_payload):
        dt_otel, mem = _make_exporter()
        run_id = str(uuid.uuid4())
        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(_event(EventType.LLM_CALLED, run_id, step_index=1, payload=called_payload))
        dt_otel.handle(
            _event(EventType.LLM_RESPONDED, run_id, step_index=1, payload=responded_payload)
        )
        dt_otel.handle(_run_completed(run_id))
        return mem.get_finished_spans()

    def test_llm_span_name_is_chat_model(self):
        spans = self._run_with_llm({"model": "gpt-4o"}, {"finish_reason": "stop"})
        assert _named(spans, "chat gpt-4o") is not None

    def test_llm_span_kind_is_client(self):
        spans = self._run_with_llm({"model": "gpt-4o"}, {"finish_reason": "stop"})
        assert _named(spans, "chat gpt-4o").kind == SpanKind.CLIENT

    def test_llm_genai_attributes(self):
        spans = self._run_with_llm(
            {"model": "claude-3-5-sonnet", "prompt_tokens": 300},
            {"finish_reason": "stop", "completion_tokens": 120},
        )
        llm = _named(spans, "chat claude-3-5-sonnet")
        assert llm.attributes["gen_ai.operation.name"] == "chat"
        assert llm.attributes["gen_ai.provider.name"] == "anthropic"
        assert llm.attributes["gen_ai.request.model"] == "claude-3-5-sonnet"
        assert llm.attributes["gen_ai.usage.input_tokens"] == 300
        assert llm.attributes["gen_ai.usage.output_tokens"] == 120

    def test_llm_output_tokens_include_reasoning(self):
        spans = self._run_with_llm(
            {"model": "o3"},
            {"finish_reason": "stop", "completion_tokens": 50, "reasoning_tokens": 200},
        )
        llm = _named(spans, "chat o3")
        assert llm.attributes["gen_ai.usage.output_tokens"] == 250
        assert llm.attributes["dunetrace.llm.reasoning_tokens"] == 200

    def test_llm_finish_reasons_is_array(self):
        spans = self._run_with_llm({"model": "gpt-4o"}, {"finish_reason": "tool_calls"})
        llm = _named(spans, "chat gpt-4o")
        assert llm.attributes["gen_ai.response.finish_reasons"] == ("tool_calls",)

    def test_llm_truncation_flagged_not_errored(self):
        spans = self._run_with_llm({"model": "gpt-4o"}, {"finish_reason": "length"})
        llm = _named(spans, "chat gpt-4o")
        assert llm.attributes["dunetrace.llm.output_truncated"] is True
        # Truncation is a soft signal, not a failed operation.
        assert llm.status.status_code != StatusCode.ERROR

    def test_llm_cost_attribute_set(self):
        spans = self._run_with_llm(
            {"model": "gpt-4o", "prompt_tokens": 1000},
            {"finish_reason": "stop", "completion_tokens": 1000},
        )
        llm = _named(spans, "chat gpt-4o")
        # gpt-4o: 1000*5e-6 + 1000*15e-6 = 0.02
        assert abs(llm.attributes["dunetrace.llm.cost_usd"] - 0.02) < 1e-9

    def test_llm_error_sets_error_status(self):
        spans = self._run_with_llm(
            {"model": "gpt-4o"}, {"finish_reason": "error", "error": "rate limited"}
        )
        llm = _named(spans, "chat gpt-4o")
        assert llm.status.status_code == StatusCode.ERROR

    def test_llm_latency_attribute(self):
        spans = self._run_with_llm(
            {"model": "gpt-4o"}, {"finish_reason": "stop", "latency_ms": 450}
        )
        assert _named(spans, "chat gpt-4o").attributes["dunetrace.llm.latency_ms"] == 450


# ── Hierarchy and orphan handling ────────────────────────────────────────────────


class TestHierarchy:
    def test_llm_span_parented_to_run(self):
        dt_otel, mem = _make_exporter()
        run_id = str(uuid.uuid4())
        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(EventType.LLM_CALLED, run_id, step_index=1, payload={"model": "gpt-4o"})
        )
        dt_otel.handle(
            _event(EventType.LLM_RESPONDED, run_id, step_index=1, payload={"finish_reason": "stop"})
        )
        dt_otel.handle(_run_completed(run_id))

        spans = mem.get_finished_spans()
        root = _named(spans, "dunetrace.run")
        llm = _named(spans, "chat gpt-4o")
        assert llm.parent.span_id == root.context.span_id
        assert llm.context.trace_id == root.context.trace_id

    def test_orphan_llm_closed_on_run_end_as_error(self):
        dt_otel, mem = _make_exporter()
        run_id = str(uuid.uuid4())
        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(EventType.LLM_CALLED, run_id, step_index=1, payload={"model": "gpt-4o"})
        )
        # Run ends with no LLM_RESPONDED.
        dt_otel.handle(_run_completed(run_id))

        llm = _named(mem.get_finished_spans(), "chat gpt-4o")
        assert llm.status.status_code == StatusCode.ERROR


# ── Tool span ────────────────────────────────────────────────────────────────────


class TestToolSpan:
    def _run_with_tool(self, called_payload, responded_payload, capture_content=True):
        mem = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(mem))
        dt_otel = DunetraceOTelExporter(tracer_provider=provider, capture_content=capture_content)
        run_id = str(uuid.uuid4())
        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(_event(EventType.TOOL_CALLED, run_id, step_index=1, payload=called_payload))
        dt_otel.handle(
            _event(EventType.TOOL_RESPONDED, run_id, step_index=1, payload=responded_payload)
        )
        dt_otel.handle(_run_completed(run_id))
        return mem.get_finished_spans()

    def test_tool_span_name_and_attributes(self):
        spans = self._run_with_tool(
            {"tool_name": "web_search", "args": "{'q': 'hello'}"},
            {"success": True, "output_length": 512, "latency_ms": 42},
        )
        tool = _named(spans, "dunetrace.tool.web_search")
        assert tool.attributes["dunetrace.tool.name"] == "web_search"
        assert tool.attributes["dunetrace.tool.args"] == "{'q': 'hello'}"
        assert tool.attributes["dunetrace.tool.result_status"] == "success"
        assert tool.attributes["dunetrace.tool.output_length"] == 512
        assert tool.attributes["dunetrace.tool.latency_ms"] == 42
        assert tool.status.status_code != StatusCode.ERROR

    def test_tool_failure_sets_error_status_and_message(self):
        spans = self._run_with_tool(
            {"tool_name": "db", "args": "{}"},
            {"success": False, "error": "connection refused"},
        )
        tool = _named(spans, "dunetrace.tool.db")
        assert tool.attributes["dunetrace.tool.result_status"] == "error"
        assert tool.attributes["dunetrace.tool.error_message"] == "connection refused"
        assert tool.status.status_code == StatusCode.ERROR

    def test_http_shaped_tool_uses_http_conventions(self):
        spans = self._run_with_tool(
            {
                "tool_name": "api.example.com",
                "args": "{'url': 'https://api.example.com/v1/x', 'method': 'get'}",
            },
            {"success": True, "output_length": 100},
        )
        tool = _named(spans, "dunetrace.tool.api.example.com")
        assert tool.kind == SpanKind.CLIENT
        assert tool.attributes["url.full"] == "https://api.example.com/v1/x"
        assert tool.attributes["server.address"] == "api.example.com"
        assert tool.attributes["http.request.method"] == "GET"

    def test_http_tool_failure_maps_status_code(self):
        spans = self._run_with_tool(
            {"tool_name": "api.example.com", "args": "{'url': 'https://api.example.com/x'}"},
            {"success": False, "error": "404"},
        )
        tool = _named(spans, "dunetrace.tool.api.example.com")
        assert tool.attributes["http.response.status_code"] == 404
        # A numeric HTTP status is not stored as a free-text error message.
        assert "dunetrace.tool.error_message" not in tool.attributes

    def test_non_http_tool_is_internal_kind(self):
        spans = self._run_with_tool({"tool_name": "calc", "args": "{'x': 1}"}, {"success": True})
        assert _named(spans, "dunetrace.tool.calc").kind == SpanKind.INTERNAL

    def test_pii_off_drops_args_and_url_and_error(self):
        spans = self._run_with_tool(
            {
                "tool_name": "api.example.com",
                "args": "{'url': 'https://api.example.com/x?token=secret'}",
            },
            {"success": False, "error": "boom with pii"},
            capture_content=False,
        )
        tool = _named(spans, "dunetrace.tool.api.example.com")
        assert "dunetrace.tool.args" not in tool.attributes
        assert "url.full" not in tool.attributes
        assert "dunetrace.tool.error_message" not in tool.attributes
        # Non-content metadata still present.
        assert tool.attributes["server.address"] == "api.example.com"
        assert tool.attributes["dunetrace.tool.result_status"] == "error"


# ── Retrieval span ───────────────────────────────────────────────────────────────


class TestRetrievalSpan:
    def _run_with_retrieval(self, called_payload, responded_payload, capture_content=True):
        mem = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(mem))
        dt_otel = DunetraceOTelExporter(tracer_provider=provider, capture_content=capture_content)
        run_id = str(uuid.uuid4())
        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(EventType.RETRIEVAL_CALLED, run_id, step_index=1, payload=called_payload)
        )
        dt_otel.handle(
            _event(EventType.RETRIEVAL_RESPONDED, run_id, step_index=1, payload=responded_payload)
        )
        dt_otel.handle(_run_completed(run_id))
        return mem.get_finished_spans()

    def test_retrieval_span_attributes(self):
        spans = self._run_with_retrieval(
            {"index_name": "pinecone-kb", "query": "how to reset password"},
            {"result_count": 5, "top_score": 0.91, "latency_ms": 30},
        )
        ret = _named(spans, "dunetrace.retrieval")
        assert ret.attributes["dunetrace.retrieval.vector_store"] == "pinecone-kb"
        assert ret.attributes["dunetrace.retrieval.query"] == "how to reset password"
        assert ret.attributes["dunetrace.retrieval.document_count"] == 5
        assert abs(ret.attributes["dunetrace.retrieval.top_score"] - 0.91) < 1e-6
        assert ret.attributes["dunetrace.retrieval.latency_ms"] == 30

    def test_retrieval_zero_results_is_not_error(self):
        spans = self._run_with_retrieval({"index_name": "kb", "query": "x"}, {"result_count": 0})
        ret = _named(spans, "dunetrace.retrieval")
        assert ret.attributes["dunetrace.retrieval.document_count"] == 0
        assert ret.status.status_code != StatusCode.ERROR

    def test_retrieval_pii_off_drops_query(self):
        spans = self._run_with_retrieval(
            {"index_name": "kb", "query": "sensitive query"},
            {"result_count": 3},
            capture_content=False,
        )
        ret = _named(spans, "dunetrace.retrieval")
        assert "dunetrace.retrieval.query" not in ret.attributes
        assert ret.attributes["dunetrace.retrieval.vector_store"] == "kb"


# ── Voice spans ──────────────────────────────────────────────────────────────────


class TestVoiceSpans:
    def _exporter(self):
        mem = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(mem))
        return DunetraceOTelExporter(tracer_provider=provider), mem

    def test_transcription_span(self):
        dt_otel, mem = self._exporter()
        run_id = str(uuid.uuid4())
        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.TRANSCRIPTION_RECEIVED,
                run_id,
                step_index=1,
                payload={
                    "text": "hello there",
                    "confidence": 0.87,
                    "latency_ms": 120,
                    "audio_seconds": 1.5,
                },
            )
        )
        dt_otel.handle(_run_completed(run_id))

        span = _named(mem.get_finished_spans(), "dunetrace.voice.transcription")
        assert abs(span.attributes["dunetrace.voice.confidence"] - 0.87) < 1e-6
        assert span.attributes["dunetrace.voice.char_count"] == len("hello there")
        assert span.attributes["dunetrace.voice.latency_ms"] == 120
        assert abs(span.attributes["dunetrace.voice.audio_seconds"] - 1.5) < 1e-6
        # Raw transcript text is never placed on the span.
        assert all("hello there" not in str(v) for v in span.attributes.values())

    def test_tts_span(self):
        dt_otel, mem = self._exporter()
        run_id = str(uuid.uuid4())
        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.TTS_GENERATED,
                run_id,
                step_index=1,
                payload={
                    "text": "your order shipped",
                    "latency_ms": 90,
                    "voice_id": "rachel",
                    "model": "eleven_turbo_v2",
                },
            )
        )
        dt_otel.handle(_run_completed(run_id))

        span = _named(mem.get_finished_spans(), "dunetrace.voice.tts")
        assert span.attributes["dunetrace.voice.char_count"] == len("your order shipped")
        assert span.attributes["dunetrace.voice.voice_id"] == "rachel"
        assert span.attributes["dunetrace.voice.model"] == "eleven_turbo_v2"
        assert span.attributes["dunetrace.voice.latency_ms"] == 90

    def test_vad_is_span_event_on_run(self):
        dt_otel, mem = self._exporter()
        run_id = str(uuid.uuid4())
        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.VOICE_ACTIVITY_DETECTED,
                run_id,
                payload={"type": "silence", "duration_ms": 800},
            )
        )
        dt_otel.handle(_run_completed(run_id))

        root = _named(mem.get_finished_spans(), "dunetrace.run")
        evt = next(e for e in root.events if e.name == "dunetrace.voice.silence")
        assert evt.attributes["dunetrace.voice.duration_ms"] == 800


# ── Cross-trace correlation (nested runs) ────────────────────────────────────────


class TestNestedRuns:
    def test_child_run_gets_own_trace_and_links_parent(self):
        dt_otel, mem = _make_exporter()
        parent_id = str(uuid.uuid4())
        child_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, parent_id, payload={}))
        dt_otel.handle(_event(EventType.RUN_STARTED, child_id, payload={}, parent_run_id=parent_id))
        dt_otel.handle(_run_completed(child_id))
        dt_otel.handle(_run_completed(parent_id))

        spans = mem.get_finished_spans()
        roots = [s for s in spans if s.name == "dunetrace.run"]
        assert len({s.context.trace_id for s in roots}) == 2  # distinct traces

        child = next(s for s in roots if s.attributes["dunetrace.run.id"] == child_id)
        assert child.attributes["dunetrace.run.parent_run_id"] == parent_id
        # A link points back to the parent run's trace.
        link_trace_ids = {link.context.trace_id for link in child.links}
        assert _trace_id(parent_id) in link_trace_ids

    def test_concurrent_runs_have_separate_traces(self):
        dt_otel, mem = _make_exporter()
        a, b = str(uuid.uuid4()), str(uuid.uuid4())
        dt_otel.handle(_event(EventType.RUN_STARTED, a, agent_id="a", payload={}))
        dt_otel.handle(_event(EventType.RUN_STARTED, b, agent_id="b", payload={}))
        dt_otel.handle(_run_completed(a))
        dt_otel.handle(_run_completed(b))

        roots = [s for s in mem.get_finished_spans() if s.name == "dunetrace.run"]
        assert len({s.context.trace_id for s in roots}) == 2

    def test_events_for_unknown_run_ignored(self):
        dt_otel, mem = _make_exporter()
        ghost = str(uuid.uuid4())
        # Never started — should not raise, should produce no spans.
        dt_otel.handle(_event(EventType.LLM_CALLED, ghost, payload={"model": "gpt-4o"}))
        dt_otel.handle(_run_completed(ghost))
        assert mem.get_finished_spans() == ()


# ── Server-side findings (signals + policies) ────────────────────────────────────


class TestServerSideFindings:
    def _tracer(self):
        mem = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(mem))
        return provider.get_tracer("dunetrace"), mem

    def test_signal_span_in_run_trace_parented_to_root(self):
        tracer, mem = self._tracer()
        run_id = str(uuid.uuid4())
        emit_signal_span(
            tracer,
            run_id,
            failure_type="TOOL_LOOP",
            severity="HIGH",
            confidence=0.92,
            detector_name="ToolLoopDetector",
            evidence={"repeats": 3, "tool": "search"},
            org_id="org_1",
        )
        span = _named(mem.get_finished_spans(), "dunetrace.signal.TOOL_LOOP")
        assert span.context.trace_id == _trace_id(run_id)
        assert span.parent.span_id == _root_span_id(run_id)
        assert span.attributes["dunetrace.signal.type"] == "TOOL_LOOP"
        assert span.attributes["dunetrace.signal.severity"] == "HIGH"
        assert abs(span.attributes["dunetrace.signal.confidence"] - 0.92) < 1e-6
        assert span.attributes["dunetrace.signal.detector"] == "ToolLoopDetector"
        assert span.attributes["dunetrace.signal.id"] == f"{run_id}:TOOL_LOOP"
        assert span.attributes["dunetrace.run.org_id"] == "org_1"
        assert span.attributes["dunetrace.signal.evidence.repeats"] == 3
        assert span.attributes["dunetrace.signal.evidence.tool"] == "search"
        # HIGH severity marks the span ERROR.
        assert span.status.status_code == StatusCode.ERROR

    def test_low_severity_signal_is_not_error(self):
        tracer, mem = self._tracer()
        emit_signal_span(
            tracer, str(uuid.uuid4()), failure_type="SLOW_STEP", severity="LOW", confidence=0.4
        )
        span = _named(mem.get_finished_spans(), "dunetrace.signal.SLOW_STEP")
        assert span.status.status_code != StatusCode.ERROR

    def test_signal_string_evidence_gated_by_capture_content(self):
        tracer, mem = self._tracer()
        emit_signal_span(
            tracer,
            str(uuid.uuid4()),
            failure_type="TOOL_LOOP",
            severity="MEDIUM",
            evidence={"repeats": 3, "sample_args": "user@example.com"},
            capture_content=False,
        )
        span = _named(mem.get_finished_spans(), "dunetrace.signal.TOOL_LOOP")
        assert span.attributes["dunetrace.signal.evidence.repeats"] == 3
        assert "dunetrace.signal.evidence.sample_args" not in span.attributes

    def test_policy_span(self):
        tracer, mem = self._tracer()
        run_id = str(uuid.uuid4())
        emit_policy_span(
            tracer,
            run_id,
            action="switch_model",
            policy_name="cost-guard",
            trigger="cost_usd",
            trigger_value=0.55,
        )
        span = _named(mem.get_finished_spans(), "dunetrace.policy.switch_model")
        assert span.context.trace_id == _trace_id(run_id)
        assert span.parent.span_id == _root_span_id(run_id)
        assert span.attributes["dunetrace.policy.action"] == "switch_model"
        assert span.attributes["dunetrace.policy.name"] == "cost-guard"
        assert span.attributes["dunetrace.policy.trigger"] == "cost_usd"
        assert span.attributes["dunetrace.policy.trigger_value"] == "0.55"

    def test_emit_run_findings_signals_and_policies(self):
        tracer, mem = self._tracer()
        run_id = str(uuid.uuid4())
        sig = FailureSignal(
            failure_type=FailureType.TOOL_LOOP,
            severity=Severity.HIGH,
            run_id=run_id,
            agent_id="a",
            agent_version="v1",
            step_index=4,
            confidence=0.9,
            evidence={"repeats": 3},
        )
        policy_event = {
            "event_type": "policy.triggered",
            "timestamp": time.time(),
            "payload": {
                "policy_name": "stopper",
                "action_type": "stop",
                "trigger": "cost_usd",
                "value": 1.0,
            },
        }
        emit_run_findings(
            tracer, run_id, signals=[sig], policy_events=[policy_event], org_id="org_9"
        )
        names = {s.name for s in mem.get_finished_spans()}
        assert "dunetrace.signal.TOOL_LOOP" in names
        assert "dunetrace.policy.stop" in names

    def test_emit_run_findings_is_best_effort(self):
        tracer, mem = self._tracer()

        class Broken:
            @property
            def failure_type(self):
                raise RuntimeError("boom")

        # A malformed signal must not raise; a valid policy still emits.
        policy_event = {"event_type": "policy.triggered", "payload": {"action_type": "block"}}
        emit_run_findings(
            tracer, str(uuid.uuid4()), signals=[Broken()], policy_events=[policy_event]
        )
        names = {s.name for s in mem.get_finished_spans()}
        assert "dunetrace.policy.block" in names


# ── Construction ─────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_accepts_prebuilt_tracer(self):
        mem = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(mem))
        tracer = provider.get_tracer("dunetrace")

        dt_otel = DunetraceOTelExporter(tracer=tracer)
        run_id = str(uuid.uuid4())
        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(_run_completed(run_id))
        assert _named(mem.get_finished_spans(), "dunetrace.run") is not None


# ── Client integration (real dt.run) ─────────────────────────────────────────────


class TestClientIntegration:
    def _client_with_inmemory(self):
        from dunetrace import Dunetrace
        from dunetrace.emitters import NoopBatchingEmitter

        mem = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(mem))
        exporter = DunetraceOTelExporter(tracer_provider=provider)
        dt = Dunetrace(emitter=NoopBatchingEmitter(), otel_exporter=exporter)
        return dt, mem

    def test_run_emits_spans_and_stamps_correlation_ids(self):
        dt, mem = self._client_with_inmemory()
        try:
            with dt.run("checkout", user_input="hi", model="gpt-4o") as run:
                run.llm_called("gpt-4o", prompt_tokens=10)
                run.llm_responded(completion_tokens=5, finish_reason="stop")
                run.final_answer()
                captured_run_id = run.run_id
                assert run.otel_trace_id == trace_id_hex(run.run_id)
                assert run.otel_span_id == root_span_id_hex(run.run_id)
        finally:
            dt.shutdown()

        spans = mem.get_finished_spans()
        root = _named(spans, "dunetrace.run")
        assert root.attributes["dunetrace.run.id"] == captured_run_id
        assert root.attributes["dunetrace.run.status"] == "completed"
        assert _named(spans, "chat gpt-4o") is not None

    def test_tool_retrieval_voice_through_real_run(self):
        dt, mem = self._client_with_inmemory()
        try:
            with dt.run("support", user_input="hi", model="gpt-4o") as run:
                run.retrieval_called("kb", query="reset password")
                run.retrieval_responded("kb", result_count=3, top_score=0.8)
                run.tool_called("lookup", {"id": 7})
                run.tool_responded("lookup", success=True, output_length=20)
                run.transcription_received("hello", confidence=0.9, latency_ms=50)
                run.tts_generated("hi back", latency_ms=40, voice_id="rachel")
                run.final_answer()
        finally:
            dt.shutdown()

        names = {s.name for s in mem.get_finished_spans()}
        assert "dunetrace.retrieval" in names
        assert "dunetrace.tool.lookup" in names
        assert "dunetrace.voice.transcription" in names
        assert "dunetrace.voice.tts" in names

    def test_run_error_marks_span_failed(self):
        dt, mem = self._client_with_inmemory()
        try:
            with dt.run("checkout", user_input="hi", model="gpt-4o"):
                raise ValueError("boom")
        except ValueError:
            pass
        finally:
            dt.shutdown()

        root = _named(mem.get_finished_spans(), "dunetrace.run")
        assert root.attributes["dunetrace.run.status"] == "failed"
        assert root.attributes["dunetrace.run.error_type"] == "ValueError"
