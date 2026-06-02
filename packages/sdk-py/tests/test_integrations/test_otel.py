"""
Integration tests for DunetraceOTelExporter.

All tests use the in-memory InMemorySpanExporter — no real OTLP endpoint needed.
"""

from __future__ import annotations

import time
import unittest
import uuid
from typing import List

try:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import StatusCode
    from dunetrace.integrations.otel import DunetraceOTelExporter, _trace_id, _root_span_id
    from dunetrace.models import AgentEvent, EventType, RunState, FailureType, Severity

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

# Skip the entire module when opentelemetry is not installed
if not _OTEL_AVAILABLE:
    raise unittest.SkipTest("opentelemetry not installed — skipping OTel exporter tests")


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_provider():
    """Return (provider, exporter) using in-memory span exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def _event(
    event_type: EventType,
    run_id: str,
    agent_id: str = "test-agent",
    step_index: int = 0,
    payload: dict | None = None,
    parent_run_id: str | None = None,
) -> AgentEvent:
    return AgentEvent(
        event_type=event_type,
        run_id=run_id,
        agent_id=agent_id,
        agent_version="1.0",
        step_index=step_index,
        timestamp=time.time(),
        payload=payload or {},
        parent_run_id=parent_run_id,
    )


def _full_run(exporter_obj: DunetraceOTelExporter, run_id: str) -> List:
    """Emit a minimal complete run and return recorded spans."""
    exporter_obj.handle(
        _event(EventType.RUN_STARTED, run_id, payload={"model": "gpt-4o", "tools": ["search"]})
    )
    exporter_obj.handle(
        _event(EventType.LLM_CALLED, run_id, step_index=1, payload={"model": "gpt-4o"})
    )
    exporter_obj.handle(
        _event(EventType.LLM_RESPONDED, run_id, step_index=1, payload={"finish_reason": "stop"})
    )
    exporter_obj.handle(
        _event(
            EventType.RUN_COMPLETED,
            run_id,
            payload={"total_steps": 1, "exit_reason": "stop", "tool_call_count": 0},
        )
    )
    return []


# ── Utility function tests ─────────────────────────────────────────────────────


class TestTraceIdDerivation:
    def test_trace_id_is_deterministic(self):
        run_id = str(uuid.uuid4())
        assert _trace_id(run_id) == _trace_id(run_id)

    def test_trace_id_differs_per_run(self):
        assert _trace_id(str(uuid.uuid4())) != _trace_id(str(uuid.uuid4()))

    def test_trace_id_is_128_bit_int(self):
        tid = _trace_id(str(uuid.uuid4()))
        assert isinstance(tid, int)
        assert 0 < tid < 2**128

    def test_root_span_id_is_64_bit(self):
        sid = _root_span_id(str(uuid.uuid4()))
        assert isinstance(sid, int)
        assert 0 < sid <= 0xFFFF_FFFF_FFFF_FFFF

    def test_trace_id_matches_uuid_int(self):
        run_id = str(uuid.uuid4())
        assert _trace_id(run_id) == uuid.UUID(run_id).int

    def test_root_span_id_is_lower_64_bits(self):
        run_id = str(uuid.uuid4())
        expected = uuid.UUID(run_id).int & 0xFFFF_FFFF_FFFF_FFFF
        assert _root_span_id(run_id) == expected


# ── Span creation ─────────────────────────────────────────────────────────────


class TestRunSpans:
    def test_run_started_creates_agent_run_span(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(
            _event(EventType.RUN_STARTED, run_id, payload={"model": "gpt-4o", "tools": ["search"]})
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 0, "exit_reason": "stop", "tool_call_count": 0},
            )
        )

        spans = mem.get_finished_spans()
        root = next(s for s in spans if s.name == "agent_run")
        assert root is not None

    def test_root_span_has_agent_id_attribute(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(
            _event(
                EventType.RUN_STARTED,
                run_id,
                agent_id="my-agent",
                payload={"model": "gpt-4o", "tools": []},
            )
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 0, "exit_reason": "stop", "tool_call_count": 0},
            )
        )

        spans = mem.get_finished_spans()
        root = next(s for s in spans if s.name == "agent_run")
        assert root.attributes["dunetrace.agent_id"] == "my-agent"

    def test_root_span_has_run_id_attribute(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(
            _event(EventType.RUN_STARTED, run_id, payload={"model": "gpt-4o", "tools": []})
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 0, "exit_reason": "stop", "tool_call_count": 0},
            )
        )

        spans = mem.get_finished_spans()
        root = next(s for s in spans if s.name == "agent_run")
        assert root.attributes["dunetrace.run_id"] == run_id

    def test_root_span_has_model_and_tools(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(
            _event(
                EventType.RUN_STARTED,
                run_id,
                payload={"model": "gpt-4o", "tools": ["search", "calc"]},
            )
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 0, "exit_reason": "stop", "tool_call_count": 0},
            )
        )

        spans = mem.get_finished_spans()
        root = next(s for s in spans if s.name == "agent_run")
        assert root.attributes["dunetrace.model"] == "gpt-4o"
        assert root.attributes["dunetrace.tools"] == "search,calc"

    def test_root_span_trace_id_matches_run_id(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 0, "exit_reason": "stop", "tool_call_count": 0},
            )
        )

        spans = mem.get_finished_spans()
        root = next(s for s in spans if s.name == "agent_run")
        assert root.context.trace_id == _trace_id(run_id)

    def test_parent_run_id_attribute_set_when_provided(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())
        parent_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}, parent_run_id=parent_id))
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 0, "exit_reason": "stop", "tool_call_count": 0},
            )
        )

        spans = mem.get_finished_spans()
        root = next(s for s in spans if s.name == "agent_run")
        assert root.attributes.get("dunetrace.parent_run_id") == parent_id

    def test_parent_run_id_not_set_when_absent(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 0, "exit_reason": "stop", "tool_call_count": 0},
            )
        )

        spans = mem.get_finished_spans()
        root = next(s for s in spans if s.name == "agent_run")
        assert "dunetrace.parent_run_id" not in root.attributes

    def test_run_completed_sets_total_steps_and_exit_reason(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 7, "exit_reason": "final_answer", "tool_call_count": 3},
            )
        )

        spans = mem.get_finished_spans()
        root = next(s for s in spans if s.name == "agent_run")
        assert root.attributes["dunetrace.total_steps"] == 7
        assert root.attributes["dunetrace.exit_reason"] == "final_answer"
        assert root.attributes["dunetrace.tool_call_count"] == 3


# ── LLM span ──────────────────────────────────────────────────────────────────


class TestLlmSpan:
    def test_llm_called_creates_llm_call_span(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.LLM_CALLED,
                run_id,
                step_index=1,
                payload={"model": "gpt-4o", "prompt_tokens": 150},
            )
        )
        dt_otel.handle(
            _event(
                EventType.LLM_RESPONDED,
                run_id,
                step_index=1,
                payload={"finish_reason": "stop", "completion_tokens": 80},
            )
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 1, "exit_reason": "stop", "tool_call_count": 0},
            )
        )

        spans = mem.get_finished_spans()
        llm_span = next(s for s in spans if s.name == "llm_call")
        assert llm_span is not None

    def test_llm_span_has_model_attribute(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(EventType.LLM_CALLED, run_id, step_index=1, payload={"model": "claude-3-sonnet"})
        )
        dt_otel.handle(
            _event(EventType.LLM_RESPONDED, run_id, step_index=1, payload={"finish_reason": "stop"})
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 1, "exit_reason": "stop", "tool_call_count": 0},
            )
        )

        spans = mem.get_finished_spans()
        llm_span = next(s for s in spans if s.name == "llm_call")
        assert llm_span.attributes["gen_ai.request.model"] == "claude-3-sonnet"

    def test_llm_span_has_input_tokens(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.LLM_CALLED,
                run_id,
                step_index=1,
                payload={"model": "gpt-4o", "prompt_tokens": 300},
            )
        )
        dt_otel.handle(
            _event(EventType.LLM_RESPONDED, run_id, step_index=1, payload={"finish_reason": "stop"})
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 1, "exit_reason": "stop", "tool_call_count": 0},
            )
        )

        spans = mem.get_finished_spans()
        llm_span = next(s for s in spans if s.name == "llm_call")
        assert llm_span.attributes["gen_ai.usage.input_tokens"] == 300

    def test_llm_span_has_output_tokens(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(EventType.LLM_CALLED, run_id, step_index=1, payload={"model": "gpt-4o"})
        )
        dt_otel.handle(
            _event(
                EventType.LLM_RESPONDED,
                run_id,
                step_index=1,
                payload={"finish_reason": "stop", "completion_tokens": 120},
            )
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 1, "exit_reason": "stop", "tool_call_count": 0},
            )
        )

        spans = mem.get_finished_spans()
        llm_span = next(s for s in spans if s.name == "llm_call")
        assert llm_span.attributes["gen_ai.usage.output_tokens"] == 120

    def test_llm_span_has_finish_reason(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(EventType.LLM_CALLED, run_id, step_index=1, payload={"model": "gpt-4o"})
        )
        dt_otel.handle(
            _event(
                EventType.LLM_RESPONDED,
                run_id,
                step_index=1,
                payload={"finish_reason": "tool_calls"},
            )
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 1, "exit_reason": "stop", "tool_call_count": 0},
            )
        )

        spans = mem.get_finished_spans()
        llm_span = next(s for s in spans if s.name == "llm_call")
        assert llm_span.attributes["gen_ai.response.finish_reason"] == "tool_calls"

    def test_llm_truncation_sets_span_error(self):
        """finish_reason=length should mark the span ERROR."""
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(EventType.LLM_CALLED, run_id, step_index=1, payload={"model": "gpt-4o"})
        )
        dt_otel.handle(
            _event(
                EventType.LLM_RESPONDED, run_id, step_index=1, payload={"finish_reason": "length"}
            )
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 1, "exit_reason": "stop", "tool_call_count": 0},
            )
        )

        spans = mem.get_finished_spans()
        llm_span = next(s for s in spans if s.name == "llm_call")
        assert llm_span.status.status_code == StatusCode.ERROR

    def test_llm_span_latency_ms_attribute(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(EventType.LLM_CALLED, run_id, step_index=1, payload={"model": "gpt-4o"})
        )
        dt_otel.handle(
            _event(
                EventType.LLM_RESPONDED,
                run_id,
                step_index=1,
                payload={"finish_reason": "stop", "latency_ms": 450},
            )
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 1, "exit_reason": "stop", "tool_call_count": 0},
            )
        )

        spans = mem.get_finished_spans()
        llm_span = next(s for s in spans if s.name == "llm_call")
        assert llm_span.attributes["dunetrace.latency_ms"] == 450


# ── Tool span ─────────────────────────────────────────────────────────────────


class TestToolSpan:
    def test_tool_called_creates_tool_call_span(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.TOOL_CALLED,
                run_id,
                step_index=2,
                payload={"tool_name": "web_search", "args_hash": "abc123"},
            )
        )
        dt_otel.handle(
            _event(
                EventType.TOOL_RESPONDED,
                run_id,
                step_index=2,
                payload={"success": True, "output_length": 512},
            )
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 2, "exit_reason": "stop", "tool_call_count": 1},
            )
        )

        spans = mem.get_finished_spans()
        tool_span = next(s for s in spans if s.name == "tool_call")
        assert tool_span is not None

    def test_tool_span_has_tool_name(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.TOOL_CALLED,
                run_id,
                step_index=2,
                payload={"tool_name": "calculator", "args_hash": "abc123"},
            )
        )
        dt_otel.handle(
            _event(EventType.TOOL_RESPONDED, run_id, step_index=2, payload={"success": True})
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 2, "exit_reason": "stop", "tool_call_count": 1},
            )
        )

        spans = mem.get_finished_spans()
        tool_span = next(s for s in spans if s.name == "tool_call")
        assert tool_span.attributes["dunetrace.tool_name"] == "calculator"

    def test_tool_span_has_args_hash(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.TOOL_CALLED,
                run_id,
                step_index=2,
                payload={"tool_name": "search", "args_hash": "deadbeef"},
            )
        )
        dt_otel.handle(
            _event(EventType.TOOL_RESPONDED, run_id, step_index=2, payload={"success": True})
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 2, "exit_reason": "stop", "tool_call_count": 1},
            )
        )

        spans = mem.get_finished_spans()
        tool_span = next(s for s in spans if s.name == "tool_call")
        assert tool_span.attributes["dunetrace.args_hash"] == "deadbeef"

    def test_tool_success_sets_ok_status(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.TOOL_CALLED,
                run_id,
                step_index=2,
                payload={"tool_name": "search", "args_hash": "x"},
            )
        )
        dt_otel.handle(
            _event(EventType.TOOL_RESPONDED, run_id, step_index=2, payload={"success": True})
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 2, "exit_reason": "stop", "tool_call_count": 1},
            )
        )

        spans = mem.get_finished_spans()
        tool_span = next(s for s in spans if s.name == "tool_call")
        assert tool_span.status.status_code != StatusCode.ERROR

    def test_tool_failure_sets_error_status(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.TOOL_CALLED,
                run_id,
                step_index=2,
                payload={"tool_name": "search", "args_hash": "x"},
            )
        )
        dt_otel.handle(
            _event(
                EventType.TOOL_RESPONDED,
                run_id,
                step_index=2,
                payload={"success": False, "error_hash": "err123"},
            )
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 2, "exit_reason": "stop", "tool_call_count": 1},
            )
        )

        spans = mem.get_finished_spans()
        tool_span = next(s for s in spans if s.name == "tool_call")
        assert tool_span.status.status_code == StatusCode.ERROR

    def test_tool_failure_records_error_hash(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.TOOL_CALLED,
                run_id,
                step_index=2,
                payload={"tool_name": "search", "args_hash": "x"},
            )
        )
        dt_otel.handle(
            _event(
                EventType.TOOL_RESPONDED,
                run_id,
                step_index=2,
                payload={"success": False, "error_hash": "abc999"},
            )
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 2, "exit_reason": "stop", "tool_call_count": 1},
            )
        )

        spans = mem.get_finished_spans()
        tool_span = next(s for s in spans if s.name == "tool_call")
        assert tool_span.attributes["dunetrace.error_hash"] == "abc999"

    def test_tool_span_has_success_attribute(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.TOOL_CALLED,
                run_id,
                step_index=2,
                payload={"tool_name": "search", "args_hash": "x"},
            )
        )
        dt_otel.handle(
            _event(
                EventType.TOOL_RESPONDED,
                run_id,
                step_index=2,
                payload={"success": True, "output_length": 200},
            )
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 2, "exit_reason": "stop", "tool_call_count": 1},
            )
        )

        spans = mem.get_finished_spans()
        tool_span = next(s for s in spans if s.name == "tool_call")
        assert tool_span.attributes["dunetrace.success"] is True
        assert tool_span.attributes["dunetrace.output_length"] == 200


# ── Retrieval span ─────────────────────────────────────────────────────────────


class TestRetrievalSpan:
    def test_retrieval_called_creates_retrieval_span(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.RETRIEVAL_CALLED,
                run_id,
                step_index=1,
                payload={"index_name": "docs", "query_hash": "qh1"},
            )
        )
        dt_otel.handle(
            _event(
                EventType.RETRIEVAL_RESPONDED,
                run_id,
                step_index=1,
                payload={"result_count": 5, "top_score": 0.9},
            )
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 1, "exit_reason": "stop", "tool_call_count": 0},
            )
        )

        spans = mem.get_finished_spans()
        ret_span = next(s for s in spans if s.name == "retrieval")
        assert ret_span is not None

    def test_retrieval_span_has_index_name(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.RETRIEVAL_CALLED,
                run_id,
                step_index=1,
                payload={"index_name": "knowledge-base", "query_hash": "qh1"},
            )
        )
        dt_otel.handle(
            _event(
                EventType.RETRIEVAL_RESPONDED,
                run_id,
                step_index=1,
                payload={"result_count": 3, "top_score": 0.8},
            )
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 1, "exit_reason": "stop", "tool_call_count": 0},
            )
        )

        spans = mem.get_finished_spans()
        ret_span = next(s for s in spans if s.name == "retrieval")
        assert ret_span.attributes["dunetrace.index_name"] == "knowledge-base"

    def test_retrieval_span_has_result_count(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.RETRIEVAL_CALLED,
                run_id,
                step_index=1,
                payload={"index_name": "docs", "query_hash": "qh1"},
            )
        )
        dt_otel.handle(
            _event(
                EventType.RETRIEVAL_RESPONDED,
                run_id,
                step_index=1,
                payload={"result_count": 7, "top_score": 0.75},
            )
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 1, "exit_reason": "stop", "tool_call_count": 0},
            )
        )

        spans = mem.get_finished_spans()
        ret_span = next(s for s in spans if s.name == "retrieval")
        assert ret_span.attributes["dunetrace.result_count"] == 7

    def test_retrieval_span_has_top_score(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.RETRIEVAL_CALLED,
                run_id,
                step_index=1,
                payload={"index_name": "docs", "query_hash": "qh1"},
            )
        )
        dt_otel.handle(
            _event(
                EventType.RETRIEVAL_RESPONDED,
                run_id,
                step_index=1,
                payload={"result_count": 4, "top_score": 0.92},
            )
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 1, "exit_reason": "stop", "tool_call_count": 0},
            )
        )

        spans = mem.get_finished_spans()
        ret_span = next(s for s in spans if s.name == "retrieval")
        assert abs(ret_span.attributes["dunetrace.top_score"] - 0.92) < 1e-6

    def test_retrieval_zero_results_sets_error_status(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.RETRIEVAL_CALLED,
                run_id,
                step_index=1,
                payload={"index_name": "docs", "query_hash": "qh1"},
            )
        )
        dt_otel.handle(
            _event(EventType.RETRIEVAL_RESPONDED, run_id, step_index=1, payload={"result_count": 0})
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 1, "exit_reason": "stop", "tool_call_count": 0},
            )
        )

        spans = mem.get_finished_spans()
        ret_span = next(s for s in spans if s.name == "retrieval")
        assert ret_span.status.status_code == StatusCode.ERROR


# ── Orphaned child span handling ───────────────────────────────────────────────


class TestOrphanedChildSpan:
    def test_orphaned_child_span_closed_on_next_call_start(self):
        """A _on_child_start without a preceding _on_child_end should close the previous child."""
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        # First LLM call — no response emitted (orphan)
        dt_otel.handle(
            _event(EventType.LLM_CALLED, run_id, step_index=1, payload={"model": "gpt-4o"})
        )
        # Second tool call starts without LLM being closed first
        dt_otel.handle(
            _event(
                EventType.TOOL_CALLED,
                run_id,
                step_index=2,
                payload={"tool_name": "search", "args_hash": "x"},
            )
        )
        dt_otel.handle(
            _event(EventType.TOOL_RESPONDED, run_id, step_index=2, payload={"success": True})
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 2, "exit_reason": "stop", "tool_call_count": 1},
            )
        )

        spans = mem.get_finished_spans()
        # Both the orphaned llm_call and the tool_call should be finished
        llm_span = next(s for s in spans if s.name == "llm_call")
        tool_span = next(s for s in spans if s.name == "tool_call")
        assert llm_span is not None
        assert tool_span is not None

    def test_orphaned_child_on_run_ended_is_closed(self):
        """A child span still open when run ends should be closed with ERROR status."""
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.TOOL_CALLED,
                run_id,
                step_index=1,
                payload={"tool_name": "search", "args_hash": "x"},
            )
        )
        # Run ends without TOOL_RESPONDED
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 1, "exit_reason": "stop", "tool_call_count": 1},
            )
        )

        spans = mem.get_finished_spans()
        tool_span = next(s for s in spans if s.name == "tool_call")
        assert tool_span.status.status_code == StatusCode.ERROR

    def test_orphaned_child_on_run_errored_is_closed(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(EventType.LLM_CALLED, run_id, step_index=1, payload={"model": "gpt-4o"})
        )
        dt_otel.handle(
            _event(
                EventType.RUN_ERRORED,
                run_id,
                payload={
                    "error_type": "TimeoutError",
                    "total_steps": 1,
                    "exit_reason": "error",
                    "tool_call_count": 0,
                },
            )
        )

        spans = mem.get_finished_spans()
        llm_span = next(s for s in spans if s.name == "llm_call")
        assert llm_span.status.status_code == StatusCode.ERROR


# ── Run errored ────────────────────────────────────────────────────────────────


class TestRunErrored:
    def test_run_errored_sets_root_span_error(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.RUN_ERRORED,
                run_id,
                payload={
                    "error_type": "ValueError",
                    "total_steps": 0,
                    "exit_reason": "error",
                    "tool_call_count": 0,
                },
            )
        )

        spans = mem.get_finished_spans()
        root = next(s for s in spans if s.name == "agent_run")
        assert root.status.status_code == StatusCode.ERROR

    def test_run_errored_sets_error_type_attribute(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.RUN_ERRORED,
                run_id,
                payload={
                    "error_type": "TimeoutError",
                    "total_steps": 0,
                    "exit_reason": "error",
                    "tool_call_count": 0,
                },
            )
        )

        spans = mem.get_finished_spans()
        root = next(s for s in spans if s.name == "agent_run")
        assert root.attributes["dunetrace.error_type"] == "TimeoutError"


# ── Failure signal annotation ──────────────────────────────────────────────────


class TestSignalAnnotation:
    def _make_tool_loop_run_state(self) -> RunState:
        """Build a RunState that triggers TOOL_LOOP (same tool 3x in a 5-tool window)."""
        from dunetrace.models import ToolCall

        state = RunState(
            run_id=str(uuid.uuid4()),
            agent_id="test",
            agent_version="1.0",
            available_tools=["search"],
        )
        args_hash = "aaaa1111"
        # 5 calls total — 3 of "search" (triggers loop) + 2 fillers within the window
        tools = ["other", "search", "search", "other2", "search"]
        for i, tool in enumerate(tools):
            h = args_hash if tool == "search" else f"filler{i}"
            state.tool_calls.append(
                ToolCall(tool_name=tool, step_index=i, args_hash=h, timestamp=time.time())
            )
        return state

    def test_high_signal_sets_root_span_error_status(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())
        run_state = self._make_tool_loop_run_state()
        run_state.run_id = run_id

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.notify_run_state(run_id, run_state)
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 3, "exit_reason": "stop", "tool_call_count": 3},
            )
        )

        spans = mem.get_finished_spans()
        root = next(s for s in spans if s.name == "agent_run")
        assert root.status.status_code == StatusCode.ERROR

    def test_signal_failure_type_annotated_on_root_span(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())
        run_state = self._make_tool_loop_run_state()
        run_state.run_id = run_id

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.notify_run_state(run_id, run_state)
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 3, "exit_reason": "stop", "tool_call_count": 3},
            )
        )

        spans = mem.get_finished_spans()
        root = next(s for s in spans if s.name == "agent_run")
        # At least one signal should be indexed on the root span
        assert any(
            k.startswith("dunetrace.signal.") and k.endswith(".failure_type")
            for k in root.attributes
        )

    def test_signal_severity_annotated_on_root_span(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())
        run_state = self._make_tool_loop_run_state()
        run_state.run_id = run_id

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.notify_run_state(run_id, run_state)
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 3, "exit_reason": "stop", "tool_call_count": 3},
            )
        )

        spans = mem.get_finished_spans()
        root = next(s for s in spans if s.name == "agent_run")
        assert any(
            k.startswith("dunetrace.signal.") and k.endswith(".severity") for k in root.attributes
        )

    def test_no_signal_annotation_when_run_state_not_provided(self):
        """Without notify_run_state, no signal attributes should appear on root span."""
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        # No notify_run_state call
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 0, "exit_reason": "stop", "tool_call_count": 0},
            )
        )

        spans = mem.get_finished_spans()
        root = next(s for s in spans if s.name == "agent_run")
        assert not any(k.startswith("dunetrace.signal.") for k in root.attributes)


# ── External signal events ─────────────────────────────────────────────────────


class TestExternalSignal:
    def test_external_signal_adds_span_event_to_child(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.TOOL_CALLED,
                run_id,
                step_index=1,
                payload={"tool_name": "api", "args_hash": "x"},
            )
        )
        dt_otel.handle(
            _event(
                EventType.EXTERNAL_SIGNAL,
                run_id,
                payload={"signal_name": "rate_limit", "source": "openai"},
            )
        )
        dt_otel.handle(
            _event(EventType.TOOL_RESPONDED, run_id, step_index=1, payload={"success": True})
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 1, "exit_reason": "stop", "tool_call_count": 1},
            )
        )

        spans = mem.get_finished_spans()
        tool_span = next(s for s in spans if s.name == "tool_call")
        event_names = [e.name for e in tool_span.events]
        assert "rate_limit" in event_names

    def test_external_signal_falls_back_to_root_span_when_no_child(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.EXTERNAL_SIGNAL,
                run_id,
                payload={"signal_name": "circuit_breaker", "source": "internal"},
            )
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 0, "exit_reason": "stop", "tool_call_count": 0},
            )
        )

        spans = mem.get_finished_spans()
        root = next(s for s in spans if s.name == "agent_run")
        event_names = [e.name for e in root.events]
        assert "circuit_breaker" in event_names

    def test_external_signal_includes_source_attribute(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(
                EventType.EXTERNAL_SIGNAL,
                run_id,
                payload={"signal_name": "rate_limit", "source": "openai"},
            )
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 0, "exit_reason": "stop", "tool_call_count": 0},
            )
        )

        spans = mem.get_finished_spans()
        root = next(s for s in spans if s.name == "agent_run")
        evt = next(e for e in root.events if e.name == "rate_limit")
        assert evt.attributes["dunetrace.source"] == "openai"


# ── Concurrent runs ────────────────────────────────────────────────────────────


class TestConcurrentRuns:
    def test_two_concurrent_runs_have_separate_trace_ids(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id_a = str(uuid.uuid4())
        run_id_b = str(uuid.uuid4())

        # Interleave two runs
        dt_otel.handle(_event(EventType.RUN_STARTED, run_id_a, agent_id="agent-a", payload={}))
        dt_otel.handle(_event(EventType.RUN_STARTED, run_id_b, agent_id="agent-b", payload={}))
        dt_otel.handle(
            _event(EventType.LLM_CALLED, run_id_a, step_index=1, payload={"model": "gpt-4o"})
        )
        dt_otel.handle(
            _event(EventType.LLM_CALLED, run_id_b, step_index=1, payload={"model": "gpt-4o"})
        )
        dt_otel.handle(
            _event(
                EventType.LLM_RESPONDED, run_id_a, step_index=1, payload={"finish_reason": "stop"}
            )
        )
        dt_otel.handle(
            _event(
                EventType.LLM_RESPONDED, run_id_b, step_index=1, payload={"finish_reason": "stop"}
            )
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id_a,
                payload={"total_steps": 1, "exit_reason": "stop", "tool_call_count": 0},
            )
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id_b,
                payload={"total_steps": 1, "exit_reason": "stop", "tool_call_count": 0},
            )
        )

        spans = mem.get_finished_spans()
        roots = [s for s in spans if s.name == "agent_run"]
        assert len(roots) == 2
        trace_ids = {s.context.trace_id for s in roots}
        assert len(trace_ids) == 2  # distinct trace IDs

    def test_events_for_unknown_run_are_silently_ignored(self):
        """Handle events for a run_id that was never started — should not raise."""
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        ghost_id = str(uuid.uuid4())

        # Should not raise
        dt_otel.handle(_event(EventType.LLM_CALLED, ghost_id, payload={"model": "gpt-4o"}))
        dt_otel.handle(
            _event(EventType.TOOL_CALLED, ghost_id, payload={"tool_name": "x", "args_hash": "y"})
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                ghost_id,
                payload={"total_steps": 0, "exit_reason": "stop", "tool_call_count": 0},
            )
        )

        # No spans should have been recorded
        spans = mem.get_finished_spans()
        assert len(spans) == 0


# ── Span hierarchy ─────────────────────────────────────────────────────────────


class TestSpanHierarchy:
    def test_child_spans_are_parented_to_root_span(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(EventType.LLM_CALLED, run_id, step_index=1, payload={"model": "gpt-4o"})
        )
        dt_otel.handle(
            _event(EventType.LLM_RESPONDED, run_id, step_index=1, payload={"finish_reason": "stop"})
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 1, "exit_reason": "stop", "tool_call_count": 0},
            )
        )

        spans = mem.get_finished_spans()
        root = next(s for s in spans if s.name == "agent_run")
        llm_span = next(s for s in spans if s.name == "llm_call")

        # llm_call's parent should be the root span
        assert llm_span.parent.span_id == root.context.span_id

    def test_multiple_child_spans_all_share_same_trace_id(self):
        provider, mem = _make_provider()
        dt_otel = DunetraceOTelExporter(provider)
        run_id = str(uuid.uuid4())

        dt_otel.handle(_event(EventType.RUN_STARTED, run_id, payload={}))
        dt_otel.handle(
            _event(EventType.LLM_CALLED, run_id, step_index=1, payload={"model": "gpt-4o"})
        )
        dt_otel.handle(
            _event(
                EventType.LLM_RESPONDED,
                run_id,
                step_index=1,
                payload={"finish_reason": "tool_calls"},
            )
        )
        dt_otel.handle(
            _event(
                EventType.TOOL_CALLED,
                run_id,
                step_index=2,
                payload={"tool_name": "search", "args_hash": "x"},
            )
        )
        dt_otel.handle(
            _event(EventType.TOOL_RESPONDED, run_id, step_index=2, payload={"success": True})
        )
        dt_otel.handle(
            _event(
                EventType.RUN_COMPLETED,
                run_id,
                payload={"total_steps": 2, "exit_reason": "stop", "tool_call_count": 1},
            )
        )

        spans = mem.get_finished_spans()
        trace_ids = {s.context.trace_id for s in spans}
        assert len(trace_ids) == 1  # all spans share one trace ID
