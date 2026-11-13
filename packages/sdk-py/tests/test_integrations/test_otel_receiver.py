"""
Tests for DunetraceOTelReceiver — the SpanExporter that translates incoming
OTel gen_ai.* spans into Dunetrace SDK calls. This receiver had no test coverage
before Phase 1.

Spans are fed through the real export() path. A capturing exporter on the
Dunetrace client records the AgentEvents the translation produces, so the tests
assert on the actual emitted events, not internals.
"""

from __future__ import annotations

import unittest

try:
    from opentelemetry.trace import StatusCode

    from dunetrace import CallableExporter, Dunetrace
    from dunetrace.emitters import NoopBatchingEmitter
    from dunetrace.integrations.otel_receiver import DunetraceOTelReceiver
    from dunetrace.models import EventType

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

if not _OTEL_AVAILABLE:
    raise unittest.SkipTest("opentelemetry not installed — skipping OTel receiver tests")


# ── Fake OTel ReadableSpan ────────────────────────────────────────────────────


class _Ctx:
    def __init__(self, trace_id: int, span_id: int):
        self.trace_id = trace_id
        self.span_id = span_id


class _Status:
    def __init__(self, code):
        self.status_code = code


class _Span:
    """Minimal stand-in for an OTel ReadableSpan: only the fields the receiver
    reads (context, parent, name, attributes, start/end time, status)."""

    def __init__(
        self,
        name,
        attributes,
        *,
        span_id=1,
        trace_id=1,
        parent=None,
        start=0,
        end=1_000_000,  # 1 ms in ns
        error=False,
    ):
        self.name = name
        self.attributes = attributes
        self.context = _Ctx(trace_id, span_id)
        self.parent = parent  # None => root
        self.start_time = start
        self.end_time = end
        self.status = _Status(StatusCode.ERROR if error else StatusCode.OK)


def _root(**kw):
    return _Span("agent", {}, span_id=1, parent=None, start=0, **kw)


def _child(name, attrs, *, error=False, start=100):
    # parent is a context object with a span_id, so _is_root() is False.
    return _Span(
        name, attrs, span_id=2, parent=_Ctx(1, 1), start=start, end=start + 500_000, error=error
    )


def _receive(child_spans):
    """Feed a root plus the given child spans through the receiver and return
    the captured AgentEvents."""
    events = []
    dt = Dunetrace(
        emitter=NoopBatchingEmitter(),
        exporters=[CallableExporter(lambda e: events.append(e))],
    )
    receiver = DunetraceOTelReceiver(dt, agent_id="test-agent")
    receiver.export([*child_spans, _root()])  # root last; order must not matter
    dt.shutdown()
    return events


def _payload(events, event_type):
    return next(e.payload for e in events if e.event_type == event_type)


# ── LLM ────────────────────────────────────────────────────────────────────────


class TestLlm:
    def test_current_token_convention(self):
        events = _receive(
            [
                _child(
                    "chat",
                    {
                        "gen_ai.request.model": "gpt-4o",
                        "gen_ai.usage.input_tokens": 100,
                        "gen_ai.usage.output_tokens": 50,
                    },
                )
            ]
        )
        assert _payload(events, EventType.LLM_CALLED)["model"] == "gpt-4o"
        assert _payload(events, EventType.LLM_CALLED)["prompt_tokens"] == 100
        assert _payload(events, EventType.LLM_RESPONDED)["completion_tokens"] == 50

    def test_legacy_token_convention_still_works(self):
        events = _receive(
            [
                _child(
                    "chat",
                    {
                        "gen_ai.request.model": "gpt-4o",
                        "gen_ai.usage.prompt_tokens": 12,
                        "gen_ai.usage.completion_tokens": 7,
                    },
                )
            ]
        )
        assert _payload(events, EventType.LLM_CALLED)["prompt_tokens"] == 12
        assert _payload(events, EventType.LLM_RESPONDED)["completion_tokens"] == 7

    def test_provider_only_span_classified_as_llm(self):
        # gen_ai.system with no model still recognized as an LLM span.
        events = _receive([_child("openai.chat", {"gen_ai.system": "openai"})])
        assert _payload(events, EventType.LLM_CALLED)["model"] == "unknown"

    def test_output_text_and_finish_reason(self):
        events = _receive(
            [
                _child(
                    "chat",
                    {
                        "gen_ai.request.model": "gpt-4o",
                        "gen_ai.completion": "Hello there",
                        "gen_ai.completion.0.finish_reason": "stop",
                    },
                )
            ]
        )
        resp = _payload(events, EventType.LLM_RESPONDED)
        assert resp["output"] == "Hello there"
        assert resp["finish_reason"] == "stop"

    def test_error_span_marks_finish_reason_error(self):
        events = _receive([_child("chat", {"gen_ai.request.model": "gpt-4o"}, error=True)])
        assert _payload(events, EventType.LLM_RESPONDED)["finish_reason"] == "error"

    def test_output_from_gen_ai_output_messages(self):
        # Traceloop's current output format (Phase 4 finding): structured
        # gen_ai.output.messages rather than a plain gen_ai.completion string.
        events = _receive(
            [
                _child(
                    "chat",
                    {
                        "gen_ai.request.model": "gpt-4o",
                        "gen_ai.output.messages": (
                            '[{"role": "assistant", "parts": '
                            '[{"content": "The capital of France is Paris.", "type": "text"}]}]'
                        ),
                    },
                )
            ]
        )
        assert (
            _payload(events, EventType.LLM_RESPONDED)["output"] == "The capital of France is Paris."
        )


# ── Tool ─────────────────────────────────────────────────────────────────────────


class TestTool:
    def test_tool_name_and_args(self):
        events = _receive(
            [
                _child(
                    "web_search",
                    {
                        "gen_ai.tool.name": "web_search",
                        "traceloop.entity.input": '{"query": "otel"}',
                    },
                )
            ]
        )
        called = _payload(events, EventType.TOOL_CALLED)
        assert called["tool_name"] == "web_search"
        assert called["args"] == '{"query": "otel"}'

    def test_tool_name_from_tool_dot_name(self):
        events = _receive([_child("search", {"tool.name": "search"})])
        assert _payload(events, EventType.TOOL_CALLED)["tool_name"] == "search"

    def test_tool_failure_recorded(self):
        events = _receive([_child("search", {"tool.name": "search"}, error=True)])
        assert _payload(events, EventType.TOOL_RESPONDED)["success"] is False


# ── Retrieval (new in Phase 1) ───────────────────────────────────────────────────


class TestRetrieval:
    def test_retrieval_translated(self):
        events = _receive(
            [
                _child(
                    "pinecone.query",
                    {
                        "retrieval.index_name": "kb",
                        "retrieval.result_count": 4,
                        "retrieval.top_score": 0.9,
                    },
                )
            ]
        )
        assert _payload(events, EventType.RETRIEVAL_CALLED)["index_name"] == "kb"
        resp = _payload(events, EventType.RETRIEVAL_RESPONDED)
        assert resp["result_count"] == 4
        assert abs(resp["top_score"] - 0.9) < 1e-6

    def test_retrieval_by_vector_db_attrs(self):
        events = _receive(
            [_child("query", {"vector_db.collection_name": "docs", "db.result_count": 2})]
        )
        assert _payload(events, EventType.RETRIEVAL_RESPONDED)["index_name"] == "docs"


# ── Robustness ───────────────────────────────────────────────────────────────────


class TestRobustness:
    def test_unknown_span_skipped_without_crash(self):
        events = _receive([_child("some.chain.step", {"custom.attr": "x"})])
        types = {e.event_type for e in events}
        # A run is still opened/closed, but no llm/tool/retrieval events.
        assert EventType.RUN_STARTED in types
        assert EventType.LLM_CALLED not in types
        assert EventType.TOOL_CALLED not in types
        assert EventType.RETRIEVAL_CALLED not in types

    def test_missing_fields_do_not_crash(self):
        # LLM span with only the marker attribute, everything else absent.
        events = _receive([_child("chat", {"gen_ai.request.model": "gpt-4o"})])
        assert _payload(events, EventType.LLM_CALLED)["prompt_tokens"] == 0

    def test_span_for_unfinished_trace_is_buffered_not_emitted(self):
        # No root span => trace stays pending, nothing emitted, no crash.
        events = []
        dt = Dunetrace(
            emitter=NoopBatchingEmitter(),
            exporters=[CallableExporter(lambda e: events.append(e))],
        )
        receiver = DunetraceOTelReceiver(dt, agent_id="a")
        receiver.export([_child("chat", {"gen_ai.request.model": "gpt-4o"})])
        dt.shutdown()
        assert events == []
