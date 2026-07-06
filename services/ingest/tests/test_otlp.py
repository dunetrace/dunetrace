"""
Unit tests for the OTLP → Dunetrace event mapper.
No DB or network required — pure unit tests of otel.py.

Run:
    cd services/ingest
    pytest tests/test_otlp.py -v
"""

from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingest_svc.otel import (
    otlp_to_events,
    protobuf_to_resource_spans,
    _classify,
    _val,
    _trace_to_uuid,
)

# ── _val ──────────────────────────────────────────────────────────────────────


def test_val_string():
    assert _val({"stringValue": "hello"}) == "hello"


def test_val_int_as_string():
    assert _val({"intValue": "42"}) == 42


def test_val_double():
    assert _val({"doubleValue": 3.14}) == pytest.approx(3.14)


def test_val_bool():
    assert _val({"boolValue": True}) is True


def test_val_array():
    v = {"arrayValue": {"values": [{"intValue": "1"}, {"intValue": "2"}]}}
    assert _val(v) == [1, 2]


# ── _classify ────────────────────────────────────────────────────────────────


def test_classify_llm_by_attribute():
    assert _classify("some.span", {"gen_ai.system": "openai"}) == "llm"


def test_classify_llm_by_model_attr():
    assert _classify("any", {"llm.request.model": "gpt-4"}) == "llm"


def test_classify_tool_by_attribute():
    assert _classify("any", {"tool.name": "search"}) == "tool"


def test_classify_retrieval_by_attribute():
    assert _classify("any", {"vector_db.vendor": "pinecone"}) == "retrieval"


def test_classify_llm_by_name():
    assert _classify("openai.chat.completions", {}) == "llm"


def test_classify_tool_by_name():
    assert _classify("tool_call", {}) == "tool"


def test_classify_retrieval_by_name():
    assert _classify("pinecone.query", {}) == "retrieval"


def test_classify_lifecycle_by_name():
    assert _classify("langchain.chain.invoke", {}) == "lifecycle"


def test_classify_skip():
    assert _classify("some.random.span", {}) == "skip"


# ── _trace_to_uuid ────────────────────────────────────────────────────────────


def test_trace_to_uuid_format():
    tid = "4bf92f3577b34da6a3ce929d0e0e4736"
    result = _trace_to_uuid(tid)
    assert result == "4bf92f35-77b3-4da6-a3ce-929d0e0e4736"
    assert len(result) == 36


# ── otlp_to_events ────────────────────────────────────────────────────────────


def _make_resource_span(trace_id, spans, service_name="my-agent", service_version="1.0"):
    return {
        "resource": {
            "attributes": [
                {"key": "service.name", "value": {"stringValue": service_name}},
                {"key": "service.version", "value": {"stringValue": service_version}},
            ]
        },
        "scopeSpans": [{"spans": spans}],
    }


def _span(span_id, parent_id, name, start_ns, end_ns, attrs=None, status_code=0):
    s = {
        "traceId": "4bf92f3577b34da6a3ce929d0e0e4736",
        "spanId": span_id,
        "parentSpanId": parent_id,
        "name": name,
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(end_ns),
        "attributes": attrs or [],
        "status": {"code": status_code},
    }
    return s


def _attr(key, val_type, val):
    return {"key": key, "value": {val_type: val}}


def test_minimal_root_span_only():
    root = _span("aaaa", "", "my-agent", 1_000_000_000_000, 2_000_000_000_000)
    events = otlp_to_events([_make_resource_span("trace1", [root])])
    types = [e["event_type"] for e in events]
    assert types == ["run.started", "run.completed"]
    assert events[0]["agent_id"] == "my-agent"
    assert events[0]["agent_version"] == "1.0"


def test_llm_span_produces_called_and_responded():
    # Timestamps in nanoseconds; 800_000_000 ns = 800 ms
    root = _span("root", "", "agent", 1_000_000_000, 3_000_000_000)
    llm = _span(
        "llm1",
        "root",
        "openai.chat.completions",
        1_100_000_000,
        1_900_000_000,
        attrs=[
            _attr("gen_ai.system", "stringValue", "openai"),
            _attr("gen_ai.request.model", "stringValue", "gpt-4o"),
            _attr("gen_ai.usage.input_tokens", "intValue", "100"),
            _attr("gen_ai.usage.output_tokens", "intValue", "50"),
        ],
    )
    events = otlp_to_events([_make_resource_span("t", [root, llm])])
    types = [e["event_type"] for e in events]
    assert "llm.called" in types
    assert "llm.responded" in types

    called = next(e for e in events if e["event_type"] == "llm.called")
    responded = next(e for e in events if e["event_type"] == "llm.responded")
    assert called["payload"]["model"] == "gpt-4o"
    assert called["payload"]["prompt_tokens"] == 100
    assert responded["payload"]["completion_tokens"] == 50
    assert responded["payload"]["latency_ms"] == 800


def test_tool_span_produces_called_and_responded():
    root = _span("root", "", "agent", 1_000_000_000_000, 3_000_000_000_000)
    tool = _span(
        "t1",
        "root",
        "search_web",
        1_200_000_000_000,
        1_500_000_000_000,
        attrs=[_attr("tool.name", "stringValue", "search_web")],
    )
    events = otlp_to_events([_make_resource_span("t", [root, tool])])
    types = [e["event_type"] for e in events]
    assert "tool.called" in types
    assert "tool.responded" in types
    called = next(e for e in events if e["event_type"] == "tool.called")
    assert called["payload"]["tool_name"] == "search_web"


def test_retrieval_span_produces_called_and_responded():
    root = _span("root", "", "agent", 1_000_000_000_000, 3_000_000_000_000)
    ret = _span(
        "r1",
        "root",
        "pinecone.query",
        1_300_000_000_000,
        1_600_000_000_000,
        attrs=[
            _attr("vector_db.vendor", "stringValue", "pinecone"),
            _attr("vector_db.collection_name", "stringValue", "my-index"),
            _attr("retrieval.result_count", "intValue", "5"),
        ],
    )
    events = otlp_to_events([_make_resource_span("t", [root, ret])])
    types = [e["event_type"] for e in events]
    assert "retrieval.called" in types
    assert "retrieval.responded" in types
    resp = next(e for e in events if e["event_type"] == "retrieval.responded")
    assert resp["payload"]["result_count"] == 5


def test_errored_root_span_produces_run_errored():
    root = _span("root", "", "agent", 1_000_000_000_000, 2_000_000_000_000, status_code=2)
    events = otlp_to_events([_make_resource_span("t", [root])])
    types = [e["event_type"] for e in events]
    assert "run.errored" in types
    assert "run.completed" not in types


def test_lifecycle_spans_not_counted_as_steps():
    root = _span("root", "", "agent", 1_000_000_000_000, 4_000_000_000_000)
    lc = _span("lc1", "root", "langchain.chain.invoke", 1_100_000_000_000, 1_200_000_000_000)
    llm = _span(
        "llm1",
        "root",
        "openai.completion",
        1_200_000_000_000,
        1_900_000_000_000,
        attrs=[_attr("gen_ai.system", "stringValue", "openai")],
    )
    events = otlp_to_events([_make_resource_span("t", [root, lc, llm])])
    # llm events should be at step 1, not step 2
    called = next(e for e in events if e["event_type"] == "llm.called")
    assert called["step_index"] == 1


def test_agent_id_override():
    root = _span("root", "", "agent", 1_000_000_000_000, 2_000_000_000_000)
    events = otlp_to_events(
        [_make_resource_span("t", [root], service_name="from-otel")],
        agent_id_override="forced-agent",
    )
    assert all(e["agent_id"] == "forced-agent" for e in events)


def test_multiple_traces_in_one_batch():
    def _make_trace(trace_id, service):
        root = {
            "traceId": trace_id,
            "spanId": "aaaa",
            "parentSpanId": "",
            "name": "root",
            "startTimeUnixNano": str(1_000_000_000_000),
            "endTimeUnixNano": str(2_000_000_000_000),
            "attributes": [],
            "status": {"code": 0},
        }
        return _make_resource_span(trace_id, [root], service_name=service)

    events = otlp_to_events(
        [
            _make_trace("aabbccdd" * 4, "agent-a"),
            _make_trace("11223344" * 4, "agent-b"),
        ]
    )
    agent_ids = {e["agent_id"] for e in events}
    run_ids = {e["run_id"] for e in events}
    assert "agent-a" in agent_ids
    assert "agent-b" in agent_ids
    assert len(run_ids) == 2


# ── protobuf_to_resource_spans (D11) ────────────────────────────────────────────


def _protobuf_export_request(agent_id="proto-agent", span_name="root"):
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )
    from opentelemetry.proto.common.v1.common_pb2 import AnyValue

    req = ExportTraceServiceRequest()
    rs = req.resource_spans.add()
    rs.resource.attributes.add(key="service.name", value=AnyValue(string_value=agent_id))
    ss = rs.scope_spans.add()
    span = ss.spans.add()
    span.trace_id = bytes.fromhex("4bf92f3577b34da6a3ce929d0e0e4736")
    span.span_id = bytes.fromhex("00f067aa0ba902b7")
    span.name = span_name
    span.start_time_unix_nano = 1_000_000_000_000
    span.end_time_unix_nano = 2_000_000_000_000
    return req


def test_protobuf_to_resource_spans_matches_json_shape():
    req = _protobuf_export_request()
    resource_spans = protobuf_to_resource_spans(req.SerializeToString())

    assert len(resource_spans) == 1
    rs = resource_spans[0]
    assert rs["resource"]["attributes"][0] == {
        "key": "service.name",
        "value": {"stringValue": "proto-agent"},
    }
    span = rs["scopeSpans"][0]["spans"][0]
    assert span["startTimeUnixNano"] == "1000000000000"
    assert span["endTimeUnixNano"] == "2000000000000"


def test_protobuf_trace_id_decoded_to_hex_not_base64():
    """MessageToDict base64-encodes bytes fields by default — traceId/spanId
    must be corrected to plain hex, matching the OTLP JSON convention and
    what _trace_to_uuid()/the rest of this module already expect."""
    req = _protobuf_export_request()
    resource_spans = protobuf_to_resource_spans(req.SerializeToString())
    span = resource_spans[0]["scopeSpans"][0]["spans"][0]
    assert span["traceId"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert span["spanId"] == "00f067aa0ba902b7"


def test_protobuf_output_produces_valid_dunetrace_events():
    """End-to-end: a protobuf-sourced resourceSpans list must produce the
    same event shape as the equivalent JSON input."""
    req = _protobuf_export_request(agent_id="proto-agent")
    resource_spans = protobuf_to_resource_spans(req.SerializeToString())
    events = otlp_to_events(resource_spans)

    assert any(e["event_type"] == "run.started" for e in events)
    assert any(e["event_type"] == "run.completed" for e in events)
    assert all(e["agent_id"] == "proto-agent" for e in events)


def test_protobuf_malformed_bytes_raises():
    with pytest.raises(Exception):
        protobuf_to_resource_spans(b"\xff\xff\xff not a real protobuf message at all")


# ── Per-resource / per-trace error isolation (D11) ──────────────────────────────


def test_malformed_resource_span_does_not_affect_others():
    """One resourceSpan with an unexpected structure must not prevent a
    sibling, well-formed resourceSpan in the same batch from producing
    events — see otlp_to_events()'s docstring."""
    good_root = _span("aaaa", "", "root", 1_000_000_000_000, 2_000_000_000_000)
    good_resource = _make_resource_span(
        "4bf92f3577b34da6a3ce929d0e0e4736", [good_root], service_name="good-agent"
    )
    # "resource" is a string instead of a dict — .get("attributes", []) on
    # a str raises AttributeError inside the mapper's resource-attrs step.
    bad_resource = {"resource": "not-a-dict", "scopeSpans": [{"spans": [good_root]}]}

    events = otlp_to_events([bad_resource, good_resource])

    assert any(e["agent_id"] == "good-agent" for e in events)
    assert any(e["event_type"] == "run.started" for e in events)


def test_malformed_trace_does_not_affect_other_traces():
    """One trace whose spans are unprocessable must not prevent a sibling
    trace in the same batch from producing events."""
    good_root = _span("aaaa", "", "root", 1_000_000_000_000, 2_000_000_000_000)
    good_resource = _make_resource_span("4bf92f3577b34da6a3ce929d0e0e4736", [good_root])

    # A span whose startTimeUnixNano can't be coerced to int — raises inside
    # the chronological sort in _events_for_trace(). Distinct traceId from
    # the good span above — _span() hardcodes traceId, so it must be
    # overridden here, otherwise both spans land in the same trace bucket
    # (otlp_to_events groups by traceId, not by which resourceSpan a span
    # came from) and this test would poison the "good" trace instead of
    # proving isolation between two separate ones.
    bad_span = _span("bbbb", "", "root", "not-a-number", 2_000_000_000_000)
    bad_span["traceId"] = "11112222" * 4
    bad_resource = _make_resource_span(bad_span["traceId"], [bad_span])

    events = otlp_to_events([bad_resource, good_resource])

    run_ids = {e["run_id"] for e in events}
    assert len(run_ids) == 1  # only the good trace survived
    assert any(e["event_type"] == "run.started" for e in events)


def test_all_malformed_returns_empty_list_not_raise():
    bad_resource = {"resource": "not-a-dict", "scopeSpans": "also-not-a-list"}
    events = otlp_to_events([bad_resource])
    assert events == []
