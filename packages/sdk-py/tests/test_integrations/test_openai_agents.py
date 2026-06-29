"""
Tests for DunetraceTracingProcessor (OpenAI Agents SDK integration).

These tests stub out the ``agents`` package so they run without installing
openai-agents, and use lightweight fakes for the SDK's Trace/Span objects.
"""

import json
import sys
import time
import types
import unittest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Stub out the agents package so we can import the processor without it
# ---------------------------------------------------------------------------
def _stub_agents() -> None:
    if "agents" in sys.modules:
        return

    agents = types.ModuleType("agents")
    tracing = types.ModuleType("agents.tracing")

    class TracingProcessor:
        pass

    def add_trace_processor(processor):  # noqa: ANN001
        _stub_agents.registered.append(processor)

    tracing.TracingProcessor = TracingProcessor
    tracing.add_trace_processor = add_trace_processor
    agents.tracing = tracing

    sys.modules["agents"] = agents
    sys.modules["agents.tracing"] = tracing

    import dunetrace.integrations.openai_agents as _mod

    _mod._AGENTS_AVAILABLE = True


_stub_agents.registered = []  # type: ignore[attr-defined]
_stub_agents()

from dunetrace.integrations.openai_agents import (  # noqa: E402
    DunetraceTracingProcessor,
    add_dunetrace_processor,
)
from dunetrace.models import EventType, hash_content  # noqa: E402


# ── Fakes for SDK Trace/Span objects ────────────────────────────────────────


class _FakeTrace:
    def __init__(self, trace_id="trace_1", name="workflow", metadata=None):
        self.trace_id = trace_id
        self.name = name
        self.metadata = metadata


class _FakeSpan:
    def __init__(self, span_id, trace_id, span_data, error=None):
        self.span_id = span_id
        self.trace_id = trace_id
        self.span_data = span_data
        self.error = error


class _GenerationData:
    type = "generation"

    def __init__(self, model="gpt-4o", output=None, usage=None, input=None):
        self.model = model
        self.input = input if input is not None else []
        self.output = output
        self.usage = usage


class _ResponseData:
    type = "response"

    def __init__(self, response=None, usage=None):
        self.response = response
        self.input = []
        self.usage = usage


class _FunctionData:
    type = "function"

    def __init__(self, name="search", input="q", output="result"):
        self.name = name
        self.input = input
        self.output = output


class _HandoffData:
    type = "handoff"

    def __init__(self, from_agent="AgentA", to_agent="AgentB"):
        self.from_agent = from_agent
        self.to_agent = to_agent


class _McpToolsData:
    type = "mcp_tools"

    def __init__(self, server="github", input="{}", output="ok"):
        self.server = server
        self.input = input
        self.output = output


def _make_processor(tools=None, model="gpt-4o"):
    client = MagicMock()
    emitted = []
    client._emit.side_effect = emitted.append
    proc = DunetraceTracingProcessor(
        client, agent_id="test-agent", model=model, tools=tools or ["search"]
    )
    return proc, emitted


def _types(emitted):
    return [e.event_type for e in emitted]


# ── Run lifecycle ────────────────────────────────────────────────────────────


class TestRunLifecycle(unittest.TestCase):
    def test_trace_start_with_metadata_emits_run_started(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "hello world"}))
        self.assertEqual(emitted[0].event_type, EventType.RUN_STARTED)
        self.assertEqual(emitted[0].payload["input_hash"], hash_content("hello world"))

    def test_trace_start_without_metadata_defers_run_started(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace())
        self.assertEqual(len(emitted), 0)

    def test_run_started_uses_span_input_when_metadata_missing(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace())
        span = _FakeSpan("s1", "trace_1", _GenerationData(input="user query"))
        proc.on_span_start(span)
        started = next(e for e in emitted if e.event_type == EventType.RUN_STARTED)
        self.assertEqual(started.payload["input_hash"], hash_content("user query"))

    def test_trace_end_emits_run_completed(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        proc.on_trace_end(_FakeTrace())
        self.assertIn(EventType.RUN_COMPLETED, _types(emitted))

    def test_run_started_payload_has_tools_and_model(self):
        proc, emitted = _make_processor(tools=["search", "calc"], model="gpt-4o")
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        payload = emitted[0].payload
        self.assertEqual(payload["tools"], ["search", "calc"])
        self.assertEqual(payload["model"], "gpt-4o")

    def test_run_id_matches_trace_id(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(trace_id="trace_abc", metadata={"input": "q"}))
        self.assertEqual(emitted[0].run_id, "trace_abc")

    def test_last_run_id_set_after_completion(self):
        proc, _ = _make_processor()
        proc.on_trace_start(_FakeTrace(trace_id="trace_xyz", metadata={"input": "q"}))
        proc.on_trace_end(_FakeTrace(trace_id="trace_xyz"))
        self.assertEqual(proc.last_run_id, "trace_xyz")

    def test_run_completed_payload(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        span = _FakeSpan("s1", "trace_1", _FunctionData())
        proc.on_span_start(span)
        proc.on_span_end(span)
        proc.on_trace_end(_FakeTrace())
        completed = next(e for e in emitted if e.event_type == EventType.RUN_COMPLETED)
        self.assertEqual(completed.payload["exit_reason"], "final_answer")
        self.assertEqual(completed.payload["total_steps"], 1)
        self.assertEqual(completed.payload["tool_call_count"], 1)

    def test_run_errored_uses_last_error_hash(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        span = _FakeSpan("s1", "trace_1", _GenerationData())
        proc.on_span_start(span)
        span.error = {"message": "boom"}
        proc.on_span_end(span)
        proc.on_trace_end(_FakeTrace())
        errored = next(e for e in emitted if e.event_type == EventType.RUN_ERRORED)
        self.assertEqual(errored.payload["error_hash"], hash_content("boom"))

    def test_state_cleared_after_completion(self):
        proc, _ = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        proc.on_trace_end(_FakeTrace())
        self.assertEqual(len(proc._runs), 0)

    def test_span_error_emits_run_errored(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        span = _FakeSpan("s1", "trace_1", _GenerationData())
        proc.on_span_start(span)
        span.error = {"message": "boom"}
        proc.on_span_end(span)
        proc.on_trace_end(_FakeTrace())
        self.assertIn(EventType.RUN_ERRORED, _types(emitted))
        self.assertNotIn(EventType.RUN_COMPLETED, _types(emitted))


# ── LLM spans ────────────────────────────────────────────────────────────────


class TestLLMSpans(unittest.TestCase):
    def test_generation_span_emits_llm_called_and_responded(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        span = _FakeSpan("s1", "trace_1", _GenerationData(model="gpt-4o-mini", output="hi"))
        proc.on_span_start(span)
        proc.on_span_end(span)
        self.assertIn(EventType.LLM_CALLED, _types(emitted))
        self.assertIn(EventType.LLM_RESPONDED, _types(emitted))

    def test_llm_called_carries_model(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        proc.on_span_start(_FakeSpan("s1", "trace_1", _GenerationData(model="gpt-4o-mini")))
        called = next(e for e in emitted if e.event_type == EventType.LLM_CALLED)
        self.assertEqual(called.payload["model"], "gpt-4o-mini")

    def test_llm_responded_hashes_output(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        secret = "secret model output"
        span = _FakeSpan("s1", "trace_1", _GenerationData(output=secret))
        proc.on_span_start(span)
        proc.on_span_end(span)
        responded = next(e for e in emitted if e.event_type == EventType.LLM_RESPONDED)
        self.assertNotIn(secret, json.dumps(responded.payload))
        self.assertIn("output_hash", responded.payload)

    def test_generation_usage_tokens(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        usage = {
            "input_tokens": 10,
            "output_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 2},
        }
        span = _FakeSpan("s1", "trace_1", _GenerationData(usage=usage))
        proc.on_span_start(span)
        proc.on_span_end(span)
        responded = next(e for e in emitted if e.event_type == EventType.LLM_RESPONDED)
        self.assertEqual(responded.payload["prompt_tokens"], 10)
        self.assertEqual(responded.payload["completion_tokens"], 5)
        self.assertEqual(responded.payload["reasoning_tokens"], 2)

    def test_response_span_pulls_model_usage_and_output(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        response = types.SimpleNamespace(
            model="gpt-4.1",
            usage=types.SimpleNamespace(input_tokens=7, output_tokens=3),
            output_text="answer text",
            status="completed",
        )
        span = _FakeSpan("s1", "trace_1", _ResponseData(response=response))
        proc.on_span_start(span)
        proc.on_span_end(span)
        called = next(e for e in emitted if e.event_type == EventType.LLM_CALLED)
        responded = next(e for e in emitted if e.event_type == EventType.LLM_RESPONDED)
        self.assertEqual(called.payload["model"], "gpt-4.1")
        self.assertEqual(responded.payload["prompt_tokens"], 7)
        self.assertEqual(responded.payload["completion_tokens"], 3)
        self.assertEqual(responded.payload["output_length"], len("answer text"))
        self.assertEqual(
            responded.payload["output_hash"], hash_content("answer text")
        )

    def test_response_span_tool_calls_finish_reason(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        response = types.SimpleNamespace(
            model="gpt-4o",
            status="completed",
            output_text="",
            output=[types.SimpleNamespace(type="function_call", name="search")],
        )
        span = _FakeSpan("s1", "trace_1", _ResponseData(response=response))
        proc.on_span_start(span)
        proc.on_span_end(span)
        responded = next(e for e in emitted if e.event_type == EventType.LLM_RESPONDED)
        self.assertEqual(responded.payload["finish_reason"], "tool_calls")

    def test_response_span_incomplete_finish_reason(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        response = types.SimpleNamespace(
            model="gpt-4o",
            status="incomplete",
            incomplete_details=types.SimpleNamespace(reason="max_output_tokens"),
            output_text="partial",
        )
        span = _FakeSpan("s1", "trace_1", _ResponseData(response=response))
        proc.on_span_start(span)
        proc.on_span_end(span)
        responded = next(e for e in emitted if e.event_type == EventType.LLM_RESPONDED)
        self.assertEqual(responded.payload["finish_reason"], "length")

    def test_generation_structured_output_text(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        output = [{"role": "assistant", "content": "structured answer"}]
        span = _FakeSpan("s1", "trace_1", _GenerationData(output=output))
        proc.on_span_start(span)
        proc.on_span_end(span)
        responded = next(e for e in emitted if e.event_type == EventType.LLM_RESPONDED)
        self.assertEqual(responded.payload["output_length"], len("structured answer"))
        self.assertEqual(
            responded.payload["output_hash"], hash_content("structured answer")
        )

    def test_llm_error_emits_error_finish_reason(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        span = _FakeSpan("s1", "trace_1", _GenerationData())
        proc.on_span_start(span)
        span.error = {"message": "rate limited"}
        proc.on_span_end(span)
        responded = next(e for e in emitted if e.event_type == EventType.LLM_RESPONDED)
        self.assertEqual(responded.payload["finish_reason"], "error")
        self.assertNotIn("rate limited", json.dumps(responded.payload))
        self.assertIn("error_hash", responded.payload)

    def test_response_failed_status_emits_run_errored(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        response = types.SimpleNamespace(
            model="gpt-4o",
            status="failed",
            output_text="",
        )
        span = _FakeSpan("s1", "trace_1", _ResponseData(response=response))
        proc.on_span_start(span)
        proc.on_span_end(span)
        proc.on_trace_end(_FakeTrace())
        responded = next(e for e in emitted if e.event_type == EventType.LLM_RESPONDED)
        self.assertEqual(responded.payload["finish_reason"], "error")
        self.assertIn(EventType.RUN_ERRORED, _types(emitted))
        self.assertNotIn(EventType.RUN_COMPLETED, _types(emitted))

    def test_response_span_skipped_when_immediately_after_generation(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        gen = _FakeSpan("g1", "trace_1", _GenerationData(output="from generation"))
        resp = _FakeSpan(
            "r1",
            "trace_1",
            _ResponseData(
                response=types.SimpleNamespace(
                    model="gpt-4o",
                    status="completed",
                    output_text="from response",
                )
            ),
        )
        proc.on_span_start(gen)
        proc.on_span_end(gen)
        proc.on_span_start(resp)
        proc.on_span_end(resp)
        llm_called = [e for e in emitted if e.event_type == EventType.LLM_CALLED]
        llm_responded = [e for e in emitted if e.event_type == EventType.LLM_RESPONDED]
        self.assertEqual(len(llm_called), 1)
        self.assertEqual(len(llm_responded), 1)
        responded = llm_responded[0]
        self.assertEqual(
            responded.payload["output_hash"], hash_content("from generation")
        )

    def test_response_span_emitted_after_generation_when_handoff_intervenes(self):
        """Mixed API backends in one trace: generation then handoff then response."""
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        gen = _FakeSpan("g1", "trace_1", _GenerationData(output="chat answer"))
        handoff = _FakeSpan("h1", "trace_1", _HandoffData(from_agent="Chat", to_agent="Resp"))
        resp = _FakeSpan(
            "r1",
            "trace_1",
            _ResponseData(
                response=types.SimpleNamespace(
                    model="gpt-4o",
                    status="completed",
                    output_text="responses answer",
                )
            ),
        )
        proc.on_span_start(gen)
        proc.on_span_end(gen)
        proc.on_span_start(handoff)
        proc.on_span_end(handoff)
        proc.on_span_start(resp)
        proc.on_span_end(resp)
        llm_called = [e for e in emitted if e.event_type == EventType.LLM_CALLED]
        llm_responded = [e for e in emitted if e.event_type == EventType.LLM_RESPONDED]
        self.assertEqual(len(llm_called), 2)
        self.assertEqual(len(llm_responded), 2)
        self.assertEqual(
            llm_responded[-1].payload["output_hash"], hash_content("responses answer")
        )


# ── Tool spans ───────────────────────────────────────────────────────────────


class TestToolSpans(unittest.TestCase):
    def test_function_span_emits_tool_called_and_responded(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        span = _FakeSpan("s1", "trace_1", _FunctionData(name="get_weather"))
        proc.on_span_start(span)
        proc.on_span_end(span)
        self.assertIn(EventType.TOOL_CALLED, _types(emitted))
        self.assertIn(EventType.TOOL_RESPONDED, _types(emitted))
        called = next(e for e in emitted if e.event_type == EventType.TOOL_CALLED)
        self.assertEqual(called.payload["tool_name"], "get_weather")

    def test_tool_args_are_hashed(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        secret = "sensitive args"
        span = _FakeSpan("s1", "trace_1", _FunctionData(input=secret))
        proc.on_span_start(span)
        called = next(e for e in emitted if e.event_type == EventType.TOOL_CALLED)
        self.assertNotIn(secret, json.dumps(called.payload))
        self.assertIn("args_hash", called.payload)

    def test_tool_count_in_run_completed(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        for i in range(3):
            span = _FakeSpan(f"s{i}", "trace_1", _FunctionData())
            proc.on_span_start(span)
            proc.on_span_end(span)
        proc.on_trace_end(_FakeTrace())
        completed = next(e for e in emitted if e.event_type == EventType.RUN_COMPLETED)
        self.assertEqual(completed.payload["tool_call_count"], 3)

    def test_tool_error_marks_success_false(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        span = _FakeSpan("s1", "trace_1", _FunctionData())
        proc.on_span_start(span)
        span.error = types.SimpleNamespace(message="tool blew up")
        proc.on_span_end(span)
        responded = next(e for e in emitted if e.event_type == EventType.TOOL_RESPONDED)
        self.assertFalse(responded.payload["success"])
        self.assertNotIn("tool blew up", json.dumps(responded.payload))

    def test_called_and_responded_share_step_index(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        # Interleave two tool spans so steps would diverge without span_steps.
        a = _FakeSpan("a", "trace_1", _FunctionData())
        b = _FakeSpan("b", "trace_1", _FunctionData())
        proc.on_span_start(a)
        proc.on_span_start(b)
        proc.on_span_end(a)
        proc.on_span_end(b)
        called = {e.payload["tool_name"]: e for e in emitted if e.event_type == EventType.TOOL_CALLED}
        # span a got step 1, span b got step 2; responded must match starts.
        ends = [e for e in emitted if e.event_type == EventType.TOOL_RESPONDED]
        self.assertEqual(ends[0].step_index, 1)  # span a
        self.assertEqual(ends[1].step_index, 2)  # span b

    def test_tool_responded_includes_latency_ms(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        span = _FakeSpan("s1", "trace_1", _FunctionData())
        proc.on_span_start(span)
        proc.on_span_end(span)
        responded = next(e for e in emitted if e.event_type == EventType.TOOL_RESPONDED)
        self.assertIn("latency_ms", responded.payload)
        self.assertGreaterEqual(responded.payload["latency_ms"], 0)

    def test_mcp_tools_span_is_not_a_tool_call(self):
        # `mcp_tools` is a list-tools span, not an invocation; it must not emit
        # tool events or inflate tool_call_count.
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        span = _FakeSpan("s1", "trace_1", _McpToolsData(server="github"))
        proc.on_span_start(span)
        proc.on_span_end(span)
        proc.on_trace_end(_FakeTrace())
        self.assertNotIn(EventType.TOOL_CALLED, _types(emitted))
        self.assertNotIn(EventType.TOOL_RESPONDED, _types(emitted))
        completed = next(e for e in emitted if e.event_type == EventType.RUN_COMPLETED)
        self.assertEqual(completed.payload["tool_call_count"], 0)


# ── Handoff spans ────────────────────────────────────────────────────────────


class TestHandoffSpans(unittest.TestCase):
    def test_handoff_emits_tool_events(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        span = _FakeSpan("s1", "trace_1", _HandoffData(from_agent="Triage", to_agent="Specialist"))
        proc.on_span_start(span)
        proc.on_span_end(span)
        called = next(e for e in emitted if e.event_type == EventType.TOOL_CALLED)
        responded = next(e for e in emitted if e.event_type == EventType.TOOL_RESPONDED)
        self.assertEqual(called.payload["tool_name"], "handoff:Specialist")
        self.assertEqual(called.payload["args_hash"], hash_content("Triage"))
        self.assertTrue(responded.payload["success"])

    def test_handoff_responded_includes_latency_ms(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        span = _FakeSpan("s1", "trace_1", _HandoffData())
        proc.on_span_start(span)
        proc.on_span_end(span)
        responded = next(e for e in emitted if e.event_type == EventType.TOOL_RESPONDED)
        self.assertIn("latency_ms", responded.payload)

    def test_handoff_does_not_increment_tool_call_count(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        span = _FakeSpan("s1", "trace_1", _HandoffData())
        proc.on_span_start(span)
        proc.on_span_end(span)
        proc.on_trace_end(_FakeTrace())
        completed = next(e for e in emitted if e.event_type == EventType.RUN_COMPLETED)
        self.assertEqual(completed.payload["tool_call_count"], 0)
        self.assertEqual(completed.payload["total_steps"], 1)


# ── Concurrency / isolation ──────────────────────────────────────────────────


class TestConcurrentTraces(unittest.TestCase):
    def test_two_traces_do_not_collide(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(trace_id="t1", metadata={"input": "q1"}))
        proc.on_trace_start(_FakeTrace(trace_id="t2", metadata={"input": "q2"}))
        proc.on_span_start(_FakeSpan("s1", "t1", _FunctionData()))
        proc.on_span_start(_FakeSpan("s2", "t2", _FunctionData()))
        proc.on_trace_end(_FakeTrace(trace_id="t1"))
        proc.on_trace_end(_FakeTrace(trace_id="t2"))
        completed = [e for e in emitted if e.event_type == EventType.RUN_COMPLETED]
        by_run = {e.run_id: e for e in completed}
        self.assertEqual(by_run["t1"].payload["tool_call_count"], 1)
        self.assertEqual(by_run["t2"].payload["tool_call_count"], 1)

    def test_span_without_known_trace_is_ignored(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(trace_id="t1", metadata={"input": "q"}))
        before = len(emitted)
        proc.on_span_start(_FakeSpan("s1", "unknown", _FunctionData()))
        self.assertEqual(len(emitted), before)


# ── Registration helper ──────────────────────────────────────────────────────


class TestRegistration(unittest.TestCase):
    def setUp(self):
        import dunetrace.integrations.openai_agents as oa_mod

        oa_mod._registered.clear()
        _stub_agents.registered.clear()

    def test_add_dunetrace_processor_registers(self):
        client = MagicMock()
        proc = add_dunetrace_processor(client, agent_id="a", model="gpt-4o")
        self.assertIs(_stub_agents.registered[-1], proc)
        self.assertIsInstance(proc, DunetraceTracingProcessor)

    def test_duplicate_registration_returns_existing(self):
        client = MagicMock()
        first = add_dunetrace_processor(client, agent_id="a", model="gpt-4o")
        second = add_dunetrace_processor(client, agent_id="a", model="gpt-4o")
        self.assertIs(first, second)
        self.assertEqual(len(_stub_agents.registered), 1)

    def test_second_agent_reuses_first_processor(self):
        # Agents SDK tracing is process-global: a second processor would re-emit
        # every trace under a second agent_id, so registration is refused.
        client = MagicMock()
        first = add_dunetrace_processor(client, agent_id="a", model="gpt-4o")
        second = add_dunetrace_processor(client, agent_id="b", model="gpt-4o")
        self.assertIs(first, second)
        self.assertEqual(len(_stub_agents.registered), 1)


# ── Robustness ───────────────────────────────────────────────────────────────


class TestRobustness(unittest.TestCase):
    def test_unknown_span_type_ignored(self):
        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        before = len(emitted)
        agent_data = types.SimpleNamespace(type="agent", name="Assistant")
        proc.on_span_start(_FakeSpan("s1", "trace_1", agent_data))
        proc.on_span_end(_FakeSpan("s1", "trace_1", agent_data))
        self.assertEqual(len(emitted), before)

    def test_emit_failure_does_not_raise(self):
        client = MagicMock()
        client._emit.side_effect = RuntimeError("network down")
        proc = DunetraceTracingProcessor(client, agent_id="a", model="gpt-4o")
        # Should not raise despite _emit failing.
        proc.on_trace_start(_FakeTrace(metadata={"input": "q"}))
        proc.on_trace_end(_FakeTrace())


class TestStalePruning(unittest.TestCase):
    def setUp(self):
        import dunetrace.integrations.openai_agents as oa_mod

        oa_mod._registered.clear()

    def test_stale_trace_pruned_without_terminal_event(self):
        import dunetrace.integrations.openai_agents as oa_mod

        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(trace_id="old", metadata={"input": "q"}))
        with proc._lock:
            proc._runs["old"].last_activity_time = time.time() - oa_mod._STALE_RUN_SECS - 1

        proc.on_trace_start(_FakeTrace(trace_id="new", metadata={"input": "q2"}))

        self.assertNotIn(EventType.RUN_ERRORED, _types(emitted))
        self.assertNotIn(EventType.RUN_COMPLETED, [e.event_type for e in emitted if e.run_id == "old"])
        with proc._lock:
            self.assertNotIn("old", proc._runs)
        self.assertIn("old", proc._stale_pruned)

    def test_stale_pruned_trace_resurrects_on_resumed_activity(self):
        import dunetrace.integrations.openai_agents as oa_mod

        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(trace_id="idle", metadata={"input": "q"}))
        with proc._lock:
            proc._runs["idle"].last_activity_time = (
                time.time() - oa_mod._STALE_RUN_SECS - 1
            )
        proc.on_trace_start(_FakeTrace(trace_id="new", metadata={"input": "q2"}))
        self.assertIn("idle", proc._stale_pruned)

        span = _FakeSpan("s1", "idle", _FunctionData())
        proc.on_span_start(span)
        proc.on_span_end(span)
        proc.on_trace_end(_FakeTrace(trace_id="idle"))

        idle_events = [e for e in emitted if e.run_id == "idle"]
        types_ = [e.event_type for e in idle_events]
        self.assertNotIn(EventType.RUN_ERRORED, types_)
        self.assertIn(EventType.TOOL_CALLED, types_)
        self.assertIn(EventType.RUN_COMPLETED, types_)
        self.assertNotIn("idle", proc._stale_pruned)

    def test_stale_sweep_skips_trace_with_recent_activity(self):
        import dunetrace.integrations.openai_agents as oa_mod

        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(trace_id="old", metadata={"input": "q"}))
        with proc._lock:
            proc._runs["old"].last_activity_time = time.time() - oa_mod._STALE_RUN_SECS - 1
        # Activity resumes before the next trace triggers the sweep.
        proc.on_span_start(_FakeSpan("s1", "old", _FunctionData()))
        proc.on_span_end(_FakeSpan("s1", "old", _FunctionData()))
        proc.on_trace_start(_FakeTrace(trace_id="new", metadata={"input": "q2"}))

        self.assertNotIn("old", proc._stale_pruned)
        with proc._lock:
            self.assertIn("old", proc._runs)

    def test_stale_sweep_skips_trace_with_open_spans(self):
        import dunetrace.integrations.openai_agents as oa_mod

        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(trace_id="old", metadata={"input": "q"}))
        proc.on_span_start(_FakeSpan("s1", "old", _FunctionData()))
        with proc._lock:
            proc._runs["old"].last_activity_time = time.time() - oa_mod._STALE_RUN_SECS - 1

        proc.on_trace_start(_FakeTrace(trace_id="new", metadata={"input": "q2"}))

        self.assertNotIn("old", proc._stale_pruned)
        with proc._lock:
            self.assertIn("old", proc._runs)
        proc.on_span_end(_FakeSpan("s1", "old", _FunctionData()))

    def test_span_end_resurrects_pruned_trace(self):
        import dunetrace.integrations.openai_agents as oa_mod

        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(trace_id="idle", metadata={"input": "q"}))
        with proc._lock:
            proc._runs["idle"].last_activity_time = time.time() - oa_mod._STALE_RUN_SECS - 1
        proc.on_trace_start(_FakeTrace(trace_id="new", metadata={"input": "q2"}))
        self.assertIn("idle", proc._stale_pruned)
        # Orphan span_end on a pruned trace should not raise.
        proc.on_span_end(_FakeSpan("orphan", "idle", _FunctionData()))
        proc.on_trace_end(_FakeTrace(trace_id="idle"))
        idle_types = [e.event_type for e in emitted if e.run_id == "idle"]
        self.assertIn(EventType.RUN_COMPLETED, idle_types)

    def test_in_flight_span_completes_after_stale_sweep(self):
        import dunetrace.integrations.openai_agents as oa_mod

        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(trace_id="old", metadata={"input": "q"}))
        span = _FakeSpan("s1", "old", _FunctionData())
        proc.on_span_start(span)
        with proc._lock:
            proc._runs["old"].last_activity_time = time.time() - oa_mod._STALE_RUN_SECS - 1
        proc.on_trace_start(_FakeTrace(trace_id="new", metadata={"input": "q2"}))
        self.assertNotIn("old", proc._stale_pruned)
        proc.on_span_end(span)
        self.assertIn(EventType.TOOL_RESPONDED, _types(emitted))

    def test_active_long_trace_not_pruned(self):
        import dunetrace.integrations.openai_agents as oa_mod

        proc, emitted = _make_processor()
        proc.on_trace_start(_FakeTrace(trace_id="long", metadata={"input": "q"}))
        with proc._lock:
            proc._runs["long"].start_time = time.time() - oa_mod._STALE_RUN_SECS - 1
            proc._runs["long"].last_activity_time = time.time()

        proc.on_trace_start(_FakeTrace(trace_id="new", metadata={"input": "q2"}))

        errored = [e for e in emitted if e.event_type == EventType.RUN_ERRORED]
        self.assertEqual(len(errored), 0)
        with proc._lock:
            self.assertIn("long", proc._runs)


if __name__ == "__main__":
    unittest.main()
