"""
Tests for DunetraceHaystackTracer (Haystack 2.x integration).

Stubs out haystack.tracing so no haystack package is required.
Run:
    cd packages/sdk-py
    python -m unittest tests.test_integrations.test_haystack -v
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Stub haystack.tracing so we can import the integration without haystack
# ---------------------------------------------------------------------------
def _stub_haystack() -> None:
    if "haystack" in sys.modules:
        return

    haystack = types.ModuleType("haystack")
    haystack_tracing = types.ModuleType("haystack.tracing")
    haystack.tracing = haystack_tracing
    sys.modules["haystack"] = haystack
    sys.modules["haystack.tracing"] = haystack_tracing


_stub_haystack()

from dunetrace import Dunetrace  # noqa: E402
from dunetrace.integrations.haystack import (  # noqa: E402
    DunetraceHaystackTracer,
    _classify_component,
    _extract_tokens,
    _extract_tool_calls,
    _flatten_input,
    _model_hint_from_type,
)
from dunetrace.models import EventType  # noqa: E402


def _make_dt() -> tuple[Dunetrace, list]:
    from dunetrace.policies import PolicyEngine

    dt = Dunetrace.__new__(Dunetrace)
    dt._ingest_url = "http://localhost:8001/v1/ingest"
    dt._api_key = ""
    dt._otel_exporter = None
    dt._policy_engine = PolicyEngine()
    emitted: list = []
    dt._emit = lambda event: emitted.append(event)
    dt.flush = lambda: None
    return dt, emitted


def _make_tracer() -> tuple[DunetraceHaystackTracer, list]:
    dt, emitted = _make_dt()
    tracer = DunetraceHaystackTracer(dt, agent_id="hs-agent", model="gpt-4o")
    return tracer, emitted


# ---------------------------------------------------------------------------
# Component classification
# ---------------------------------------------------------------------------
class TestClassifyComponent(unittest.TestCase):
    def test_openai_chat_generator_is_llm(self):
        self.assertEqual(_classify_component("OpenAIChatGenerator"), "llm")

    def test_retriever_is_retrieval(self):
        self.assertEqual(_classify_component("InMemoryBM25Retriever"), "retrieval")

    def test_tool_invoker_is_tool(self):
        self.assertEqual(_classify_component("ToolInvoker"), "tool")

    def test_agent_is_other(self):
        self.assertEqual(_classify_component("Agent"), "other")

    def test_pipeline_is_other(self):
        self.assertEqual(_classify_component("Pipeline"), "other")

    def test_unknown_is_other(self):
        self.assertEqual(_classify_component("CustomComponent"), "other")

    def test_llm_suffix(self):
        self.assertEqual(_classify_component("AnthropicChatGenerator"), "llm")

    def test_converter_is_other(self):
        self.assertEqual(_classify_component("DocumentConverter"), "other")


class TestModelHintFromType(unittest.TestCase):
    def test_openai_returns_gpt(self):
        self.assertIn("gpt", _model_hint_from_type("OpenAIChatGenerator"))

    def test_anthropic_returns_claude(self):
        self.assertIn("claude", _model_hint_from_type("AnthropicChatGenerator"))

    def test_unknown_returns_unknown(self):
        self.assertEqual(_model_hint_from_type("SomeNewGenerator"), "unknown")

    def test_google_returns_gemini(self):
        self.assertIn("gemini", _model_hint_from_type("GoogleAIGeminiChatGenerator"))


# ---------------------------------------------------------------------------
# _extract_tokens
# ---------------------------------------------------------------------------
class TestExtractTokens(unittest.TestCase):
    def _make_reply(self, prompt=100, completion=40, model="gpt-4o", finish="stop"):
        reply = MagicMock()
        reply.meta = {
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
            "finish_reason": finish,
            "model": model,
        }
        return reply

    def test_extracts_prompt_and_completion_tokens(self):
        output = {"replies": [self._make_reply(150, 60)]}
        pt, ct, finish, model = _extract_tokens(output)
        self.assertEqual(pt, 150)
        self.assertEqual(ct, 60)

    def test_extracts_finish_reason(self):
        output = {"replies": [self._make_reply(finish="length")]}
        _, _, finish, _ = _extract_tokens(output)
        self.assertEqual(finish, "length")

    def test_extracts_model(self):
        output = {"replies": [self._make_reply(model="gpt-4o-mini")]}
        _, _, _, model = _extract_tokens(output)
        self.assertEqual(model, "gpt-4o-mini")

    def test_empty_output_returns_nones(self):
        pt, ct, finish, model = _extract_tokens({})
        self.assertIsNone(pt)
        self.assertIsNone(ct)

    def test_non_dict_output_returns_nones(self):
        pt, ct, _, _ = _extract_tokens("string")
        self.assertIsNone(pt)
        self.assertIsNone(ct)

    def test_alternative_key_input_tokens(self):
        reply = MagicMock()
        reply.meta = {"usage": {"input_tokens": 80, "output_tokens": 20}}
        output = {"replies": [reply]}
        pt, ct, _, _ = _extract_tokens(output)
        self.assertEqual(pt, 80)
        self.assertEqual(ct, 20)


# ---------------------------------------------------------------------------
# _flatten_input
# ---------------------------------------------------------------------------
class TestFlattenInput(unittest.TestCase):
    def test_plain_string(self):
        self.assertEqual(_flatten_input("hello"), "hello")

    def test_dict_with_text_value(self):
        self.assertEqual(_flatten_input({"query": "search term"}), "search term")

    def test_object_with_text_attr(self):
        obj = MagicMock()
        obj.text = "my question"
        self.assertEqual(_flatten_input(obj), "my question")

    def test_nested_dict(self):
        result = _flatten_input({"messages": [{"content": "the content"}]})
        self.assertEqual(result, "the content")

    def test_empty_dict_returns_empty(self):
        self.assertEqual(_flatten_input({}), "")


# ---------------------------------------------------------------------------
# _extract_tool_calls
# ---------------------------------------------------------------------------
class TestExtractToolCalls(unittest.TestCase):
    def _make_tool_call_result(self, tool_name, args, error=False):
        origin = MagicMock()
        origin.tool_name = tool_name
        origin.arguments = args
        tcr = MagicMock()
        tcr.origin = origin
        tcr.error = error
        return tcr

    def test_extracts_tool_from_output_tool_messages(self):
        msg = MagicMock()
        msg.tool_call_results = [self._make_tool_call_result("web_search", {"q": "test"})]
        output = {"tool_messages": [msg]}
        calls = _extract_tool_calls({}, output, None)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "web_search")

    def test_failed_tool_has_ok_false(self):
        msg = MagicMock()
        msg.tool_call_results = [self._make_tool_call_result("fetch", {}, error=True)]
        output = {"tool_messages": [msg]}
        calls = _extract_tool_calls({}, output, None)
        self.assertFalse(calls[0][2])

    def test_success_tool_has_ok_true(self):
        msg = MagicMock()
        msg.tool_call_results = [self._make_tool_call_result("calculator", {}, error=False)]
        output = {"tool_messages": [msg]}
        calls = _extract_tool_calls({}, output, None)
        self.assertTrue(calls[0][2])

    def test_no_tool_messages_fallback_to_input(self):
        tc = MagicMock()
        tc.tool_name = "read_file"
        tc.arguments = {"path": "/etc/hosts"}
        msg = MagicMock()
        msg.tool_calls = [tc]
        inp = {"messages": [msg]}
        calls = _extract_tool_calls(inp, {}, None)
        self.assertEqual(calls[0][0], "read_file")

    def test_exception_exc_type_marks_fallback_as_failed(self):
        tc = MagicMock()
        tc.tool_name = "t"
        tc.arguments = {}
        msg = MagicMock()
        msg.tool_calls = [tc]
        inp = {"messages": [msg]}
        calls = _extract_tool_calls(inp, {}, RuntimeError)
        self.assertFalse(calls[0][2])

    def test_empty_input_and_output_returns_empty(self):
        calls = _extract_tool_calls({}, {}, None)
        self.assertEqual(calls, [])


# ---------------------------------------------------------------------------
# Tracer span lifecycle
# ---------------------------------------------------------------------------
class TestTracerPipelineSpan(unittest.TestCase):
    def test_pipeline_run_emits_run_started(self):
        tracer, emitted = _make_tracer()
        with tracer.trace("haystack.pipeline.run", tags={}):
            pass
        started = [e for e in emitted if e.event_type == EventType.RUN_STARTED]
        self.assertEqual(len(started), 1)

    def test_pipeline_run_emits_run_completed_on_success(self):
        tracer, emitted = _make_tracer()
        with tracer.trace("haystack.pipeline.run", tags={}):
            pass
        completed = [e for e in emitted if e.event_type == EventType.RUN_COMPLETED]
        self.assertEqual(len(completed), 1)

    def test_pipeline_run_emits_run_errored_on_exception(self):
        tracer, emitted = _make_tracer()
        with self.assertRaises(RuntimeError):
            with tracer.trace("haystack.pipeline.run", tags={}):
                raise RuntimeError("pipeline failed")
        errored = [e for e in emitted if e.event_type == EventType.RUN_ERRORED]
        self.assertEqual(len(errored), 1)

    def test_run_errored_has_error_hash(self):
        tracer, emitted = _make_tracer()
        with self.assertRaises(ValueError):
            with tracer.trace("haystack.pipeline.run", tags={}):
                raise ValueError("bad input")
        errored = [e for e in emitted if e.event_type == EventType.RUN_ERRORED][0]
        self.assertIn("error_hash", errored.payload)

    def test_input_hash_in_run_started(self):
        tracer, emitted = _make_tracer()
        with tracer.trace(
            "haystack.pipeline.run", tags={"haystack.pipeline.input_data": {"query": "hello"}}
        ):
            pass
        started = [e for e in emitted if e.event_type == EventType.RUN_STARTED][0]
        self.assertIn("input_hash", started.payload)

    def test_nested_pipeline_reuses_outer_run(self):
        """Nested haystack.pipeline.run must not emit a second RUN_STARTED."""
        tracer, emitted = _make_tracer()
        with tracer.trace("haystack.pipeline.run", tags={}):
            # No parent_span arg → ContextVar picks up the outer span automatically
            with tracer.trace("haystack.pipeline.run", tags={}):
                pass

        started = [e for e in emitted if e.event_type == EventType.RUN_STARTED]
        completed = [e for e in emitted if e.event_type == EventType.RUN_COMPLETED]
        self.assertEqual(len(started), 1)
        self.assertEqual(len(completed), 1)

    def test_two_independent_pipelines_emit_two_runs(self):
        tracer, emitted = _make_tracer()
        with tracer.trace("haystack.pipeline.run", tags={}):
            pass
        with tracer.trace("haystack.pipeline.run", tags={}):
            pass
        run_ids = {e.run_id for e in emitted if e.event_type == EventType.RUN_STARTED}
        self.assertEqual(len(run_ids), 2)


class TestTracerLLMComponentSpan(unittest.TestCase):
    def _run_llm_span(self, tracer, emitted, output=None):
        with tracer.trace("haystack.pipeline.run", tags={}):
            comp = tracer.trace(
                "haystack.component.run",
                tags={
                    "haystack.component.type": "OpenAIChatGenerator",
                    "haystack.component.name": "llm",
                },
            )
            comp.__enter__()
            if output is not None:
                comp.set_content_tag("haystack.component.output", output)
            comp.__exit__(None, None, None)

        called = [e for e in emitted if e.event_type == EventType.LLM_CALLED]
        responded = [e for e in emitted if e.event_type == EventType.LLM_RESPONDED]
        return called, responded

    def test_llm_component_emits_llm_called_and_responded(self):
        tracer, emitted = _make_tracer()
        called, responded = self._run_llm_span(tracer, emitted)
        self.assertEqual(len(called), 1)
        self.assertEqual(len(responded), 1)

    def test_llm_called_has_no_prompt_tokens(self):
        """Haystack puts token counts in llm.responded — llm.called must not have them."""
        tracer, emitted = _make_tracer()
        called, _ = self._run_llm_span(tracer, emitted)
        self.assertNotIn("prompt_tokens", called[0].payload)

    def test_tokens_from_output_in_llm_responded(self):
        reply = MagicMock()
        reply.meta = {"usage": {"prompt_tokens": 120, "completion_tokens": 45}}
        reply.text = "Paris"
        tracer, emitted = _make_tracer()
        _, responded = self._run_llm_span(tracer, emitted, output={"replies": [reply]})
        self.assertEqual(responded[0].payload.get("prompt_tokens"), 120)
        self.assertEqual(responded[0].payload.get("completion_tokens"), 45)

    def test_no_tokens_when_output_missing(self):
        """Without output, prompt_tokens and completion_tokens are absent from responded."""
        tracer, emitted = _make_tracer()
        _, responded = self._run_llm_span(tracer, emitted)
        self.assertNotIn("prompt_tokens", responded[0].payload)
        self.assertNotIn("completion_tokens", responded[0].payload)

    def test_llm_error_emits_error_finish_reason(self):
        tracer, emitted = _make_tracer()
        with self.assertRaises(RuntimeError):
            with tracer.trace("haystack.pipeline.run", tags={}):
                comp = tracer.trace(
                    "haystack.component.run",
                    tags={
                        "haystack.component.type": "OpenAIChatGenerator",
                        "haystack.component.name": "llm",
                    },
                )
                comp.__enter__()
                comp.__exit__(RuntimeError, RuntimeError("timeout"), None)
                raise RuntimeError("timeout")

        responded = [e for e in emitted if e.event_type == EventType.LLM_RESPONDED][0]
        self.assertEqual(responded.payload.get("finish_reason"), "error")

    def test_provider_tag_overrides_output_tokens(self):
        """Provider child span (haystack.openai.*) propagates token tags to component parent."""
        tracer, emitted = _make_tracer()
        with tracer.trace("haystack.pipeline.run", tags={}):
            comp = tracer.trace(
                "haystack.component.run",
                tags={
                    "haystack.component.type": "OpenAIChatGenerator",
                    "haystack.component.name": "llm",
                },
            )
            comp.__enter__()
            # Simulate provider child span: must use parent_span= so propagation works
            provider = tracer.trace("haystack.openai.chat_completion", tags={}, parent_span=comp)
            provider.__enter__()
            provider.set_tag("response.usage.prompt_tokens", 999)
            provider.set_tag("response.usage.completion_tokens", 77)
            provider.__exit__(None, None, None)
            comp.__exit__(None, None, None)

        responded = [e for e in emitted if e.event_type == EventType.LLM_RESPONDED][0]
        self.assertEqual(responded.payload.get("prompt_tokens"), 999)
        self.assertEqual(responded.payload.get("completion_tokens"), 77)


class TestTracerRetrievalSpan(unittest.TestCase):
    def _run_retrieval(self, tracer, emitted, docs=None, raise_exc=None):
        with tracer.trace("haystack.pipeline.run", tags={}):
            comp = tracer.trace(
                "haystack.component.run",
                tags={
                    "haystack.component.type": "InMemoryBM25Retriever",
                    "haystack.component.name": "retriever",
                },
            )
            comp.__enter__()
            if docs is not None:
                comp.set_content_tag("haystack.component.output", {"documents": docs})
            if raise_exc:
                comp.__exit__(type(raise_exc), raise_exc, None)
            else:
                comp.__exit__(None, None, None)

    def test_retrieval_emits_called_and_responded(self):
        tracer, emitted = _make_tracer()
        self._run_retrieval(tracer, emitted, docs=[])
        types_seen = [e.event_type for e in emitted]
        self.assertIn(EventType.RETRIEVAL_CALLED, types_seen)
        self.assertIn(EventType.RETRIEVAL_RESPONDED, types_seen)

    def test_result_count_from_docs(self):
        tracer, emitted = _make_tracer()
        docs = [MagicMock(score=0.8), MagicMock(score=0.6), MagicMock(score=0.4)]
        self._run_retrieval(tracer, emitted, docs=docs)
        responded = [e for e in emitted if e.event_type == EventType.RETRIEVAL_RESPONDED][0]
        self.assertEqual(responded.payload.get("result_count"), 3)

    def test_empty_retrieval_result_count_zero(self):
        tracer, emitted = _make_tracer()
        self._run_retrieval(tracer, emitted, docs=[])
        responded = [e for e in emitted if e.event_type == EventType.RETRIEVAL_RESPONDED][0]
        self.assertEqual(responded.payload.get("result_count"), 0)

    def test_top_score_is_max(self):
        tracer, emitted = _make_tracer()
        docs = [MagicMock(score=0.4), MagicMock(score=0.9), MagicMock(score=0.7)]
        self._run_retrieval(tracer, emitted, docs=docs)
        responded = [e for e in emitted if e.event_type == EventType.RETRIEVAL_RESPONDED][0]
        self.assertAlmostEqual(responded.payload.get("top_score"), 0.9)

    def test_retrieval_error_emits_zero_count(self):
        tracer, emitted = _make_tracer()
        self._run_retrieval(tracer, emitted, raise_exc=RuntimeError("retrieval failed"))
        responded = [e for e in emitted if e.event_type == EventType.RETRIEVAL_RESPONDED][0]
        self.assertEqual(responded.payload.get("result_count"), 0)


class TestTracerToolSpan(unittest.TestCase):
    def _make_tool_msg(self, tool_name, args, error=False):
        origin = MagicMock()
        origin.tool_name = tool_name
        origin.arguments = args
        tcr = MagicMock()
        tcr.origin = origin
        tcr.error = error
        msg = MagicMock()
        msg.tool_call_results = [tcr]
        return msg

    def _run_tool(self, tracer, emitted, output=None):
        with tracer.trace("haystack.pipeline.run", tags={}):
            comp = tracer.trace(
                "haystack.component.run",
                tags={
                    "haystack.component.type": "ToolInvoker",
                    "haystack.component.name": "tool_invoker",
                },
            )
            comp.__enter__()
            if output is not None:
                comp.set_content_tag("haystack.component.output", output)
            comp.__exit__(None, None, None)

    def test_tool_success_emits_called_and_responded(self):
        tracer, emitted = _make_tracer()
        msg = self._make_tool_msg("web_search", {"q": "test"}, error=False)
        self._run_tool(tracer, emitted, output={"tool_messages": [msg]})
        types_seen = [e.event_type for e in emitted]
        self.assertIn(EventType.TOOL_CALLED, types_seen)
        self.assertIn(EventType.TOOL_RESPONDED, types_seen)

    def test_tool_success_marks_success_true(self):
        tracer, emitted = _make_tracer()
        msg = self._make_tool_msg("calculator", {}, error=False)
        self._run_tool(tracer, emitted, output={"tool_messages": [msg]})
        responded = [e for e in emitted if e.event_type == EventType.TOOL_RESPONDED][0]
        self.assertTrue(responded.payload.get("success"))

    def test_tool_failure_marks_success_false(self):
        tracer, emitted = _make_tracer()
        msg = self._make_tool_msg("fetch", {}, error=True)
        self._run_tool(tracer, emitted, output={"tool_messages": [msg]})
        responded = [e for e in emitted if e.event_type == EventType.TOOL_RESPONDED][0]
        self.assertFalse(responded.payload.get("success"))

    def test_fallback_comp_name_when_no_tool_messages(self):
        """Empty output falls back to component name as tool name."""
        tracer, emitted = _make_tracer()
        self._run_tool(tracer, emitted, output={})
        called = [e for e in emitted if e.event_type == EventType.TOOL_CALLED]
        self.assertEqual(len(called), 1)
        self.assertEqual(called[0].payload.get("tool_name"), "tool_invoker")


class TestTracerCurrentSpan(unittest.TestCase):
    def test_current_span_none_outside_trace(self):
        tracer, _ = _make_tracer()
        self.assertIsNone(tracer.current_span())

    def test_current_span_returns_active_span(self):
        tracer, _ = _make_tracer()
        with tracer.trace("haystack.pipeline.run", tags={}) as span:
            self.assertIs(tracer.current_span(), span)
        self.assertIsNone(tracer.current_span())


if __name__ == "__main__":
    unittest.main(verbosity=2)
