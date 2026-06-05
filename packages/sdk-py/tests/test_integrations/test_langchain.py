"""
Tests for DunetraceCallbackHandler.

These tests mock LangChain's callback interface so they run without
installing langchain.
"""

import sys
import time
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Stub out langchain so we can import the handler without the real package
# ---------------------------------------------------------------------------
def _stub_langchain() -> None:
    if "langchain" in sys.modules:
        return

    lc = types.ModuleType("langchain")
    lc_cb = types.ModuleType("langchain.callbacks")
    lc_cb_base = types.ModuleType("langchain.callbacks.base")

    class BaseCallbackHandler:
        def __init__(self):
            pass

    lc_cb_base.BaseCallbackHandler = BaseCallbackHandler
    lc.callbacks = lc_cb
    lc_cb.base = lc_cb_base

    sys.modules["langchain"] = lc
    sys.modules["langchain.callbacks"] = lc_cb
    sys.modules["langchain.callbacks.base"] = lc_cb_base

    # Patch the availability flag so handler.__init__ doesn't raise
    import dunetrace.integrations.langchain as _lc_mod

    _lc_mod._LANGCHAIN_AVAILABLE = True


_stub_langchain()

from dunetrace.integrations.langchain import DunetraceCallbackHandler
from dunetrace.models import EventType


def _make_handler(tools=None, model="gpt-4o"):
    client = MagicMock()
    emitted = []
    client._emit.side_effect = emitted.append
    handler = DunetraceCallbackHandler(
        client, agent_id="test-agent", model=model, tools=tools or ["search"]
    )
    return handler, emitted


def _types(emitted):
    return [e.event_type for e in emitted]


# ── Run lifecycle ──────────────────────────────────────────────────────────────


class TestRunLifecycle(unittest.TestCase):
    def test_chain_start_emits_run_started(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "hello"}, run_id="lc-1")
        self.assertEqual(emitted[0].event_type, EventType.RUN_STARTED)

    def test_chain_end_emits_run_completed(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "hello"}, run_id="lc-1")
        handler.on_chain_end({}, run_id="lc-1")
        self.assertIn(EventType.RUN_COMPLETED, _types(emitted))

    def test_chain_error_emits_run_errored(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "hello"}, run_id="lc-1")
        handler.on_chain_error(ValueError("boom"), run_id="lc-1")
        self.assertIn(EventType.RUN_ERRORED, _types(emitted))

    def test_run_started_payload_has_tools_and_model(self):
        handler, emitted = _make_handler(tools=["search", "calc"], model="gpt-4o")
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        payload = emitted[0].payload
        self.assertEqual(payload["tools"], ["search", "calc"])
        self.assertEqual(payload["model"], "gpt-4o")

    def test_run_started_hashes_input(self):
        handler, emitted = _make_handler()
        secret = "my secret query"
        handler.on_chain_start({}, {"input": secret}, run_id="lc-1")
        import json

        self.assertNotIn(secret, json.dumps(emitted[0].payload))
        self.assertIn("input_hash", emitted[0].payload)

    def test_run_completed_payload_has_exit_reason(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        handler.on_chain_end({}, run_id="lc-1")
        completed = next(e for e in emitted if e.event_type == EventType.RUN_COMPLETED)
        self.assertEqual(completed.payload["exit_reason"], "final_answer")

    def test_run_errored_payload_has_error_type(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        handler.on_chain_error(RuntimeError("fail"), run_id="lc-1")
        errored = next(e for e in emitted if e.event_type == EventType.RUN_ERRORED)
        self.assertEqual(errored.payload["error_type"], "RuntimeError")

    def test_run_errored_does_not_include_raw_error_message(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        secret_msg = "secret internal error detail"
        handler.on_chain_error(RuntimeError(secret_msg), run_id="lc-1")
        import json

        self.assertNotIn(secret_msg, json.dumps([e.payload for e in emitted]))

    def test_sub_chain_start_is_ignored(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "root"}, run_id="lc-root")
        count_before = len(emitted)
        handler.on_chain_start(
            {}, {"input": "sub"}, run_id="lc-sub", parent_run_id="lc-root"
        )
        self.assertEqual(len(emitted), count_before)

    def test_handler_resets_after_completion(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "run1"}, run_id="lc-1")
        handler.on_chain_end({}, run_id="lc-1")
        handler.on_chain_start({}, {"input": "run2"}, run_id="lc-2")
        handler.on_chain_end({}, run_id="lc-2")
        started = [e for e in emitted if e.event_type == EventType.RUN_STARTED]
        self.assertEqual(len(started), 2)

    def test_handler_resets_after_error(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "run1"}, run_id="lc-1")
        handler.on_chain_error(ValueError("oops"), run_id="lc-1")
        handler.on_chain_start({}, {"input": "run2"}, run_id="lc-2")
        handler.on_chain_end({}, run_id="lc-2")
        started = [e for e in emitted if e.event_type == EventType.RUN_STARTED]
        self.assertEqual(len(started), 2)

    def test_concurrent_runs_have_separate_run_ids(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "a"}, run_id="lc-A")
        handler.on_chain_start({}, {"input": "b"}, run_id="lc-B")
        started = [e for e in emitted if e.event_type == EventType.RUN_STARTED]
        run_ids = {e.run_id for e in started}
        self.assertEqual(len(run_ids), 2)

    def test_input_extracted_from_messages_format(self):
        """LangGraph passes input as {'messages': [('human', text)]}"""
        handler, emitted = _make_handler()
        handler.on_chain_start(
            {}, {"messages": [("human", "secret message")]}, run_id="lc-1"
        )
        import json

        self.assertNotIn("secret message", json.dumps(emitted[0].payload))
        self.assertIn("input_hash", emitted[0].payload)


# ── LLM callbacks ─────────────────────────────────────────────────────────────


class TestLlmCallbacks(unittest.TestCase):
    def _run_with_llm(self, finish_reason="stop", output_text="Answer"):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")

        # on_chat_model_start (chat models)
        handler.on_chat_model_start(
            {"kwargs": {"model_name": "gpt-4o"}},
            [[]],
            run_id="lc-llm",
            parent_run_id="lc-1",
        )

        gen = MagicMock()
        gen.text = output_text
        gen.generation_info = {"finish_reason": finish_reason}
        gen.message = None
        response = MagicMock()
        response.generations = [[gen]]
        response.llm_output = {
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 50}
        }
        handler.on_llm_end(response, run_id="lc-llm", parent_run_id="lc-1")

        return handler, emitted

    def test_chat_model_start_emits_llm_called(self):
        _, emitted = self._run_with_llm()
        self.assertIn(EventType.LLM_CALLED, _types(emitted))

    def test_llm_end_emits_llm_responded(self):
        _, emitted = self._run_with_llm()
        self.assertIn(EventType.LLM_RESPONDED, _types(emitted))

    def test_llm_responded_has_finish_reason(self):
        _, emitted = self._run_with_llm(finish_reason="length")
        responded = next(e for e in emitted if e.event_type == EventType.LLM_RESPONDED)
        self.assertEqual(responded.payload["finish_reason"], "length")

    def test_llm_responded_has_output_length(self):
        _, emitted = self._run_with_llm(output_text="Hello world")
        responded = next(e for e in emitted if e.event_type == EventType.LLM_RESPONDED)
        self.assertEqual(responded.payload["output_length"], len("Hello world"))

    def test_llm_responded_has_prompt_tokens(self):
        _, emitted = self._run_with_llm()
        responded = next(e for e in emitted if e.event_type == EventType.LLM_RESPONDED)
        self.assertEqual(responded.payload["prompt_tokens"], 100)

    def test_llm_responded_has_latency_ms(self):
        _, emitted = self._run_with_llm()
        responded = next(e for e in emitted if e.event_type == EventType.LLM_RESPONDED)
        self.assertIn("latency_ms", responded.payload)
        self.assertGreaterEqual(responded.payload["latency_ms"], 0)

    def test_llm_responded_does_not_include_raw_output(self):
        _, emitted = self._run_with_llm(output_text="secret output")
        import json

        self.assertNotIn("secret output", json.dumps([e.payload for e in emitted]))

    def test_llm_error_emits_responded_with_error_finish_reason(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        handler.on_chat_model_start({}, [[]], run_id="lc-llm", parent_run_id="lc-1")
        handler.on_llm_error(
            RuntimeError("timeout"), run_id="lc-llm", parent_run_id="lc-1"
        )
        responded = next(
            (e for e in emitted if e.event_type == EventType.LLM_RESPONDED), None
        )
        self.assertIsNotNone(responded)
        self.assertEqual(responded.payload["finish_reason"], "error")

    def test_on_llm_start_emits_llm_called(self):
        """on_llm_start fires for text-completion LLMs, not chat models."""
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        handler.on_llm_start(
            {"kwargs": {"model_name": "text-davinci-003"}},
            ["prompt"],
            run_id="lc-llm",
            parent_run_id="lc-1",
        )
        self.assertIn(EventType.LLM_CALLED, _types(emitted))


# ── Tool callbacks ─────────────────────────────────────────────────────────────


class TestToolCallbacks(unittest.TestCase):
    def test_tool_start_emits_tool_called(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        handler.on_tool_start(
            {"name": "web_search"}, "query text", run_id="lc-tool", parent_run_id="lc-1"
        )
        self.assertIn(EventType.TOOL_CALLED, _types(emitted))

    def test_tool_called_payload_has_tool_name(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        handler.on_tool_start(
            {"name": "calculator"}, "1+1", run_id="lc-tool", parent_run_id="lc-1"
        )
        called = next(e for e in emitted if e.event_type == EventType.TOOL_CALLED)
        self.assertEqual(called.payload["tool_name"], "calculator")

    def test_tool_called_hashes_args(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        handler.on_tool_start(
            {"name": "search"}, "secret args", run_id="lc-tool", parent_run_id="lc-1"
        )
        import json

        self.assertNotIn("secret args", json.dumps([e.payload for e in emitted]))

    def test_tool_end_emits_tool_responded_success(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        handler.on_tool_start(
            {"name": "search"}, "q", run_id="lc-tool", parent_run_id="lc-1"
        )
        handler.on_tool_end("result", run_id="lc-tool", parent_run_id="lc-1")
        responded = next(
            (e for e in emitted if e.event_type == EventType.TOOL_RESPONDED), None
        )
        self.assertIsNotNone(responded)
        self.assertTrue(responded.payload["success"])

    def test_tool_error_emits_tool_responded_failure(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        handler.on_tool_start(
            {"name": "search"}, "q", run_id="lc-tool", parent_run_id="lc-1"
        )
        handler.on_tool_error(
            RuntimeError("timeout"), run_id="lc-tool", parent_run_id="lc-1"
        )
        responded = next(
            (e for e in emitted if e.event_type == EventType.TOOL_RESPONDED), None
        )
        self.assertIsNotNone(responded)
        self.assertFalse(responded.payload["success"])

    def test_tool_end_with_error_kwarg_emits_failure(self):
        """handle_tool_error=True / LangGraph ToolNode calls on_tool_end with error= kwarg."""
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        handler.on_tool_start(
            {"name": "search"}, "q", run_id="lc-tool", parent_run_id="lc-1"
        )
        handler.on_tool_end(
            "Error: timeout",
            run_id="lc-tool",
            parent_run_id="lc-1",
            name="search",
            error=RuntimeError("timeout"),
        )
        responded = next(
            (e for e in emitted if e.event_type == EventType.TOOL_RESPONDED), None
        )
        self.assertIsNotNone(responded)
        self.assertFalse(responded.payload["success"])
        self.assertIn("error_hash", responded.payload)

    def test_tool_end_without_error_kwarg_emits_success(self):
        """Normal on_tool_end (no error kwarg) still reports success=True."""
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        handler.on_tool_start(
            {"name": "search"}, "q", run_id="lc-tool", parent_run_id="lc-1"
        )
        handler.on_tool_end(
            "result text", run_id="lc-tool", parent_run_id="lc-1", name="search"
        )
        responded = next(
            (e for e in emitted if e.event_type == EventType.TOOL_RESPONDED), None
        )
        self.assertIsNotNone(responded)
        self.assertTrue(responded.payload["success"])

    def test_tool_error_does_not_include_raw_error(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        handler.on_tool_start(
            {"name": "search"}, "q", run_id="lc-tool", parent_run_id="lc-1"
        )
        handler.on_tool_error(
            RuntimeError("secret error details"), run_id="lc-tool", parent_run_id="lc-1"
        )
        import json

        self.assertNotIn(
            "secret error details", json.dumps([e.payload for e in emitted])
        )

    def test_on_agent_action_emits_tool_called(self):
        """AgentExecutor path (LangChain < 1.x)."""
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        action = MagicMock()
        action.tool = "web_search"
        action.tool_input = {"query": "France"}
        handler.on_agent_action(action, run_id="lc-1")
        self.assertIn(EventType.TOOL_CALLED, _types(emitted))

    def test_completed_run_has_tool_call_count(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        handler.on_tool_start({"name": "s"}, "q1", run_id="t1", parent_run_id="lc-1")
        handler.on_tool_end("r1", run_id="t1", parent_run_id="lc-1")
        handler.on_tool_start({"name": "s"}, "q2", run_id="t2", parent_run_id="lc-1")
        handler.on_tool_end("r2", run_id="t2", parent_run_id="lc-1")
        handler.on_chain_end({}, run_id="lc-1")
        completed = next(e for e in emitted if e.event_type == EventType.RUN_COMPLETED)
        self.assertEqual(completed.payload["tool_call_count"], 2)


# ── Retrieval callbacks ────────────────────────────────────────────────────────


class TestRetrievalCallbacks(unittest.TestCase):
    def test_retriever_start_emits_retrieval_called(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        handler.on_retriever_start(
            {"name": "my-index"}, "search query", run_id="lc-ret", parent_run_id="lc-1"
        )
        self.assertIn(EventType.RETRIEVAL_CALLED, _types(emitted))

    def test_retriever_end_emits_retrieval_responded(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        handler.on_retriever_start(
            {"name": "idx"}, "q", run_id="lc-ret", parent_run_id="lc-1"
        )
        doc = MagicMock()
        doc.metadata = {"score": 0.9}
        handler.on_retriever_end([doc, doc], run_id="lc-ret", parent_run_id="lc-1")
        responded = next(
            (e for e in emitted if e.event_type == EventType.RETRIEVAL_RESPONDED), None
        )
        self.assertIsNotNone(responded)
        self.assertEqual(responded.payload["result_count"], 2)
        self.assertAlmostEqual(responded.payload["top_score"], 0.9)

    def test_retriever_end_with_no_documents(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        handler.on_retriever_start(
            {"name": "idx"}, "q", run_id="lc-ret", parent_run_id="lc-1"
        )
        handler.on_retriever_end([], run_id="lc-ret", parent_run_id="lc-1")
        responded = next(
            e for e in emitted if e.event_type == EventType.RETRIEVAL_RESPONDED
        )
        self.assertEqual(responded.payload["result_count"], 0)
        self.assertIsNone(responded.payload["top_score"])

    def test_retriever_error_emits_retrieval_responded_with_zero_count(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        handler.on_retriever_start(
            {"name": "idx"}, "q", run_id="lc-ret", parent_run_id="lc-1"
        )
        handler.on_retriever_error(
            RuntimeError("index down"), run_id="lc-ret", parent_run_id="lc-1"
        )
        responded = next(
            e for e in emitted if e.event_type == EventType.RETRIEVAL_RESPONDED
        )
        self.assertEqual(responded.payload["result_count"], 0)

    def test_retriever_hashes_query(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        handler.on_retriever_start(
            {"name": "idx"}, "secret query", run_id="lc-ret", parent_run_id="lc-1"
        )
        import json

        self.assertNotIn("secret query", json.dumps([e.payload for e in emitted]))


# ── Step counter ───────────────────────────────────────────────────────────────


class TestStepCounter(unittest.TestCase):
    def test_step_increments_on_llm_call(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        handler.on_chat_model_start({}, [[]], run_id="lc-llm1", parent_run_id="lc-1")
        llm_called = next(e for e in emitted if e.event_type == EventType.LLM_CALLED)
        self.assertEqual(llm_called.step_index, 1)

    def test_step_increments_on_tool_call(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        handler.on_tool_start({"name": "s"}, "q", run_id="lc-t", parent_run_id="lc-1")
        tool_called = next(e for e in emitted if e.event_type == EventType.TOOL_CALLED)
        self.assertEqual(tool_called.step_index, 1)

    def test_steps_accumulate_across_llm_and_tools(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        handler.on_chat_model_start({}, [[]], run_id="lc-llm", parent_run_id="lc-1")
        handler.on_tool_start({"name": "s"}, "q", run_id="lc-t", parent_run_id="lc-1")
        handler.on_chat_model_start({}, [[]], run_id="lc-llm2", parent_run_id="lc-1")
        steps = [
            e.step_index
            for e in emitted
            if e.event_type in (EventType.LLM_CALLED, EventType.TOOL_CALLED)
        ]
        self.assertEqual(steps, [1, 2, 3])


# ── Privacy ────────────────────────────────────────────────────────────────────


class TestPrivacy(unittest.TestCase):
    def test_no_raw_content_in_any_event(self):
        import json

        handler, emitted = _make_handler()
        secret = "the secret user query"
        handler.on_chain_start({}, {"input": secret}, run_id="lc-1")
        handler.on_tool_start(
            {"name": "search"}, f"search: {secret}", run_id="lc-t", parent_run_id="lc-1"
        )
        for event in emitted:
            self.assertNotIn(secret, json.dumps(event.payload))

    def test_all_events_share_same_run_id(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        handler.on_tool_start({"name": "s"}, "q", run_id="lc-t", parent_run_id="lc-1")
        handler.on_chain_end({}, run_id="lc-1")
        run_ids = {e.run_id for e in emitted}
        self.assertEqual(len(run_ids), 1)

    def test_all_events_share_same_agent_id(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="lc-1")
        handler.on_chain_end({}, run_id="lc-1")
        for event in emitted:
            self.assertEqual(event.agent_id, "test-agent")


# ── Stale run pruning ──────────────────────────────────────────────────────────


class TestStalePruning(unittest.TestCase):
    def test_stale_run_pruned_on_new_start(self):
        import dunetrace.integrations.langchain as lc_mod

        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="old-run")

        # Back-date the run's start_time past the stale threshold
        with handler._lock:
            handler._runs["old-run"].start_time = (
                time.time() - lc_mod._STALE_RUN_SECS - 1
            )

        # Starting a new run should prune the stale one
        handler.on_chain_start({}, {"input": "q2"}, run_id="new-run")
        with handler._lock:
            self.assertNotIn("old-run", handler._runs)

    def test_active_run_not_pruned(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "q"}, run_id="active-run")
        handler.on_chain_start({}, {"input": "q2"}, run_id="another-run")
        with handler._lock:
            self.assertIn("active-run", handler._runs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
