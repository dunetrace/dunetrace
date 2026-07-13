from __future__ import annotations

import unittest

from semantic_svc.evaluators.run_extract import build_evaluation_input


def evt(event_type: str, payload: dict | None = None) -> dict:
    return {"event_type": event_type, "payload": payload or {}}


class TestBuildEvaluationInput(unittest.TestCase):
    def test_none_when_no_input_text(self):
        events = [evt("llm.responded", {"output": "hi"})]
        self.assertIsNone(build_evaluation_input(events))

    def test_none_when_no_output_text(self):
        events = [evt("run.started", {"input_text": "hello"})]
        self.assertIsNone(build_evaluation_input(events))

    def test_basic_input_and_output(self):
        events = [
            evt("run.started", {"input_text": "What is 2+2?"}),
            evt("llm.responded", {"output": "4"}),
        ]
        result = build_evaluation_input(events)
        self.assertEqual(result.input_text, "What is 2+2?")
        self.assertEqual(result.actual_output, "4")

    def test_last_llm_output_wins(self):
        events = [
            evt("run.started", {"input_text": "q"}),
            evt("llm.responded", {"output": "first"}),
            evt("llm.responded", {"output": "final"}),
        ]
        result = build_evaluation_input(events)
        self.assertEqual(result.actual_output, "final")

    def test_tool_output_used_as_fallback_when_no_llm_output(self):
        events = [
            evt("run.started", {"input_text": "q"}),
            evt("tool.called", {"tool_name": "search"}),
            evt("tool.responded", {"tool_name": "search", "output": "search result text"}),
        ]
        result = build_evaluation_input(events)
        self.assertEqual(result.actual_output, "search result text")

    def test_retrieval_content_collected_into_context(self):
        events = [
            evt("run.started", {"input_text": "q"}),
            evt("retrieval.responded", {"content": "doc 1"}),
            evt("retrieval.responded", {"content": "doc 2"}),
            evt("llm.responded", {"output": "answer"}),
        ]
        result = build_evaluation_input(events)
        self.assertEqual(result.context, ["doc 1", "doc 2"])

    def test_no_retrieval_events_gives_empty_context(self):
        events = [
            evt("run.started", {"input_text": "q"}),
            evt("llm.responded", {"output": "answer"}),
        ]
        result = build_evaluation_input(events)
        self.assertEqual(result.context, [])

    def test_tool_calls_backfilled_with_output(self):
        events = [
            evt("run.started", {"input_text": "q"}),
            evt("tool.called", {"tool_name": "search", "args": {"query": "x"}}),
            evt("tool.responded", {"tool_name": "search", "output": "result"}),
            evt("llm.responded", {"output": "answer"}),
        ]
        result = build_evaluation_input(events)
        self.assertEqual(len(result.tools_called), 1)
        tc = result.tools_called[0]
        self.assertEqual(tc.name, "search")
        self.assertEqual(tc.input_parameters, {"query": "x"})
        self.assertEqual(tc.output, "result")

    def test_non_dict_args_wrapped_as_raw(self):
        events = [
            evt("run.started", {"input_text": "q"}),
            evt("tool.called", {"tool_name": "search", "args": "raw string args"}),
            evt("llm.responded", {"output": "answer"}),
        ]
        result = build_evaluation_input(events)
        self.assertEqual(result.tools_called[0].input_parameters, {"raw": "raw string args"})

    def test_sequential_calls_to_same_tool_each_matched_to_their_own_response(self):
        # call/respond pairs interleaved (the common case) — each response
        # matches the immediately preceding unmatched call of that name.
        events = [
            evt("run.started", {"input_text": "q"}),
            evt("tool.called", {"tool_name": "search", "args": {"query": "a"}}),
            evt("tool.responded", {"tool_name": "search", "output": "result-a"}),
            evt("tool.called", {"tool_name": "search", "args": {"query": "b"}}),
            evt("tool.responded", {"tool_name": "search", "output": "result-b"}),
            evt("llm.responded", {"output": "answer"}),
        ]
        result = build_evaluation_input(events)
        outputs = [tc.output for tc in result.tools_called]
        self.assertEqual(outputs, ["result-a", "result-b"])


if __name__ == "__main__":
    unittest.main()
