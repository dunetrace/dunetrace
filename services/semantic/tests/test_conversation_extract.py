from __future__ import annotations

import unittest

from semantic_svc.evaluators.conversation_extract import build_conversation_evaluation_input


def evt(event_type: str, payload: dict | None = None) -> dict:
    return {"event_type": event_type, "payload": payload or {}}


def _run_events(input_text: str, output: str) -> list[dict]:
    return [
        evt("run.started", {"input_text": input_text}),
        evt("llm.responded", {"output": output}),
    ]


class TestBuildConversationEvaluationInput(unittest.TestCase):
    def test_none_when_no_runs(self):
        self.assertIsNone(build_conversation_evaluation_input([]))

    def test_none_when_no_run_has_extractable_content(self):
        runs = [("run-1", [evt("llm.responded", {"output": "hi"})])]  # no input_text
        self.assertIsNone(build_conversation_evaluation_input(runs))

    def test_builds_one_turn_per_run_in_order(self):
        runs = [
            ("run-1", _run_events("hello", "hi there")),
            ("run-2", _run_events("how are you", "doing well")),
        ]
        result = build_conversation_evaluation_input(runs)
        self.assertEqual(len(result.turns), 2)
        self.assertEqual(result.turns[0].run_id, "run-1")
        self.assertEqual(result.turns[0].input_text, "hello")
        self.assertEqual(result.turns[0].actual_output, "hi there")
        self.assertEqual(result.turns[1].run_id, "run-2")

    def test_run_without_content_is_skipped_not_fatal(self):
        runs = [
            ("run-1", _run_events("hello", "hi there")),
            ("run-2", [evt("run.started", {"input_text": "q"})]),  # no output
            ("run-3", _run_events("bye", "goodbye")),
        ]
        result = build_conversation_evaluation_input(runs)
        self.assertEqual([t.run_id for t in result.turns], ["run-1", "run-3"])


if __name__ == "__main__":
    unittest.main()
