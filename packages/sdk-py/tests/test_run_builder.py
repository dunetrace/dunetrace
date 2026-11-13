"""
The canonical run builder — one implementation, shared by every service.

Lives with the SDK because it depends on nothing but dunetrace.models and is the
inverse of what the SDK emits. Two hand-copied forks previously drifted; see
dunetrace/run_builder.py's docstring.

Run: cd packages/sdk-py && python -m unittest tests.test_run_builder -v
"""

from __future__ import annotations

import time
import unittest

from dunetrace.run_builder import build_run_state


def _evt(event_type, step, payload):
    return {
        "event_type": event_type,
        "run_id": "run-1",
        "agent_id": "agent-1",
        "agent_version": "v1",
        "step_index": step,
        "timestamp": time.time(),
        "payload": payload,
    }


class TestReconstructionFidelity(unittest.TestCase):
    """Every field here was dropped by one of the two forks at some point."""

    def setUp(self):
        self.state = build_run_state(
            [
                _evt(
                    "run.started",
                    0,
                    {"input_text": "hi", "system_prompt": "be careful", "tools": ["s"]},
                ),
                _evt("llm.called", 1, {"model": "gpt-4o", "prompt_tokens": 10, "call_id": 0}),
                _evt(
                    "llm.responded",
                    2,
                    {
                        "call_id": 0,
                        "completion_tokens": 5,
                        "finish_reason": "stop",
                        "output": "42",
                        "output_length": 2,
                    },
                ),
                _evt("tool.called", 3, {"tool_name": "s", "args": "q"}),
                _evt("tool.responded", 4, {"tool_name": "s", "success": True, "output": "body"}),
                _evt("memory.written", 5, {"key": "k", "value": "v", "source": "retrieval"}),
                _evt("run.completed", 6, {"exit_reason": "final_answer"}),
            ]
        )

    def test_system_prompt(self):
        self.assertEqual(self.state.system_prompt, "be careful")

    def test_llm_output_text(self):
        self.assertEqual(self.state.llm_calls[0].output_text, "42")

    def test_tool_output(self):
        self.assertEqual(self.state.tool_calls[0].output, "body")

    def test_memory_events(self):
        self.assertEqual(len(self.state.memory_events), 1)

    def test_exit_reason(self):
        self.assertEqual(self.state.exit_reason, "final_answer")

    def test_empty_events_rejected(self):
        with self.assertRaises(ValueError):
            build_run_state([])


if __name__ == "__main__":
    unittest.main()
