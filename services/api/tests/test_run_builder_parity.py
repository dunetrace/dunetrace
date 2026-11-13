"""
The API and detector run builders must be one implementation.

They were hand-copied forks and drifted: the API copy had lost memory events,
the system prompt, tool output, retrieval content and LLM output text. Replay
runs on the API side, so five detectors that read those fields saw them as
absent and reported "resolved" for any modification, including a no-op.

This file is the API half; the detector half is
``services/detector/tests/test_run_builder_parity.py``. Each service asserts
its own module IS ``dunetrace.run_builder``, in the suite where that module is
importable, so neither assertion has to skip.

Run:
    PYTHONPATH=packages/sdk-py:services/explainer:services/api \
      python -m pytest services/api/tests/test_run_builder_parity.py -v
"""

from __future__ import annotations

import time
import unittest

import api_svc.run_builder as api_builder
import dunetrace.run_builder as canonical


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


def _events():
    return [
        _evt(
            "run.started",
            0,
            {
                "input_text": "hello",
                "system_prompt": "You are a careful assistant.",
                "tools": ["search"],
            },
        ),
        _evt("llm.called", 1, {"model": "gpt-4o", "prompt_tokens": 100, "call_id": 0}),
        _evt(
            "llm.responded",
            2,
            {
                "call_id": 0,
                "completion_tokens": 20,
                "finish_reason": "stop",
                "output": "the answer is 42",
                "output_length": 16,
            },
        ),
        _evt("tool.called", 3, {"tool_name": "search", "args": "q"}),
        _evt(
            "tool.responded",
            4,
            {
                "tool_name": "search",
                "success": True,
                "output": "result body",
            },
        ),
        _evt("memory.written", 5, {"key": "pref", "value": "dark mode", "source": "user_input"}),
        _evt("run.completed", 6, {"exit_reason": "final_answer"}),
    ]


class TestBuildersAreOneImplementation(unittest.TestCase):
    def test_api_builder_is_the_canonical_one(self):
        self.assertIs(api_builder.build_run_state, canonical.build_run_state)

    # The detector half of this guard lives in
    # services/detector/tests/test_run_builder_parity.py, where detector_svc is
    # importable. It used to sit here behind
    # `except ImportError: self.skipTest("detector_svc not on PYTHONPATH")` —
    # and detector_svc is on no CI job's PYTHONPATH for this suite, so the
    # assertion never ran anywhere while its passing sibling made the class
    # look covered. Both halves compare against dunetrace.run_builder, so
    # between them the two services are still pinned to one implementation.


class TestReplayStateIsFullFidelity(unittest.TestCase):
    """Each of these fields was dropped by the API fork, and each is read by at
    least one detector that replay claims to re-evaluate."""

    def setUp(self):
        self.state = api_builder.build_run_state(_events())

    def test_system_prompt_survives(self):
        self.assertEqual(self.state.system_prompt, "You are a careful assistant.")

    def test_llm_output_text_survives(self):
        self.assertEqual(self.state.llm_calls[0].output_text, "the answer is 42")

    def test_tool_output_survives(self):
        self.assertEqual(self.state.tool_calls[0].output, "result body")

    def test_memory_events_survive(self):
        self.assertEqual(len(self.state.memory_events), 1)
        self.assertEqual(self.state.memory_events[0].source, "user_input")


if __name__ == "__main__":
    unittest.main()
