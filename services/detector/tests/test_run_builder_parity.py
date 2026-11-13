"""
The detector's run builder must BE the canonical one, not a copy of it.

This is the detector half of the guard; the API half lives in
``services/api/tests/test_run_builder_parity.py``. Both compare against
``dunetrace.run_builder``, so between them they pin the two services to one
implementation.

It used to be a single test in the API suite that imported ``detector_svc``
inside a ``try`` and skipped on ImportError — and ``detector_svc`` is on no CI
job's PYTHONPATH for that suite, so the assertion never ran anywhere. Its
sibling passed, which made the class look covered. What it protects: the two
builders were hand-copied forks that drifted, the API copy lost memory events,
the system prompt, tool output, retrieval content and LLM output text, and
replay ran on that copy — so five detectors that read those fields reported
"resolved" for any modification, including a no-op.

Run:
    PYTHONPATH=packages/schemas-py:packages/sdk-py:services/detector \
      python -m pytest services/detector/tests/test_run_builder_parity.py -v
"""

from __future__ import annotations

import time
import unittest

import detector_svc.run_builder as detector_builder
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
    def test_detector_builder_is_the_canonical_one(self):
        self.assertIs(detector_builder.build_run_state, canonical.build_run_state)


class TestDetectorStateIsFullFidelity(unittest.TestCase):
    """Each of these fields was dropped by one of the forks, and each is read by
    at least one detector the worker runs off this state."""

    def setUp(self):
        self.state = detector_builder.build_run_state(_events())

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
