"""
Tests for native_explain.py — builds the root-cause explain prompt from
Dunetrace's own events, no Langfuse trace needed.

No DB, no network.
"""

from __future__ import annotations

import unittest

from api_svc.native_explain import (
    _extract_system_prompt,
    _format_events,
    build_native_explain_prompt,
)


def _event(event_type: str, step_index: int = 0, **payload) -> dict:
    return {"event_type": event_type, "step_index": step_index, "payload": payload}


class TestExtractSystemPrompt(unittest.TestCase):
    def test_returns_none_today(self):
        """Known, disclosed gap — see native_explain.py's docstring on this
        function. run.started's payload doesn't carry system_prompt text."""
        events = [_event("run.started", input_text="hi", model="gpt-4o")]
        self.assertIsNone(_extract_system_prompt(events))

    def test_returns_value_if_a_future_sdk_version_adds_it(self):
        events = [_event("run.started", system_prompt="Be helpful.")]
        self.assertEqual(_extract_system_prompt(events), "Be helpful.")

    def test_no_run_started_event_returns_none(self):
        events = [_event("tool.called", tool_name="x")]
        self.assertIsNone(_extract_system_prompt(events))

    def test_empty_events_returns_none(self):
        self.assertIsNone(_extract_system_prompt([]))


class TestFormatEvents(unittest.TestCase):
    def test_empty_events_returns_placeholder(self):
        self.assertEqual(_format_events([], 0, 0), "(no events recorded)")

    def test_includes_step_and_event_type(self):
        events = [_event("tool.called", step_index=2, tool_name="web_search")]
        text = _format_events(events, 2, 2)
        self.assertIn("[step 2]", text)
        self.assertIn("tool.called", text)

    def test_includes_raw_payload_fields(self):
        events = [_event("tool.called", step_index=1, tool_name="web_search", args="{'q': 'x'}")]
        text = _format_events(events, 1, 1)
        self.assertIn("tool_name: web_search", text)
        self.assertIn("args: {'q': 'x'}", text)

    def test_truncates_long_values(self):
        events = [_event("llm.responded", step_index=1, output="x" * 1000)]
        text = _format_events(events, 1, 1, signal_steps=[])
        # non-focus limit is 150 chars
        line = next(l for l in text.splitlines() if l.strip().startswith("output:"))
        self.assertLessEqual(len(line), len("      output: ") + 150)

    def test_focus_steps_get_higher_limit(self):
        events = [_event("llm.responded", step_index=5, output="x" * 1000)]
        text = _format_events(events, 5, 5, signal_steps=[5])
        line = next(l for l in text.splitlines() if l.strip().startswith("output:"))
        self.assertGreater(len(line), len("      output: ") + 150)


class TestBuildNativeExplainPrompt(unittest.IsolatedAsyncioTestCase):
    async def test_includes_failure_type_and_confidence(self):
        signal = {"failure_type": "TOOL_LOOP", "confidence": 0.88, "evidence": {}}
        prompt = await build_native_explain_prompt(signal, [])
        self.assertIn("TOOL_LOOP", prompt)
        self.assertIn("88%", prompt)

    async def test_empty_events_shows_placeholder(self):
        signal = {"failure_type": "TOOL_LOOP", "confidence": 0.5, "evidence": {}}
        prompt = await build_native_explain_prompt(signal, [])
        self.assertIn("(no events recorded)", prompt)

    async def test_events_are_rendered_in_prompt(self):
        signal = {
            "failure_type": "TOOL_LOOP",
            "confidence": 0.9,
            "evidence": {"first_step": 1, "last_step": 3},
        }
        events = [
            _event("run.started", step_index=0, input_text="find AI news"),
            _event("tool.called", step_index=1, tool_name="web_search", args="q1"),
            _event("tool.called", step_index=2, tool_name="web_search", args="q1"),
        ]
        prompt = await build_native_explain_prompt(signal, events)
        self.assertIn("web_search", prompt)
        self.assertIn("find AI news", prompt)

    async def test_step_range_uses_evidence_fields(self):
        signal = {
            "failure_type": "TOOL_LOOP",
            "confidence": 0.9,
            "evidence": {"first_step": 2, "last_step": 10},
        }
        prompt = await build_native_explain_prompt(signal, [])
        self.assertIn("2–10", prompt)

    async def test_no_step_range_detector_falls_back_to_signal_step_index(self):
        signal = {
            "failure_type": "TOOL_AVOIDANCE",
            "confidence": 0.7,
            "evidence": {},
            "step_index": 4,
        }
        prompt = await build_native_explain_prompt(signal, [])
        self.assertIn("Failing steps: 4", prompt)

    async def test_missing_system_prompt_says_not_found(self):
        signal = {"failure_type": "TOOL_LOOP", "confidence": 0.5, "evidence": {}}
        events = [_event("run.started", input_text="hi")]
        prompt = await build_native_explain_prompt(signal, events)
        self.assertIn("System prompt: (not found in events)", prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
