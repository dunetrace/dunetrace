"""
Tests for DunetraceHermesPlugin.

These tests stub out hermes_cli so no hermes-agent package is required.
Run:
    cd packages/sdk-py
    python -m unittest tests.test_integrations.test_hermes -v
"""

from __future__ import annotations

import sys
import types
import time
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Stub hermes_cli so we can test without the hermes-agent package
# ---------------------------------------------------------------------------
def _stub_hermes_cli() -> None:
    if "hermes_cli" in sys.modules:
        return

    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli_plugins = types.ModuleType("hermes_cli.plugins")

    class _FakePluginManager:
        def __init__(self):
            self._hooks: dict = {}

    _mgr = _FakePluginManager()

    hermes_cli_plugins.get_plugin_manager = lambda: _mgr
    hermes_cli.plugins = hermes_cli_plugins

    sys.modules["hermes_cli"] = hermes_cli
    sys.modules["hermes_cli.plugins"] = hermes_cli_plugins


_stub_hermes_cli()

from dunetrace import Dunetrace  # noqa: E402
from dunetrace.integrations.hermes import DunetraceHermesPlugin  # noqa: E402
from dunetrace.models import EventType  # noqa: E402


def _make_plugin() -> tuple[DunetraceHermesPlugin, list]:
    """Return (plugin, emitted_events) where emitted_events accumulates every _emit call."""
    dt = Dunetrace.__new__(Dunetrace)
    dt._ingest_url = "http://localhost:8001/v1/ingest"
    dt._api_key = ""
    emitted = []
    dt._emit = lambda event: emitted.append(event)

    plugin = DunetraceHermesPlugin(dt, agent_id="test-agent", model="gpt-4o")
    return plugin, emitted


class TestHermesTokenNoDuplication(unittest.TestCase):
    """Regression: prompt_tokens must not appear in llm.responded to prevent double-counting."""

    def _do_llm_round_trip(
        self,
        approx_input: int,
        actual_input: int,
        output_tokens: int,
    ) -> list:
        plugin, emitted = _make_plugin()
        tid = "session:task:abc"
        plugin._pre_llm_call(turn_id=tid, user_message="hi", model="gpt-4o")
        plugin._pre_api_request(
            turn_id=tid,
            api_request_id="req-1",
            api_call_count=1,
            model="gpt-4o",
            approx_input_tokens=approx_input,
        )
        plugin._post_api_request(
            turn_id=tid,
            api_request_id="req-1",
            api_duration=0.3,
            finish_reason="stop",
            usage={"input_tokens": actual_input, "output_tokens": output_tokens},
        )
        plugin._on_session_end(turn_id=tid, completed=True)
        return emitted

    def test_llm_responded_has_no_prompt_tokens(self):
        """llm.responded payload must NOT contain prompt_tokens."""
        events = self._do_llm_round_trip(250, 260, 80)
        responded = [e for e in events if e.event_type == EventType.LLM_RESPONDED]
        self.assertEqual(len(responded), 1)
        self.assertNotIn("prompt_tokens", responded[0].payload)

    def test_llm_called_has_prompt_tokens(self):
        """prompt_tokens must appear in llm.called (approx value)."""
        events = self._do_llm_round_trip(250, 260, 80)
        called = [e for e in events if e.event_type == EventType.LLM_CALLED]
        self.assertEqual(len(called), 1)
        self.assertEqual(called[0].payload.get("prompt_tokens"), 250)

    def test_completion_tokens_in_llm_responded(self):
        """completion_tokens must appear in llm.responded."""
        events = self._do_llm_round_trip(250, 260, 80)
        responded = [e for e in events if e.event_type == EventType.LLM_RESPONDED]
        self.assertEqual(responded[0].payload.get("completion_tokens"), 80)

    def test_zero_approx_tokens_still_no_duplication(self):
        """When approx_input_tokens=0, llm.called emits no prompt_tokens; llm.responded also none."""
        plugin, emitted = _make_plugin()
        tid = "s:t:x"
        plugin._pre_llm_call(turn_id=tid, user_message="hi")
        plugin._pre_api_request(
            turn_id=tid,
            api_request_id="r",
            api_call_count=1,
            approx_input_tokens=0,
        )
        plugin._post_api_request(
            turn_id=tid,
            api_request_id="r",
            api_duration=0.1,
            finish_reason="stop",
            usage={"input_tokens": 100, "output_tokens": 30},
        )
        plugin._on_session_end(turn_id=tid, completed=True)

        responded = [e for e in emitted if e.event_type == EventType.LLM_RESPONDED]
        self.assertNotIn("prompt_tokens", responded[0].payload)


class TestHermesEventSequence(unittest.TestCase):
    """Verify the hook→event mapping produces the expected event sequence."""

    def _run_happy(self) -> list:
        plugin, emitted = _make_plugin()
        tid = "s:t:happy"
        plugin._pre_llm_call(turn_id=tid, user_message="What is 2+2?", model="gpt-4o")
        plugin._pre_api_request(
            turn_id=tid,
            api_request_id="r1",
            api_call_count=1,
            model="gpt-4o",
            approx_input_tokens=150,
        )
        plugin._pre_tool_call(
            turn_id=tid, tool_name="calculator", args={"expr": "2+2"}, tool_call_id="tc1"
        )
        plugin._post_tool_call(
            turn_id=tid,
            tool_name="calculator",
            tool_call_id="tc1",
            duration_ms=50,
            status="ok",
            result="4",
        )
        plugin._post_api_request(
            turn_id=tid,
            api_request_id="r1",
            api_duration=0.4,
            finish_reason="stop",
            usage={"input_tokens": 155, "output_tokens": 60},
        )
        plugin._on_session_end(turn_id=tid, completed=True)
        return emitted

    def test_event_types_in_order(self):
        events = self._run_happy()
        types_seen = [e.event_type for e in events]
        self.assertEqual(
            types_seen,
            [
                EventType.RUN_STARTED,
                EventType.LLM_CALLED,
                EventType.TOOL_CALLED,
                EventType.TOOL_RESPONDED,
                EventType.LLM_RESPONDED,
                EventType.RUN_COMPLETED,
            ],
        )

    def test_all_events_share_run_id(self):
        events = self._run_happy()
        run_ids = {e.run_id for e in events}
        self.assertEqual(len(run_ids), 1)

    def test_run_started_has_input_hash(self):
        events = self._run_happy()
        started = events[0]
        self.assertIn("input_hash", started.payload)
        self.assertNotEqual(started.payload["input_hash"], "")

    def test_tool_called_has_args_hash(self):
        events = self._run_happy()
        tc = [e for e in events if e.event_type == EventType.TOOL_CALLED][0]
        self.assertIn("args_hash", tc.payload)

    def test_tool_responded_success_true(self):
        events = self._run_happy()
        tr = [e for e in events if e.event_type == EventType.TOOL_RESPONDED][0]
        self.assertTrue(tr.payload["success"])

    def test_tool_failure_emits_error_hash(self):
        plugin, emitted = _make_plugin()
        tid = "s:t:fail"
        plugin._pre_llm_call(turn_id=tid)
        plugin._pre_api_request(turn_id=tid, api_request_id="r", api_call_count=1)
        plugin._pre_tool_call(turn_id=tid, tool_name="fetch", args={}, tool_call_id="tc-f")
        plugin._post_tool_call(
            turn_id=tid,
            tool_name="fetch",
            tool_call_id="tc-f",
            duration_ms=20,
            status="error",
            error_message="timeout",
        )
        plugin._on_session_end(turn_id=tid, completed=False)

        tr = [e for e in emitted if e.event_type == EventType.TOOL_RESPONDED][0]
        self.assertFalse(tr.payload["success"])
        self.assertIn("error_hash", tr.payload)

    def test_interrupted_run_emits_run_errored(self):
        plugin, emitted = _make_plugin()
        tid = "s:t:int"
        plugin._pre_llm_call(turn_id=tid)
        plugin._on_session_end(turn_id=tid, completed=False, interrupted=True)
        last = emitted[-1]
        self.assertEqual(last.event_type, EventType.RUN_ERRORED)
        self.assertTrue(last.payload.get("interrupted"))

    def test_completed_run_emits_run_completed(self):
        plugin, emitted = _make_plugin()
        tid = "s:t:ok"
        plugin._pre_llm_call(turn_id=tid)
        plugin._on_session_end(turn_id=tid, completed=True)
        last = emitted[-1]
        self.assertEqual(last.event_type, EventType.RUN_COMPLETED)


class TestHermesEdgeCases(unittest.TestCase):
    """Edge cases: empty turn_id, missing ctx, duplicate attach."""

    def test_empty_turn_id_no_crash(self):
        plugin, _ = _make_plugin()
        plugin._pre_api_request(turn_id="", api_request_id="x", api_call_count=1)
        plugin._post_tool_call(turn_id="", tool_name="t", tool_call_id="x", status="ok")
        plugin._on_session_end(turn_id="", completed=True)

    def test_hook_after_session_end_no_crash(self):
        """Hook fired after on_session_end pops the ctx — should be silently ignored."""
        plugin, emitted = _make_plugin()
        tid = "s:t:orphan"
        plugin._pre_llm_call(turn_id=tid)
        plugin._on_session_end(turn_id=tid, completed=True)
        before = len(emitted)
        plugin._pre_api_request(turn_id=tid, api_request_id="late", api_call_count=2)
        self.assertEqual(len(emitted), before)  # no new events emitted

    def test_attach_registers_six_hooks(self):
        from hermes_cli.plugins import get_plugin_manager

        mgr = get_plugin_manager()
        mgr._hooks.clear()
        plugin, _ = _make_plugin()
        plugin.attach()
        self.assertEqual(len(mgr._hooks), 6)

    def test_multiple_concurrent_turns_isolated(self):
        """Two concurrent turn_ids must not share state."""
        plugin, emitted = _make_plugin()
        tid1, tid2 = "s:1:a", "s:2:b"
        plugin._pre_llm_call(turn_id=tid1, user_message="q1")
        plugin._pre_llm_call(turn_id=tid2, user_message="q2")
        plugin._on_session_end(turn_id=tid1, completed=True)
        plugin._on_session_end(turn_id=tid2, completed=True)
        run_ids = [e.run_id for e in emitted]
        self.assertEqual(len(set(run_ids)), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
