"""
Tests for Phase 1.3 voice policy actions (stop_current_tts,
escalate_to_human, inject_recovery_prompt, slow_response_pace). These follow the model_override
contract: a policy fires during _check_policies (after llm_responded / tool
hooks), Dunetrace sets a structured attribute on the run, and the agent's own
loop reads it. No network — the client's shipper is stubbed.

Run: python -m unittest tests.test_voice_policy_actions -v
"""

from __future__ import annotations

import unittest

from dunetrace.client import DunetraceClient


def _make_client(**kwargs) -> DunetraceClient:
    defaults = dict(api_key="dt_test", debug=False)
    defaults.update(kwargs)
    c = DunetraceClient(**defaults)
    c._ship = lambda batch: None
    return c


class TestStopCurrentTts(unittest.TestCase):
    def test_sets_stop_tts_flag(self):
        c = _make_client()
        c.add_policy(
            name="latency-stop",
            condition={"trigger": "llm_latency_ms", "operator": "gt", "value": 1000},
            action={"type": "stop_current_tts"},
        )
        with c.run("voice-agent") as run:
            self.assertFalse(run.stop_tts)
            run.llm_called("gpt-4o-realtime")
            run.llm_responded(latency_ms=5000, finish_reason="stop")
            self.assertTrue(run.stop_tts)
        c.shutdown(timeout=2)

    def test_not_set_when_policy_does_not_fire(self):
        c = _make_client()
        c.add_policy(
            name="latency-stop",
            condition={"trigger": "llm_latency_ms", "operator": "gt", "value": 10000},
            action={"type": "stop_current_tts"},
        )
        with c.run("voice-agent") as run:
            run.llm_called("gpt-4o-realtime")
            run.llm_responded(latency_ms=500, finish_reason="stop")
            self.assertFalse(run.stop_tts)
        c.shutdown(timeout=2)


class TestEscalateToHuman(unittest.TestCase):
    def test_sets_flag_and_reason(self):
        c = _make_client()
        c.add_policy(
            name="escalate-on-errors",
            condition={"trigger": "error_count", "operator": "gte", "value": 2},
            action={"type": "escalate_to_human", "params": {"reason": "too many failures"}},
        )
        with c.run("voice-agent") as run:
            run.tool_called("a", {})
            run.tool_responded("a", success=False)
            run.tool_called("b", {})
            run.tool_responded("b", success=False)
            self.assertTrue(run.escalate_to_human)
            self.assertEqual(run.escalation_reason, "too many failures")
        c.shutdown(timeout=2)

    def test_reason_optional(self):
        c = _make_client()
        c.add_policy(
            name="escalate",
            condition={"trigger": "error_count", "operator": "gte", "value": 1},
            action={"type": "escalate_to_human"},
        )
        with c.run("voice-agent") as run:
            run.tool_called("a", {})
            run.tool_responded("a", success=False)
            self.assertTrue(run.escalate_to_human)
            self.assertIsNone(run.escalation_reason)
        c.shutdown(timeout=2)


class TestInjectRecoveryPrompt(unittest.TestCase):
    def test_sets_recovery_prompt(self):
        c = _make_client()
        c.add_policy(
            name="recover-on-latency",
            condition={"trigger": "llm_latency_ms", "operator": "gt", "value": 1000},
            action={
                "type": "inject_recovery_prompt",
                "params": {"prompt": "Sorry for the delay, let me try again."},
            },
        )
        with c.run("voice-agent") as run:
            run.llm_called("gpt-4o-realtime")
            run.llm_responded(latency_ms=6000, finish_reason="stop")
            self.assertEqual(run.recovery_prompt, "Sorry for the delay, let me try again.")
        c.shutdown(timeout=2)

    def test_does_not_touch_prompt_additions(self):
        """Recovery prompt is a distinct attribute, not the text-agent
        prompt_additions list."""
        c = _make_client()
        c.add_policy(
            name="recover",
            condition={"trigger": "llm_latency_ms", "operator": "gt", "value": 1000},
            action={"type": "inject_recovery_prompt", "params": {"prompt": "one moment"}},
        )
        with c.run("voice-agent") as run:
            run.llm_called("gpt-4o-realtime")
            run.llm_responded(latency_ms=6000, finish_reason="stop")
            self.assertEqual(run.recovery_prompt, "one moment")
            self.assertEqual(run.prompt_additions, [])
        c.shutdown(timeout=2)


class TestSlowResponsePace(unittest.TestCase):
    def test_sets_pace_from_params(self):
        c = _make_client()
        c.add_policy(
            name="pace-on-latency",
            condition={"trigger": "llm_latency_ms", "operator": "gt", "value": 1000},
            action={"type": "slow_response_pace", "params": {"pace": "slower"}},
        )
        with c.run("voice-agent") as run:
            self.assertIsNone(run.response_pace)
            run.llm_called("gpt-4o-realtime")
            run.llm_responded(latency_ms=4000, finish_reason="stop")
            self.assertEqual(run.response_pace, "slower")
        c.shutdown(timeout=2)

    def test_pace_defaults_to_slow_when_absent(self):
        c = _make_client()
        c.add_policy(
            name="pace",
            condition={"trigger": "llm_latency_ms", "operator": "gt", "value": 1000},
            action={"type": "slow_response_pace"},
        )
        with c.run("voice-agent") as run:
            run.llm_called("gpt-4o-realtime")
            run.llm_responded(latency_ms=4000, finish_reason="stop")
            self.assertEqual(run.response_pace, "slow")
        c.shutdown(timeout=2)

    def test_not_set_when_policy_does_not_fire(self):
        c = _make_client()
        c.add_policy(
            name="pace",
            condition={"trigger": "llm_latency_ms", "operator": "gt", "value": 10000},
            action={"type": "slow_response_pace"},
        )
        with c.run("voice-agent") as run:
            run.llm_called("gpt-4o-realtime")
            run.llm_responded(latency_ms=500, finish_reason="stop")
            self.assertIsNone(run.response_pace)
        c.shutdown(timeout=2)


class TestBackwardCompat(unittest.TestCase):
    def test_defaults_are_inert(self):
        """A run with no voice policies leaves every voice attribute at its
        inert default."""
        c = _make_client()
        with c.run("agent") as run:
            run.llm_called("gpt-4o")
            run.llm_responded(latency_ms=100, finish_reason="stop")
            self.assertFalse(run.stop_tts)
            self.assertFalse(run.escalate_to_human)
            self.assertIsNone(run.escalation_reason)
            self.assertIsNone(run.recovery_prompt)
            self.assertIsNone(run.response_pace)
        c.shutdown(timeout=2)

    def test_existing_switch_model_action_unaffected(self):
        c = _make_client()
        c.add_policy(
            name="downgrade",
            condition={"trigger": "llm_latency_ms", "operator": "gt", "value": 1000},
            action={"type": "switch_model", "params": {"model": "gpt-4o-mini"}},
        )
        with c.run("agent") as run:
            run.llm_called("gpt-4o")
            run.llm_responded(latency_ms=6000, finish_reason="stop")
            self.assertEqual(run.model_override, "gpt-4o-mini")
        c.shutdown(timeout=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
