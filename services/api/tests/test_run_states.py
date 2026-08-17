"""
Tests for state-machine reconstruction (Capability 3, Phase 3.1). Pure logic,
no DB — the whole point of run_states.py is that the event→state mapping is
testable in isolation.

Run: PYTHONPATH=../../packages/sdk-py:../explainer:. python -m unittest tests.test_run_states -v
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from api_svc.routers.runs import get_run_states
from api_svc.run_states import reconstruct_states


def _e(event_type, ts, step=0, **payload):
    return {"event_type": event_type, "timestamp": ts, "step_index": step, "payload": payload}


class TestBasicStates(unittest.TestCase):
    def test_llm_call_becomes_thinking_segment(self):
        events = [
            _e("run.started", 0.0),
            _e("llm.called", 1.0, model="gpt-4o"),
            _e("llm.responded", 2.5),
            _e("run.completed", 3.0),
        ]
        out = reconstruct_states(events)
        segs = out["segments"]
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0]["state"], "thinking")
        self.assertEqual(segs[0]["label"], "gpt-4o")
        self.assertEqual(segs[0]["duration_ms"], 1500)
        self.assertTrue(segs[0]["closed"])

    def test_tool_call_becomes_acting(self):
        events = [
            _e("tool.called", 1.0, tool_name="web_search"),
            _e("tool.responded", 1.2),
        ]
        segs = reconstruct_states(events)["segments"]
        self.assertEqual(segs[0]["state"], "acting")
        self.assertEqual(segs[0]["label"], "web_search")
        self.assertEqual(segs[0]["duration_ms"], 200)

    def test_retrieval_and_approval_states(self):
        events = [
            _e("retrieval.called", 1.0, index_name="docs"),
            _e("retrieval.responded", 1.1),
            _e("approval.requested", 2.0, tool_name="wire_money"),
            _e("approval.granted", 9.0),
        ]
        segs = reconstruct_states(events)["segments"]
        states = [s["state"] for s in segs]
        self.assertEqual(states, ["retrieving", "waiting_approval"])
        self.assertEqual(segs[1]["label"], "wire_money")
        self.assertEqual(segs[1]["duration_ms"], 7000)

    def test_approval_denied_closes_waiting(self):
        events = [
            _e("approval.requested", 1.0, tool_name="wire_money"),
            _e("approval.denied", 4.0),
        ]
        segs = reconstruct_states(events)["segments"]
        self.assertEqual(segs[0]["state"], "waiting_approval")
        self.assertTrue(segs[0]["closed"])


class TestOrderingAndSequence(unittest.TestCase):
    def test_full_sequence_in_order(self):
        events = [
            _e("run.started", 0.0),
            _e("llm.called", 1.0, model="gpt-4o"),
            _e("llm.responded", 2.0),
            _e("tool.called", 2.1, tool_name="search"),
            _e("tool.responded", 2.4),
            _e("llm.called", 2.5, model="gpt-4o"),
            _e("llm.responded", 3.0),
            _e("run.completed", 3.1),
        ]
        segs = reconstruct_states(events)["segments"]
        self.assertEqual([s["state"] for s in segs], ["thinking", "acting", "thinking"])

    def test_unsorted_events_are_sorted(self):
        events = [
            _e("llm.responded", 2.0),
            _e("llm.called", 1.0, model="gpt-4o"),
        ]
        segs = reconstruct_states(events)["segments"]
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0]["duration_ms"], 1000)


class TestForgiving(unittest.TestCase):
    def test_missing_response_marked_not_closed(self):
        # run crashed mid-tool: tool.called with no tool.responded, ends at run.errored
        events = [
            _e("tool.called", 1.0, tool_name="flaky"),
            _e("run.errored", 5.0),
        ]
        segs = reconstruct_states(events)["segments"]
        self.assertEqual(len(segs), 1)
        self.assertFalse(segs[0]["closed"])
        self.assertEqual(segs[0]["duration_ms"], 4000)

    def test_new_opener_closes_dangling_previous(self):
        events = [
            _e("llm.called", 1.0, model="gpt-4o"),  # no llm.responded
            _e("tool.called", 2.0, tool_name="x"),  # opens while thinking still open
            _e("tool.responded", 2.5),
        ]
        segs = reconstruct_states(events)["segments"]
        self.assertEqual([s["state"] for s in segs], ["thinking", "acting"])
        self.assertFalse(segs[0]["closed"])  # thinking got cut off
        self.assertTrue(segs[1]["closed"])

    def test_open_at_end_with_no_terminal_closes_at_last_event(self):
        events = [
            _e("llm.called", 1.0, model="gpt-4o"),
            _e("external.signal", 3.0, signal_name="rate_limit"),  # point event
        ]
        segs = reconstruct_states(events)["segments"]
        self.assertEqual(len(segs), 1)
        self.assertFalse(segs[0]["closed"])
        self.assertEqual(segs[0]["duration_ms"], 2000)  # to last event

    def test_orphan_closer_ignored(self):
        segs = reconstruct_states([_e("tool.responded", 1.0)])["segments"]
        self.assertEqual(segs, [])

    def test_point_events_are_not_states(self):
        events = [
            _e("run.started", 0.0),
            _e("voice_activity.detected", 1.0, type="speech_start"),
            _e("policy.triggered", 1.1, policy_name="p"),
            _e("run.completed", 2.0),
        ]
        self.assertEqual(reconstruct_states(events)["segments"], [])

    def test_negative_duration_guarded(self):
        # Clock skew: responded timestamp before called.
        events = [
            _e("llm.called", 5.0, model="gpt-4o"),
            _e("llm.responded", 4.0),
        ]
        segs = reconstruct_states(events)["segments"]
        # sorted by ts, so responded(4.0) comes first → orphan closer; called(5.0)
        # opens and never closes → ends at last event (5.0), duration 0.
        self.assertGreaterEqual(segs[0]["duration_ms"], 0)


class TestSummary(unittest.TestCase):
    def test_by_state_totals_and_span(self):
        events = [
            _e("run.started", 0.0),
            _e("llm.called", 1.0, model="gpt-4o"),
            _e("llm.responded", 2.0),  # thinking 1000ms
            _e("tool.called", 3.0, tool_name="x"),
            _e("tool.responded", 3.5),  # acting 500ms
            _e("run.completed", 4.0),
        ]
        summary = reconstruct_states(events)["summary"]
        self.assertEqual(summary["by_state"]["thinking"], 1000)
        self.assertEqual(summary["by_state"]["acting"], 500)
        self.assertEqual(summary["segment_count"], 2)
        self.assertEqual(summary["total_ms"], 4000)  # span 0.0 → 4.0

    def test_empty_events(self):
        out = reconstruct_states([])
        self.assertEqual(out["segments"], [])
        self.assertEqual(out["summary"]["total_ms"], 0)
        self.assertEqual(out["summary"]["segment_count"], 0)


class TestGetRunStatesEndpoint(unittest.IsolatedAsyncioTestCase):
    async def test_returns_reconstructed_states(self):
        detail = {
            "events": [
                _e("llm.called", 1.0, model="gpt-4o"),
                _e("llm.responded", 2.0),
            ]
        }
        with patch(
            "api_svc.routers.runs.get_run_detail", AsyncMock(return_value=detail)
        ) as mock_detail:
            out = await get_run_states("run-1", org_id="org")
        # org-scoped fetch reused
        self.assertEqual(mock_detail.call_args.args[0], "org")
        self.assertEqual(out["segments"][0]["state"], "thinking")

    async def test_missing_run_is_404(self):
        with patch("api_svc.routers.runs.get_run_detail", AsyncMock(return_value=None)):
            with self.assertRaises(HTTPException) as ctx:
                await get_run_states("nope", org_id="org")
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
