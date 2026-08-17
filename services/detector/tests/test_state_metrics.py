"""
Tests for per-run state metrics (Capability 3, Phase 3.3): the pure
summarize_states reducer and the write_run_state_metrics DB helper. Mirrors the
per-run timeline cases in api_svc's test_run_states.py to keep the two
reconstructions in sync.

Run: PYTHONPATH=../../packages/sdk-py:. python -m unittest discover -s tests -p "test_state_metrics.py"
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock

from detector_svc.state_metrics import summarize_states


def _e(event_type, ts, step=0, **payload):
    return {"event_type": event_type, "timestamp": ts, "step_index": step, "payload": payload}


class TestSummarizeStates(unittest.TestCase):
    def test_thinking_and_acting_totals(self):
        events = [
            _e("run.started", 0.0),
            _e("llm.called", 1.0, model="gpt-4o"),
            _e("llm.responded", 2.0),  # thinking 1000
            _e("tool.called", 3.0, tool_name="x"),
            _e("tool.responded", 3.5),  # acting 500
            _e("run.completed", 4.0),
        ]
        out = summarize_states(events)
        self.assertEqual(out["states"]["thinking"]["total_ms"], 1000)
        self.assertEqual(out["states"]["acting"]["total_ms"], 500)
        self.assertEqual(out["states"]["thinking"]["count"], 1)
        self.assertEqual(out["run_started_ts"], 0.0)

    def test_repeated_state_accumulates(self):
        events = [
            _e("llm.called", 1.0, model="m"),
            _e("llm.responded", 2.0),  # 1000
            _e("llm.called", 3.0, model="m"),
            _e("llm.responded", 3.5),  # 500
        ]
        out = summarize_states(events)
        self.assertEqual(out["states"]["thinking"]["total_ms"], 1500)
        self.assertEqual(out["states"]["thinking"]["count"], 2)

    def test_waiting_approval(self):
        events = [
            _e("approval.requested", 1.0, tool_name="wire"),
            _e("approval.granted", 9.0),
        ]
        out = summarize_states(events)
        self.assertEqual(out["states"]["waiting_approval"]["total_ms"], 8000)

    def test_missing_response_still_counted(self):
        events = [_e("tool.called", 1.0, tool_name="x"), _e("run.errored", 5.0)]
        out = summarize_states(events)
        self.assertEqual(out["states"]["acting"]["total_ms"], 4000)

    def test_point_events_ignored(self):
        events = [
            _e("run.started", 0.0),
            _e("voice_activity.detected", 1.0, type="silence"),
            _e("run.completed", 2.0),
        ]
        self.assertEqual(summarize_states(events)["states"], {})

    def test_empty(self):
        out = summarize_states([])
        self.assertEqual(out["states"], {})
        self.assertIsNone(out["run_started_ts"])

    def test_float_rounding(self):
        # 0.2s should be 200ms, not 199.
        events = [_e("tool.called", 1.0, tool_name="x"), _e("tool.responded", 1.2)]
        self.assertEqual(summarize_states(events)["states"]["acting"]["total_ms"], 200)


class TestWriteRunStateMetrics(unittest.IsolatedAsyncioTestCase):
    async def _run_write(self, states):
        import detector_svc.db as db

        conn = AsyncMock()

        class _Acq:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *a):
                return False

        class _Pool:
            def acquire(self):
                return _Acq()

        orig = db._pool
        db._pool = _Pool()
        try:
            await db.write_run_state_metrics("run-1", "org", "agent", 1000.0, states)
        finally:
            db._pool = orig
        return conn

    async def test_upserts_one_row_per_state(self):
        conn = await self._run_write(
            {"thinking": {"total_ms": 1500, "count": 2}, "acting": {"total_ms": 500, "count": 1}}
        )
        self.assertEqual(conn.execute.await_count, 2)
        # each row carries run_id, org, agent, state, total, count
        states_written = {c.args[4] for c in conn.execute.await_args_list}
        self.assertEqual(states_written, {"thinking", "acting"})

    async def test_empty_states_writes_nothing(self):
        conn = await self._run_write({})
        conn.execute.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
