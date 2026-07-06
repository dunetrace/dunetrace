"""
Tests for issue persistence logic in detector_svc.

Tests the lifecycle: open → hit → resolve → reopen.
All DB calls are mocked — no real Postgres needed.

Run:
    cd services/detector
    python -m pytest tests/test_issues.py -v
"""

from __future__ import annotations

import sys
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
for _p in [
    os.path.join(_ROOT, "packages/sdk-py"),
    os.path.join(_ROOT, "services/detector"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import detector_svc.db as db_module
from detector_svc.db import CLEAN_RUNS_THRESHOLD

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_pool():
    """Return a mock asyncpg pool whose acquire() returns a usable async context manager."""
    conn = AsyncMock()
    conn.execute = AsyncMock()
    conn.executemany = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=_AsyncCtx(conn))
    pool._conn = conn  # expose for assertions
    return pool


class _AsyncCtx:
    def __init__(self, obj):
        self._obj = obj

    async def __aenter__(self):
        return self._obj

    async def __aexit__(self, *_):
        pass


# ── upsert_fired_issues ────────────────────────────────────────────────────────


class TestUpsertFiredIssues(unittest.IsolatedAsyncioTestCase):
    async def test_noop_when_no_fired_types(self):
        pool = _make_pool()
        with patch.object(db_module, "_pool", pool):
            await db_module.upsert_fired_issues("org-1", "agent-1", [])
        pool._conn.executemany.assert_not_called()

    async def test_noop_when_pool_is_none(self):
        with patch.object(db_module, "_pool", None):
            # Should not raise
            await db_module.upsert_fired_issues("org-1", "agent-1", ["TOOL_LOOP"])

    async def test_executemany_called_once_per_fired_type(self):
        pool = _make_pool()
        fired = ["TOOL_LOOP", "RETRY_STORM"]
        with patch.object(db_module, "_pool", pool):
            await db_module.upsert_fired_issues("org-1", "agent-1", fired)
        pool._conn.executemany.assert_called_once()
        call_args = pool._conn.executemany.call_args
        rows = call_args[0][1]
        self.assertEqual(len(rows), 2)
        self.assertIn(("org-1", "agent-1", "TOOL_LOOP"), rows)
        self.assertIn(("org-1", "agent-1", "RETRY_STORM"), rows)

    async def test_single_fired_type(self):
        pool = _make_pool()
        with patch.object(db_module, "_pool", pool):
            await db_module.upsert_fired_issues("org-1", "agent-x", ["CONTEXT_BLOAT"])
        pool._conn.executemany.assert_called_once()
        rows = pool._conn.executemany.call_args[0][1]
        self.assertEqual(rows, [("org-1", "agent-x", "CONTEXT_BLOAT")])


# ── advance_clean_runs ─────────────────────────────────────────────────────────


class TestAdvanceCleanRuns(unittest.IsolatedAsyncioTestCase):
    async def test_noop_when_pool_is_none(self):
        with patch.object(db_module, "_pool", None):
            await db_module.advance_clean_runs("org-1", "agent-1", [])

    async def test_execute_called_with_agent_id_and_threshold(self):
        pool = _make_pool()
        with patch.object(db_module, "_pool", pool):
            await db_module.advance_clean_runs("org-1", "agent-1", ["TOOL_LOOP"])
        pool._conn.execute.assert_called_once()
        sql, org_id, agent_id, fired, threshold = pool._conn.execute.call_args[0]
        self.assertEqual(org_id, "org-1")
        self.assertEqual(agent_id, "agent-1")
        self.assertEqual(fired, ["TOOL_LOOP"])
        self.assertEqual(threshold, CLEAN_RUNS_THRESHOLD)

    async def test_empty_fired_types_passes_none(self):
        """Empty fired_types → pass None so ALL open issues get their clean counter incremented."""
        pool = _make_pool()
        with patch.object(db_module, "_pool", pool):
            await db_module.advance_clean_runs("org-1", "agent-1", [])
        _, _, _, fired, _ = pool._conn.execute.call_args[0]
        self.assertIsNone(fired)

    async def test_threshold_constant_is_five(self):
        self.assertEqual(CLEAN_RUNS_THRESHOLD, 5)


# ── Issue lifecycle via worker ─────────────────────────────────────────────────


class TestWorkerIssueIntegration(unittest.IsolatedAsyncioTestCase):
    """Tests that the detector worker calls issue functions after writing signals."""

    def _make_worker_mocks(self, signals):
        import sys
        import types

        # Minimal FailureSignal stub
        class FakeSignal:
            def __init__(self, ft):
                self.failure_type = types.SimpleNamespace(value=ft)
                self.severity = types.SimpleNamespace(value="HIGH")
                self.run_id = "run-1"
                self.agent_id = "agent-1"
                self.agent_version = "v1"
                self.step_index = 1
                self.confidence = 0.9
                self.evidence = {}

        return [FakeSignal(ft) for ft in signals]

    async def test_upsert_called_when_signals_fired(self):
        import detector_svc.worker as w

        signals = self._make_worker_mocks(["TOOL_LOOP", "RETRY_STORM"])

        with (
            patch(
                "detector_svc.worker.fetch_run_events",
                AsyncMock(return_value=[{"event_type": "run.started", "payload": {}}]),
            ),
            patch(
                "detector_svc.worker.build_run_state",
                return_value=MagicMock(baseline_p75_steps=None),
            ),
            patch(
                "detector_svc.worker.fetch_step_count_baseline",
                AsyncMock(return_value=None),
            ),
            patch("detector_svc.worker.run_detectors", return_value=signals),
            patch(
                "detector_svc.worker.RiskEngine",
                return_value=MagicMock(evaluate=MagicMock(return_value=MagicMock(severity=None))),
            ),
            patch("detector_svc.worker._injection_signal_from_events", return_value=None),
            patch("detector_svc.worker.write_signals", AsyncMock(return_value=1)),
            patch("detector_svc.worker.mark_run_processed", AsyncMock()),
            patch("detector_svc.worker.upsert_fired_issues", AsyncMock()) as mock_upsert,
            patch("detector_svc.worker.advance_clean_runs", AsyncMock()) as mock_advance,
        ):
            await w.process_run("run-1", "agent-1", "v1", "completed", "org-1")

        mock_upsert.assert_called_once()
        org_id, agent_id, fired = mock_upsert.call_args[0]
        self.assertEqual(org_id, "org-1")
        self.assertEqual(agent_id, "agent-1")
        self.assertIn("TOOL_LOOP", fired)
        self.assertIn("RETRY_STORM", fired)
        mock_advance.assert_called_once()

    async def test_advance_called_even_when_no_signals(self):
        import detector_svc.worker as w

        with (
            patch(
                "detector_svc.worker.fetch_run_events",
                AsyncMock(return_value=[{"event_type": "run.started", "payload": {}}]),
            ),
            patch(
                "detector_svc.worker.build_run_state",
                return_value=MagicMock(baseline_p75_steps=None),
            ),
            patch(
                "detector_svc.worker.fetch_step_count_baseline",
                AsyncMock(return_value=None),
            ),
            patch("detector_svc.worker.run_detectors", return_value=[]),
            patch(
                "detector_svc.worker.RiskEngine",
                return_value=MagicMock(evaluate=MagicMock(return_value=MagicMock(severity=None))),
            ),
            patch("detector_svc.worker._injection_signal_from_events", return_value=None),
            patch("detector_svc.worker.write_signals", AsyncMock(return_value=0)),
            patch("detector_svc.worker.mark_run_processed", AsyncMock()),
            patch("detector_svc.worker.upsert_fired_issues", AsyncMock()) as mock_upsert,
            patch("detector_svc.worker.advance_clean_runs", AsyncMock()) as mock_advance,
        ):
            await w.process_run("run-1", "agent-1", "v1", "completed", "org-1")

        mock_upsert.assert_not_called()  # nothing to upsert
        mock_advance.assert_called_once_with("org-1", "agent-1", [])

    async def test_issue_tracking_failure_does_not_break_run_processing(self):
        """If upsert_fired_issues raises, the run is still marked processed."""
        import detector_svc.worker as w

        signals = self._make_worker_mocks(["TOOL_LOOP"])

        with (
            patch(
                "detector_svc.worker.fetch_run_events",
                AsyncMock(return_value=[{"event_type": "run.started", "payload": {}}]),
            ),
            patch(
                "detector_svc.worker.build_run_state",
                return_value=MagicMock(baseline_p75_steps=None),
            ),
            patch(
                "detector_svc.worker.fetch_step_count_baseline",
                AsyncMock(return_value=None),
            ),
            patch("detector_svc.worker.run_detectors", return_value=signals),
            patch(
                "detector_svc.worker.RiskEngine",
                return_value=MagicMock(evaluate=MagicMock(return_value=MagicMock(severity=None))),
            ),
            patch("detector_svc.worker._injection_signal_from_events", return_value=None),
            patch("detector_svc.worker.write_signals", AsyncMock(return_value=1)),
            patch("detector_svc.worker.mark_run_processed", AsyncMock()) as mock_mark,
            patch(
                "detector_svc.worker.upsert_fired_issues",
                AsyncMock(side_effect=Exception("DB down")),
            ),
            patch("detector_svc.worker.advance_clean_runs", AsyncMock()),
        ):
            result = await w.process_run("run-1", "agent-1", "v1", "completed", "org-1")

        mock_mark.assert_called_once()
        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
