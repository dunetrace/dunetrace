"""
Tests for signal claiming — what makes the alerts worker safe to replicate.

Before claiming existed, `alerted = FALSE` was the only thing marking a signal as
"needs an alert", and it isn't set until *after* a successful send. Two workers
polling the same window therefore both saw the same rows as outstanding, both
delivered, and the customer got duplicate Slack messages. The `alert_dedup` window
doesn't help: it's read before either worker writes it, so both pass the check.

These tests cover the claim/release contract and the shard threading. The atomicity
itself lives in one SQL statement (db.py::claim_unalerted_signals) and is asserted
structurally here — verifying it against a live Postgres belongs in an integration
test, since the guarantee is FOR UPDATE SKIP LOCKED's, not ours.

No DB, no real HTTP calls.

Run:
    cd services/alerts
    python -m unittest tests.test_claiming -v
"""

from __future__ import annotations

import importlib
import os
import sys
import time
import unittest
from unittest.mock import AsyncMock, patch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
for _p in [
    os.path.join(_ROOT, "packages/sdk-py"),
    os.path.join(_ROOT, "services/explainer"),
    os.path.join(_ROOT, "services/alerts"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import alerts_svc.config as config_module
import alerts_svc.db as db_module
import alerts_svc.worker as worker_module
from alerts_svc.sender import SendResult


def _row(signal_id: int, *, agent_id: str = "agent-a", failure_type: str = "TOOL_LOOP") -> dict:
    return {
        "id": signal_id,
        "failure_type": failure_type,
        "severity": "HIGH",
        "run_id": f"run-{signal_id}",
        "agent_id": agent_id,
        "org_id": "default",
        "agent_version": "v1",
        "step_index": 3,
        "confidence": 0.9,
        "evidence": {},
        "detected_at": __import__("datetime").datetime(2026, 7, 1, 12, 0, 0),
        "source": "structural",
    }


class TestClaimQueryContract(unittest.IsolatedAsyncioTestCase):
    """The claim statement's shape is the correctness argument — assert the parts
    that make concurrent replicas safe are actually in the SQL."""

    async def test_noop_without_pool(self):
        with patch.object(db_module, "_pool", None):
            self.assertEqual(await db_module.claim_unalerted_signals(), [])
            await db_module.release_claims([1, 2])  # must not raise
            await db_module.ensure_alert_claim_columns()  # must not raise

    async def test_claim_sql_locks_rows_and_skips_contended_ones(self):
        captured = {}

        class _Conn:
            async def fetch(self, sql, *args):
                captured["sql"] = sql
                captured["args"] = args
                return []

        await self._run_claim(_Conn(), captured)

        sql = captured["sql"]
        # Without SKIP LOCKED a second worker blocks instead of moving on; without
        # FOR UPDATE it doesn't lock at all and both read the same rows.
        self.assertIn("FOR UPDATE SKIP LOCKED", sql)
        # The claim must be written by the same statement that selects, or the
        # gap between select and update is the duplicate-delivery window again.
        self.assertIn("UPDATE failure_signals", sql)
        self.assertIn("alert_claimed_at = NOW()", sql)
        # Shadow signals are never alerted on.
        self.assertIn("COALESCE(shadow, TRUE) = FALSE", sql)
        # Expired claims must be reclaimable or a crashed worker strands them.
        self.assertIn("alert_claimed_at IS NULL", sql)

    async def test_shard_filter_present_and_parameterized(self):
        captured = {}

        class _Conn:
            async def fetch(self, sql, *args):
                captured["sql"] = sql
                captured["args"] = args
                return []

        await self._run_claim(_Conn(), captured, shard_count=4, shard_index=2)

        self.assertIn("abs(hashtext(agent_id)) % $2 = $3", captured["sql"])
        limit, shard_count, shard_index, worker_id, timeout = captured["args"]
        self.assertEqual(shard_count, 4)
        self.assertEqual(shard_index, 2)
        self.assertEqual(worker_id, "w-1")
        # Interval is built from a text parameter, never string-interpolated.
        self.assertEqual(timeout, "42.0")

    async def test_shard_count_one_short_circuits(self):
        """SHARD_COUNT=1 must bypass the hash filter entirely so a single-instance
        deployment does no modulo work — same contract as detector_svc."""
        captured = {}

        class _Conn:
            async def fetch(self, sql, *args):
                captured["sql"] = sql
                captured["args"] = args
                return []

        await self._run_claim(_Conn(), captured, shard_count=1)
        self.assertIn("$2::int = 1 OR", captured["sql"])
        self.assertEqual(captured["args"][1], 1)

    async def test_rows_returned_oldest_first(self):
        """RETURNING doesn't preserve the CTE's ORDER BY; the caller's
        best-signal-per-group logic reads against oldest-first."""
        import datetime

        newer = _row(2)
        newer["detected_at"] = datetime.datetime(2026, 7, 2, 12, 0, 0)
        older = _row(1)
        older["detected_at"] = datetime.datetime(2026, 7, 1, 12, 0, 0)

        class _Conn:
            async def fetch(self, sql, *args):
                return [newer, older]

        out = await self._run_claim(_Conn(), {})
        self.assertEqual([r["id"] for r in out], [1, 2])

    async def _run_claim(self, conn, captured, **kwargs):
        class _Acquire:
            async def __aenter__(self_inner):
                return conn

            async def __aexit__(self_inner, *a):
                return False

        class _Pool:
            def acquire(self_inner):
                return _Acquire()

        params = dict(
            limit=50, shard_count=1, shard_index=0, worker_id="w-1", claim_timeout_secs=42.0
        )
        params.update(kwargs)
        with patch.object(db_module, "_pool", _Pool()):
            return await db_module.claim_unalerted_signals(**params)


class TestReleaseClaimGuard(unittest.IsolatedAsyncioTestCase):
    async def test_release_only_touches_unalerted_rows(self):
        """Releasing must never un-claim a row that was already delivered, or the
        next poll re-delivers it."""
        captured = {}

        class _Conn:
            async def execute(self, sql, *args):
                captured["sql"] = sql

        class _Acquire:
            async def __aenter__(self_inner):
                return _Conn()

            async def __aexit__(self_inner, *a):
                return False

        class _Pool:
            def acquire(self_inner):
                return _Acquire()

        with patch.object(db_module, "_pool", _Pool()):
            await db_module.release_claims([1, 2, 3])

        self.assertIn("alert_claimed_at = NULL", captured["sql"])
        self.assertIn("alerted = FALSE", captured["sql"])

    async def test_empty_list_is_a_noop(self):
        class _Pool:
            def acquire(self_inner):
                raise AssertionError("must not touch the pool for an empty list")

        with patch.object(db_module, "_pool", _Pool()):
            await db_module.release_claims([])


class TestPollOnceReleasesClaims(unittest.IsolatedAsyncioTestCase):
    """Claims taken by a poll that doesn't deliver must be handed back, or the
    signal waits out CLAIM_TIMEOUT_SECS before anyone retries it."""

    async def _poll(self, rows, deliver_results):
        release_mock = AsyncMock()
        with (
            patch("alerts_svc.worker.claim_unalerted_signals", AsyncMock(return_value=rows)),
            patch("alerts_svc.worker.release_claims", release_mock),
            patch("alerts_svc.worker.mark_alerted_batch", AsyncMock()),
            patch("alerts_svc.worker.fetch_dedup_states", AsyncMock(return_value={})),
            patch("alerts_svc.worker.fetch_agent_overrides", AsyncMock(return_value={})),
            patch("alerts_svc.worker.fetch_signal_rate_context", AsyncMock(return_value={})),
            patch("alerts_svc.worker.fetch_run_tokens", AsyncMock(return_value={})),
            patch("alerts_svc.worker.record_alert_sent", AsyncMock()),
            patch("alerts_svc.worker.increment_suppressed_count", AsyncMock()),
            patch(
                "alerts_svc.worker.evaluate_alert_policy",
                AsyncMock(return_value=(True, "immediate")),
            ),
            patch("alerts_svc.worker._resolve_slack_destination", AsyncMock(return_value=None)),
            patch("alerts_svc.worker._resolve_linear_config", AsyncMock(return_value=None)),
            patch("alerts_svc.worker.deliver", return_value=deliver_results),
        ):
            found, delivered = await worker_module.poll_once()
        return found, delivered, release_mock

    async def test_failed_delivery_releases_the_claim(self):
        rows = [_row(11)]
        found, delivered, release_mock = await self._poll(
            rows,
            {"slack": SendResult(success=False, destination="slack", attempts=3, error="boom")},
        )
        self.assertEqual(delivered, 0)
        release_mock.assert_awaited()
        self.assertIn(11, release_mock.await_args.args[0])

    async def test_successful_delivery_still_calls_release_harmlessly(self):
        """release_claims is guarded on alerted = FALSE in SQL, so calling it with
        delivered ids is a no-op — that's what lets poll_once release
        unconditionally instead of tracking which rows settled."""
        rows = [_row(12)]
        found, delivered, release_mock = await self._poll(
            rows, {"slack": SendResult(success=True, destination="slack", attempts=1)}
        )
        self.assertEqual(delivered, 1)
        release_mock.assert_awaited()

    async def test_no_rows_claims_nothing_and_releases_nothing(self):
        release_mock = AsyncMock()
        with (
            patch("alerts_svc.worker.claim_unalerted_signals", AsyncMock(return_value=[])),
            patch("alerts_svc.worker.release_claims", release_mock),
        ):
            found, delivered = await worker_module.poll_once()
        self.assertEqual((found, delivered), (0, 0))
        release_mock.assert_not_awaited()


class TestShardConfigValidation(unittest.TestCase):
    """Misconfigured replicas must crash at import, not silently claim nothing —
    the same guard detector_svc has."""

    def _reload_with(self, **env):
        original = {k: os.environ.get(k) for k in env}
        os.environ.update({k: str(v) for k, v in env.items()})
        try:
            importlib.reload(config_module)
        finally:
            for k, v in original.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            importlib.reload(config_module)

    def test_shard_count_zero_rejected(self):
        with self.assertRaises(ValueError):
            self._reload_with(SHARD_COUNT=0)

    def test_shard_index_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            self._reload_with(SHARD_COUNT=2, SHARD_INDEX=2)

    def test_negative_shard_index_rejected(self):
        with self.assertRaises(ValueError):
            self._reload_with(SHARD_COUNT=2, SHARD_INDEX=-1)

    def test_zero_claim_timeout_rejected(self):
        with self.assertRaises(ValueError):
            self._reload_with(CLAIM_TIMEOUT_SECS=0)

    def test_valid_shard_config_accepted(self):
        self._reload_with(SHARD_COUNT=4, SHARD_INDEX=3)

    def test_alerts_prefixed_vars_win_over_shared_ones(self):
        """A SHARD_COUNT meant for the detector must not silently shard the alerts
        worker — otherwise a lone alerts worker claims one bucket and the rest of
        the signals are never delivered."""
        self._reload_with(SHARD_COUNT=4, SHARD_INDEX=2, ALERTS_SHARD_COUNT=1, ALERTS_SHARD_INDEX=0)
        # Reloaded back to ambient by _reload_with, so assert inside a fresh load.
        original = {
            k: os.environ.get(k)
            for k in ("SHARD_COUNT", "SHARD_INDEX", "ALERTS_SHARD_COUNT", "ALERTS_SHARD_INDEX")
        }
        os.environ.update(
            {
                "SHARD_COUNT": "4",
                "SHARD_INDEX": "2",
                "ALERTS_SHARD_COUNT": "1",
                "ALERTS_SHARD_INDEX": "0",
            }
        )
        try:
            importlib.reload(config_module)
            self.assertEqual(config_module.settings.SHARD_COUNT, 1)
            self.assertEqual(config_module.settings.SHARD_INDEX, 0)
        finally:
            for k, v in original.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            importlib.reload(config_module)

    def test_shared_vars_used_when_no_prefix_set(self):
        original = {k: os.environ.get(k) for k in ("SHARD_COUNT", "SHARD_INDEX")}
        os.environ.update({"SHARD_COUNT": "3", "SHARD_INDEX": "1"})
        os.environ.pop("ALERTS_SHARD_COUNT", None)
        os.environ.pop("ALERTS_SHARD_INDEX", None)
        try:
            importlib.reload(config_module)
            self.assertEqual(config_module.settings.SHARD_COUNT, 3)
            self.assertEqual(config_module.settings.SHARD_INDEX, 1)
        finally:
            for k, v in original.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            importlib.reload(config_module)


if __name__ == "__main__":
    unittest.main()
