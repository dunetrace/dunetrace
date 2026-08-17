"""
Tests for cross-run state analytics (Capability 3, Phase 3.3): the pure math
(percentiles, aggregates, outliers, trends) and the endpoint. No DB.

Run: PYTHONPATH=../../packages/sdk-py:../explainer:. python -m unittest tests.test_state_analytics -v
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from api_svc.routers.performance_trends import get_state_analytics
from api_svc.state_analytics import (
    aggregate_by_state,
    compute_state_analytics,
    daily_trend,
    detect_outliers,
    percentile,
)


def _row(run_id, state, total_ms, started=None):
    return {"run_id": run_id, "state": state, "total_ms": total_ms, "run_started_at": started}


class TestPercentile(unittest.TestCase):
    def test_empty_is_zero(self):
        self.assertEqual(percentile([], 95), 0.0)

    def test_single_value(self):
        self.assertEqual(percentile([42], 95), 42.0)

    def test_median(self):
        self.assertEqual(percentile([10, 20, 30], 50), 20.0)

    def test_p95_interpolates(self):
        self.assertAlmostEqual(percentile([0, 100], 95), 95.0)


class TestAggregateByState(unittest.TestCase):
    def test_per_state_stats(self):
        rows = [
            _row("r1", "thinking", 1000),
            _row("r2", "thinking", 2000),
            _row("r1", "acting", 500),
        ]
        agg = aggregate_by_state(rows)
        self.assertEqual(agg["thinking"]["run_count"], 2)
        self.assertEqual(agg["thinking"]["total_ms"], 3000)
        self.assertEqual(agg["thinking"]["avg_ms"], 1500)
        self.assertEqual(agg["acting"]["run_count"], 1)

    def test_empty(self):
        self.assertEqual(aggregate_by_state([]), {})


class TestDetectOutliers(unittest.TestCase):
    def test_flags_run_far_above_median(self):
        rows = [
            _row("r1", "acting", 1000),
            _row("r2", "acting", 1100),
            _row("r3", "acting", 900),
            _row("r4", "acting", 9000),  # ~9x median
        ]
        out = detect_outliers(rows, factor=3.0)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["run_id"], "r4")
        self.assertEqual(out[0]["state"], "acting")
        self.assertGreater(out[0]["ratio"], 3.0)

    def test_no_outliers_when_uniform(self):
        rows = [_row(f"r{i}", "thinking", 1000) for i in range(5)]
        self.assertEqual(detect_outliers(rows), [])

    def test_low_baseline_states_not_flagged_on_noise(self):
        # median well under the floor — a 3x spike on a near-zero state is noise.
        rows = [_row("r1", "retrieving", 10), _row("r2", "retrieving", 50)]
        self.assertEqual(detect_outliers(rows, min_baseline_ms=100), [])


class TestDailyTrend(unittest.TestCase):
    def test_buckets_by_run_date_with_gaps_filled(self):
        now = datetime(2026, 7, 12, tzinfo=timezone.utc)
        rows = [
            _row("r1", "thinking", 1000, now),
            _row("r2", "thinking", 3000, now),
            _row("r3", "thinking", 2000, now - timedelta(days=2)),
        ]
        trend = daily_trend(rows, window_days=3, now=now)
        series = trend["thinking"]
        self.assertEqual(len(series), 3)  # 3 days, oldest -> newest
        self.assertEqual(series[-1]["avg_ms"], 2000)  # today: (1000+3000)/2
        self.assertEqual(series[-1]["run_count"], 2)
        self.assertEqual(series[1]["avg_ms"], 0)  # middle day empty


class TestCompute(unittest.TestCase):
    def test_full_payload(self):
        now = datetime(2026, 7, 12, tzinfo=timezone.utc)
        rows = [
            _row("r1", "thinking", 1000, now),
            _row("r1", "acting", 500, now),
            _row("r2", "thinking", 1200, now),
        ]
        out = compute_state_analytics(rows, window_days=7, now=now)
        self.assertEqual(out["run_count"], 2)  # distinct runs
        self.assertIn("thinking", out["by_state"])
        self.assertIn("thinking", out["trend"])
        self.assertEqual(out["window_days"], 7)


class TestEndpoint(unittest.IsolatedAsyncioTestCase):
    async def test_returns_analytics(self):
        payload = {"window_days": 30, "run_count": 3, "by_state": {}, "trend": {}, "outliers": []}
        with patch(
            "api_svc.routers.performance_trends.agent_state_analytics",
            AsyncMock(return_value=payload),
        ) as mock_q:
            result = await get_state_analytics("agent-1", window=30, org_id="org")
        self.assertEqual(result["agent_id"], "agent-1")
        self.assertEqual(result["run_count"], 3)
        self.assertEqual(mock_q.call_args.args[0], "org")

    async def test_invalid_window_is_422(self):
        with self.assertRaises(HTTPException) as ctx:
            await get_state_analytics("agent-1", window=13, org_id="org")
        self.assertEqual(ctx.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main(verbosity=2)
