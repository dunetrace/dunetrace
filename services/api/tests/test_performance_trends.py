"""
Tests for Phase 4.4's pure performance-trends math
(api_svc/performance_trends.py) — day bucketing, rate/delta computation,
and self-baseline comparison. No DB, no I/O.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from api_svc.performance_trends import (
    build_day_buckets,
    compute_baseline_comparisons,
    compute_daily_points,
    compute_failure_mode_deltas,
)


class TestBuildDayBuckets(unittest.TestCase):
    def test_returns_correct_count(self):
        buckets = build_day_buckets(7, now=datetime(2026, 7, 12, tzinfo=timezone.utc))
        self.assertEqual(len(buckets), 7)

    def test_oldest_to_newest_order(self):
        buckets = build_day_buckets(3, now=datetime(2026, 7, 12, tzinfo=timezone.utc))
        self.assertEqual(buckets, ["2026-07-10", "2026-07-11", "2026-07-12"])

    def test_last_bucket_is_today(self):
        now = datetime(2026, 7, 12, 15, 30, tzinfo=timezone.utc)
        buckets = build_day_buckets(90, now=now)
        self.assertEqual(buckets[-1], "2026-07-12")
        self.assertEqual(len(buckets), 90)


class TestComputeDailyPoints(unittest.TestCase):
    def test_zero_fills_days_with_no_runs(self):
        buckets = ["2026-07-10", "2026-07-11", "2026-07-12"]
        points = compute_daily_points(buckets, {}, {}, {}, {}, {})
        self.assertEqual(len(points), 3)
        for p in points:
            self.assertEqual(p["total_runs"], 0)
            self.assertEqual(p["structural_signal_rate"], 0.0)
            self.assertEqual(p["semantic_signal_rate"], 0.0)
            self.assertEqual(p["cost_usd"], 0.0)
            self.assertIsNone(p["avg_latency_ms"])

    def test_computes_rates_correctly(self):
        buckets = ["2026-07-12"]
        points = compute_daily_points(
            buckets,
            runs_by_day={"2026-07-12": 10},
            structural_by_day={"2026-07-12": 3},
            semantic_by_day={"2026-07-12": 2},
            cost_by_day={"2026-07-12": 1.5},
            latency_by_day={"2026-07-12": 842.0},
        )
        p = points[0]
        self.assertEqual(p["total_runs"], 10)
        self.assertEqual(p["structural_signal_rate"], 0.3)
        self.assertEqual(p["semantic_signal_rate"], 0.2)
        self.assertEqual(p["cost_usd"], 1.5)
        self.assertEqual(p["avg_latency_ms"], 842.0)

    def test_missing_day_in_partial_data_defaults_safely(self):
        buckets = ["2026-07-11", "2026-07-12"]
        points = compute_daily_points(
            buckets,
            runs_by_day={"2026-07-12": 5},
            structural_by_day={"2026-07-12": 1},
            semantic_by_day={},
            cost_by_day={},
            latency_by_day={},
        )
        self.assertEqual(points[0]["total_runs"], 0)
        self.assertEqual(points[1]["total_runs"], 5)
        self.assertEqual(points[1]["structural_signal_rate"], 0.2)


class TestComputeFailureModeDeltas(unittest.TestCase):
    def test_computes_the_brief_example(self):
        """ "your Tool Loop rate went from 5% to 12%" """
        deltas = compute_failure_mode_deltas(
            current_counts={"TOOL_LOOP": 12},
            previous_counts={"TOOL_LOOP": 5},
            current_total=100,
            previous_total=100,
        )
        self.assertEqual(len(deltas), 1)
        d = deltas[0]
        self.assertEqual(d["current_rate"], 0.12)
        self.assertEqual(d["previous_rate"], 0.05)
        self.assertEqual(d["delta"], 0.07)

    def test_failure_type_only_in_previous_period(self):
        deltas = compute_failure_mode_deltas(
            current_counts={},
            previous_counts={"RETRY_STORM": 4},
            current_total=50,
            previous_total=40,
        )
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["current_rate"], 0.0)
        self.assertEqual(deltas[0]["previous_rate"], 0.1)
        self.assertEqual(deltas[0]["delta"], -0.1)

    def test_zero_totals_do_not_divide_by_zero(self):
        deltas = compute_failure_mode_deltas(
            current_counts={"TOOL_LOOP": 0},
            previous_counts={},
            current_total=0,
            previous_total=0,
        )
        self.assertEqual(deltas[0]["current_rate"], 0.0)
        self.assertEqual(deltas[0]["previous_rate"], 0.0)

    def test_sorted_by_absolute_delta_descending(self):
        deltas = compute_failure_mode_deltas(
            current_counts={"A": 1, "B": 50},
            previous_counts={"A": 0, "B": 0},
            current_total=100,
            previous_total=100,
        )
        self.assertEqual(deltas[0]["failure_type"], "B")
        self.assertEqual(deltas[1]["failure_type"], "A")


class TestComputeBaselineComparisons(unittest.TestCase):
    def test_matches_the_brief_example(self):
        """ "your Hallucination rate is 2x industry median" -> self-baseline
        version: "2x your own baseline"."""
        comparisons = compute_baseline_comparisons(
            current_rates={"HALLUCINATION": 0.20},
            baseline_counts={"HALLUCINATION": 3},
            baseline_total_runs=30,
        )
        self.assertEqual(len(comparisons), 1)
        c = comparisons[0]
        self.assertEqual(c["baseline_rate"], 0.1)
        self.assertEqual(c["ratio"], 2.0)

    def test_not_ready_below_min_baseline_runs(self):
        comparisons = compute_baseline_comparisons(
            current_rates={"TOOL_LOOP": 0.1},
            baseline_counts={"TOOL_LOOP": 1},
            baseline_total_runs=5,
        )
        self.assertFalse(comparisons[0]["baseline_ready"])

    def test_ready_at_or_above_min_baseline_runs(self):
        comparisons = compute_baseline_comparisons(
            current_rates={"TOOL_LOOP": 0.1},
            baseline_counts={"TOOL_LOOP": 3},
            baseline_total_runs=30,
        )
        self.assertTrue(comparisons[0]["baseline_ready"])

    def test_ratio_none_when_baseline_rate_is_zero(self):
        comparisons = compute_baseline_comparisons(
            current_rates={"TOOL_LOOP": 0.05},
            baseline_counts={},
            baseline_total_runs=40,
        )
        self.assertIsNone(comparisons[0]["ratio"])

    def test_baseline_total_zero_does_not_divide_by_zero(self):
        comparisons = compute_baseline_comparisons(
            current_rates={"TOOL_LOOP": 0.05},
            baseline_counts={},
            baseline_total_runs=0,
        )
        self.assertEqual(comparisons[0]["baseline_rate"], 0.0)
        self.assertIsNone(comparisons[0]["ratio"])
        self.assertFalse(comparisons[0]["baseline_ready"])


if __name__ == "__main__":
    unittest.main()
