"""
Endpoint-level tests for Phase 4.4's performance-trends endpoint
(api_svc/routers/performance_trends.py). Calls the route function directly
(this codebase's established pattern), mocked DB call. No network, no DB.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from api_svc.routers.performance_trends import get_performance_trends


def _trends_result():
    return {
        "points": [
            {
                "day": "2026-07-12",
                "total_runs": 10,
                "structural_signal_rate": 0.3,
                "semantic_signal_rate": 0.1,
                "cost_usd": 1.23,
                "avg_latency_ms": 842.0,
            }
        ],
        "failure_mode_deltas": [
            {
                "failure_type": "TOOL_LOOP",
                "current_rate": 0.12,
                "previous_rate": 0.05,
                "delta": 0.07,
                "current_affected_runs": 12,
                "previous_affected_runs": 5,
            }
        ],
        "baseline_comparisons": [
            {
                "failure_type": "HALLUCINATION",
                "current_rate": 0.2,
                "baseline_rate": 0.1,
                "ratio": 2.0,
                "baseline_sample_runs": 30,
                "baseline_ready": True,
            }
        ],
    }


class TestGetPerformanceTrends(unittest.IsolatedAsyncioTestCase):
    async def test_valid_window_returns_shaped_response(self):
        with patch(
            "api_svc.routers.performance_trends.agent_performance_trends",
            AsyncMock(return_value=_trends_result()),
        ):
            result = await get_performance_trends("agent-1", window=30, org_id="org-1")

        self.assertEqual(result.agent_id, "agent-1")
        self.assertEqual(result.window_days, 30)
        self.assertEqual(len(result.points), 1)
        self.assertEqual(result.points[0].day, "2026-07-12")
        self.assertEqual(result.failure_mode_deltas[0].failure_type, "TOOL_LOOP")
        self.assertEqual(result.baseline_comparisons[0].ratio, 2.0)

    async def test_invalid_window_returns_422(self):
        with self.assertRaises(HTTPException) as ctx:
            await get_performance_trends("agent-1", window=14, org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 422)

    async def test_default_window_is_30(self):
        with patch(
            "api_svc.routers.performance_trends.agent_performance_trends",
            AsyncMock(return_value=_trends_result()),
        ) as mock_fn:
            result = await get_performance_trends("agent-1", org_id="org-1")

        self.assertEqual(result.window_days, 30)
        mock_fn.assert_awaited_once_with("org-1", "agent-1", 30)

    async def test_window_7_and_90_both_accepted(self):
        for window in (7, 90):
            with patch(
                "api_svc.routers.performance_trends.agent_performance_trends",
                AsyncMock(return_value=_trends_result()),
            ):
                result = await get_performance_trends("agent-1", window=window, org_id="org-1")
            self.assertEqual(result.window_days, window)

    async def test_empty_result_returns_empty_lists(self):
        with patch(
            "api_svc.routers.performance_trends.agent_performance_trends",
            AsyncMock(
                return_value={"points": [], "failure_mode_deltas": [], "baseline_comparisons": []}
            ),
        ):
            result = await get_performance_trends("agent-1", window=7, org_id="org-1")
        self.assertEqual(result.points, [])
        self.assertEqual(result.failure_mode_deltas, [])
        self.assertEqual(result.baseline_comparisons, [])


if __name__ == "__main__":
    unittest.main()
