"""
Tests for Phase 1.5's pure projection math (api_svc/semantic_usage.py) and
the GET /v1/orgs/semantic-usage endpoint (api_svc/routers/orgs.py). Calls the
route function directly (this codebase's established pattern) with mocked DB
calls. No network, no DB.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from api_svc.semantic_usage import current_month, project_month_end
from api_svc.routers.orgs import get_semantic_usage


class TestCurrentMonth(unittest.TestCase):
    def test_formats_as_yyyy_mm(self):
        now = datetime(2026, 7, 15, tzinfo=timezone.utc)
        self.assertEqual(current_month(now), "2026-07")


class TestProjectMonthEnd(unittest.TestCase):
    def test_halfway_through_30_day_month_doubles(self):
        # April has 30 days; day 15 -> 15 days elapsed -> straightforward 2x.
        now = datetime(2026, 4, 15, tzinfo=timezone.utc)
        self.assertAlmostEqual(project_month_end(100, now), 200.0)

    def test_first_day_of_month_uses_day_one_not_zero(self):
        now = datetime(2026, 4, 1, tzinfo=timezone.utc)
        # days_elapsed floors at 1, not 0 — no division by zero.
        self.assertAlmostEqual(project_month_end(10, now), 10.0 * 30)

    def test_last_day_of_month_returns_used_so_far(self):
        now = datetime(2026, 4, 30, tzinfo=timezone.utc)
        self.assertAlmostEqual(project_month_end(300, now), 300.0)

    def test_zero_usage_projects_to_zero(self):
        now = datetime(2026, 4, 15, tzinfo=timezone.utc)
        self.assertEqual(project_month_end(0, now), 0.0)

    def test_february_leap_year(self):
        now = datetime(2028, 2, 14, tzinfo=timezone.utc)  # 2028 is a leap year, Feb has 29 days
        self.assertAlmostEqual(project_month_end(14, now), 14.0 / 14 * 29)


class TestGetSemanticUsageEndpoint(unittest.IsolatedAsyncioTestCase):
    async def test_computes_remaining_and_projection(self):
        usage = {
            "quota": 1000,
            "allow_overage": False,
            "used_this_month": 500,
            "cost_so_far_usd": 10.0,
        }
        with (
            patch("api_svc.routers.orgs.get_org_semantic_usage", AsyncMock(return_value=usage)),
            patch("api_svc.routers.orgs.current_month", return_value="2026-04"),
            patch(
                "api_svc.routers.orgs.project_month_end",
                side_effect=lambda x: x * 2,  # deterministic stand-in for the real formula
            ),
        ):
            result = await get_semantic_usage(org_id="org-1")

        self.assertEqual(result.quota, 1000)
        self.assertEqual(result.used_this_month, 500)
        self.assertEqual(result.remaining, 500)
        self.assertFalse(result.allow_overage)
        self.assertEqual(result.projected_month_end_usage, 1000)
        self.assertAlmostEqual(result.projected_month_end_cost_usd, 20.0)

    async def test_remaining_never_negative_when_over_quota(self):
        usage = {
            "quota": 100,
            "allow_overage": True,
            "used_this_month": 150,
            "cost_so_far_usd": 3.0,
        }
        with patch("api_svc.routers.orgs.get_org_semantic_usage", AsyncMock(return_value=usage)):
            result = await get_semantic_usage(org_id="org-1")

        self.assertEqual(result.remaining, 0)
        self.assertTrue(result.allow_overage)

    async def test_org_id_passed_through_to_query(self):
        usage = {
            "quota": 1000,
            "allow_overage": False,
            "used_this_month": 0,
            "cost_so_far_usd": 0.0,
        }
        with patch(
            "api_svc.routers.orgs.get_org_semantic_usage", AsyncMock(return_value=usage)
        ) as query_mock:
            await get_semantic_usage(org_id="org-42")
        query_mock.assert_awaited_once()
        self.assertEqual(query_mock.call_args.args[0], "org-42")


if __name__ == "__main__":
    unittest.main()
