"""
Tests for baseline fetch functions in detector_svc.db.
No real DB — asyncpg pool is mocked.

Run:
    cd services/detector
    PYTHONPATH=packages/sdk-py:services/detector \
        python -m pytest tests/test_baselines.py -v
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


# ── Pool helpers ───────────────────────────────────────────────────────────────


def _make_pool(row: dict | None) -> MagicMock:
    """Build a mock asyncpg pool whose fetchrow returns the given row dict."""
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=row)
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    return pool


def _row(sample_size: int, p75: float | None) -> dict:
    """Construct a fake fetchrow result matching the baseline query schema."""
    return {"sample_size": sample_size, "p75": p75}


# ── fetch_step_count_baseline ─────────────────────────────────────────────────


class TestFetchStepCountBaseline(unittest.IsolatedAsyncioTestCase):
    FUNC = staticmethod(db_module.fetch_step_count_baseline)
    ARGS = ("org-1", "agent-1", "v1", "run-excluded")

    async def test_returns_none_when_no_pool(self):
        """Without a pool the function must return None, not raise."""
        with patch.object(db_module, "_pool", None):
            result = await self.FUNC(*self.ARGS)
        self.assertIsNone(result)

    async def test_returns_none_on_empty_table(self):
        """Zero rows → sample_size=0 < min_runs → None."""
        pool = _make_pool(_row(0, None))
        with patch.object(db_module, "_pool", pool):
            result = await self.FUNC(*self.ARGS, min_runs=1)
        self.assertIsNone(result)

    async def test_returns_none_when_below_min_runs(self):
        """Fewer than min_runs qualifying runs → None."""
        pool = _make_pool(_row(5, 12.0))
        with patch.object(db_module, "_pool", pool):
            result = await self.FUNC(*self.ARGS, min_runs=20)
        self.assertIsNone(result)

    async def test_single_value_returns_that_value(self):
        """Single-run sample: P75 = the single step count value."""
        pool = _make_pool(_row(1, 7.0))
        with patch.object(db_module, "_pool", pool):
            result = await self.FUNC(*self.ARGS, min_runs=1)
        self.assertAlmostEqual(result, 7.0)

    async def test_all_identical_values_returns_that_value(self):
        """When every run has the same step count, P75 equals that count."""
        pool = _make_pool(_row(10, 5.0))  # Postgres computed P75=5.0
        with patch.object(db_module, "_pool", pool):
            result = await self.FUNC(*self.ARGS, min_runs=1)
        self.assertAlmostEqual(result, 5.0)

    async def test_normal_distribution_returns_float(self):
        """10 diverse values → P75 is a float returned by the DB."""
        pool = _make_pool(_row(10, 14.25))  # Postgres computed P75
        with patch.object(db_module, "_pool", pool):
            result = await self.FUNC(*self.ARGS, min_runs=1)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 14.25)

    async def test_returns_none_when_p75_is_none_despite_sample(self):
        """If Postgres returns NULL for P75 (empty sub-table), treat as None."""
        pool = _make_pool(_row(20, None))
        with patch.object(db_module, "_pool", pool):
            result = await self.FUNC(*self.ARGS, min_runs=1)
        self.assertIsNone(result)


# ── fetch_latency_baseline ────────────────────────────────────────────────────


class TestFetchLatencyBaseline(unittest.IsolatedAsyncioTestCase):
    FUNC = staticmethod(db_module.fetch_latency_baseline)
    ARGS = ("org-1", "agent-1", "v1", "run-excluded", "tool.called")

    async def test_returns_none_when_no_pool(self):
        with patch.object(db_module, "_pool", None):
            result = await self.FUNC(*self.ARGS)
        self.assertIsNone(result)

    async def test_returns_none_on_empty_table(self):
        pool = _make_pool(_row(0, None))
        with patch.object(db_module, "_pool", pool):
            result = await self.FUNC(*self.ARGS, min_runs=1)
        self.assertIsNone(result)

    async def test_single_value_returns_that_value(self):
        pool = _make_pool(_row(1, 500.0))
        with patch.object(db_module, "_pool", pool):
            result = await self.FUNC(*self.ARGS, min_runs=1)
        self.assertAlmostEqual(result, 500.0)

    async def test_all_identical_values_returns_that_value(self):
        pool = _make_pool(_row(25, 300.0))
        with patch.object(db_module, "_pool", pool):
            result = await self.FUNC(*self.ARGS, min_runs=1)
        self.assertAlmostEqual(result, 300.0)

    async def test_normal_distribution_returns_p75(self):
        pool = _make_pool(_row(20, 12500.0))
        with patch.object(db_module, "_pool", pool):
            result = await self.FUNC(*self.ARGS, min_runs=1)
        self.assertAlmostEqual(result, 12500.0)

    async def test_below_min_runs_returns_none(self):
        pool = _make_pool(_row(15, 400.0))
        with patch.object(db_module, "_pool", pool):
            result = await self.FUNC(*self.ARGS, min_runs=20)
        self.assertIsNone(result)


# ── fetch_token_growth_baseline ────────────────────────────────────────────────


class TestFetchTokenGrowthBaseline(unittest.IsolatedAsyncioTestCase):
    FUNC = staticmethod(db_module.fetch_token_growth_baseline)
    ARGS = ("org-1", "agent-1", "v1", "run-excluded")

    async def test_returns_none_when_no_pool(self):
        with patch.object(db_module, "_pool", None):
            result = await self.FUNC(*self.ARGS)
        self.assertIsNone(result)

    async def test_returns_none_on_empty_table(self):
        pool = _make_pool(_row(0, None))
        with patch.object(db_module, "_pool", pool):
            result = await self.FUNC(*self.ARGS, min_runs=1)
        self.assertIsNone(result)

    async def test_single_value_returns_that_value(self):
        pool = _make_pool(_row(1, 1.5))
        with patch.object(db_module, "_pool", pool):
            result = await self.FUNC(*self.ARGS, min_runs=1)
        self.assertAlmostEqual(result, 1.5)

    async def test_all_identical_values_returns_that_value(self):
        pool = _make_pool(_row(20, 2.0))
        with patch.object(db_module, "_pool", pool):
            result = await self.FUNC(*self.ARGS, min_runs=1)
        self.assertAlmostEqual(result, 2.0)

    async def test_normal_distribution(self):
        pool = _make_pool(_row(20, 3.2))
        with patch.object(db_module, "_pool", pool):
            result = await self.FUNC(*self.ARGS, min_runs=1)
        self.assertAlmostEqual(result, 3.2)


# ── fetch_duration_baseline ────────────────────────────────────────────────────


class TestFetchDurationBaseline(unittest.IsolatedAsyncioTestCase):
    FUNC = staticmethod(db_module.fetch_duration_baseline)
    ARGS = ("org-1", "agent-1", "v1", "run-excluded")

    async def test_returns_none_when_no_pool(self):
        with patch.object(db_module, "_pool", None):
            result = await self.FUNC(*self.ARGS)
        self.assertIsNone(result)

    async def test_returns_none_on_empty_table(self):
        pool = _make_pool(_row(0, None))
        with patch.object(db_module, "_pool", pool):
            result = await self.FUNC(*self.ARGS, min_runs=1)
        self.assertIsNone(result)

    async def test_single_value_returns_that_value(self):
        pool = _make_pool(_row(1, 30.0))
        with patch.object(db_module, "_pool", pool):
            result = await self.FUNC(*self.ARGS, min_runs=1)
        self.assertAlmostEqual(result, 30.0)

    async def test_all_identical_returns_that_value(self):
        pool = _make_pool(_row(20, 45.0))
        with patch.object(db_module, "_pool", pool):
            result = await self.FUNC(*self.ARGS, min_runs=1)
        self.assertAlmostEqual(result, 45.0)

    async def test_normal_distribution(self):
        pool = _make_pool(_row(20, 62.5))
        with patch.object(db_module, "_pool", pool):
            result = await self.FUNC(*self.ARGS, min_runs=1)
        self.assertAlmostEqual(result, 62.5)


# ── fetch_total_tokens_baseline ────────────────────────────────────────────────


class TestFetchTotalTokensBaseline(unittest.IsolatedAsyncioTestCase):
    FUNC = staticmethod(db_module.fetch_total_tokens_baseline)
    ARGS = ("org-1", "agent-1", "v1", "run-excluded")

    async def test_returns_none_when_no_pool(self):
        with patch.object(db_module, "_pool", None):
            result = await self.FUNC(*self.ARGS)
        self.assertIsNone(result)

    async def test_returns_none_on_empty_table(self):
        pool = _make_pool(_row(0, None))
        with patch.object(db_module, "_pool", pool):
            result = await self.FUNC(*self.ARGS, min_runs=1)
        self.assertIsNone(result)

    async def test_single_value_returns_that_value(self):
        pool = _make_pool(_row(1, 10000.0))
        with patch.object(db_module, "_pool", pool):
            result = await self.FUNC(*self.ARGS, min_runs=1)
        self.assertAlmostEqual(result, 10000.0)

    async def test_all_identical_returns_that_value(self):
        pool = _make_pool(_row(20, 8000.0))
        with patch.object(db_module, "_pool", pool):
            result = await self.FUNC(*self.ARGS, min_runs=1)
        self.assertAlmostEqual(result, 8000.0)

    async def test_normal_distribution(self):
        pool = _make_pool(_row(20, 15000.0))
        with patch.object(db_module, "_pool", pool):
            result = await self.FUNC(*self.ARGS, min_runs=1)
        self.assertAlmostEqual(result, 15000.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
