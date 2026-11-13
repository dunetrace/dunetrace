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


# ── SQL semantics ─────────────────────────────────────────────────────────────
#
# Every test above mocks fetchrow and asserts on the row it hands back, so none
# of them can see the query. These capture the SQL the function actually issues.
# They are drift guards, not a substitute for running the SQL — the semantics
# were verified against a real Postgres 16 when the predicates were written.


async def _capture_sql(func, *args, **kwargs) -> str:
    """Run a baseline function against a mock pool and return the SQL it issued."""
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=_row(0, None))
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    with patch.object(db_module, "_pool", pool):
        await func(*args, **kwargs)
    return conn.fetchrow.call_args[0][0]


_ALL_BASELINES = (
    ("step_count", db_module.fetch_step_count_baseline, ("org-1", "agent-1", "v1", "run-x")),
    (
        "latency",
        db_module.fetch_latency_baseline,
        ("org-1", "agent-1", "v1", "run-x", "tool.called"),
    ),
    ("token_growth", db_module.fetch_token_growth_baseline, ("org-1", "agent-1", "v1", "run-x")),
    (
        "llm_tool_ratio",
        db_module.fetch_llm_tool_ratio_baseline,
        ("org-1", "agent-1", "v1", "run-x"),
    ),
    ("total_tokens", db_module.fetch_total_tokens_baseline, ("org-1", "agent-1", "v1", "run-x")),
    ("duration", db_module.fetch_duration_baseline, ("org-1", "agent-1", "v1", "run-x")),
)


class TestBaselinesLearnFromCleanRunsOnly(unittest.IsolatedAsyncioTestCase):
    """A run that fired a live signal must not define what normal looks like.

    Without this a chronically looping agent raises its own step and token
    baselines until the loop reads as typical and the detector goes quiet.
    """

    async def test_every_baseline_excludes_runs_with_a_live_signal(self):
        for name, func, args in _ALL_BASELINES:
            with self.subTest(baseline=name):
                sql = await _capture_sql(func, *args)
                self.assertIn("NOT EXISTS", sql, f"{name} has no exclusion clause")
                self.assertIn("failure_signals fs", sql)
                self.assertIn("fs.shadow = FALSE", sql)

    async def test_every_baseline_still_excludes_errored_runs(self):
        """The pre-existing completed-only filter must survive the change —
        errored runs drag P75 down and cause false STEP_COUNT_INFLATION."""
        for name, func, args in _ALL_BASELINES:
            with self.subTest(baseline=name):
                sql = await _capture_sql(func, *args)
                self.assertIn("run.completed", sql)

    async def test_signal_exclusion_is_org_scoped(self):
        """run_id is caller-supplied and collides across tenants, so an
        unscoped exclusion would let another org's signal disqualify a run."""
        for name, func, args in _ALL_BASELINES:
            with self.subTest(baseline=name):
                sql = await _capture_sql(func, *args)
                self.assertIn("fs.org_id = pr.org_id", sql)

    async def test_every_event_join_is_org_scoped(self):
        """run_id is caller-supplied and collides across tenants, so a join on
        run_id alone lets another org's events feed this org's baseline."""
        import re

        for name, func, args in _ALL_BASELINES:
            with self.subTest(baseline=name):
                sql = await _capture_sql(func, *args)
                # Every reference to the events table must sit in a scope that
                # also constrains org_id.
                self.assertNotRegex(
                    sql,
                    r"JOIN events e ON e\.run_id = r\.run_id\s*\n",
                    "events join is not org-scoped",
                )
                for m in re.finditer(r"e\.run_id IN \(SELECT run_id FROM recent\)", sql):
                    window = sql[max(0, m.start() - 120) : m.end()]
                    self.assertIn("e.org_id", window, "IN-subquery is not org-scoped")

    async def test_completed_run_probe_is_org_scoped(self):
        """Unscoped, another tenant's run.completed marks this tenant's
        errored run as completed — and errored runs are exactly what the
        completed filter exists to keep out."""
        for name, func, args in _ALL_BASELINES:
            with self.subTest(baseline=name):
                sql = await _capture_sql(func, *args)
                probe = sql[sql.index("run.completed") - 200 : sql.index("run.completed")]
                self.assertIn("e.org_id = pr.org_id", probe)

    async def test_shadow_signals_do_not_disqualify_a_run(self):
        """Shadow detectors are unvalidated by definition. If a shadow signal
        shrank the baseline population, enabling one for evaluation would
        silently retune six unrelated live detectors."""
        for name, func, args in _ALL_BASELINES:
            with self.subTest(baseline=name):
                sql = await _capture_sql(func, *args)
                # Only shadow=FALSE rows disqualify; the clause must not match
                # every signal regardless of shadow.
                self.assertNotIn("fs.shadow = TRUE", sql)
                self.assertIn("fs.shadow = FALSE", sql)


class TestTokenGrowthMatchesTheDetector(unittest.IsolatedAsyncioTestCase):
    """ContextBloatDetector computes calls_with_tokens[-1] / [0]. The baseline
    it is compared against has to compute the same quantity, or the effective
    threshold is not the configured one."""

    async def _sql(self) -> str:
        return await _capture_sql(
            db_module.fetch_token_growth_baseline, "org-1", "agent-1", "v1", "run-x"
        )

    async def test_uses_positional_first_and_last_not_min_and_max(self):
        sql = await self._sql()
        self.assertIn("ARRAY_AGG(prompt_tokens ORDER BY step_index)", sql)
        self.assertIn("ARRAY_AGG(prompt_tokens ORDER BY step_index DESC)", sql)
        # MIN/MAX only agrees with first/last when context grows monotonically,
        # and is strictly larger whenever a run compacts mid-run.
        self.assertNotIn("MIN(prompt_tokens) AS first_tokens", sql)
        self.assertNotIn("MAX(prompt_tokens) AS last_tokens", sql)

    async def test_reads_prompt_tokens_from_both_event_types(self):
        """run_builder overrides the llm.called estimate with llm.responded's
        exact count. Reading llm.called alone built the baseline from estimates
        while the detector compared exacts."""
        sql = await self._sql()
        self.assertIn("'llm.called', 'llm.responded'", sql)

    async def test_prefers_responded_over_called_for_the_same_call(self):
        sql = await self._sql()
        self.assertIn("DISTINCT ON", sql)
        self.assertIn("(e.event_type = 'llm.responded') DESC", sql)

    async def test_guards_apply_after_first_last_are_resolved(self):
        """The detector's guards are on first/last, not on min/max — so they
        belong in an outer WHERE, not the grouping HAVING."""
        sql = await self._sql()
        self.assertIn("WHERE first_tokens >= 10", sql)
        self.assertIn("AND last_tokens  >= 2000", sql)
        self.assertNotIn("MIN(prompt_tokens) >= 10", sql)
        self.assertNotIn("MAX(prompt_tokens) >= 2000", sql)

    async def test_min_calls_guard_survives(self):
        sql = await self._sql()
        self.assertIn("HAVING COUNT(*) >= 3", sql)


class TestSharedCteIsActuallyShared(unittest.IsolatedAsyncioTestCase):
    """The run-selection CTE was copy-pasted into six queries, which is how the
    same predicate came to need fixing in six places. It now has one home."""

    async def test_all_six_render_the_same_run_selection_cte(self):
        rendered = set()
        for _name, func, args in _ALL_BASELINES:
            sql = await _capture_sql(func, *args)
            head = sql.split("),", 1)[0]
            rendered.add(head.strip())
        self.assertEqual(len(rendered), 1, "baseline queries no longer share one run-selection CTE")


# ── Durable metrics (retention independence) ─────────────────────────────────


class TestBuildBaselineMetrics(unittest.TestCase):
    """Metrics are computed from RunState — the same object the detectors read —
    so a stored metric cannot drift from the metric it is compared against."""

    def _state(self, **kw):
        from dunetrace.models import AgentEvent, EventType, LlmCall, RunState, ToolCall

        state = RunState(run_id="r1", agent_id="a1", agent_version="v1")
        for i, (etype, ts) in enumerate(kw.get("events", [])):
            state.events.append(
                AgentEvent(
                    event_type=EventType(etype),
                    run_id="r1",
                    agent_id="a1",
                    agent_version="v1",
                    step_index=i,
                    timestamp=ts,
                )
            )
        for tok in kw.get("prompt_tokens", []):
            state.llm_calls.append(
                LlmCall(
                    model="m",
                    prompt_tokens=tok,
                    finish_reason="stop",
                    latency_ms=0,
                    step_index=0,
                    timestamp=0.0,
                )
            )
        for i in range(kw.get("tool_calls", 0)):
            state.tool_calls.append(ToolCall(tool_name="t", args="{}", step_index=i, timestamp=0.0))
        return state

    def test_token_growth_is_positional_last_over_first(self):
        from detector_svc.baseline_metrics import build_baseline_metrics

        # Rises to 9000 then compacts to 3000. Positional last/first = 3.0;
        # MAX/MIN would be 9.0 — the bug this replaced.
        m = build_baseline_metrics(self._state(prompt_tokens=[1000, 9000, 3000]))
        self.assertAlmostEqual(m["token_growth"], 3.0)

    def test_token_growth_none_when_below_min_calls(self):
        from detector_svc.baseline_metrics import build_baseline_metrics

        m = build_baseline_metrics(self._state(prompt_tokens=[1000, 5000]))
        self.assertIsNone(m["token_growth"], "a run the detector skips must not feed its baseline")

    def test_token_growth_none_when_last_below_min_tokens(self):
        from detector_svc.baseline_metrics import build_baseline_metrics

        m = build_baseline_metrics(self._state(prompt_tokens=[10, 50, 100]))
        self.assertIsNone(m["token_growth"])

    def test_token_growth_none_when_first_below_floor(self):
        from detector_svc.baseline_metrics import build_baseline_metrics

        m = build_baseline_metrics(self._state(prompt_tokens=[5, 1000, 5000]))
        self.assertIsNone(m["token_growth"])

    def test_llm_tool_ratio_none_below_min_llm_calls(self):
        from detector_svc.baseline_metrics import build_baseline_metrics

        m = build_baseline_metrics(self._state(prompt_tokens=[100] * 4, tool_calls=1))
        self.assertIsNone(m["llm_tool_ratio"])

    def test_llm_tool_ratio_divides_by_one_when_no_tools(self):
        from detector_svc.baseline_metrics import build_baseline_metrics

        m = build_baseline_metrics(self._state(prompt_tokens=[100] * 6, tool_calls=0))
        self.assertAlmostEqual(m["llm_tool_ratio"], 6.0)

    def test_duration_none_for_a_single_event(self):
        from detector_svc.baseline_metrics import build_baseline_metrics

        m = build_baseline_metrics(self._state(events=[("run.started", 5.0)]))
        self.assertIsNone(m["duration_s"])

    def test_gap_is_time_until_the_next_event(self):
        from detector_svc.baseline_metrics import build_baseline_metrics

        m = build_baseline_metrics(
            self._state(events=[("tool.called", 0.0), ("tool.responded", 2.0)])
        )
        self.assertAlmostEqual(m["gap_p75_tool_ms"], 2000.0)

    def test_p75_matches_postgres_percentile_cont(self):
        """The cross-run aggregate uses PERCENTILE_CONT, so a run's own
        contribution has to be computed the same way."""
        from detector_svc.baseline_metrics import _p75

        self.assertAlmostEqual(_p75([1, 2, 3, 4]), 3.25)
        self.assertAlmostEqual(_p75([10]), 10.0)
        self.assertIsNone(_p75([]))


class TestFetchMetricBaseline(unittest.IsolatedAsyncioTestCase):
    ARGS = ("org-1", "agent-1", "v1", "run-x")

    async def test_rejects_a_column_outside_the_whitelist(self):
        """The column is interpolated into SQL because asyncpg cannot bind an
        identifier, so the whitelist is the injection boundary."""
        pool = _make_pool(_row(50, 9.0))
        with patch.object(db_module, "_pool", pool):
            result = await db_module.fetch_metric_baseline(
                *self.ARGS, "step_count; DROP TABLE events", min_runs=1
            )
        self.assertIsNone(result)

    async def test_reads_only_clean_runs(self):
        sql = await _capture_sql(db_module.fetch_metric_baseline, *self.ARGS, "step_count")
        self.assertIn("AND clean", sql)
        self.assertIn("run_baseline_metrics", sql)

    async def test_skips_rows_where_the_metric_is_null(self):
        """NULL means the run didn't qualify for THAT baseline; it must be
        skipped, not counted, or the sample size lies."""
        sql = await _capture_sql(db_module.fetch_metric_baseline, *self.ARGS, "token_growth")
        self.assertIn("token_growth IS NOT NULL", sql)

    async def test_has_no_age_bound(self):
        """The whole point: a low-traffic agent's history must survive past
        EVENT_RETENTION_DAYS, so nothing here may filter on age."""
        sql = await _capture_sql(db_module.fetch_metric_baseline, *self.ARGS, "step_count")
        self.assertNotIn("INTERVAL", sql.upper())
        self.assertNotIn("NOW()", sql.upper())


class TestBaselineFallsBackToEvents(unittest.IsolatedAsyncioTestCase):
    """Transitional: an existing deployment must keep its baselines on the
    deploy that introduces the table, not drop to static thresholds for 20 runs."""

    async def test_uses_stored_metrics_when_they_suffice(self):
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=_row(50, 7.0))
        pool = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch.object(db_module, "_pool", pool):
            result = await db_module.fetch_step_count_baseline("o", "a", "v", "r")
        self.assertAlmostEqual(result, 7.0)
        self.assertEqual(conn.fetchrow.await_count, 1, "should not have consulted events")

    async def test_falls_back_to_the_events_query_when_stored_is_short(self):
        conn = AsyncMock()
        # first call: metrics table has too few rows; second: legacy query
        conn.fetchrow = AsyncMock(side_effect=[_row(3, 5.0), _row(50, 11.0)])
        pool = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch.object(db_module, "_pool", pool):
            result = await db_module.fetch_step_count_baseline("o", "a", "v", "r")
        self.assertAlmostEqual(result, 11.0)
        self.assertEqual(conn.fetchrow.await_count, 2)

    async def test_latency_picks_the_column_matching_the_event_type(self):
        for event_type, column in (
            ("tool.called", "gap_p75_tool_ms"),
            ("llm.called", "gap_p75_llm_ms"),
        ):
            with self.subTest(event_type=event_type):
                conn = AsyncMock()
                conn.fetchrow = AsyncMock(return_value=_row(50, 1.0))
                pool = MagicMock()
                pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
                pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
                with patch.object(db_module, "_pool", pool):
                    await db_module.fetch_latency_baseline("o", "a", "v", "r", event_type)
                self.assertIn(column, conn.fetchrow.call_args[0][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
