"""
The shared P75 baseline SQL (dunetrace_schemas.baselines) — string builders
and the row-to-response mapping. The statements are executed against a real
Postgres by the services that use them; here we pin the contract both
services rely on: the placeholder order, the clean-run filter, one window per
column, and the per-column minimum-sample rule.

Run:
    PYTHONPATH=packages/schemas-py python -m pytest packages/schemas-py/tests/test_baselines.py -v
"""

from __future__ import annotations

import re
import unittest

from dunetrace_schemas.baselines import (
    BASELINE_COLUMNS,
    BASELINE_RESPONSE_KEYS,
    MIN_BASELINE_RUNS,
    all_metric_baselines_sql,
    baselines_from_row,
    latest_agent_version_sql,
    metric_baseline_sql,
    recent_clean_metric_cte,
)


def _norm(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip()


class TestBuilders(unittest.TestCase):
    def test_every_column_has_a_response_key(self):
        self.assertEqual(set(BASELINE_COLUMNS), set(BASELINE_RESPONSE_KEYS))
        self.assertEqual(len(BASELINE_COLUMNS), 7)

    def test_unknown_column_is_refused(self):
        """The column name is interpolated, so the closed set is a safety rule."""
        with self.assertRaises(ValueError):
            metric_baseline_sql("step_count; DROP TABLE runs")
        with self.assertRaises(ValueError):
            recent_clean_metric_cte("recorded_at", "x")

    def test_single_column_sql_has_clean_filter_and_five_placeholders(self):
        sql = _norm(metric_baseline_sql("step_count"))
        for fragment in (
            "FROM run_baseline_metrics",
            "org_id = $1",
            "agent_id = $2",
            "agent_version = $3",
            "run_id != $4",
            "AND clean",
            "step_count IS NOT NULL",
            "ORDER BY recorded_at DESC LIMIT $5",
            "PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY v) AS p75",
            "COUNT(*) AS sample_size",
        ):
            self.assertIn(fragment, sql)
        self.assertNotIn("$6", sql)

    def test_all_columns_sql_samples_each_column_in_its_own_window(self):
        sql = all_metric_baselines_sql()
        norm = _norm(sql)
        for column in BASELINE_COLUMNS:
            self.assertIn(f"r_{column} AS ( SELECT {column} AS v", norm)
            self.assertIn(f"AND {column} IS NOT NULL", norm)
            self.assertIn(f"AS {column}_n", norm)
            self.assertIn(f"AS {column}_p75", norm)
        # One LIMIT per column — every metric takes the newest N runs that carry it.
        self.assertEqual(norm.count("LIMIT $5"), len(BASELINE_COLUMNS))
        self.assertEqual(norm.count("AND clean"), len(BASELINE_COLUMNS))
        self.assertNotIn("$6", norm)

    def test_all_columns_window_matches_single_column_window(self):
        """The aggregate must select runs exactly as the per-column read does."""
        single = _norm(recent_clean_metric_cte("total_tokens", "recent"))
        multi = _norm(recent_clean_metric_cte("total_tokens", "r_total_tokens"))
        self.assertEqual(single.replace("recent AS", "r_total_tokens AS"), multi)
        self.assertIn(multi, _norm(all_metric_baselines_sql()))

    def test_latest_version_sql_is_org_scoped(self):
        sql = _norm(latest_agent_version_sql())
        self.assertIn("org_id = $1 AND agent_id = $2", sql)
        self.assertIn("ORDER BY recorded_at DESC LIMIT 1", sql)


class TestBaselinesFromRow(unittest.TestCase):
    def _row(self, **per_column):
        row = {}
        for column in BASELINE_COLUMNS:
            n, p75 = per_column.get(column, (0, None))
            row[f"{column}_n"] = n
            row[f"{column}_p75"] = p75
        return row

    def test_none_row_is_none(self):
        self.assertIsNone(baselines_from_row(None))

    def test_no_history_is_none_not_dict_of_nones(self):
        self.assertIsNone(baselines_from_row(self._row()))

    def test_each_column_judged_on_its_own_sample_size(self):
        row = self._row(
            step_count=(MIN_BASELINE_RUNS, 12.0),
            total_tokens=(MIN_BASELINE_RUNS - 1, 5000.0),  # too few — None
            duration_s=(40, 3.5),
        )
        result = baselines_from_row(row)
        self.assertEqual(result["p75_steps"], 12.0)
        self.assertIsNone(result["p75_total_tokens"])
        self.assertEqual(result["p75_duration_s"], 3.5)
        self.assertIsNone(result["p75_latency_tool_ms"])
        self.assertEqual(set(result), set(BASELINE_RESPONSE_KEYS.values()))

    def test_values_are_floats(self):
        from decimal import Decimal

        row = self._row(step_count=(25, Decimal("11.5")))
        self.assertEqual(
            baselines_from_row(row),
            {**{k: None for k in BASELINE_RESPONSE_KEYS.values()}, "p75_steps": 11.5},
        )
        self.assertIs(type(baselines_from_row(row)["p75_steps"]), float)

    def test_min_runs_override(self):
        row = self._row(step_count=(3, 9.0))
        self.assertIsNone(baselines_from_row(row))
        self.assertEqual(baselines_from_row(row, min_runs=3)["p75_steps"], 9.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
