"""
prompt_tokens must be OVERRIDDEN across the paired LLM events, never summed.

The same LLM call reports prompt_tokens twice: `llm.called` carries the SDK's
chars//4 estimate at call time, `llm.responded` the exact count once the
provider reports it. CLAUDE.md states the rule outright — "consumers must
override, not sum" — and the canonical run builder does exactly that.

Four query-layer aggregates summed them instead. Measured against a real
database, agent "agent" (12,342 runs, 30d) reported 399,504 prompt tokens where
156,970 were used — 152% over — and the Trends cost chart therefore disagreed
with the run-detail page about the very same runs.

The subtlety that let it survive one round of fixing: a
`SUM(CASE WHEN responded THEN x WHEN called THEN x END)` looks like a choice but
is not one. CASE picks a value per ROW; SUM then adds across rows, so a run with
both events still gets estimate + exact. The override has to happen between two
aggregates — `COALESCE(NULLIF(SUM(responded), 0), SUM(called))` — not inside one.

This is a source guard in the style of test_run_scoping.py: it reads the SQL
literals in the query layer, so it needs no database and cannot be satisfied by
a passing query that happens to hit empty tables.

Run:
  PYTHONPATH=packages/schemas-py:packages/sdk-py:services/explainer:services/api \
    python -m pytest services/api/tests/test_prompt_token_double_count.py -v
"""

from __future__ import annotations

import inspect
import re
import unittest

import api_svc.db.queries as queries

_SQL = re.compile(r'"""(.*?)"""', re.S)

# An aggregate that ADDS prompt_tokens without discriminating the event type.
# `SUM(COALESCE((payload->>'prompt_tokens')...))` with no CASE inside is the
# shape that double-counts.
_NAKED_SUM = re.compile(r"SUM\(\s*COALESCE\(\s*\(\s*[\w.]*payload\s*->>\s*'prompt_tokens'", re.I)
# The override marker: an outer COALESCE/NULLIF choosing between two SUMs.
_OVERRIDE = re.compile(r"NULLIF\(\s*SUM\(", re.I)


def _defers_override_to_python(sql: str) -> bool:
    """The other correct shape: select the two totals under distinct names and
    override in the pure layer (agent_performance_trends does this, so the
    arithmetic stays unit-testable without a database). Correct as long as the
    two are kept apart in SQL — which is what this checks."""
    return "prompt_tokens_responded" in sql and "prompt_tokens_called" in sql


def _sql_literals(source: str):
    return [m.group(1) for m in _SQL.finditer(source)]


def _spans_both_events(sql: str) -> bool:
    return "'llm.called'" in sql and "'llm.responded'" in sql


def _summing_statements(source: str):
    """SQL literals that SUM prompt_tokens across both paired event types."""
    out = []
    for sql in _sql_literals(source):
        if not _spans_both_events(sql) or "prompt_tokens" not in sql:
            continue
        if "SUM(" not in sql.upper():
            continue
        out.append(sql)
    return out


class TestNoNakedPromptTokenSum(unittest.TestCase):
    """Module-wide, so the idiom cannot come back a third time."""

    def setUp(self):
        self.source = inspect.getsource(queries)

    def test_no_sql_literal_sums_prompt_tokens_across_both_event_types(self):
        offenders = []
        for sql in _summing_statements(self.source):
            if _NAKED_SUM.search(sql):
                offenders.append(" ".join(sql.split())[:160])
        self.assertEqual(
            offenders,
            [],
            "these SUM prompt_tokens over llm.called AND llm.responded without "
            f"discriminating the event type, which double-counts: {offenders}",
        )

    def test_every_such_statement_uses_the_two_aggregate_override(self):
        offenders = [
            " ".join(sql.split())[:160]
            for sql in _summing_statements(self.source)
            if not _OVERRIDE.search(sql) and not _defers_override_to_python(sql)
        ]
        self.assertEqual(
            offenders,
            [],
            "a statement that totals prompt_tokens across both paired events must "
            "either use COALESCE(NULLIF(SUM(responded),0), SUM(called)) or select "
            f"the two totals separately and override in Python: {offenders}",
        )

    def test_the_guard_actually_matches_the_shape_it_is_guarding(self):
        """Without this, a typo in the regex makes every assertion above vacuous."""
        bad = """
            SELECT run_id,
                   SUM(COALESCE((payload->>'prompt_tokens')::int, 0)) AS prompt_tokens
            FROM events WHERE event_type IN ('llm.called', 'llm.responded')
        """
        self.assertTrue(_spans_both_events(bad))
        self.assertTrue(_NAKED_SUM.search(bad), "regex no longer detects the naked sum")
        self.assertFalse(_OVERRIDE.search(bad))

    def test_the_guard_accepts_the_corrected_shape(self):
        good = """
            SELECT run_id,
                   COALESCE(
                       NULLIF(SUM(CASE WHEN event_type = 'llm.responded'
                                       THEN COALESCE((payload->>'prompt_tokens')::int, 0)
                                       ELSE 0 END), 0),
                       SUM(CASE WHEN event_type = 'llm.called'
                                THEN COALESCE((payload->>'prompt_tokens')::int, 0)
                                ELSE 0 END)
                   ) AS prompt_tokens
            FROM events WHERE event_type IN ('llm.called', 'llm.responded')
        """
        self.assertTrue(_spans_both_events(good))
        self.assertFalse(_NAKED_SUM.search(good))
        self.assertTrue(_OVERRIDE.search(good))

    def test_the_previously_accepted_case_form_is_rejected(self):
        """The shape that shipped as a 'fix' and still added. It has a CASE, so a
        naive 'must contain CASE' check would pass it — the guard must not."""
        subtly_bad = """
            SELECT SUM(
                       CASE WHEN e.event_type = 'llm.responded'
                                 AND (e.payload->>'prompt_tokens') IS NOT NULL
                            THEN COALESCE((e.payload->>'prompt_tokens')::integer, 0)
                            WHEN e.event_type = 'llm.called'
                            THEN COALESCE((e.payload->>'prompt_tokens')::integer, 0)
                            ELSE 0 END
                   ) AS prompt_tokens
            FROM events WHERE e.event_type IN ('llm.called', 'llm.responded')
        """
        self.assertTrue(_spans_both_events(subtly_bad))
        self.assertFalse(
            _OVERRIDE.search(subtly_bad),
            "the CASE-inside-SUM form must not satisfy the override check",
        )


class TestKnownCallSitesAreCovered(unittest.TestCase):
    """Pins the specific functions that carried the bug, so a rename or a new
    copy does not quietly fall outside the module-wide scan."""

    FIXED = ("list_runs", "agent_cost_stats", "agent_token_stats")

    def test_each_fixed_function_uses_the_override(self):
        for name in self.FIXED:
            with self.subTest(function=name):
                source = inspect.getsource(getattr(queries, name))
                stmts = _summing_statements(source)
                self.assertTrue(stmts, f"{name} no longer totals prompt_tokens — update this test")
                for sql in stmts:
                    self.assertTrue(
                        _OVERRIDE.search(sql),
                        f"{name} lost the override: {' '.join(sql.split())[:160]}",
                    )
                    self.assertFalse(_NAKED_SUM.search(sql), f"{name} regressed to a naked SUM")

    def test_performance_trends_keeps_the_two_sums_apart(self):
        """agent_performance_trends does the override in Python (the pure layer),
        so it selects the two totals separately instead."""
        source = inspect.getsource(queries.agent_performance_trends)
        stmts = _summing_statements(source)
        self.assertTrue(stmts, "trends no longer totals prompt_tokens — update this test")
        for sql in stmts:
            self.assertTrue(_defers_override_to_python(sql), "trends stopped splitting the totals")
            self.assertFalse(_NAKED_SUM.search(sql), "trends regressed to a naked SUM")
        # ...and the pure layer it defers to still overrides.
        from api_svc.performance_trends import run_prompt_tokens

        self.assertEqual(run_prompt_tokens(responded=12, called=1), 12)


if __name__ == "__main__":
    unittest.main()
