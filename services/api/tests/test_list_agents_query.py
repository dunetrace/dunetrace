"""
list_agents' SQL shape — the aggregate that made the dashboard's first request slow.

`/v1/agents` is on the critical path: nothing else on the dashboard starts until
it returns, and it was taking 2.5s warm (11s under load). Two causes, both in
this one function:

  1. `COUNT(DISTINCT run_id) ... GROUP BY agent_id` over `events`. Reading the
     rows is not the cost — making the planner sort all 149,907 of them by
     (agent_id, run_id) is. It chose an external merge and spilled 10MB to disk:
     743ms. Collapsing to one row per (agent, run) first lets both levels hash,
     for the identical numbers in 152ms with no spill.
  2. A separate `COUNT(DISTINCT agent_id)` scan of the whole events table just to
     fill `page.total`. `event_agg` already holds exactly one row per agent, so a
     window function reads the total off the scan that is already running.

Equivalence was checked against the live database with a FULL JOIN of the old and
new forms: 69 agents, 0 rows differing on either run_count or last_seen.

Source guard — no DB needed, and it cannot be satisfied by a query that happens
to hit empty tables.

Run:
  PYTHONPATH=packages/schemas-py:packages/sdk-py:services/explainer:services/api \
    python -m pytest services/api/tests/test_list_agents_query.py -v
"""

from __future__ import annotations

import inspect
import re
import unittest

import api_svc.db.queries as queries


def _strip_comments(source: str) -> str:
    """Drop SQL (`--`) and Python (`#`) comment lines.

    The comments in this function explain the very forms the guards below reject,
    so matching raw source would fail on the prose rather than the code. Testing
    the executable text is also the point: a comment mentioning
    COUNT(DISTINCT run_id) is documentation, not a regression.
    """
    out = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("--") or stripped.startswith("#"):
            continue
        out.append(line)
    return "\n".join(out)


class TestListAgentsAggregate(unittest.TestCase):
    def setUp(self):
        self.raw = inspect.getsource(queries.list_agents)
        self.source = _strip_comments(self.raw)

    def test_no_count_distinct_run_id(self):
        """The form that forced the sort-and-spill. Reintroducing it is a 4.3x
        regression on the dashboard's first request."""
        self.assertNotRegex(
            self.source,
            r"COUNT\s*\(\s*DISTINCT\s+run_id",
            "COUNT(DISTINCT run_id) makes the planner sort every event row; "
            "collapse to one row per (agent_id, run_id) and COUNT(*) instead",
        )

    def test_collapses_per_agent_and_run_before_counting(self):
        self.assertRegex(self.source, r"GROUP BY agent_id,\s*run_id")
        self.assertRegex(self.source, r"COUNT\(\*\)\s+AS run_count")

    def test_total_comes_from_the_same_scan(self):
        """Not a second COUNT(DISTINCT agent_id) pass over events."""
        self.assertRegex(self.source, r"COUNT\(\*\)\s*OVER\s*\(\s*\)\s*AS _total")

    def test_the_only_remaining_distinct_agent_count_is_the_empty_page_fallback(self):
        """An offset past the end returns no row to read the window value from,
        so that one case still needs a direct count — but only that case."""
        occurrences = re.findall(r"COUNT\(DISTINCT agent_id\)", self.source)
        self.assertEqual(len(occurrences), 1, "expected exactly one, in the empty-page branch")
        self.assertIn("if rows:", self.raw)
        self.assertIn('total = int(rows[0]["_total"])', self.raw)

    def test_internal_total_column_is_not_leaked_to_callers(self):
        """_total rides along on every row; the response model has no such field."""
        self.assertIn('if k != "_total"', self.raw)

    def test_fallback_runs_inside_the_connection_scope(self):
        """The fallback awaits on `conn`, so it must sit inside the `async with`
        that acquired it — outside, the connection is already back in the pool."""
        body = self.source[self.source.index("async with _pool.acquire()") :]
        fallback = body.index("COUNT(DISTINCT agent_id)")
        dedent = body.index("\n    return ")
        self.assertLess(fallback, dedent, "fallback escaped the connection scope")

    def test_still_scoped_by_org(self):
        for stmt in re.findall(r'"""(.*?)"""', self.raw, re.S):
            if "FROM events" in stmt or "FROM failure_signals" in stmt:
                self.assertIn("org_id = $1", stmt)


if __name__ == "__main__":
    unittest.main()
