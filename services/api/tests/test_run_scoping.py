"""
Run-scoped reads must carry org_id, not just authorise on one table.

run_id is caller-supplied — the SDK exposes `run_id=` and the OTLP path derives
it from a caller-supplied trace id — so two tenants can hold the same one. Since
the composite (org_id, run_id) key they legitimately both exist, which makes an
unscoped `WHERE run_id = $1` return the other tenant's raw system prompts, tool
arguments and LLM output rather than merely the wrong row.

Run:
    PYTHONPATH=packages/schemas-py:packages/sdk-py:services/explainer:services/api \
      python -m pytest services/api/tests/test_run_scoping.py -v
"""

from __future__ import annotations

import inspect
import re
import unittest

import api_svc.db.queries as queries


def _sql_literals(source: str):
    return [m.group(1) for m in re.finditer(r'"""(.*?)"""', source, re.S)]


class TestGetRunDetailIsFullyScoped(unittest.TestCase):
    def setUp(self):
        self.source = inspect.getsource(queries.get_run_detail)

    def test_every_run_scoped_statement_also_filters_by_org(self):
        offenders = [
            " ".join(sql.split())[:120]
            for sql in _sql_literals(self.source)
            if re.search(r"\brun_id\s*=\s*\$\d", sql) and "org_id" not in sql
        ]
        self.assertEqual(offenders, [], f"unscoped run-id reads: {offenders}")

    def test_events_read_is_scoped(self):
        self.assertRegex(self.source, r"FROM events WHERE run_id = \$1 AND org_id = \$2")

    def test_signals_read_is_scoped(self):
        self.assertRegex(self.source, r"WHERE run_id = \$1 AND org_id = \$2")

    def test_runs_read_is_scoped(self):
        self.assertRegex(self.source, r"FROM runs WHERE run_id = \$1 AND org_id = \$2")


class TestNoUnscopedRunIdAnywhereInTheQueryLayer(unittest.TestCase):
    """A guard for the whole module, so the idiom cannot come back."""

    def test_no_sql_literal_filters_on_run_id_without_org_id(self):
        source = inspect.getsource(queries)
        offenders = [
            " ".join(sql.split())[:120]
            for sql in _sql_literals(source)
            if re.search(r"\brun_id\s*=\s*\$\d", sql) and "org_id" not in sql
        ]
        self.assertEqual(offenders, [], f"unscoped run-id reads: {offenders}")


if __name__ == "__main__":
    unittest.main()
