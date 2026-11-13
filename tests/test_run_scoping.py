"""
Cross-service guard: no SQL reads a run without also filtering by org_id.

`run_id` is caller-supplied — the SDK exposes `run_id=`, and the OTLP path
derives it from a caller-supplied trace id — so two tenants legitimately hold
the same one. That is why `runs` and `processed_runs` are keyed
`(org_id, run_id)`. A read that filters on `run_id` alone returns the other
tenant's rows.

This file exists because the previous guard did not cover the ground it was
written for. `services/api/tests/test_run_scoping.py` inspects exactly one
module — the customer API's query layer — so the detector, semantic and
integrations query layers were never checked, and every one of them had an
unscoped read. Its pattern also matched only the literal `run_id = $N` form,
so `run_id = ANY($1::text[])` slipped past inside the one module it did scan.
Both holes are fixed here: every service, and every comparison form.

Deliberately source-text based rather than import based, so it needs no
service on the PYTHONPATH and runs from the repo root under
`make test-schema-parity`.

Run: python -m pytest tests/test_run_scoping.py -v
"""

from __future__ import annotations

import pathlib
import re
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]

# The query layers that read run-scoped data. Adding a service here is cheap;
# leaving one out is how this class of bug survived a guard that existed.
QUERY_MODULES = [
    "services/api/api_svc/db/queries.py",
    "services/detector/detector_svc/db.py",
    "services/semantic/semantic_svc/db.py",
    "services/integrations/integrations_svc/db.py",
    "services/alerts/alerts_svc/db.py",
    "services/ingest/ingest_svc/db/postgres.py",
]

# Every way a statement can compare run_id to a bound parameter:
#   run_id = $1
#   e.run_id = $2
#   run_id = ANY($1::text[])        <- the form the voice queries used
#   run_id IN ($1, $2)
_RUN_ID_PREDICATE = re.compile(
    r"\b(?:\w+\.)?run_id\s*(?:=|\bIN\b)\s*(?:ANY\s*)?\(?\s*\$\d",
    re.IGNORECASE,
)

# A correlated join between two tables on run_id alone, e.g.
#   JOIN events e ON e.run_id = r.run_id
#   WHERE p.run_id = e.run_id
# These are what let one tenant's processed_runs row suppress another's run.
_RUN_ID_CORRELATION = re.compile(r"\b(\w+)\.run_id\s*=\s*(\w+)\.run_id", re.IGNORECASE)

_SQL_LITERAL = re.compile(r'"""(.*?)"""', re.S)

# Statements that legitimately carry no org_id, each with the reason. Matched
# as a substring of the normalised SQL. Keep this list short and justified —
# it is the escape hatch that makes the guard honest, not a place to park
# inconvenient failures.
ALLOWED = [
    # The org_id backfills exist precisely to populate org_id from events, so
    # they cannot filter on the column they are filling.
    "SET org_id =",
    "org_id IS NULL",
    # Retention prune: deletes processed_runs rows whose events are gone
    # entirely. The NOT EXISTS is org-agnostic on purpose and errs toward
    # keeping rows, so it can only under-delete, never cross tenants.
    "FROM events e WHERE e.run_id = pr.run_id",
    "FROM events e WHERE e.run_id = p.run_id",
]


def _normalise(sql: str) -> str:
    # Strip -- comments first: they routinely mention run_id in prose and would
    # otherwise both trip the scan and bury the real predicate.
    return " ".join(re.sub(r"--[^\n]*", "", sql).split())


def _is_update_target(before: str) -> bool:
    """True when the predicate that follows is an UPDATE's SET assignment.

    `UPDATE t SET run_id = $3 ... WHERE id = $1` writes the column; it is not a
    tenant filter. Decided by looking back for an UPDATE with no intervening
    WHERE, which is the only shape that puts run_id on the left of an
    assignment in these modules.
    """
    upd = None
    for m in re.finditer(r"\bUPDATE\b", before, re.IGNORECASE):
        upd = m
    if upd is None:
        return False
    return re.search(r"\bWHERE\b", before[upd.end() :], re.IGNORECASE) is None


def _read_predicates_only(sql: str) -> str:
    """The part of a statement where run_id can act as a tenant filter.

    In an UPDATE, `SET ... run_id = $3` assigns the column; it is a write
    target, not a read predicate, so only the WHERE clause is scannable.
    """
    if re.match(r"^\s*UPDATE\b", sql, re.IGNORECASE):
        where = re.search(r"\bWHERE\b", sql, re.IGNORECASE)
        return sql[where.start() :] if where else ""
    return sql


# How far either side of a run_id predicate to look for its org_id when the
# statement is not one triple-quoted block. Comfortably wider than the longest
# single statement in these modules, narrow enough not to borrow the org_id of
# a neighbouring query.
_WINDOW = 500


def _offending_statements(source: str) -> list[str]:
    """Every SQL statement in `source` that reads a run without an org filter.

    Two passes, because SQL is written two ways in this codebase. Triple-quoted
    blocks are checked whole. Statements assembled from ADJACENT single-line
    literals are invisible to that pass — several of the reads fixed in this
    round were exactly that shape — so a second pass windows around each
    predicate in the raw source. Comment lines are stripped first: they mention
    run_id in prose constantly.
    """
    out: list[str] = []
    seen: set[str] = set()

    for raw in _SQL_LITERAL.findall(source):
        sql = _normalise(raw)
        if "run_id" not in sql or "org_id" in sql:
            continue
        if any(allowed in sql for allowed in ALLOWED):
            continue
        scannable = _read_predicates_only(sql)
        if _RUN_ID_PREDICATE.search(scannable) or _RUN_ID_CORRELATION.search(scannable):
            snippet = sql[:160]
            if snippet not in seen:
                seen.add(snippet)
                out.append(snippet)

    # Pass 2: adjacent-literal SQL. Drop Python and SQL comments, then window.
    stripped = re.sub(r"^\s*#[^\n]*$", "", source, flags=re.M)
    stripped = re.sub(r"--[^\n]*", "", stripped)
    for m in _RUN_ID_PREDICATE.finditer(stripped):
        before = stripped[max(0, m.start() - _WINDOW) : m.start()]
        if _is_update_target(before):
            continue
        window = before + stripped[m.start() : m.end() + _WINDOW]
        if "org_id" in window:
            continue
        if any(allowed in " ".join(window.split()) for allowed in ALLOWED):
            continue
        snippet = " ".join(stripped[max(0, m.start() - 80) : m.end() + 80].split())[:160]
        if any(snippet in s or s in snippet for s in seen):
            continue
        seen.add(snippet)
        out.append(snippet)
    return out


class TestNoUnscopedRunReadsInAnyService(unittest.TestCase):
    def test_every_query_module_scopes_run_reads_by_org(self):
        offenders: dict[str, list[str]] = {}
        for rel in QUERY_MODULES:
            path = REPO / rel
            self.assertTrue(path.exists(), f"{rel} moved — update QUERY_MODULES")
            found = _offending_statements(path.read_text())
            if found:
                offenders[rel] = found
        self.assertEqual(
            offenders,
            {},
            "SQL reads a run without filtering by org_id — run_id is "
            f"caller-supplied and collides across tenants:\n{offenders}",
        )

    def test_the_guard_actually_catches_each_form(self):
        """A guard nobody has seen fail is a guard nobody can trust."""
        cases = {
            "plain": 'q = """SELECT x FROM events WHERE run_id = $1"""',
            "qualified": 'q = """SELECT x FROM events e WHERE e.run_id = $1"""',
            "any-array": 'q = """SELECT x FROM events WHERE run_id = ANY($1::text[])"""',
            "in-list": 'q = """SELECT x FROM events WHERE run_id IN ($1, $2)"""',
            "correlation": ('q = """SELECT 1 FROM processed_runs p WHERE p.run_id = e.run_id"""'),
        }
        for name, src in cases.items():
            with self.subTest(form=name):
                # At least one: a triple-quoted statement is legitimately seen
                # by both passes. The real assertion is that it is seen at all.
                self.assertGreaterEqual(len(_offending_statements(src)), 1)

    def test_the_guard_passes_a_properly_scoped_statement(self):
        scoped = 'q = """SELECT x FROM events WHERE run_id = $1 AND org_id = $2"""'
        self.assertEqual(_offending_statements(scoped), [])

    def test_adjacent_string_literal_sql_is_scanned_too(self):
        """The shape the previous guard could not see."""
        src = (
            "row = await conn.fetchrow(\n"
            '    "SELECT DISTINCT failure_type FROM failure_signals "\n'
            '    "WHERE run_id = $1",\n'
            "    run_id,\n"
            ")\n"
        )
        self.assertEqual(len(_offending_statements(src)), 1)
        scoped = src.replace("WHERE run_id = $1", "WHERE run_id = $1 AND org_id = $2")
        self.assertEqual(_offending_statements(scoped), [])

    def test_an_update_target_is_not_mistaken_for_a_read_predicate(self):
        """`SET run_id = $3` writes the column; it does not read across orgs."""
        upd = 'q = """UPDATE gen SET run_id = $3, agent_id = $4 WHERE id = $1"""'
        self.assertEqual(_offending_statements(upd), [])


if __name__ == "__main__":
    unittest.main()
