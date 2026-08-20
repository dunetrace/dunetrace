"""
Every ON CONFLICT target must match a real unique constraint.

Postgres resolves an ON CONFLICT column list against declared PRIMARY KEY /
UNIQUE constraints. A target that matches none raises
InvalidColumnReferenceError at execution time — not at deploy, not at import,
and not in any test that doesn't touch a live database.

That failure mode already bit this service twice. Migration 2
(composite_run_key) re-keyed `runs` and `processed_runs` to (org_id, run_id),
but two INSERTs kept `ON CONFLICT (run_id)`. Both raised on every call from the
moment the migration applied:

  - mark_run_processed  -> runs were never marked processed, so every completed
                           run in the poll window was re-detected on each cycle,
                           writing duplicate signals and re-alerting them.
  - upsert_run_and_conversation -> the runs registry stopped being written, so
                           GET /v1/runs/{id} reported "no signals" for runs that
                           had them. A false clean verdict.

Both were silent: each call site swallows the exception into a WARNING, and the
detector kept reporting healthy cycles throughout.

This is a static check — it parses the SQL and the DDL, so it runs with no
database and catches the mismatch at the moment someone edits either side.

Run:
    PYTHONPATH=packages/schemas-py:packages/sdk-py:services/detector \
      python -m pytest services/detector/tests/test_conflict_targets.py -v
"""

from __future__ import annotations

import inspect
import re
import unittest

import detector_svc.db as db
from dunetrace_schemas import migrations

# INSERT INTO <table> ... ON CONFLICT (<cols>) — the table name may be followed
# by a column list on the same or a later line, so scan forward to the target.
_INSERT_RE = re.compile(
    r"INSERT\s+INTO\s+([a-z_][a-z0-9_]*)(.*?)ON\s+CONFLICT\s*\(([^)]*)\)",
    re.I | re.S,
)
_PK_RE = re.compile(r"PRIMARY\s+KEY\s*\(([^)]*)\)", re.I)
_INLINE_PK_RE = re.compile(r"^\s*([a-z_][a-z0-9_]*)\s+[A-Z].*?\bPRIMARY\s+KEY\b", re.I | re.M)
_UNIQUE_RE = re.compile(r"\bUNIQUE\s*\(([^)]*)\)", re.I)
_CREATE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-z_][a-z0-9_]*)\s*\((.*?)\n\)",
    re.I | re.S,
)
# A constraint can also be (re)declared by a later ALTER — including ones issued
# from inside a function body rather than a module-level DDL constant, e.g. the
# org_id widening of issues' UNIQUE and the detector_watermarks pkey repair.
_ALTER_CONSTRAINT_RE = re.compile(
    r"ALTER\s+TABLE\s+([a-z_][a-z0-9_]*)\s+ADD\s+CONSTRAINT\s+\S+\s+"
    r"(?:PRIMARY\s+KEY|UNIQUE)\s*\(([^)]*)\)",
    re.I | re.S,
)
_CREATE_UNIQUE_IDX_RE = re.compile(
    r"CREATE\s+UNIQUE\s+INDEX\s+(?:CONCURRENTLY\s+)?(?:IF\s+NOT\s+EXISTS\s+)?\S+\s+"
    r"ON\s+([a-z_][a-z0-9_]*)\s*\(([^)]*)\)",
    re.I | re.S,
)


def _cols(raw: str) -> frozenset:
    return frozenset(c.strip().strip('"') for c in raw.split(",") if c.strip())


def _ddl_sources() -> str:
    """All DDL this service's tables can be declared by.

    The WHOLE module source, not just its UPPERCASE schema constants: some
    constraints are (re)declared by ALTERs issued from inside functions — the
    org_id widening of issues' UNIQUE lives in _backfill_org_id, and the
    detector_watermarks pkey repair in _WATERMARK_SCHEMA. Reading only the
    constants made this check report the former as a violation when the live
    database was in fact correct.
    """
    return "\n".join([inspect.getsource(db)] + [sql for _n, _name, sql in migrations.MIGRATIONS])


def _declared_unique_sets() -> dict:
    """table -> {frozenset(cols), ...} for every PRIMARY KEY / UNIQUE declared."""
    ddl = _ddl_sources()
    out: dict = {}
    for table, body in _CREATE_RE.findall(ddl):
        keys = out.setdefault(table.lower(), set())
        for m in _PK_RE.findall(body):
            keys.add(_cols(m))
        for m in _UNIQUE_RE.findall(body):
            keys.add(_cols(m))
        for col in _INLINE_PK_RE.findall(body):
            keys.add(frozenset({col.lower()}))
    for table, raw in _ALTER_CONSTRAINT_RE.findall(ddl):
        out.setdefault(table.lower(), set()).add(_cols(raw))
    for table, raw in _CREATE_UNIQUE_IDX_RE.findall(ddl):
        out.setdefault(table.lower(), set()).add(_cols(raw))
    return out


def _conflict_targets() -> list:
    """(function, table, cols) for every ON CONFLICT in this service's SQL."""
    found = []
    for name, obj in vars(db).items():
        if not callable(obj) or not hasattr(obj, "__code__"):
            continue
        try:
            src = inspect.getsource(obj)
        except (OSError, TypeError):
            continue
        for table, _mid, raw in _INSERT_RE.findall(src):
            if raw.strip():
                found.append((name, table.lower(), _cols(raw)))
    return found


class TestOnConflictTargetsMatchDeclaredConstraints(unittest.TestCase):
    def test_at_least_one_conflict_target_is_discovered(self):
        """Guards the parser itself — a regex that silently matches nothing
        would make every assertion below vacuously pass."""
        self.assertGreater(len(_conflict_targets()), 3)

    def test_every_conflict_target_matches_a_unique_constraint(self):
        declared = _declared_unique_sets()
        offenders = []
        for fn, table, cols in _conflict_targets():
            keys = declared.get(table)
            if keys is None:
                continue  # table declared by another service — not ours to assert
            if cols not in keys:
                offenders.append(
                    f"{fn}: ON CONFLICT {sorted(cols)} on `{table}`, "
                    f"which declares {[sorted(k) for k in keys]}"
                )
        self.assertEqual(
            offenders,
            [],
            "ON CONFLICT targets with no matching constraint:\n  " + "\n  ".join(offenders),
        )

    def test_run_keyed_tables_are_composite(self):
        """The specific regression: run_id is caller-supplied, so anything keyed
        on it alone is both a cross-tenant collision and a broken conflict
        target."""
        declared = _declared_unique_sets()
        for table in ("runs", "processed_runs"):
            self.assertIn(
                frozenset({"org_id", "run_id"}),
                declared.get(table, set()),
                f"`{table}` must be keyed (org_id, run_id)",
            )
        for fn, table, cols in _conflict_targets():
            if table in ("runs", "processed_runs"):
                self.assertEqual(
                    cols,
                    frozenset({"org_id", "run_id"}),
                    f"{fn} targets {sorted(cols)} on `{table}`",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
