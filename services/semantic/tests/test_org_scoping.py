"""
Every run-scoped read in semantic_svc.db must carry org_id.

run_id is caller-supplied — the SDK exposes `run_id=` and the OTLP path derives
it from a caller-supplied trace id — so two tenants legitimately hold the same
one. That is why `runs` and `processed_runs` are keyed (org_id, run_id), and
this worker's own `semantic_processed_runs` now is too.

Two concrete failures these guard:
  * fetch_run_events unscoped put the *other* tenant's system prompt, LLM output
    and tool content into the payload sent to the evaluator provider, and the
    reasoning quoting it landed in this tenant's failure_signals.evidence.
  * a run_id-keyed semantic_processed_runs let one tenant's sampling decision
    swallow another's via ON CONFLICT DO NOTHING, after which the anti-join in
    fetch_unevaluated_runs read that run as already decided — so it was never
    sampled, silently and permanently.

The reads are exercised against a tiny in-memory Postgres stand-in that really
applies the predicates in the SQL, so a query that drops org_id genuinely
returns the other tenant's rows and the test fails on the data, not on a
string match.

Run: make test-semantic
"""

from __future__ import annotations

import re
import unittest

import semantic_svc.db as db

# ── A very small SQL stand-in ──────────────────────────────────────────────────
#
# Supports exactly the predicate forms this module's single-table reads use:
# `col = $n`, `col = ANY($n::t[])`, `col = 'lit'`, `col IN ('a','b')` and
# `col IS NOT NULL`. Enough to prove that dropping an org_id predicate changes
# which rows come back.

_ANY_PARAM = re.compile(r"(?:\w+\.)?(\w+)\s*=\s*ANY\(\$(\d+)::\w+\[\]\)")
_EQ_PARAM = re.compile(r"(?:\w+\.)?(\w+)\s*=\s*\$(\d+)")
_EQ_LIT = re.compile(r"(?:\w+\.)?(\w+)\s*=\s*'([^']*)'")
_IN_LIT = re.compile(r"(?:\w+\.)?(\w+)\s+IN\s*\(\s*'")
_IN_BODY = re.compile(r"(?:\w+\.)?(\w+)\s+IN\s*\(([^)]*)\)")
_NOT_NULL = re.compile(r"(?:\w+\.)?(\w+)\s+IS\s+NOT\s+NULL")
_FROM = re.compile(r"FROM\s+(\w+)")


def _matches(sql: str, args: tuple, row: dict) -> bool:
    for col, n in _ANY_PARAM.findall(sql):
        if row.get(col) not in args[int(n) - 1]:
            return False
    for col, n in _EQ_PARAM.findall(sql):
        if row.get(col) != args[int(n) - 1]:
            return False
    for col, lit in _EQ_LIT.findall(sql):
        if row.get(col) != lit:
            return False
    if _IN_LIT.search(sql):
        for col, body in _IN_BODY.findall(sql):
            if row.get(col) not in re.findall(r"'([^']*)'", body):
                return False
    for col in _NOT_NULL.findall(sql):
        if row.get(col) is None:
            return False
    return True


class _Conn:
    def __init__(self, tables: dict[str, list[dict]]):
        self.tables = tables

    def _select(self, sql, args):
        table = _FROM.search(sql).group(1)
        return [r for r in self.tables.get(table, []) if _matches(sql, args, r)]

    async def fetch(self, sql, *args):
        return self._select(sql, args)

    async def fetchrow(self, sql, *args):
        rows = self._select(sql, args)
        return rows[0] if rows else None

    async def fetchval(self, sql, *args):
        rows = self._select(sql, args)
        if re.match(r"\s*SELECT\s+EXISTS", sql, re.I):
            return bool(rows)
        if not rows:
            return None
        col = re.match(r"\s*SELECT\s+([\w.]+)\s+FROM", sql, re.I).group(1).split(".")[-1]
        return rows[0][col]


class _Pool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        conn = self._conn

        class _Acquire:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *exc):
                return False

        return _Acquire()


class _PoolPatch:
    def __init__(self, tables: dict[str, list[dict]]):
        self.conn = _Conn(tables)

    def __enter__(self):
        self._orig = db._pool
        db._pool = _Pool(self.conn)
        return self.conn

    def __exit__(self, *exc):
        db._pool = self._orig
        return False


# Same caller-supplied run_id in two tenants — the whole point.
RUN = "run-42"
MINE, THEIRS = "org-mine", "org-theirs"


def _event(org_id, **over):
    row = {
        "org_id": org_id,
        "run_id": RUN,
        "event_type": "llm.responded",
        "agent_id": f"agent-{org_id}",
        "agent_version": "v1",
        "step_index": 0,
        "timestamp": 1.0,
        "payload": {"output": f"secret belonging to {org_id}"},
        "conversation_id": None,
    }
    row.update(over)
    return row


class TestFetchRunEventsIsOrgScoped(unittest.IsolatedAsyncioTestCase):
    """The evaluator prompt is built from these rows, so an unscoped read ships
    the other tenant's content to the LLM provider and into our evidence."""

    async def test_returns_only_this_orgs_events(self):
        tables = {"events": [_event(MINE), _event(THEIRS)]}
        with _PoolPatch(tables):
            rows = await db.fetch_run_events(MINE, RUN)

        self.assertEqual([r["org_id"] for r in rows], [MINE])
        self.assertNotIn(
            f"secret belonging to {THEIRS}",
            str(rows),
            "another tenant's LLM output reached the evaluator input",
        )


class TestSamplingProbesAreOrgScoped(unittest.IsolatedAsyncioTestCase):
    """A colliding run_id from another tenant must not decide whether this
    tenant's run is sampled — either direction wastes or withholds an LLM
    evaluation the customer is billed for."""

    async def test_structural_signal_probe_ignores_other_orgs(self):
        tables = {
            "failure_signals": [
                {"org_id": THEIRS, "run_id": RUN, "source": "structural"},
            ]
        }
        with _PoolPatch(tables):
            self.assertFalse(await db.has_structural_signal(MINE, RUN))
            self.assertTrue(await db.has_structural_signal(THEIRS, RUN))

    async def test_retrieval_probe_ignores_other_orgs(self):
        tables = {
            "events": [_event(THEIRS, event_type="retrieval.called")],
        }
        with _PoolPatch(tables):
            self.assertFalse(await db.has_retrieval_event(MINE, RUN))
            self.assertTrue(await db.has_retrieval_event(THEIRS, RUN))

    async def test_conversation_id_lookup_ignores_other_orgs(self):
        tables = {
            "events": [
                _event(THEIRS, conversation_id="conv-theirs"),
                _event(MINE, conversation_id="conv-mine"),
            ]
        }
        with _PoolPatch(tables):
            self.assertEqual(await db.fetch_run_conversation_id(MINE, RUN), "conv-mine")
            self.assertEqual(await db.fetch_run_conversation_id(THEIRS, RUN), "conv-theirs")


class TestProcessedRunsTableIsOrgKeyed(unittest.TestCase):
    """A run_id-only key silently dropped the second tenant's run forever."""

    def test_primary_key_is_composite(self):
        self.assertIn("PRIMARY KEY (org_id, run_id)", db._SEMANTIC_SCHEMA)
        self.assertNotRegex(db._SEMANTIC_SCHEMA, r"run_id\s+TEXT\s+PRIMARY KEY")

    def test_legacy_single_column_key_is_repaired(self):
        """Existing installs keep their rows: CREATE TABLE IF NOT EXISTS is a
        no-op there, so the widening has to be an explicit guarded ALTER."""
        schema = db._SEMANTIC_SCHEMA
        self.assertIn("semantic_processed_runs_pkey", schema)
        self.assertIn("i.indnatts = 1", schema)
        self.assertIn(
            "ADD CONSTRAINT semantic_processed_runs_pkey PRIMARY KEY (org_id, run_id)", schema
        )
        self.assertNotIn("DROP TABLE", schema)

    def test_mark_run_processed_conflicts_on_the_full_key(self):
        import inspect

        source = inspect.getsource(db.mark_run_processed)
        self.assertIn("ON CONFLICT (org_id, run_id) DO NOTHING", source)

    def test_unevaluated_anti_join_is_org_scoped(self):
        import inspect

        source = inspect.getsource(db.fetch_unevaluated_runs)
        self.assertIn("p.org_id = e.org_id", source)


class TestNoUnscopedRunIdInTheSemanticQueryLayer(unittest.TestCase):
    """Module-wide guard, mirroring api_svc's test_run_scoping.py, so the
    idiom cannot come back in a new query.

    Scans around every `run_id = $n` rather than only inside triple-quoted
    blocks: api_svc's version misses single-line SQL built from adjacent
    string literals, and three of the reads fixed here were exactly that
    shape."""

    def test_every_run_id_predicate_sits_near_an_org_id_one(self):
        import inspect

        source = inspect.getsource(db)
        offenders = []
        for m in re.finditer(r"\brun_id\s*=\s*\$\d", source):
            window = source[max(0, m.start() - 400) : m.end() + 400]
            if "org_id" not in window:
                offenders.append(" ".join(window[380:520].split()))
        self.assertEqual(offenders, [], f"unscoped run-id reads: {offenders}")


if __name__ == "__main__":
    unittest.main()
