"""
The voice call reads must carry org_id, not authorise on `conversations` and
then fan out on bare run ids.

run_id is caller-supplied — the SDK exposes `run_id=` and the OTLP path derives
it from a caller-supplied trace id — so two tenants legitimately hold the same
one, which is why `runs` is keyed (org_id, run_id). Authorising the
conversation by org and then reading `FROM events WHERE run_id = ANY($1)` put
the other tenant's transcription.received (caller speech), tts.generated
(spoken text), llm.called/llm.responded payloads and recording.available URLs
into this tenant's call timeline, recordings list and cost breakdown. The
failure_signals half of the same function was already scoped; the events half
was not.

Exercised against a tiny in-memory Postgres stand-in that really applies the
predicates in the SQL, so a query that drops org_id genuinely returns the other
tenant's rows and the test fails on the data, not on a string match.

Run: make test-api
"""

from __future__ import annotations

import datetime
import inspect
import re
import unittest

import api_svc.db.queries as q

UTC = datetime.timezone.utc
T0 = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

# One caller-supplied run id, held by both tenants — the whole point.
RUN = "run-shared"
MINE, THEIRS = "org-mine", "org-theirs"
THEIR_SECRET = "my social security number is 000-00-0000"


# ── A very small SQL stand-in ──────────────────────────────────────────────────
#
# Supports the predicate forms these reads use: `col = $n`,
# `col = ANY($n::t[])` and `col LIKE 'prefix%'`.

_ANY_PARAM = re.compile(r"(?:\w+\.)?(\w+)\s*=\s*ANY\(\$(\d+)::\w+\[\]\)")
_EQ_PARAM = re.compile(r"(?:\w+\.)?(\w+)\s*=\s*\$(\d+)")
_LIKE = re.compile(r"(?:\w+\.)?(\w+)\s+LIKE\s+'([^%']*)%'")
_FROM = re.compile(r"FROM\s+(\w+)")


def _joined_literals(func) -> str:
    """Source with adjacent string literals spliced back together, so a query
    written as `"SELECT ... " "WHERE ..."` reads as one statement."""
    return re.sub(r'"\s*\n\s*"', "", inspect.getsource(func))


def _matches(sql, args, row):
    for col, n in _ANY_PARAM.findall(sql):
        if row.get(col) not in args[int(n) - 1]:
            return False
    for col, n in _EQ_PARAM.findall(sql):
        if row.get(col) != args[int(n) - 1]:
            return False
    for col, prefix in _LIKE.findall(sql):
        if not str(row.get(col, "")).startswith(prefix):
            return False
    return True


class _Conn:
    def __init__(self, tables):
        self.tables = tables
        self.queries: list[tuple[str, tuple]] = []

    def _select(self, sql, args):
        self.queries.append((sql, args))
        table = _FROM.search(sql).group(1)
        return [r for r in self.tables.get(table, []) if _matches(sql, args, r)]

    async def fetch(self, sql, *args):
        return self._select(sql, args)

    async def fetchrow(self, sql, *args):
        rows = self._select(sql, args)
        return rows[0] if rows else None


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
    def __init__(self, tables):
        self.conn = _Conn(tables)

    def __enter__(self):
        self._orig = q._pool
        q._pool = _Pool(self.conn)
        return self.conn

    def __exit__(self, *exc):
        q._pool = self._orig
        return False


def _event(org_id, event_type, secs, payload, step_index=0):
    return {
        "org_id": org_id,
        "run_id": RUN,
        "event_type": event_type,
        "payload": payload,
        "received_at": T0 + datetime.timedelta(seconds=secs),
        "step_index": step_index,
    }


def _both_tenants_events():
    return [
        _event(THEIRS, "transcription.received", 0, {"text": THEIR_SECRET}),
        _event(THEIRS, "recording.available", 1, {"url": "https://audio/theirs.mp3"}),
        _event(MINE, "transcription.received", 2, {"text": "what is my balance"}),
        _event(MINE, "recording.available", 3, {"url": "https://audio/mine.mp3"}),
    ]


class TestFetchCallEventsIsOrgScoped(unittest.IsolatedAsyncioTestCase):
    async def test_events_read_returns_only_this_orgs_rows(self):
        patcher = _PoolPatch({"events": _both_tenants_events(), "failure_signals": []})
        with patcher:
            event_rows, _sigs = await q._fetch_call_events_and_signals(patcher.conn, MINE, [RUN])

        self.assertEqual({r["org_id"] for r in event_rows}, {MINE})
        self.assertNotIn(THEIR_SECRET, str(event_rows))

    async def test_signals_read_stays_scoped(self):
        """The half that was already correct must not regress."""
        signals = [
            {"org_id": THEIRS, "run_id": RUN, "failure_type": "VOICE_DEAD_AIR", "step_index": 0},
            {"org_id": MINE, "run_id": RUN, "failure_type": "VOICE_BARGE_IN", "step_index": 0},
        ]
        patcher = _PoolPatch({"events": [], "failure_signals": signals})
        with patcher:
            _events, sig_rows = await q._fetch_call_events_and_signals(patcher.conn, MINE, [RUN])

        self.assertEqual([s["failure_type"] for s in sig_rows], ["VOICE_BARGE_IN"])


class TestGetCallDetailIsOrgScoped(unittest.IsolatedAsyncioTestCase):
    """End to end: authorising the conversation by org is not enough — the run
    ids it yields are caller-supplied and collide."""

    def _tables(self):
        return {
            "conversations": [
                {
                    "id": 1,
                    "org_id": MINE,
                    "agent_id": "agent-mine",
                    "external_id": "conv-1",
                    "first_run_at": T0,
                    "last_run_at": T0 + datetime.timedelta(seconds=10),
                    "run_count": 1,
                }
            ],
            "runs": [
                {
                    "run_id": RUN,
                    "conversation_id": 1,
                    "org_id": MINE,
                    "agent_version": "v1",
                    "started_at": T0,
                },
                {
                    "run_id": RUN,
                    "conversation_id": 1,
                    "org_id": THEIRS,
                    "agent_version": "v9",
                    "started_at": T0,
                },
            ],
            "events": _both_tenants_events(),
            "failure_signals": [],
        }

    async def test_timeline_and_recordings_exclude_the_other_tenant(self):
        with _PoolPatch(self._tables()):
            detail = await q.get_call_detail(MINE, 1)

        self.assertIsNotNone(detail)
        rendered = str(detail)
        self.assertNotIn(THEIR_SECRET, rendered)
        self.assertNotIn("https://audio/theirs.mp3", rendered)
        self.assertEqual([r["url"] for r in detail["recordings"]], ["https://audio/mine.mp3"])
        self.assertEqual(
            [t["payload"].get("text") for t in detail["timeline"]], ["what is my balance"]
        )
        self.assertEqual([r["agent_version"] for r in detail["runs"]], ["v1"])

    async def test_another_orgs_conversation_id_still_returns_none(self):
        with _PoolPatch(self._tables()):
            self.assertIsNone(await q.get_call_detail(THEIRS, 1))


class TestListCallsCandidateFilterIsOrgScoped(unittest.TestCase):
    """The EXISTS that decides whether a conversation is a call at all joins
    `runs` to `events`; on a bare run_id that let another org's voice events
    make the decision."""

    def test_events_join_carries_org_id(self):
        source = inspect.getsource(q.list_calls)
        self.assertIn("JOIN events e ON e.run_id = r.run_id AND e.org_id = r.org_id", source)
        self.assertIn("r.org_id = c.org_id", source)
        self.assertNotRegex(source, r"JOIN events e ON e\.run_id = r\.run_id\s*\n")

    def test_run_harvest_is_org_scoped(self):
        """The run ids this yields are fed straight to the events fan-out, so
        the org filter has to travel with them."""
        self.assertIn(
            "FROM runs WHERE conversation_id = ANY($1::bigint[]) AND org_id = $2",
            _joined_literals(q.list_calls),
        )

    def test_call_detail_run_read_is_org_scoped(self):
        self.assertIn(
            "FROM runs WHERE conversation_id = $1 AND org_id = $2",
            _joined_literals(q.get_call_detail),
        )


if __name__ == "__main__":
    unittest.main()
