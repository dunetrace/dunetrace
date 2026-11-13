"""
The trace-id correlation read must be scoped to the polling org.

trace_id, like run_id, is caller-supplied — the SDK sets it and the OTLP path
derives it from the exporter's own trace id — so two tenants legitimately hold
the same one. Unscoped, `... WHERE trace_id = $1 LIMIT 1` could return the
other tenant's row, and _poll_one then wrote the signal with the *polling*
org's org_id but that row's agent_id / agent_version / run_id: one org's agent
naming and run identifiers surfacing in another org's dashboard.

The read is exercised against a tiny in-memory Postgres stand-in that really
applies the predicates in the SQL, so dropping org_id genuinely returns the
wrong tenant's row and the test fails on the data, not on a string match.

Run: make test-integrations
"""

from __future__ import annotations

import re
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import integrations_svc.db as db
import integrations_svc.worker  # noqa: F401  (so patch() can resolve worker attrs)
from integrations_svc.worker import _poll_one

from tests.test_worker import _evaluation, _integration, _mock_provider_cls

_EQ_PARAM = re.compile(r"(?:\w+\.)?(\w+)\s*=\s*\$(\d+)")
_FROM = re.compile(r"FROM\s+(\w+)")


class _Conn:
    def __init__(self, tables):
        self.tables = tables

    def _select(self, sql, args):
        table = _FROM.search(sql).group(1)
        rows = []
        for row in self.tables.get(table, []):
            if all(row.get(c) == args[int(n) - 1] for c, n in _EQ_PARAM.findall(sql)):
                rows.append(row)
        return rows

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
        self._orig = db._pool
        db._pool = _Pool(self.conn)
        return self.conn

    def __exit__(self, *exc):
        db._pool = self._orig
        return False


TRACE = "trace-shared"


def _event(org_id):
    return {
        "org_id": org_id,
        "trace_id": TRACE,
        "run_id": f"run-{org_id}",
        "agent_id": f"agent-{org_id}",
        "agent_version": f"v-{org_id}",
    }


class TestFetchRunByTraceIdIsOrgScoped(unittest.IsolatedAsyncioTestCase):
    async def test_returns_only_the_polling_orgs_run(self):
        # The other org's event is listed first, so an unscoped LIMIT 1 picks it.
        tables = {"events": [_event("org-theirs"), _event("org-mine")]}
        with _PoolPatch(tables):
            row = await db.fetch_run_by_trace_id("org-mine", TRACE)

        self.assertEqual(row["org_id"], "org-mine")
        self.assertEqual(row["run_id"], "run-org-mine")
        self.assertEqual(row["agent_id"], "agent-org-mine")

    async def test_returns_none_when_only_another_org_has_the_trace(self):
        """An uncorrelated evaluation is the correct outcome here — _poll_one
        marks it processed and writes nothing, rather than attributing another
        tenant's run."""
        tables = {"events": [_event("org-theirs")]}
        with _PoolPatch(tables):
            self.assertIsNone(await db.fetch_run_by_trace_id("org-mine", TRACE))


class TestWorkerPassesThePollingOrg(unittest.IsolatedAsyncioTestCase):
    """The org the signal is written under and the org the run is looked up in
    must be the same one, or the two disagree in the row that lands."""

    async def test_lookup_is_called_with_the_integrations_org(self):
        provider = MagicMock()
        provider.fetch_new_evaluations = AsyncMock(return_value=[_evaluation()])

        with (
            patch(
                "integrations_svc.worker.decrypt_credentials",
                return_value={"public_key": "pk", "secret_key": "sk"},
            ),
            patch.dict(
                "integrations_svc.worker._PROVIDER_CLASSES",
                {"langfuse": _mock_provider_cls(provider)},
            ),
            patch("integrations_svc.worker.has_processed", AsyncMock(return_value=False)),
            patch(
                "integrations_svc.worker.fetch_run_by_trace_id",
                AsyncMock(
                    return_value={
                        "run_id": "run-1",
                        "agent_id": "agent-1",
                        "agent_version": "v1",
                        "org_id": "org-7",
                    }
                ),
            ) as fetch_mock,
            patch("integrations_svc.worker.write_external_signal", AsyncMock()) as write_mock,
            patch("integrations_svc.worker.mark_processed", AsyncMock()),
            patch("integrations_svc.worker.record_poll_success", AsyncMock()),
        ):
            await _poll_one("langfuse", _integration(org_id="org-7"))

        fetch_mock.assert_awaited_once_with("org-7", "trace-1")
        self.assertEqual(write_mock.call_args.kwargs["org_id"], "org-7")


if __name__ == "__main__":
    unittest.main()
