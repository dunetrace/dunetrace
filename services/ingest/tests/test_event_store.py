"""
Tests for the EventStore abstraction (ingest_svc/db/event_store.py).

No Postgres needed — PostgresEventStore delegation is tested with mocked
free functions, and InMemoryEventStore is exercised directly.

Run:
    cd services/ingest
    PYTHONPATH=packages/sdk-py:services/ingest python -m pytest tests/test_event_store.py -v
"""

from __future__ import annotations

import sys
import os
import unittest
from unittest.mock import AsyncMock, patch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
for _p in [
    os.path.join(_ROOT, "packages/sdk-py"),
    os.path.join(_ROOT, "services/ingest"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ingest_svc.db.event_store import (
    EventStore,
    InMemoryEventStore,
    PostgresEventStore,
    get_event_store,
    set_event_store,
)


class TestPostgresEventStoreDelegation(unittest.IsolatedAsyncioTestCase):
    """PostgresEventStore is a thin delegate — the real SQL logic stays in
    postgres.py (and is exercised by its own tests); these confirm the
    delegation itself is wired correctly."""

    async def test_insert_events_delegates_to_postgres_module(self):
        store = PostgresEventStore()
        with patch(
            "ingest_svc.db.postgres.insert_events", AsyncMock(return_value=3)
        ) as mock_insert:
            result = await store.insert_events(["e1", "e2", "e3"], "batch-1", "org-1")
        mock_insert.assert_awaited_once_with(["e1", "e2", "e3"], "batch-1", "org-1")
        self.assertEqual(result, 3)

    async def test_prune_old_events_delegates_to_postgres_module(self):
        store = PostgresEventStore()
        with patch(
            "ingest_svc.db.postgres.prune_old_events", AsyncMock(return_value=2)
        ) as mock_prune:
            result = await store.prune_old_events(90)
        mock_prune.assert_awaited_once_with(90)
        self.assertEqual(result, 2)


class TestInMemoryEventStore(unittest.IsolatedAsyncioTestCase):
    async def test_insert_events_returns_count(self):
        store = InMemoryEventStore()
        result = await store.insert_events(["e1", "e2"], "batch-1", "org-1")
        self.assertEqual(result, 2)

    async def test_inserted_events_are_retrievable(self):
        store = InMemoryEventStore()
        await store.insert_events(["e1", "e2"], "batch-1", "org-1")
        await store.insert_events(["e3"], "batch-2", "org-2")
        self.assertEqual(store.all_events, ["e1", "e2", "e3"])

    async def test_batches_record_org_id(self):
        store = InMemoryEventStore()
        await store.insert_events(["e1"], "batch-1", "org-a")
        self.assertEqual(store.batches[0]["org_id"], "org-a")

    async def test_prune_removes_old_batches_keeps_recent(self):
        store = InMemoryEventStore()
        await store.insert_events(["old"], "batch-old", "org-1")
        store.batches[0]["inserted_at"] -= 200 * 86400  # 200 days ago
        await store.insert_events(["recent"], "batch-recent", "org-1")

        dropped = await store.prune_old_events(retention_days=90)

        self.assertEqual(dropped, 1)
        self.assertEqual(store.all_events, ["recent"])

    async def test_prune_returns_zero_when_nothing_is_old(self):
        store = InMemoryEventStore()
        await store.insert_events(["recent"], "batch-1", "org-1")
        dropped = await store.prune_old_events(retention_days=90)
        self.assertEqual(dropped, 0)
        self.assertEqual(store.all_events, ["recent"])


class TestGetSetEventStore(unittest.TestCase):
    def tearDown(self):
        # Reset the module-level singleton so other test files see the
        # default PostgresEventStore, not whatever a test here installed.
        import ingest_svc.db.event_store as es_mod

        es_mod._store = None

    def test_default_is_postgres_event_store(self):
        store = get_event_store()
        self.assertIsInstance(store, PostgresEventStore)

    def test_get_returns_same_instance_on_repeated_calls(self):
        first = get_event_store()
        second = get_event_store()
        self.assertIs(first, second)

    def test_set_event_store_swaps_the_singleton(self):
        fake = InMemoryEventStore()
        set_event_store(fake)
        self.assertIs(get_event_store(), fake)

    def test_custom_store_must_implement_the_abstract_contract(self):
        with self.assertRaises(TypeError):
            EventStore()  # abstract — cannot be instantiated directly


if __name__ == "__main__":
    unittest.main(verbosity=2)
