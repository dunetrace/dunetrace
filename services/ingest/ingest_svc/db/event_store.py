"""
EventStore — storage abstraction for ingest_svc's write path.

ingest_svc is the only service that writes to `events`; the four read-side
services (detector_svc, alerts_svc, api_svc, explainer_svc) each query
Postgres directly and are deliberately left alone — this abstraction covers
only the write path ingest_svc itself owns, not a shared cross-service
interface (see CLAUDE.md: "Five services communicate only through a shared
Postgres database" — no shared business logic between them beyond the SDK
and schemas packages).

Two things are swappable behind this interface: writing a batch
(`insert_events`) and enforcing retention (`prune_old_events`). Partition
creation is deliberately *not* part of the abstract contract — it's a
Postgres-specific implementation detail of how `PostgresEventStore` keeps
itself ready to receive writes, not something every backend would even have
a concept of (`InMemoryEventStore` has no partitions).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class EventStore(ABC):
    @abstractmethod
    async def insert_events(self, events: list, batch_id: str, org_id: str) -> int:
        """Persist a batch of events. Returns the number of rows written."""
        ...

    @abstractmethod
    async def prune_old_events(self, retention_days: int) -> int:
        """Discard events older than retention_days. Returns some measure of
        how much was discarded (partitions dropped, for Postgres; rows
        removed, for an in-memory store) — callers only log it, they don't
        depend on its exact unit."""
        ...

    async def insert_policy_evaluations(self, events: list, batch_id: str, org_id: str) -> int:
        """Persist policy-evaluation observability records (Phase 5) into the
        policy_evaluations table. Concrete (not abstract) with a no-op default so
        existing EventStore implementations keep working unchanged; only the
        stores that care override it. Returns rows written."""
        return 0


class PostgresEventStore(EventStore):
    """Delegates to ingest_svc.db.postgres — the actual SQL/partition logic
    stays there (it needs the shared connection pool and is exercised
    directly by its own tests); this class only provides the swappable
    interface boundary."""

    async def insert_events(self, events: list, batch_id: str, org_id: str) -> int:
        from ingest_svc.db.postgres import insert_events as _insert_events

        return await _insert_events(events, batch_id, org_id)

    async def prune_old_events(self, retention_days: int) -> int:
        from ingest_svc.db.postgres import prune_old_events as _prune_old_events

        return await _prune_old_events(retention_days)

    async def insert_policy_evaluations(self, events: list, batch_id: str, org_id: str) -> int:
        from ingest_svc.db.postgres import (
            insert_policy_evaluations as _insert_policy_evaluations,
        )

        return await _insert_policy_evaluations(events, batch_id, org_id)


class InMemoryEventStore(EventStore):
    """Fully in-process fake — no Postgres, no asyncpg, no partitions. Lets a
    test assert on what actually got "written" instead of just mocking a
    free function and trusting the call happened."""

    def __init__(self) -> None:
        self.batches: list[dict] = []  # {"batch_id", "org_id", "events", "inserted_at"}
        self.policy_evaluations: list[dict] = []  # {"batch_id", "org_id", "events"}

    async def insert_events(self, events: list, batch_id: str, org_id: str) -> int:
        import time

        self.batches.append(
            {
                "batch_id": batch_id,
                "org_id": org_id,
                "events": list(events),
                "inserted_at": time.time(),
            }
        )
        return len(events)

    async def prune_old_events(self, retention_days: int) -> int:
        import time

        cutoff = time.time() - retention_days * 86400
        before = len(self.batches)
        self.batches = [b for b in self.batches if b["inserted_at"] >= cutoff]
        return before - len(self.batches)

    async def insert_policy_evaluations(self, events: list, batch_id: str, org_id: str) -> int:
        self.policy_evaluations.append(
            {"batch_id": batch_id, "org_id": org_id, "events": list(events)}
        )
        return len(events)

    @property
    def all_events(self) -> list:
        """Flattened convenience view for assertions."""
        return [e for batch in self.batches for e in batch["events"]]

    @property
    def all_policy_evaluations(self) -> list:
        return [e for batch in self.policy_evaluations for e in batch["events"]]


_store: Optional[EventStore] = None


def get_event_store() -> EventStore:
    global _store
    if _store is None:
        _store = PostgresEventStore()
    return _store


def set_event_store(store: EventStore) -> None:
    """Swap the active store — real usage is Postgres by default; tests use
    this to install an InMemoryEventStore instead of monkeypatching
    individual free functions."""
    global _store
    _store = store
