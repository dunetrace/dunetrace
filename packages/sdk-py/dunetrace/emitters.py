"""
Pluggable batch-shipping strategies for the SDK's drain thread.

BatchingEmitter is the seam between "the drain thread pulled a batch of
events off the ring buffer" and "deliver them somewhere." HttpBatchingEmitter
(the default) posts to the ingest API; Noop/Console/File exist for local
debugging, offline/audit trails, and explicitly disabling network shipping;
DurableRetryEmitter wraps any of them to survive a backend outage across
process restarts.

No external dependencies — matches the rest of the core SDK. sqlite3 is
stdlib, not a dependency.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sqlite3
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from contextlib import contextmanager
from threading import Lock
from typing import Dict, Iterator, List, Optional

from dunetrace.models import AgentEvent, EventType

logger = logging.getLogger("dunetrace")

try:
    from importlib.metadata import PackageNotFoundError as _PkgNotFoundError
    from importlib.metadata import version as _pkg_version

    _SDK_VERSION = _pkg_version("dunetrace")
except _PkgNotFoundError:
    _SDK_VERSION = "0.0.0"  # running from source without installing

# Every outbound request identifies itself. Without this, urllib's default
# "Python-urllib/x.y" User-Agent gets fingerprinted and blocked (HTTP 403) by
# Cloudflare's bot protection in front of app.dunetrace.com — confirmed via a
# direct request with that exact UA string returning Cloudflare error 1010.
USER_AGENT = f"dunetrace-sdk-py/{_SDK_VERSION}"


class BatchingEmitter(ABC):
    """Strategy for delivering one drained batch of events.

    ship() must never raise — catch and log internally, returning False on
    failure so a caller (e.g. a durable-retry wrapper) can react without
    needing to parse exceptions. Returning True means "handled" — for
    NoopBatchingEmitter that means "intentionally discarded," not "delivered."
    """

    @abstractmethod
    def ship(self, batch: List[AgentEvent]) -> bool:
        raise NotImplementedError


class HttpBatchingEmitter(BatchingEmitter):
    """POSTs each batch to the ingest API. The default emitter.

    Usable standalone (not just via Dunetrace): HttpBatchingEmitter("http://localhost:8001", api_key="...").
    """

    def __init__(self, endpoint: str, api_key: str = "") -> None:
        self._ingest_url = endpoint.rstrip("/") + "/v1/ingest"
        self._api_key = api_key

    def _auth_headers(self) -> Dict[str, str]:
        """Authorization header for gateway-fronted deployments (e.g. the
        Dunetrace Cloud gateway's tenancy middleware, which resolves the
        calling org from ``Authorization: Bearer <api_key>`` and nothing
        else). Self-hosted ingest_svc (no gateway) still also accepts
        ``api_key`` in the request body, so this header is additive.
        """
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def ship(self, batch: List[AgentEvent]) -> bool:
        payload = json.dumps(
            {
                "api_key": self._api_key,  # self-hosted ingest_svc compat; see _auth_headers
                "agent_id": batch[0].agent_id if batch else "",
                "events": [e.to_dict() for e in batch],
            }
        ).encode()

        req = urllib.request.Request(
            self._ingest_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-Dunetrace-Agent": batch[0].agent_id if batch else "",
                "User-Agent": USER_AGENT,
                **self._auth_headers(),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                logger.debug("Shipped %d events. status=%d", len(batch), resp.status)
            return True
        except urllib.error.URLError as exc:
            if "Connection refused" in str(exc):
                logger.warning(
                    "DuneTrace backend not reachable at %s — is it running?\n"
                    "  Start it with: docker compose up -d\n"
                    "  %d events dropped.",
                    self._ingest_url,
                    len(batch),
                )
            else:
                logger.warning("Failed to ship %d events: %s", len(batch), exc)
            return False
        except Exception as exc:
            logger.warning("Failed to ship %d events: %s", len(batch), exc)
            return False


class NoopBatchingEmitter(BatchingEmitter):
    """Discards every batch. The explicit way to disable HTTP shipping —
    e.g. for SDK-only OTel or NDJSON deployments that don't run the ingest API.

    Usage::

        dt = Dunetrace(emitter=NoopBatchingEmitter())
    """

    def ship(self, batch: List[AgentEvent]) -> bool:
        return True


class ConsoleBatchingEmitter(BatchingEmitter):
    """Prints each event in a batch as a JSON line to stdout.

    Distinct from Dunetrace(emit_as_json=True): that writes one NDJSON line
    per event, synchronously, the instant each event is created (for
    Loki/Promtail). This writes once per drain cycle, batched — meant for ad
    hoc local debugging of what the drain thread is actually shipping.
    """

    def __init__(self) -> None:
        self._lock = Lock()

    def ship(self, batch: List[AgentEvent]) -> bool:
        with self._lock:
            for event in batch:
                print(json.dumps(event.to_dict(), separators=(",", ":")))
        return True


class FileBatchingEmitter(BatchingEmitter):
    """Appends each event in a batch as a JSON line to a local file.

    For offline/audit trails, or environments where neither the ingest API
    nor stdout capture is available. Opens and closes the file per batch
    rather than holding a handle open — the simplest way to stay safe across
    process forks and concurrent instances without extra coordination.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = Lock()

    def ship(self, batch: List[AgentEvent]) -> bool:
        try:
            with self._lock:
                with open(self._path, "a") as f:
                    for event in batch:
                        f.write(json.dumps(event.to_dict(), separators=(",", ":")) + "\n")
            return True
        except OSError as exc:
            logger.warning("FileBatchingEmitter: failed to write to %s: %s", self._path, exc)
            return False


def _batch_to_json(batch: List[AgentEvent]) -> str:
    return json.dumps([e.to_dict() for e in batch])


def _batch_from_json(payload: str) -> List[AgentEvent]:
    events = []
    for d in json.loads(payload):
        d = dict(d)
        d["event_type"] = EventType(d["event_type"])
        events.append(AgentEvent(**d))
    return events


DEFAULT_QUEUE_PATH = os.path.expanduser("~/.dunetrace/queue.db")


class DurableRetryEmitter(BatchingEmitter):
    """Wraps any BatchingEmitter to survive a backend outage across process
    restarts. On ship() failure, the batch is persisted to a local SQLite
    queue instead of dropped; on every ship() call, any due backlog is
    retried first (oldest batch first), before the current new batch.

    Composable with any inner emitter — most commonly HttpBatchingEmitter::

        dt = Dunetrace(emitter=DurableRetryEmitter(HttpBatchingEmitter(endpoint, api_key)))

    Queue location: `queue_path` constructor arg, else `DUNETRACE_QUEUE_PATH`
    env var, else `~/.dunetrace/queue.db`. The env var override matters for
    servers running as a non-root user with no HOME set.

    Bounded: evicts the oldest queued batch(es) once either
    `max_queue_events` or `max_queue_bytes` is exceeded — same "never grow
    unboundedly, drop oldest first" philosophy as the in-memory RingBuffer.
    Eviction is logged at WARNING, rate-limited to once per minute (with the
    count of batches evicted since the last warning) so a long outage doesn't
    spam logs while still surfacing that data is being lost silently.

    Retry cadence is decoupled from the normal flush interval — draining the
    backlog on every drain cycle (default every 200ms) would hammer a
    still-down backend with a full-timeout connection attempt every cycle.
    Default: every 30s ± 5s jitter (prevents a thundering herd if many SDK
    instances come back online together after a shared outage). A retry
    attempt stops at the first failure rather than skipping ahead — if the
    backend is still down, later batches will fail for the same reason, and
    preserving order matters more than attempting all of them.
    """

    def __init__(
        self,
        inner: BatchingEmitter,
        *,
        queue_path: Optional[str] = None,
        max_queue_events: int = 100_000,
        max_queue_bytes: int = 100 * 1024 * 1024,  # 100MB
        retry_interval_s: float = 30.0,
        retry_jitter_s: float = 5.0,
        max_batches_per_retry: int = 50,
    ) -> None:
        self._inner = inner
        self._path = queue_path or os.environ.get("DUNETRACE_QUEUE_PATH") or DEFAULT_QUEUE_PATH
        self._max_queue_events = max_queue_events
        self._max_queue_bytes = max_queue_bytes
        self._retry_interval_s = retry_interval_s
        self._retry_jitter_s = retry_jitter_s
        self._max_batches_per_retry = max_batches_per_retry

        self._lock = Lock()
        self._next_retry_at = 0.0  # monotonic time; 0 == due immediately
        self._evicted_since_warning = 0
        self._last_eviction_warning: Optional[float] = None

        self._db_ok = self._init_db()

    def _init_db(self) -> bool:
        try:
            directory = os.path.dirname(self._path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with self._connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS queue (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        enqueued_at REAL NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        payload TEXT NOT NULL
                    )
                    """
                )
            return True
        except OSError as exc:
            logger.warning(
                "DurableRetryEmitter: could not initialize queue at %s: %s. "
                "Failed batches will be dropped instead of queued.",
                self._path,
                exc,
            )
            return False

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, timeout=5)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def ship(self, batch: List[AgentEvent]) -> bool:
        self._maybe_retry_backlog()
        if self._inner.ship(batch):
            return True
        return self._enqueue(batch)

    def _enqueue(self, batch: List[AgentEvent]) -> bool:
        if not self._db_ok:
            return False
        payload = _batch_to_json(batch)
        size = len(payload.encode())
        try:
            with self._lock, self._connection() as conn:
                conn.execute(
                    "INSERT INTO queue (enqueued_at, size_bytes, payload) VALUES (?, ?, ?)",
                    (time.time(), size, payload),
                )
                self._evict_if_over_bounds(conn)
            return True
        except sqlite3.Error as exc:
            logger.warning("DurableRetryEmitter: failed to queue batch: %s", exc)
            return False

    def _evict_if_over_bounds(self, conn: sqlite3.Connection) -> None:
        count, total_bytes = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM queue"
        ).fetchone()
        evicted = 0
        while count > self._max_queue_events or total_bytes > self._max_queue_bytes:
            row = conn.execute(
                "SELECT id, size_bytes FROM queue ORDER BY id ASC LIMIT 1"
            ).fetchone()
            if row is None:
                break
            oldest_id, oldest_size = row
            conn.execute("DELETE FROM queue WHERE id = ?", (oldest_id,))
            count -= 1
            total_bytes -= oldest_size
            evicted += 1

        if evicted:
            self._evicted_since_warning += evicted
            now = time.monotonic()
            # time.monotonic()'s epoch is unspecified (often time-since-boot on
            # Linux) — comparing against a 0.0 sentinel for "never warned yet"
            # silently skips the first warning on a freshly booted machine
            # where monotonic() itself is still under 60. None is unambiguous
            # regardless of the clock's starting point.
            if self._last_eviction_warning is None or now - self._last_eviction_warning >= 60:
                logger.warning(
                    "DurableRetryEmitter: evicted %d oldest batch(es) from the disk queue "
                    "at %s (over the %d-event / %d-byte cap) — this is silent data loss "
                    "during a long outage.",
                    self._evicted_since_warning,
                    self._path,
                    self._max_queue_events,
                    self._max_queue_bytes,
                )
                self._evicted_since_warning = 0
                self._last_eviction_warning = now

    def _maybe_retry_backlog(self) -> None:
        now = time.monotonic()
        if now < self._next_retry_at:
            return
        self._next_retry_at = now + random.uniform(
            self._retry_interval_s - self._retry_jitter_s,
            self._retry_interval_s + self._retry_jitter_s,
        )
        if not self._db_ok:
            return

        try:
            with self._lock, self._connection() as conn:
                rows = conn.execute(
                    "SELECT id, payload FROM queue ORDER BY id ASC LIMIT ?",
                    (self._max_batches_per_retry,),
                ).fetchall()
                for row_id, payload in rows:
                    batch = _batch_from_json(payload)
                    if self._inner.ship(batch):
                        conn.execute("DELETE FROM queue WHERE id = ?", (row_id,))
                    else:
                        break  # still down — stop rather than skip ahead out of order
        except sqlite3.Error as exc:
            logger.warning("DurableRetryEmitter: retry attempt failed: %s", exc)
