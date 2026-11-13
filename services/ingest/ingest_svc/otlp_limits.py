"""
OTLP receiver failure isolation and rate limiting (Phase 3).

Three independent guards for the /v1/otlp/traces path:

  SpanRateLimiter  per-org token bucket over span count, so one org's burst
                   can't starve another's ingestion.
  InflightGuard    bounds how many batches are translated/persisted at once, so
                   the receiver applies backpressure instead of growing an
                   unbounded background backlog.
  PersistRetry     buffers batches whose DB persist failed and retries them, with
                   a circuit breaker so a dead DB isn't hammered. The receiver
                   keeps accepting spans (200) while the DB is unreachable.

All state is per process and in memory. With multiple ingest workers each
enforces its own limits; the goal is per-org isolation and self-protection, not
a globally exact cap (same tradeoff as the request rate limiter).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Awaitable, Callable, Optional

from ingest_svc.config import settings

logger = logging.getLogger("dunetrace.ingest.otlp.limits")


# ── Per-org span rate limiter ────────────────────────────────────────────────────


class SpanRateLimiter:
    """Per-org token bucket measured in spans. Refills at ``rate`` spans/sec up to
    ``burst``. A single batch larger than the burst is allowed once when the
    bucket is full (so a legitimately large export isn't rejected forever), then
    drains the bucket to zero."""

    def __init__(self, rate_per_sec: int, burst: Optional[int] = None):
        self._rate = max(1, rate_per_sec)
        self._burst = burst if burst is not None else self._rate * 2
        self._buckets: dict[str, tuple[float, float]] = {}  # org -> (tokens, last_ts)
        self._lock = threading.Lock()

    def allow(self, org_id: str, span_count: int) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds). retry_after is 0 when allowed."""
        if span_count <= 0:
            return True, 0
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(org_id, (float(self._burst), now))
            tokens = min(float(self._burst), tokens + (now - last) * self._rate)
            # A single batch can't require more than a full bucket, so an
            # oversized export goes through once the bucket has filled.
            required = min(span_count, float(self._burst))
            if tokens >= required:
                self._buckets[org_id] = (max(0.0, tokens - span_count), now)
                return True, 0
            self._buckets[org_id] = (tokens, now)
            retry_after = max(1, int((required - tokens) / self._rate) + 1)
            return False, retry_after

    def evict_stale(self, idle_secs: float = 300.0) -> None:
        cutoff = time.monotonic() - idle_secs
        with self._lock:
            stale = [org for org, (_, last) in self._buckets.items() if last < cutoff]
            for org in stale:
                del self._buckets[org]


# ── Backpressure ─────────────────────────────────────────────────────────────────


class InflightGuard:
    """Counts in-flight OTLP batches. reserve() before scheduling one, release()
    when it finishes. Over the cap, reserve() returns False and the caller sheds
    load with a 429 rather than queueing unbounded background work."""

    def __init__(self, max_inflight: int):
        self._max = max(1, max_inflight)
        self._count = 0
        self._lock = threading.Lock()

    def try_reserve(self) -> bool:
        with self._lock:
            if self._count >= self._max:
                return False
            self._count += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._count > 0:
                self._count -= 1

    @property
    def inflight(self) -> int:
        return self._count


# ── Persist with circuit breaker + retry buffer ──────────────────────────────────

InsertFn = Callable[[list, str, str], Awaitable[int]]


class PersistRetry:
    """Persist OTLP event batches, buffering and retrying on DB failure.

    When an insert fails the batch is buffered (drop-oldest when full) and a
    background loop retries it. After FAILURE_THRESHOLD failures within WINDOW
    the circuit opens: batches go straight to the buffer for COOLDOWN seconds
    without a DB attempt, so a dead DB isn't hammered on every request while the
    receiver keeps returning 200.
    """

    FAILURE_THRESHOLD = 5
    WINDOW = 60.0
    COOLDOWN = 30.0

    def __init__(self, insert_fn: InsertFn, max_batches: int):
        self._insert = insert_fn
        self._buffer: deque = deque(maxlen=max(1, max_batches))
        self._failures: deque = deque()
        self._open_until = 0.0

    def _record_failure(self, now: float) -> None:
        self._failures.append(now)
        while self._failures and now - self._failures[0] > self.WINDOW:
            self._failures.popleft()
        if len(self._failures) >= self.FAILURE_THRESHOLD:
            self._open_until = now + self.COOLDOWN
            self._failures.clear()

    def _enqueue(self, events: list, batch_id: str, org_id: str) -> None:
        if len(self._buffer) >= (self._buffer.maxlen or 0):
            logger.warning("OTLP retry buffer full; dropping oldest batch to make room.")
        self._buffer.append((events, batch_id, org_id))

    async def persist(self, events: list, batch_id: str, org_id: str) -> None:
        now = time.monotonic()
        if now < self._open_until:
            # Circuit open: skip the DB, buffer for the retry loop.
            self._enqueue(events, batch_id, org_id)
            return
        try:
            await self._insert(events, batch_id, org_id)
        except Exception as exc:
            logger.warning(
                "OTLP persist failed, buffering for retry. batch_id=%s error=%s",
                batch_id,
                type(exc).__name__,
            )
            self._record_failure(now)
            self._enqueue(events, batch_id, org_id)

    async def retry_once(self) -> int:
        """One drain pass over the retry buffer. Stops at the first failure (the
        DB is still down). Returns how many batches were flushed. Called by a
        background loop."""
        flushed = 0
        for _ in range(len(self._buffer)):
            try:
                item = self._buffer.popleft()
            except IndexError:
                break
            events, batch_id, org_id = item
            try:
                await self._insert(events, batch_id, org_id)
                flushed += 1
            except Exception:
                self._buffer.append(item)  # requeue; try again next pass
                break
        return flushed

    @property
    def buffered(self) -> int:
        return len(self._buffer)


# ── Singletons ───────────────────────────────────────────────────────────────────

_span_limiter: Optional[SpanRateLimiter] = None
_inflight: Optional[InflightGuard] = None
_persist_retry: Optional[PersistRetry] = None


def get_span_limiter() -> SpanRateLimiter:
    global _span_limiter
    if _span_limiter is None:
        _span_limiter = SpanRateLimiter(settings.OTLP_MAX_SPANS_PER_SEC)
    return _span_limiter


def get_inflight_guard() -> InflightGuard:
    global _inflight
    if _inflight is None:
        _inflight = InflightGuard(settings.OTLP_MAX_INFLIGHT)
    return _inflight


async def _default_insert(events: list, batch_id: str, org_id: str) -> int:
    from ingest_svc.db import get_event_store

    return await get_event_store().insert_events(events, batch_id, org_id)


def get_persist_retry() -> PersistRetry:
    global _persist_retry
    if _persist_retry is None:
        _persist_retry = PersistRetry(_default_insert, settings.OTLP_RETRY_BUFFER_BATCHES)
    return _persist_retry
