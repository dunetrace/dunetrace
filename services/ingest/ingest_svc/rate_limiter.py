"""
Per-key sliding-window rate limiter (in-memory, with cross-process coordination).

Each API key gets an independent 60-second window sized by its `rate_limit_rpm`
value loaded from the DB at first use and cached for CACHE_TTL seconds. The
per-request hot path (`is_allowed`) never touches the DB — cross-process
coordination is approximate, not exact: see `_heartbeat` below.

Cross-process coordination — why, and the tradeoff:
    This limiter is a per-process in-memory singleton. Running ingest with
    multiple uvicorn workers or multiple container replicas means each
    process would otherwise enforce the *full* configured rpm independently,
    silently multiplying the effective limit by however many processes are
    running. `_heartbeat` closes that gap by having every worker periodically
    report itself alive in a shared `rate_limit_workers` table and read back
    how many workers are currently active; `is_allowed` then divides the
    configured rpm by that count, so N workers collectively approximate the
    configured limit instead of each enforcing it in full.

    This is deliberately approximate rather than exact: an exact answer would
    mean a synchronous Postgres round-trip on every single ingest request,
    which conflicts with the ingest API's "return 202 fast" design and its
    ~1000 req/sec throughput target. Instead, coordination happens on a
    slower heartbeat cadence (`_HEARTBEAT_INTERVAL`), so the active-worker
    count — and therefore each worker's effective rpm — can lag reality by up
    to that interval (e.g. right after a worker starts or is killed). With a
    single worker (today's actual deployment — see docker-compose.yml), the
    active-worker count is always 1 and enforcement is exactly as before.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict, deque
from typing import NamedTuple, Optional

logger = logging.getLogger("dunetrace.ingest.ratelimit")

_CACHE_TTL = 300  # seconds before re-fetching rpm/quota from DB
_HEARTBEAT_INTERVAL = 10.0  # how often each worker reports itself alive
_HEARTBEAT_STALE_AFTER = 30.0  # 3x the interval — tolerates a couple of missed beats

# Default share of a key's sustained rpm any single agent_id under that key
# may consume, absent an explicit per-agent override (see
# db/postgres.py::get_agent_quota / set_agent_quota, and the
# /admin/keys/{key_id}/agents/{agent_id}/quota endpoint). Solves one runaway
# agent under a shared key starving its siblings' share of the same budget.
_DEFAULT_AGENT_QUOTA_PCT = 0.20


class RateLimitResult(NamedTuple):
    allowed: bool
    retry_after: int  # seconds; 0 when allowed
    key_remaining: int  # requests left in the key-level window this minute
    agent_remaining: Optional[int]  # None when no agent_id was given to is_allowed()


class RateLimiter:
    def __init__(self, default_rpm: int = 600):
        self._default_rpm = default_rpm
        self._windows: dict[str, deque] = defaultdict(deque)
        self._agent_windows: dict[tuple[str, str], deque] = defaultdict(deque)
        self._rpm_cache: dict[str, tuple[int, float]] = {}  # key → (rpm, fetched_at)
        self._quota_cache: dict[
            tuple[str, str], tuple[float, float]
        ] = {}  # (key, agent) → (pct, fetched_at)
        self._lock = asyncio.Lock()
        self._worker_id = str(uuid.uuid4())
        self._active_workers = 1  # updated periodically by _heartbeat; 1 == today's exact behavior

    async def _get_rpm(self, api_key: str) -> int:
        """Return rpm for key — cached, with DB fallback."""
        now = time.monotonic()
        cached = self._rpm_cache.get(api_key)
        if cached and now - cached[1] < _CACHE_TTL:
            return cached[0]

        rpm = self._default_rpm
        try:
            from ingest_svc.db.postgres import get_pool

            pool = get_pool()
            if pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT rate_limit_rpm FROM api_keys WHERE key=$1 AND active=TRUE",
                        api_key,
                    )
                if row:
                    rpm = row["rate_limit_rpm"]
        except Exception as exc:
            # Type name only, not the exception object — same reasoning as
            # verify_api_key() in db/postgres.py: this query is also bound
            # with the raw api_key, which some DB drivers echo into error
            # message strings.
            logger.debug("rpm lookup failed: %s", type(exc).__name__)

        self._rpm_cache[api_key] = (rpm, now)
        return rpm

    async def _get_agent_quota_pct(self, api_key: str, agent_id: str) -> float:
        """Return this agent's share of the key's rpm — cached, with DB fallback
        to _DEFAULT_AGENT_QUOTA_PCT. Same cache-then-DB-on-miss shape as _get_rpm."""
        cache_key = (api_key, agent_id)
        now = time.monotonic()
        cached = self._quota_cache.get(cache_key)
        if cached and now - cached[1] < _CACHE_TTL:
            return cached[0]

        quota_pct = _DEFAULT_AGENT_QUOTA_PCT
        try:
            from ingest_svc.db.postgres import get_agent_quota_by_key

            override = await get_agent_quota_by_key(api_key, agent_id)
            if override is not None:
                quota_pct = override
        except Exception as exc:
            logger.debug("agent quota lookup failed: %s", type(exc).__name__)

        self._quota_cache[cache_key] = (quota_pct, now)
        return quota_pct

    async def is_allowed(self, api_key: str, agent_id: Optional[str] = None) -> RateLimitResult:
        """
        Checks the key-level limit first — always, regardless of agent_id — then,
        if agent_id is given, an additional per-(key, agent) sub-limit within it.
        Both windows share the same 60s sliding window and are recorded together
        on success; either one denying denies the request as a whole.

        agent_remaining is None (not 0) when agent_id wasn't given — distinguishes
        "no agent-level check happened" from "the agent's quota is exhausted."
        """
        rpm = await self._get_rpm(api_key)
        effective_rpm = max(1, rpm // self._active_workers)
        agent_rpm: Optional[int] = None
        if agent_id:
            quota_pct = await self._get_agent_quota_pct(api_key, agent_id)
            agent_rpm = max(1, int(effective_rpm * quota_pct))

        now = time.monotonic()
        window_start = now - 60.0

        async with self._lock:
            dq = self._windows[api_key]
            while dq and dq[0] < window_start:
                dq.popleft()
            key_remaining = max(0, effective_rpm - len(dq))

            if len(dq) >= effective_rpm:
                retry_after = max(1, int(60.0 - (now - dq[0])) + 1)
                return RateLimitResult(False, retry_after, key_remaining, None)

            agent_remaining: Optional[int] = None
            if agent_id and agent_rpm is not None:
                adq = self._agent_windows[(api_key, agent_id)]
                while adq and adq[0] < window_start:
                    adq.popleft()
                agent_remaining = max(0, agent_rpm - len(adq))
                if len(adq) >= agent_rpm:
                    retry_after = max(1, int(60.0 - (now - adq[0])) + 1)
                    return RateLimitResult(False, retry_after, key_remaining, agent_remaining)
                adq.append(now)
                agent_remaining -= 1

            dq.append(now)
            return RateLimitResult(True, 0, key_remaining - 1, agent_remaining)

    def evict_stale(self) -> None:
        """Discard windows for keys/agents idle for more than 2 minutes (called periodically)."""
        cutoff = time.monotonic() - 120.0
        stale = [k for k, dq in self._windows.items() if not dq or dq[-1] < cutoff]
        for k in stale:
            del self._windows[k]
            self._rpm_cache.pop(k, None)

        stale_agents = [k for k, dq in self._agent_windows.items() if not dq or dq[-1] < cutoff]
        for k in stale_agents:
            del self._agent_windows[k]
            self._quota_cache.pop(k, None)

    async def _heartbeat(self) -> None:
        """Report this worker alive, reap stale workers, and refresh
        self._active_workers from the resulting count. Called periodically
        by a background task (see main.py's _rate_limit_heartbeat_loop) —
        never from the request hot path.

        On a transient DB failure, self._active_workers is left at its last
        known value rather than reset to 1 — resetting would let every
        worker briefly enforce the *full* rpm again, causing exactly the
        aggregate overshoot this mechanism exists to prevent. Only a
        definitively absent pool (no DB configured at all — single-process
        / test usage) resets it to 1, matching pre-coordination behavior.
        """
        try:
            from ingest_svc.db.postgres import get_pool

            pool = get_pool()
            if not pool:
                self._active_workers = 1
                return
            now = time.time()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO rate_limit_workers (worker_id, last_seen)
                    VALUES ($1, $2)
                    ON CONFLICT (worker_id) DO UPDATE SET last_seen = EXCLUDED.last_seen
                    """,
                    self._worker_id,
                    now,
                )
                await conn.execute(
                    "DELETE FROM rate_limit_workers WHERE last_seen < $1",
                    now - _HEARTBEAT_STALE_AFTER,
                )
                count = await conn.fetchval("SELECT COUNT(*) FROM rate_limit_workers")
            self._active_workers = max(1, count or 1)
        except Exception as exc:
            logger.debug("rate-limit heartbeat failed: %s", exc)


_limiter: Optional[RateLimiter] = None


def get_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        import os

        default_rpm = int(os.getenv("RATE_LIMIT_RPM", "600"))
        _limiter = RateLimiter(default_rpm=default_rpm)
    return _limiter
