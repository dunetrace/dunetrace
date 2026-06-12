"""
Per-key sliding-window rate limiter (in-memory).

Each API key gets an independent 60-second window sized by its `rate_limit_rpm`
value loaded from the DB at first use and cached for CACHE_TTL seconds.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Optional

logger = logging.getLogger("dunetrace.ingest.ratelimit")

_CACHE_TTL = 300  # seconds before re-fetching rpm from DB


class RateLimiter:
    def __init__(self, default_rpm: int = 600):
        self._default_rpm = default_rpm
        self._windows: dict[str, deque] = defaultdict(deque)
        self._rpm_cache: dict[str, tuple[int, float]] = {}  # key → (rpm, fetched_at)
        self._lock = asyncio.Lock()

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
            logger.debug("rpm lookup failed for key %s…: %s", api_key[:10], exc)

        self._rpm_cache[api_key] = (rpm, now)
        return rpm

    async def is_allowed(self, api_key: str) -> tuple[bool, int]:
        """
        Returns (allowed, retry_after_seconds).
        retry_after is 0 when allowed, else seconds until the oldest request leaves the window.
        """
        rpm = await self._get_rpm(api_key)
        now = time.monotonic()
        window_start = now - 60.0

        async with self._lock:
            dq = self._windows[api_key]
            while dq and dq[0] < window_start:
                dq.popleft()
            if len(dq) >= rpm:
                retry_after = max(1, int(60.0 - (now - dq[0])) + 1)
                return False, retry_after
            dq.append(now)
            return True, 0

    def evict_stale(self) -> None:
        """Discard windows for keys idle for more than 2 minutes (called periodically)."""
        cutoff = time.monotonic() - 120.0
        stale = [k for k, dq in self._windows.items() if not dq or dq[-1] < cutoff]
        for k in stale:
            del self._windows[k]
            self._rpm_cache.pop(k, None)


_limiter: Optional[RateLimiter] = None


def get_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        import os
        default_rpm = int(os.getenv("RATE_LIMIT_RPM", "600"))
        _limiter = RateLimiter(default_rpm=default_rpm)
    return _limiter
