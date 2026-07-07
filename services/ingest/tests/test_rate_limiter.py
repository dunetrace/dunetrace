"""
Tests for the in-memory sliding-window rate limiter.
No DB, no network — fully offline.

Run:
    cd services/ingest
    PYTHONPATH=packages/sdk-py:services/ingest python -m pytest tests/test_rate_limiter.py -v
"""

from __future__ import annotations

import asyncio
import sys
import os
import time
import unittest
from unittest.mock import patch, AsyncMock

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
for _p in [
    os.path.join(_ROOT, "packages/sdk-py"),
    os.path.join(_ROOT, "services/ingest"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ingest_svc.rate_limiter import RateLimiter


class TestRateLimiterAllow(unittest.IsolatedAsyncioTestCase):
    """Basic allow / deny behaviour."""

    async def test_first_request_always_allowed(self):
        limiter = RateLimiter(default_rpm=10)
        allowed, retry_after, *_ = await limiter.is_allowed("key1")
        self.assertTrue(allowed)
        self.assertEqual(retry_after, 0)

    async def test_requests_within_limit_are_allowed(self):
        limiter = RateLimiter(default_rpm=5)
        for _ in range(5):
            allowed, *_ = await limiter.is_allowed("key1")
            self.assertTrue(allowed)

    async def test_request_exceeding_limit_is_denied(self):
        limiter = RateLimiter(default_rpm=3)
        for _ in range(3):
            await limiter.is_allowed("key1")
        # 4th request should be denied
        allowed, retry_after, *_ = await limiter.is_allowed("key1")
        self.assertFalse(allowed)
        self.assertGreater(retry_after, 0)

    async def test_retry_after_is_positive_when_denied(self):
        limiter = RateLimiter(default_rpm=1)
        await limiter.is_allowed("key1")
        _, retry_after, *_ = await limiter.is_allowed("key1")
        self.assertGreaterEqual(retry_after, 1)

    async def test_different_keys_are_independent(self):
        limiter = RateLimiter(default_rpm=1)
        await limiter.is_allowed("key1")
        # key1 is now exhausted, but key2 should still be allowed
        allowed, *_ = await limiter.is_allowed("key2")
        self.assertTrue(allowed)

    async def test_single_rpm_limits_to_one_per_window(self):
        limiter = RateLimiter(default_rpm=1)
        ok1, *_ = await limiter.is_allowed("k")
        ok2, *_ = await limiter.is_allowed("k")
        self.assertTrue(ok1)
        self.assertFalse(ok2)


class TestRateLimiterSlidingWindow(unittest.IsolatedAsyncioTestCase):
    """Sliding window boundary — old timestamps fall out after 60s."""

    async def test_requests_outside_window_are_allowed_again(self):
        limiter = RateLimiter(default_rpm=2)
        now = time.monotonic()
        # Manually insert stale timestamps (>60s ago) into the deque
        with patch("ingest_svc.rate_limiter.time") as mock_time:
            mock_time.monotonic.return_value = now - 65.0
            await limiter.is_allowed("key1")
            await limiter.is_allowed("key1")  # fills window 65s ago

        # Back to real time — window should have slid, requests should be allowed again
        with patch("ingest_svc.rate_limiter.time") as mock_time:
            mock_time.monotonic.return_value = now
            allowed, *_ = await limiter.is_allowed("key1")
            self.assertTrue(allowed)

    async def test_window_exactly_60_seconds(self):
        """A request at exactly 60s ago is NOT evicted (window uses strict <).
        The eviction condition is dq[0] < (now - 60.0), so dq[0] == now - 60.0 stays."""
        limiter = RateLimiter(default_rpm=1)
        now = time.monotonic()
        # Insert a timestamp exactly 60s ago
        with patch("ingest_svc.rate_limiter.time") as mock_time:
            mock_time.monotonic.return_value = now - 60.0
            await limiter.is_allowed("key1")

        # At current time, the 60s-old request sits at the boundary and is NOT evicted
        with patch("ingest_svc.rate_limiter.time") as mock_time:
            mock_time.monotonic.return_value = now
            allowed, *_ = await limiter.is_allowed("key1")
            self.assertFalse(allowed)  # still counted — boundary is inclusive

    async def test_request_just_inside_window_counts(self):
        """A request 59s ago should still count against the rate limit."""
        limiter = RateLimiter(default_rpm=1)
        now = time.monotonic()
        # Insert a timestamp 59s ago (inside the 60s window)
        with patch("ingest_svc.rate_limiter.time") as mock_time:
            mock_time.monotonic.return_value = now - 59.0
            await limiter.is_allowed("key1")

        # At current time, 59s-old request is still in window — should be denied
        with patch("ingest_svc.rate_limiter.time") as mock_time:
            mock_time.monotonic.return_value = now
            allowed, *_ = await limiter.is_allowed("key1")
            self.assertFalse(allowed)


class TestRateLimiterEviction(unittest.IsolatedAsyncioTestCase):
    """evict_stale() removes idle keys from memory."""

    async def test_evict_stale_removes_inactive_keys(self):
        limiter = RateLimiter(default_rpm=10)
        now = time.monotonic()
        # Manually set a stale timestamp (> 120s ago) in the window deque
        with patch("ingest_svc.rate_limiter.time") as mock_time:
            mock_time.monotonic.return_value = now - 130.0
            await limiter.is_allowed("stale_key")

        self.assertIn("stale_key", limiter._windows)
        # evict_stale uses real monotonic, so patch it to return `now`
        with patch("ingest_svc.rate_limiter.time") as mock_time:
            mock_time.monotonic.return_value = now
            limiter.evict_stale()

        self.assertNotIn("stale_key", limiter._windows)

    async def test_evict_stale_removes_rpm_cache_for_stale_keys(self):
        limiter = RateLimiter(default_rpm=10)
        now = time.monotonic()
        with patch("ingest_svc.rate_limiter.time") as mock_time:
            mock_time.monotonic.return_value = now - 130.0
            await limiter.is_allowed("stale_key")
        # Simulate a cached rpm entry
        limiter._rpm_cache["stale_key"] = (60, now - 400)

        with patch("ingest_svc.rate_limiter.time") as mock_time:
            mock_time.monotonic.return_value = now
            limiter.evict_stale()

        self.assertNotIn("stale_key", limiter._rpm_cache)

    async def test_evict_stale_keeps_active_keys(self):
        limiter = RateLimiter(default_rpm=10)
        await limiter.is_allowed("active_key")
        # active_key just had a request — its last timestamp is very recent
        limiter.evict_stale()
        self.assertIn("active_key", limiter._windows)

    async def test_evict_stale_removes_empty_window_keys(self):
        """Keys whose deque is empty (all requests expired) should also be evicted."""
        limiter = RateLimiter(default_rpm=10)
        now = time.monotonic()
        # Use a stale timestamp so the deque ages out
        with patch("ingest_svc.rate_limiter.time") as mock_time:
            mock_time.monotonic.return_value = now - 130.0
            await limiter.is_allowed("old_key")

        with patch("ingest_svc.rate_limiter.time") as mock_time:
            mock_time.monotonic.return_value = now
            limiter.evict_stale()

        self.assertNotIn("old_key", limiter._windows)


class TestRateLimiterConcurrency(unittest.IsolatedAsyncioTestCase):
    """Concurrent coroutines must never corrupt the window deque."""

    async def test_concurrent_requests_respect_limit(self):
        """10 concurrent coroutines hitting a limit of 5 — exactly 5 should pass."""
        limiter = RateLimiter(default_rpm=5)
        results = await asyncio.gather(*[limiter.is_allowed("shared") for _ in range(10)])
        allowed_count = sum(1 for ok, *_ in results if ok)
        self.assertEqual(allowed_count, 5)

    async def test_concurrent_different_keys_do_not_interfere(self):
        """Each key has its own quota; concurrent calls on different keys must not interfere."""
        limiter = RateLimiter(default_rpm=3)
        tasks = []
        for i in range(5):
            for _ in range(3):
                tasks.append(limiter.is_allowed(f"key{i}"))
        results = await asyncio.gather(*tasks)
        # All 5*3 = 15 requests should be allowed (each key gets its own 3)
        allowed_count = sum(1 for ok, *_ in results if ok)
        self.assertEqual(allowed_count, 15)

    async def test_high_concurrency_no_crash(self):
        """100 concurrent coroutines must not raise or deadlock."""
        limiter = RateLimiter(default_rpm=1000)
        results = await asyncio.gather(*[limiter.is_allowed("flood") for _ in range(100)])
        self.assertEqual(len(results), 100)


class TestRateLimiterRpmCache(unittest.IsolatedAsyncioTestCase):
    """_get_rpm uses a cache with TTL; fallback to default_rpm when no pool."""

    async def test_defaults_to_default_rpm_with_no_pool(self):
        """Without a DB pool, rpm should be the constructor default.

        Exercises the real _get_rpm code path (patches get_pool at its real
        source, ingest_svc.db.postgres, where _get_rpm imports it from) —
        not a stand-in for _get_rpm itself, which would prove nothing about
        the actual no-pool fallback logic inside it.
        """
        limiter = RateLimiter(default_rpm=42)
        with patch("ingest_svc.db.postgres.get_pool", return_value=None):
            rpm = await limiter._get_rpm("some_key")
        self.assertEqual(rpm, 42)

    async def test_db_lookup_failure_falls_back_to_default(self):
        """A DB error (not just an absent pool) must also fall back gracefully."""
        limiter = RateLimiter(default_rpm=17)
        with patch("ingest_svc.db.postgres.get_pool", side_effect=RuntimeError("boom")):
            rpm = await limiter._get_rpm("some_key")
        self.assertEqual(rpm, 17)

    async def test_db_error_never_logs_the_raw_api_key(self):
        """CodeQL: clear-text logging of sensitive info — same finding, same
        fix, as verify_api_key() in db/postgres.py: this query is also bound
        with the raw api_key, so the exception logged on failure must never
        include str(exc) itself, only its type name."""
        secret_key = "dt_live_super_secret_value_12345"

        class _LeakyError(Exception):
            def __str__(self):
                return f"query failed with param={secret_key!r}"

        class _FakeConn:
            async def fetchrow(self, query, key):
                raise _LeakyError()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        class _FakePool:
            def acquire(self):
                return _FakeConn()

        limiter = RateLimiter(default_rpm=99)
        with (
            self.assertLogs("dunetrace.ingest.ratelimit", level="DEBUG") as cm,
            patch("ingest_svc.db.postgres.get_pool", return_value=_FakePool()),
        ):
            rpm = await limiter._get_rpm(secret_key)

        self.assertEqual(rpm, 99)
        logged = "\n".join(cm.output)
        self.assertNotIn(secret_key, logged)
        self.assertIn("_LeakyError", logged)

    async def test_db_rpm_used_when_pool_present(self):
        """When the pool is present and the key is found, the DB's rpm wins
        over the constructor default."""
        limiter = RateLimiter(default_rpm=10)

        class _FakeConn:
            async def fetchrow(self, query, key):
                return {"rate_limit_rpm": 250}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        class _FakePool:
            def acquire(self):
                return _FakeConn()

        with patch("ingest_svc.db.postgres.get_pool", return_value=_FakePool()):
            rpm = await limiter._get_rpm("db_key")
        self.assertEqual(rpm, 250)

    async def test_rpm_is_cached_after_lookup(self):
        """A second _get_rpm call within TTL should hit the cache, not the DB."""
        limiter = RateLimiter(default_rpm=10)
        # Pre-populate the cache (simulating a recent successful DB lookup)
        limiter._rpm_cache["cached_key"] = (999, time.monotonic())
        rpm = await limiter._get_rpm("cached_key")
        self.assertEqual(rpm, 999)


class TestRateLimiterHeartbeat(unittest.IsolatedAsyncioTestCase):
    """_heartbeat coordinates cross-process rate limiting via active worker count."""

    async def test_no_pool_resets_active_workers_to_one(self):
        limiter = RateLimiter()
        limiter._active_workers = 7  # simulate a stale prior reading
        with patch("ingest_svc.db.postgres.get_pool", return_value=None):
            await limiter._heartbeat()
        self.assertEqual(limiter._active_workers, 1)

    async def test_db_failure_preserves_last_known_active_workers(self):
        """A transient DB error must NOT reset active_workers to 1 — that
        would let every worker briefly enforce the full rpm again, causing
        exactly the aggregate overshoot heartbeat coordination prevents."""
        limiter = RateLimiter()
        limiter._active_workers = 4
        with patch("ingest_svc.db.postgres.get_pool", side_effect=RuntimeError("boom")):
            await limiter._heartbeat()
        self.assertEqual(limiter._active_workers, 4)

    async def test_active_workers_set_from_worker_count_query(self):
        limiter = RateLimiter()

        class _FakeConn:
            async def execute(self, query, *args):
                return None

            async def fetchval(self, query):
                return 3

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        class _FakePool:
            def acquire(self):
                return _FakeConn()

        with patch("ingest_svc.db.postgres.get_pool", return_value=_FakePool()):
            await limiter._heartbeat()
        self.assertEqual(limiter._active_workers, 3)

    async def test_zero_count_floors_to_one(self):
        """A COUNT(*) of 0 (shouldn't happen — this worker just inserted its
        own row — but guard against it) must never floor active_workers below 1."""
        limiter = RateLimiter()

        class _FakeConn:
            async def execute(self, query, *args):
                return None

            async def fetchval(self, query):
                return 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        class _FakePool:
            def acquire(self):
                return _FakeConn()

        with patch("ingest_svc.db.postgres.get_pool", return_value=_FakePool()):
            await limiter._heartbeat()
        self.assertEqual(limiter._active_workers, 1)


class TestEffectiveRpmScalesWithActiveWorkers(unittest.IsolatedAsyncioTestCase):
    """is_allowed() divides the configured rpm by the current active-worker
    count, so N workers collectively approximate the configured limit."""

    async def test_single_worker_matches_full_rpm(self):
        """active_workers defaults to 1 — identical to pre-coordination behavior."""
        limiter = RateLimiter(default_rpm=10)
        for _ in range(10):
            allowed, *_ = await limiter.is_allowed("key1")
            self.assertTrue(allowed)
        allowed, *_ = await limiter.is_allowed("key1")
        self.assertFalse(allowed)

    async def test_two_workers_each_get_half_rpm(self):
        limiter = RateLimiter(default_rpm=10)
        limiter._active_workers = 2
        for _ in range(5):
            allowed, *_ = await limiter.is_allowed("key1")
            self.assertTrue(allowed)
        allowed, *_ = await limiter.is_allowed("key1")
        self.assertFalse(allowed)

    async def test_effective_rpm_never_floors_below_one(self):
        """More workers than configured rpm must still allow at least 1 req/window,
        not zero."""
        limiter = RateLimiter(default_rpm=3)
        limiter._active_workers = 100
        allowed, *_ = await limiter.is_allowed("key1")
        self.assertTrue(allowed)
        allowed, *_ = await limiter.is_allowed("key1")
        self.assertFalse(allowed)


class TestPerAgentSubLimit(unittest.IsolatedAsyncioTestCase):
    """B6: one runaway agent under a key must not starve its siblings' share
    of the same key's budget."""

    async def test_agent_remaining_is_none_without_agent_id(self):
        limiter = RateLimiter(default_rpm=10)
        result = await limiter.is_allowed("key1")
        self.assertIsNone(result.agent_remaining)

    async def test_one_agent_hitting_its_quota_does_not_affect_a_sibling(self):
        # default_rpm=10, default quota 20% -> agent_rpm = max(1, int(10*0.2)) = 2
        limiter = RateLimiter(default_rpm=10)
        r1 = await limiter.is_allowed("key1", "agent-a")
        r2 = await limiter.is_allowed("key1", "agent-a")
        self.assertTrue(r1.allowed)
        self.assertTrue(r2.allowed)
        # agent-a is now at its 2-request quota — a third request must be denied
        # even though the key overall (10 rpm) has plenty of room left.
        r3 = await limiter.is_allowed("key1", "agent-a")
        self.assertFalse(r3.allowed)

        # agent-b, same key, must be unaffected — it has its own quota.
        r4 = await limiter.is_allowed("key1", "agent-b")
        self.assertTrue(r4.allowed)

    async def test_key_level_limit_still_fires_when_total_exceeded(self):
        # Many distinct agents, each well under their own quota, must still
        # collectively trip the key-level cap once the key's own rpm is hit.
        limiter = RateLimiter(default_rpm=3)
        results = [await limiter.is_allowed("key1", f"agent-{i}") for i in range(3)]
        self.assertTrue(all(r.allowed for r in results))
        # 4th request, a brand-new agent under its own fresh quota, must still
        # be denied — the key-level 3 rpm is exhausted regardless of agent.
        r4 = await limiter.is_allowed("key1", "agent-new")
        self.assertFalse(r4.allowed)

    async def test_default_quota_is_20_percent_of_effective_rpm(self):
        limiter = RateLimiter(default_rpm=10)
        for _ in range(2):
            r = await limiter.is_allowed("key1", "agent-a")
            self.assertTrue(r.allowed)
        r = await limiter.is_allowed("key1", "agent-a")
        self.assertFalse(r.allowed)

    async def test_quota_override_widens_the_agents_share(self):
        limiter = RateLimiter(default_rpm=10)
        with patch(
            "ingest_svc.rate_limiter.RateLimiter._get_agent_quota_pct",
            AsyncMock(return_value=0.5),
        ):
            # override -> agent_rpm = max(1, int(10*0.5)) = 5
            for _ in range(5):
                r = await limiter.is_allowed("key1", "agent-a")
                self.assertTrue(r.allowed)
            r = await limiter.is_allowed("key1", "agent-a")
            self.assertFalse(r.allowed)

    async def test_key_remaining_reported_correctly(self):
        limiter = RateLimiter(default_rpm=10)
        result = await limiter.is_allowed("key1")
        self.assertEqual(result.key_remaining, 9)  # 10 - the 1 just recorded

    async def test_agent_remaining_reported_correctly(self):
        limiter = RateLimiter(default_rpm=10)
        result = await limiter.is_allowed("key1", "agent-a")
        self.assertEqual(result.agent_remaining, 1)  # quota 2 - the 1 just recorded

    async def test_evict_stale_also_clears_agent_windows_and_quota_cache(self):
        limiter = RateLimiter(default_rpm=10)
        await limiter.is_allowed("stale_key", "stale_agent")
        self.assertIn(("stale_key", "stale_agent"), limiter._agent_windows)

        # Backdate the agent window's only entry past the 2-minute cutoff.
        limiter._agent_windows[("stale_key", "stale_agent")][0] -= 200
        limiter.evict_stale()

        self.assertNotIn(("stale_key", "stale_agent"), limiter._agent_windows)
        self.assertNotIn(("stale_key", "stale_agent"), limiter._quota_cache)


class TestAgentQuotaCache(unittest.IsolatedAsyncioTestCase):
    async def test_defaults_to_20_percent_with_no_pool(self):
        limiter = RateLimiter()
        pct = await limiter._get_agent_quota_pct("key1", "agent-a")
        self.assertEqual(pct, 0.20)

    async def test_db_override_used_when_present(self):
        limiter = RateLimiter()
        with patch("ingest_svc.db.postgres.get_agent_quota_by_key", AsyncMock(return_value=0.5)):
            pct = await limiter._get_agent_quota_pct("key1", "agent-a")
        self.assertEqual(pct, 0.5)

    async def test_result_is_cached_within_ttl(self):
        limiter = RateLimiter()
        mock_lookup = AsyncMock(return_value=0.5)
        with patch("ingest_svc.db.postgres.get_agent_quota_by_key", mock_lookup):
            await limiter._get_agent_quota_pct("key1", "agent-a")
            await limiter._get_agent_quota_pct("key1", "agent-a")
        mock_lookup.assert_called_once()

    async def test_lookup_failure_falls_back_to_default(self):
        limiter = RateLimiter()
        with patch(
            "ingest_svc.db.postgres.get_agent_quota_by_key",
            side_effect=RuntimeError("boom"),
        ):
            pct = await limiter._get_agent_quota_pct("key1", "agent-a")
        self.assertEqual(pct, 0.20)

    async def test_different_agents_under_the_same_key_cached_independently(self):
        limiter = RateLimiter()

        async def fake_lookup(api_key, agent_id):
            return 0.5 if agent_id == "agent-a" else None

        with patch("ingest_svc.db.postgres.get_agent_quota_by_key", fake_lookup):
            pct_a = await limiter._get_agent_quota_pct("key1", "agent-a")
            pct_b = await limiter._get_agent_quota_pct("key1", "agent-b")
        self.assertEqual(pct_a, 0.5)
        self.assertEqual(pct_b, 0.20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
