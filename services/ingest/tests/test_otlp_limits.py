"""
Phase 3 unit tests for the OTLP failure-isolation primitives: the per-org span
rate limiter, the backpressure guard, and the persist circuit breaker + retry
buffer. Pure unit tests, no HTTP.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingest_svc.otlp_limits import InflightGuard, PersistRetry, SpanRateLimiter


# ── SpanRateLimiter ──────────────────────────────────────────────────────────────


class TestSpanRateLimiter:
    def test_allows_within_budget(self):
        lim = SpanRateLimiter(rate_per_sec=1000)  # burst 2000
        allowed, retry_after = lim.allow("org", 500)
        assert allowed is True and retry_after == 0

    def test_denies_once_drained(self):
        lim = SpanRateLimiter(rate_per_sec=100)  # burst 200
        assert lim.allow("org", 200)[0] is True  # drains the bucket
        allowed, retry_after = lim.allow("org", 200)
        assert allowed is False and retry_after >= 1

    def test_per_org_isolation(self):
        lim = SpanRateLimiter(rate_per_sec=100)  # burst 200
        lim.allow("noisy", 200)  # drain the noisy org
        assert lim.allow("noisy", 200)[0] is False
        # A quiet org's budget is untouched.
        assert lim.allow("quiet", 100)[0] is True

    def test_oversized_batch_allowed_once_then_denied(self):
        lim = SpanRateLimiter(rate_per_sec=100)  # burst 200
        # A single batch bigger than the burst goes through when the bucket is
        # full, so a legitimately large export isn't rejected forever.
        assert lim.allow("org", 1000)[0] is True
        assert lim.allow("org", 1)[0] is False  # then drained

    def test_caps_a_burst_of_requests(self):
        # 100 requests of 50 spans = 5000 spans against a burst of 2000: the
        # limiter admits roughly one burst worth, then sheds the rest.
        lim = SpanRateLimiter(rate_per_sec=1000)  # burst 2000
        admitted = sum(lim.allow("org", 50)[0] for _ in range(100))
        assert 40 <= admitted < 100


# ── InflightGuard ────────────────────────────────────────────────────────────────


class TestInflightGuard:
    def test_bounds_and_releases(self):
        g = InflightGuard(max_inflight=2)
        assert g.try_reserve() is True
        assert g.try_reserve() is True
        assert g.try_reserve() is False  # at cap, shed
        g.release()
        assert g.try_reserve() is True
        assert g.inflight == 2


# ── PersistRetry ─────────────────────────────────────────────────────────────────


class TestPersistRetry:
    @pytest.mark.asyncio
    async def test_success_persists_without_buffering(self):
        seen = []

        async def insert(events, batch_id, org_id):
            seen.append(batch_id)
            return len(events)

        pr = PersistRetry(insert, max_batches=10)
        await pr.persist(["e"], "b1", "org")
        assert seen == ["b1"]
        assert pr.buffered == 0

    @pytest.mark.asyncio
    async def test_failure_buffers_for_retry(self):
        async def insert(events, batch_id, org_id):
            raise RuntimeError("db down")

        pr = PersistRetry(insert, max_batches=10)
        await pr.persist(["e"], "b1", "org")
        assert pr.buffered == 1  # accepted, buffered, not lost

    @pytest.mark.asyncio
    async def test_retry_flushes_on_recovery(self):
        state = {"fail": True}

        async def insert(events, batch_id, org_id):
            if state["fail"]:
                raise RuntimeError("down")
            return 1

        pr = PersistRetry(insert, max_batches=10)
        await pr.persist(["e"], "b1", "org")
        assert pr.buffered == 1
        state["fail"] = False
        flushed = await pr.retry_once()
        assert flushed == 1 and pr.buffered == 0

    @pytest.mark.asyncio
    async def test_circuit_opens_and_stops_hitting_db(self):
        calls = []

        async def insert(events, batch_id, org_id):
            calls.append(batch_id)
            raise RuntimeError("down")

        pr = PersistRetry(insert, max_batches=100)
        for i in range(PersistRetry.FAILURE_THRESHOLD):
            await pr.persist(["e"], f"b{i}", "org")
        calls_at_open = len(calls)
        # Circuit is open now: the next batch is buffered without a DB attempt.
        await pr.persist(["e"], "later", "org")
        assert len(calls) == calls_at_open
        assert pr.buffered == PersistRetry.FAILURE_THRESHOLD + 1

    @pytest.mark.asyncio
    async def test_buffer_drops_oldest_when_full(self):
        async def insert(events, batch_id, org_id):
            raise RuntimeError("down")

        pr = PersistRetry(insert, max_batches=2)
        for i in range(5):
            await pr.persist(["e"], f"b{i}", "org")
        assert pr.buffered == 2  # bounded, not unbounded growth
