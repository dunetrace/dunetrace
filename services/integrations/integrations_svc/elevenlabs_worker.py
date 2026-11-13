"""Polling worker for the ElevenLabs pull integration (Phase 4.3).

Separate from worker.py by design. worker.py polls evaluation providers whose
results become failure_signals correlated by trace_id. This worker pulls TTS
generation history and stores it (elevenlabs_generations) for Phase 4.4 to
correlate to tts.generated events by timestamp/character-count/voice. Running
it as its own process keeps the two failure domains isolated: ElevenLabs being
down or rate-limited never affects the evaluation poller, and vice versa.

Trigger mechanism is the same as every other worker here: wake on a fixed
interval, check which orgs are due on their own poll_interval_secs (default 5
minutes, conservative), poll each. No message broker.
"""

from __future__ import annotations

import asyncio
import logging
import time

from dunetrace_schemas import metrics as _metrics
from dunetrace_schemas.migrations import CURRENT_SCHEMA_VERSION

import integrations_svc.db as _db
from integrations_svc.config import settings
from integrations_svc.correlation import correlate_once
from integrations_svc.crypto import decrypt_credentials
from integrations_svc.providers.elevenlabs import ElevenLabsProvider
from integrations_svc.db import (
    close_pool,
    ensure_elevenlabs_schema,
    fetch_due_elevenlabs_integrations,
    init_pool,
    record_elevenlabs_alert_sent,
    record_elevenlabs_poll_failure,
    record_elevenlabs_poll_success,
    store_generation,
    write_integration_down_signal,
)

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("dunetrace.elevenlabs")

# ── Metrics & readiness ───────────────────────────────────────────────────────
# Served by dunetrace_schemas.metrics' stdlib HTTP server on ELEVENLABS_METRICS_PORT
# (/metrics, /ready, /health). run_worker() starts it in the enabled path only:
# a disabled worker logs one line and exits 0, and there is nothing to scrape.
_M_BACKLOG = _metrics.gauge(
    "dunetrace_elevenlabs_backlog",
    "ElevenLabs integrations (accounts) due for a history poll at the most recent wake.",
)
_M_PROCESSED = _metrics.counter(
    "dunetrace_elevenlabs_processed_total",
    "ElevenLabs integrations polled, by outcome (ok | error).",
    ("result",),
)
_M_FAILURES = _metrics.counter(
    "dunetrace_elevenlabs_failures_total",
    "Errors the worker contained, by where they happened.",
    ("where",),
)
_M_EXTERNAL_SECONDS = _metrics.histogram(
    "dunetrace_elevenlabs_external_call_seconds",
    "Wall-clock seconds per outbound provider fetch (all pages of one poll).",
    ("provider",),
    buckets=_metrics.DEFAULT_LATENCY_BUCKETS,
)
_M_EXTERNAL_CALLS = _metrics.counter(
    "dunetrace_elevenlabs_external_calls_total",
    "Outbound provider fetches, by provider and outcome (ok | error).",
    ("provider", "status"),
)
_M_POLL_SECONDS = _metrics.histogram(
    "dunetrace_elevenlabs_poll_seconds",
    "Wall-clock seconds per wake cycle.",
    buckets=_metrics.DEFAULT_LATENCY_BUCKETS,
)

# /ready reports the poll loop stale once this many WAKE_INTERVALs have passed
# without a successful cycle.
_STALE_AFTER_INTERVALS = 3
# A cycle that is STILL RUNNING gets this much longer before it reads as wedged.
# Freshness used to be computed purely from the last COMPLETED cycle against
# 3 x the poll interval, but one cycle pages an entire call history per configured account, then correlates —
# so the worker reported unhealthy for most of every busy cycle, which is
# precisely when it has work. An orchestrator acting on that restarts the
# busiest worker mid-batch. The grace is a bound, not an exemption: a genuinely
# wedged loop still goes stale, just later.
_INFLIGHT_GRACE_SECS = 300.0
# time.monotonic() of the last successful poll cycle. run_worker() seeds it
# once the schema is ready, so the first cycle gets the full grace window;
# None means the worker has not reached its loop yet.
_last_poll_ok_at: float | None = None
# time.monotonic() of when the current (or last) cycle STARTED. A slow cycle and
# a dead loop look identical from completion times alone; this is what separates
# them.
_poll_started_at: float | None = None


def _mark_poll_started() -> None:
    global _poll_started_at
    _poll_started_at = time.monotonic()


def _mark_poll_ok() -> None:
    global _last_poll_ok_at
    _last_poll_ok_at = time.monotonic()


def _poll_freshness(last_ok_at: float | None, now: float, interval: float) -> tuple[bool, dict]:
    """Whether the poll loop is still turning.

    Fresh while the last SUCCESSFUL cycle is within _STALE_AFTER_INTERVALS ×
    interval — only successful cycles count, so a loop whose every cycle fails
    (DB gone, a wedged evaluator) reads as not-ready rather than as healthy
    because the process happens to be alive.

    A cycle that is STILL RUNNING is judged against _INFLIGHT_GRACE_SECS
    instead, so a long-but-healthy cycle is not mistaken for a dead one. See
    the constant for why that distinction is load-bearing here.
    """
    in_flight = _poll_started_at is not None and (
        last_ok_at is None or _poll_started_at > last_ok_at
    )
    if in_flight:
        age = max(0.0, now - float(_poll_started_at))
        limit = max(_STALE_AFTER_INTERVALS * interval, _INFLIGHT_GRACE_SECS)
        fresh = age <= limit
        return fresh, {
            "poll": "ok" if fresh else "stale",
            "in_flight": True,
            "last_poll_age_seconds": round(age, 3),
            "stale_after_seconds": limit,
        }
    if last_ok_at is None:
        return False, {"poll": "never", "in_flight": False}
    age = max(0.0, now - last_ok_at)
    limit = _STALE_AFTER_INTERVALS * interval
    fresh = age <= limit
    return fresh, {
        "poll": "ok" if fresh else "stale",
        "in_flight": False,
        "last_poll_age_seconds": round(age, 3),
        "stale_after_seconds": limit,
    }


async def _ready_check() -> tuple[bool, dict]:
    """GET /ready: the DB answers at the required schema version AND a poll
    cycle completed recently. The metrics server thread schedules this onto the
    worker's event loop with a short timeout, so a wedged loop is a 503 rather
    than a hung probe."""
    pool = _db._pool
    if pool is None:
        db_ok, info = False, {"db": "no_pool", "required": CURRENT_SCHEMA_VERSION}
    else:
        db_ok, info = await _metrics.db_ready(pool, CURRENT_SCHEMA_VERSION)
    fresh, poll_info = _poll_freshness(_last_poll_ok_at, time.monotonic(), settings.WAKE_INTERVAL)
    info.update(poll_info)
    return db_ok and fresh, info


async def _observed_call(provider: str, fn, *args):
    """Await one outbound provider call and observe its duration and outcome.

    The exception is re-raised: _poll_one's own except decides what a failure
    costs (failure bookkeeping, the >30min operational signal), this only
    observes it.
    """
    label = str(provider or "unknown")
    started = time.perf_counter()
    status = "error"
    try:
        result = await fn(*args)
        status = "ok"
        return result
    finally:
        _M_EXTERNAL_SECONDS.labels(provider=label).observe(time.perf_counter() - started)
        _M_EXTERNAL_CALLS.labels(provider=label, status=status).inc()


_PROVIDER = "elevenlabs"

# Re-fetch a window before the high-water mark to tolerate any lag in a
# generation appearing in the history list. Dedup on (org_id, generation_id)
# makes the re-fetch cheap and idempotent. Same rationale as worker.py's
# _OVERLAP_SECS, sized generously since a missed generation is not otherwise
# recoverable without a customer noticing correlation never happened.
_OVERLAP_SECS = 300

# Mirror worker.py's operational-alert policy exactly: once a provider has been
# unreachable this long, write one operational signal (no dedicated operator
# channel exists in this codebase — a documented gap), rate-limited so we don't
# re-write it every cycle while still down.
_FAILURE_ALERT_THRESHOLD_SECS = 30 * 60
_ALERT_RATE_LIMIT_SECS = 60 * 60


async def _poll_one(integration: dict) -> None:
    org_id = integration["org_id"]

    try:
        creds = decrypt_credentials(integration["encrypted_credentials"])
        provider = ElevenLabsProvider(**creds)

        last_seen = integration["last_seen_generation_at"]
        # First poll (no high-water mark yet) starts from when the integration
        # was connected, NOT the beginning of the customer's ElevenLabs history:
        # we correlate forward from connect rather than backfilling an unbounded
        # history on first run.
        base_epoch = last_seen if last_seen is not None else integration["created_at"].timestamp()
        since = base_epoch - _OVERLAP_SECS

        generations = await _observed_call(_PROVIDER, provider.fetch_generations, since)

        stored = 0
        newest = last_seen or 0.0
        for gen in generations:
            if await store_generation(org_id, gen):
                stored += 1
            if gen.generated_at > newest:
                newest = gen.generated_at

        # Advance the high-water mark only when we actually saw generations;
        # None tells record_* to leave it unchanged.
        new_hwm = newest if generations else None
        await record_elevenlabs_poll_success(integration["id"], new_hwm)
        _M_PROCESSED.labels(result="ok").inc()
        if stored:
            logger.info("elevenlabs poll — org=%s new_generations=%d", org_id, stored)

    except Exception as exc:
        _M_PROCESSED.labels(result="error").inc()
        _M_FAILURES.labels(where="integration").inc()
        logger.warning("elevenlabs poll failed — org=%s error=%s", org_id, exc)
        state = await record_elevenlabs_poll_failure(integration["id"])
        first_failure_at = state["first_failure_at"]
        last_alerted_at = state["last_alerted_at"]

        if (
            first_failure_at
            and (time.time() - first_failure_at.timestamp()) > _FAILURE_ALERT_THRESHOLD_SECS
        ):
            already_alerted_recently = (
                last_alerted_at
                and (time.time() - last_alerted_at.timestamp()) < _ALERT_RATE_LIMIT_SECS
            )
            if not already_alerted_recently:
                await write_integration_down_signal(org_id, _PROVIDER, str(exc))
                await record_elevenlabs_alert_sent(integration["id"])
                logger.error(
                    "elevenlabs integration down >30min — org=%s consecutive_failures=%d",
                    org_id,
                    state["consecutive_failures"],
                )


async def poll_once() -> int:
    """Poll every due ElevenLabs integration. Returns how many were polled.
    Different orgs use different API keys (different ElevenLabs accounts, so
    independent concurrency limits), so polling them in parallel is safe; each
    org's own history pagination stays sequential, keeping us at concurrency 1
    per account."""
    integrations = await fetch_due_elevenlabs_integrations()
    _M_BACKLOG.set(len(integrations))
    if not integrations:
        return 0
    await asyncio.gather(*[_poll_one(i) for i in integrations])
    return len(integrations)


async def run_worker() -> None:
    if not settings.ELEVENLABS_WORKER_ENABLED:
        logger.info(
            "ElevenLabs worker disabled (ELEVENLABS_WORKER_ENABLED=false) — "
            "exiting without opening a DB connection."
        )
        return

    # Observability comes up first so a slow DB start reads as a 503 /ready
    # rather than a connection refused. start_metrics_server never raises —
    # metrics are observability, not a dependency; a port of 0 disables it.
    _metrics.register_standard("elevenlabs", settings.APP_VERSION)
    metrics_server = _metrics.start_metrics_server(
        settings.ELEVENLABS_METRICS_PORT, _ready_check, asyncio.get_running_loop()
    )

    await init_pool()
    await ensure_elevenlabs_schema()
    # ensure_elevenlabs_schema applied, then required, CURRENT_SCHEMA_VERSION —
    # so that is the version this process is now running against.
    _metrics.set_schema_version(CURRENT_SCHEMA_VERSION)
    _mark_poll_ok()  # seed /ready freshness: the first cycle gets the full grace window
    logger.info(
        "ElevenLabs worker started. wake_interval=%ss metrics_port=%s",
        settings.WAKE_INTERVAL,
        settings.ELEVENLABS_METRICS_PORT,
    )
    try:
        while True:
            started = time.perf_counter()
            _mark_poll_started()
            try:
                count = await poll_once()
                _mark_poll_ok()
                if count:
                    logger.info("Cycle complete. integrations_polled=%d", count)
            except Exception as exc:
                _M_FAILURES.labels(where="cycle").inc()
                logger.error("Poll cycle failed: %s", exc)
            # Correlation runs after polling, over already-stored generations, so
            # it never blocks or delays the fetch/store of new ElevenLabs data.
            # Its own failures are isolated from polling and from the loop.
            try:
                await correlate_once()
            except Exception as exc:
                _M_FAILURES.labels(where="correlation").inc()
                logger.error("Correlation pass failed: %s", exc)
            _M_POLL_SECONDS.observe(time.perf_counter() - started)
            await asyncio.sleep(settings.WAKE_INTERVAL)
    except asyncio.CancelledError:
        logger.info("ElevenLabs worker cancelled")
    finally:
        if metrics_server is not None:
            try:
                metrics_server.shutdown()
                metrics_server.server_close()
            except Exception:  # shutdown must never mask the real exit reason
                logger.debug("metrics server shutdown failed", exc_info=True)
        await close_pool()


if __name__ == "__main__":
    asyncio.run(run_worker())
