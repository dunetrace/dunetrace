"""
Polling worker for external evaluation integrations (Phase 2.1: Langfuse;
Phase 2.2: LangSmith; 2.3 adds Braintrust as a further sibling provider under
the same shape).

Trigger mechanism mirrors every other worker in this codebase: poll on a
fixed interval, no message broker. Distinct from semantic_worker/detector_worker
in what it polls, though — not Dunetrace's own `events` table for work to do,
but each configured org's *external* provider API, on that org's own
poll_interval_secs (independent of this worker's own wake cadence).
"""

from __future__ import annotations

import asyncio
import logging
import time

from dunetrace_schemas import metrics as _metrics
from dunetrace_schemas.migrations import CURRENT_SCHEMA_VERSION

import integrations_svc.db as _db
from integrations_svc.config import settings
from integrations_svc.crypto import decrypt_credentials
from integrations_svc.providers.braintrust import BraintrustProvider
from integrations_svc.providers.langfuse import LangfuseProvider
from integrations_svc.providers.langsmith import LangSmithProvider
from integrations_svc.db import (
    close_pool,
    ensure_integrations_schema,
    fetch_due_integrations,
    fetch_run_by_trace_id,
    has_processed,
    init_pool,
    mark_processed,
    record_alert_sent,
    record_poll_failure,
    record_poll_success,
    write_external_signal,
    write_integration_down_signal,
)

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("dunetrace.integrations")

# ── Metrics & readiness ───────────────────────────────────────────────────────
# Served by dunetrace_schemas.metrics' stdlib HTTP server on INTEGRATIONS_METRICS_PORT
# (/metrics, /ready, /health). run_worker() starts it in the enabled path only:
# a disabled worker logs one line and exits 0, and there is nothing to scrape.
_M_BACKLOG = _metrics.gauge(
    "dunetrace_integrations_backlog",
    "Evaluation integrations due for a poll at the most recent wake, across every provider.",
)
_M_PROCESSED = _metrics.counter(
    "dunetrace_integrations_processed_total",
    "Evaluation integrations polled, by outcome (ok | error).",
    ("result",),
)
_M_FAILURES = _metrics.counter(
    "dunetrace_integrations_failures_total",
    "Errors the worker contained, by where they happened.",
    ("where",),
)
_M_EXTERNAL_SECONDS = _metrics.histogram(
    "dunetrace_integrations_external_call_seconds",
    "Wall-clock seconds per outbound provider fetch (all pages of one poll).",
    ("provider",),
    buckets=_metrics.DEFAULT_LATENCY_BUCKETS,
)
_M_EXTERNAL_CALLS = _metrics.counter(
    "dunetrace_integrations_external_calls_total",
    "Outbound provider fetches, by provider and outcome (ok | error).",
    ("provider", "status"),
)
_M_POLL_SECONDS = _metrics.histogram(
    "dunetrace_integrations_poll_seconds",
    "Wall-clock seconds per wake cycle.",
    buckets=_metrics.DEFAULT_LATENCY_BUCKETS,
)

# /ready reports the poll loop stale once this many WAKE_INTERVALs have passed
# without a successful cycle.
_STALE_AFTER_INTERVALS = 3
# A cycle that is STILL RUNNING gets this much longer before it reads as wedged.
# Freshness used to be computed purely from the last COMPLETED cycle against
# 3 x the poll interval, but one cycle pages an entire provider history per configured integration —
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


# Each provider's class, keyed by the same `provider` string stored in
# external_evaluation_integrations. The constructor is always called as
# provider_cls(endpoint_url, **decrypted_credentials) — every provider's
# __init__ kwarg names must exactly match what its own api_svc config
# endpoint stored in encrypted_credentials (see routers/integrations.py's
# per-provider request bodies).
_PROVIDER_CLASSES = {
    "langfuse": LangfuseProvider,
    "langsmith": LangSmithProvider,
    "braintrust": BraintrustProvider,
}

# Tolerate each provider's own indexing lag with a large safety margin —
# measured ~20-30s for Langfuse's scores endpoint, ~5s for LangSmith's
# feedback endpoint, no measured lag at all for Braintrust's fetch endpoint
# (all three against real test projects — see BACKLOG.md). Re-fetching a few
# already-processed evaluations is cheap (has_processed() skips them);
# missing a delayed one because the window was too tight is not recoverable
# without a customer noticing a signal never arrived.
_OVERLAP_SECS = 300

# Once a provider has been unreachable for this long, write an operational
# signal (see db.write_integration_down_signal). No dedicated operator-alert
# channel exists anywhere in this codebase to push to instead — confirmed
# during Phase 2.1 discovery; this is a documented gap, not an oversight.
_FAILURE_ALERT_THRESHOLD_SECS = 30 * 60
# Don't re-write the operational signal every single poll cycle while still down.
_ALERT_RATE_LIMIT_SECS = 60 * 60


async def _poll_one(provider_name: str, integration: dict) -> None:
    org_id = integration["org_id"]

    try:
        creds = decrypt_credentials(integration["encrypted_credentials"])
        provider_cls = _PROVIDER_CLASSES[provider_name]
        provider = provider_cls(integration["endpoint_url"], **creds)

        since_dt = integration["last_success_at"] or integration["created_at"]
        since = since_dt.timestamp() - _OVERLAP_SECS

        evaluations = await _observed_call(provider_name, provider.fetch_new_evaluations, since)
        written = 0
        for ev in evaluations:
            if await has_processed(org_id, provider_name, ev.external_id):
                continue

            run = await fetch_run_by_trace_id(org_id, ev.trace_id)
            if run is None:
                # Can't correlate to a Dunetrace run (predates trace_id
                # support, wasn't instrumented with it, or isn't a Dunetrace
                # run at all) — mark processed anyway so this same
                # evaluation isn't re-checked forever.
                await mark_processed(org_id, provider_name, ev.external_id)
                continue

            # Not every provider's score is guaranteed to be 0-1 (categorical/
            # arbitrary-scale numeric scores exist) — only trust a numeric
            # value in range as a confidence proxy; otherwise fall back to a
            # neutral midpoint and let the raw value speak for itself in
            # evidence, rather than fabricating false precision.
            confidence = ev.value if (ev.value is not None and 0.0 <= ev.value <= 1.0) else 0.5

            await write_external_signal(
                org_id=org_id,
                agent_id=run["agent_id"],
                agent_version=run["agent_version"],
                run_id=run["run_id"],
                provider=provider_name,
                failure_type=f"{provider_name.upper()}_{ev.name.upper()}",
                confidence=confidence,
                evidence={
                    "raw_value": ev.value,
                    "string_value": ev.string_value,
                    "comment": ev.comment,
                    "source_url": ev.source_url,
                },
            )
            await mark_processed(org_id, provider_name, ev.external_id)
            written += 1

        await record_poll_success(integration["id"])
        _M_PROCESSED.labels(result="ok").inc()
        if written:
            logger.info("%s poll — org=%s new_signals=%d", provider_name, org_id, written)

    except Exception as exc:
        _M_PROCESSED.labels(result="error").inc()
        _M_FAILURES.labels(where="integration").inc()
        logger.warning("%s poll failed — org=%s error=%s", provider_name, org_id, exc)
        state = await record_poll_failure(integration["id"])
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
                await write_integration_down_signal(org_id, provider_name, str(exc))
                await record_alert_sent(integration["id"])
                logger.error(
                    "%s integration down >30min — org=%s consecutive_failures=%d",
                    provider_name,
                    org_id,
                    state["consecutive_failures"],
                )


async def poll_once() -> int:
    """Returns the total number of integrations polled this cycle, across
    every configured provider."""
    total = 0
    for provider_name in _PROVIDER_CLASSES:
        integrations = await fetch_due_integrations(provider_name)
        if not integrations:
            continue
        await asyncio.gather(*[_poll_one(provider_name, i) for i in integrations])
        total += len(integrations)
    _M_BACKLOG.set(total)
    return total


async def run_worker() -> None:
    if not settings.INTEGRATIONS_WORKER_ENABLED:
        logger.info(
            "Integrations worker disabled (INTEGRATIONS_WORKER_ENABLED=false) — "
            "exiting without opening a DB connection."
        )
        return

    # Observability comes up first so a slow DB start reads as a 503 /ready
    # rather than a connection refused. start_metrics_server never raises —
    # metrics are observability, not a dependency; a port of 0 disables it.
    _metrics.register_standard("integrations", settings.APP_VERSION)
    metrics_server = _metrics.start_metrics_server(
        settings.INTEGRATIONS_METRICS_PORT, _ready_check, asyncio.get_running_loop()
    )

    await init_pool()
    await ensure_integrations_schema()
    # ensure_integrations_schema applied, then required, CURRENT_SCHEMA_VERSION
    # — so that is the version this process is now running against.
    _metrics.set_schema_version(CURRENT_SCHEMA_VERSION)
    _mark_poll_ok()  # seed /ready freshness: the first cycle gets the full grace window
    logger.info(
        "Integrations worker started. wake_interval=%ss metrics_port=%s",
        settings.WAKE_INTERVAL,
        settings.INTEGRATIONS_METRICS_PORT,
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
            except Exception:
                _M_FAILURES.labels(where="cycle").inc()
                logger.exception("Poll cycle failed")
            _M_POLL_SECONDS.observe(time.perf_counter() - started)
            await asyncio.sleep(settings.WAKE_INTERVAL)
    except asyncio.CancelledError:
        logger.info("Integrations worker cancelled")
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
