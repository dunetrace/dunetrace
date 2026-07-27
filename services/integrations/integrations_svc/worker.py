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

        evaluations = await provider.fetch_new_evaluations(since)
        written = 0
        for ev in evaluations:
            if await has_processed(org_id, provider_name, ev.external_id):
                continue

            run = await fetch_run_by_trace_id(ev.trace_id)
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
        if written:
            logger.info("%s poll — org=%s new_signals=%d", provider_name, org_id, written)

    except Exception as exc:
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
    return total


async def run_worker() -> None:
    if not settings.INTEGRATIONS_WORKER_ENABLED:
        logger.info(
            "Integrations worker disabled (INTEGRATIONS_WORKER_ENABLED=false) — "
            "exiting without opening a DB connection."
        )
        return

    await init_pool()
    await ensure_integrations_schema()
    logger.info("Integrations worker started. wake_interval=%ss", settings.WAKE_INTERVAL)
    try:
        while True:
            try:
                count = await poll_once()
                if count:
                    logger.info("Cycle complete. integrations_polled=%d", count)
            except Exception:
                logger.exception("Poll cycle failed")
            await asyncio.sleep(settings.WAKE_INTERVAL)
    except asyncio.CancelledError:
        logger.info("Integrations worker cancelled")
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(run_worker())
