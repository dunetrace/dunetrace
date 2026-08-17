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

        generations = await provider.fetch_generations(since)

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
        if stored:
            logger.info("elevenlabs poll — org=%s new_generations=%d", org_id, stored)

    except Exception as exc:
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

    await init_pool()
    await ensure_elevenlabs_schema()
    logger.info("ElevenLabs worker started. wake_interval=%ss", settings.WAKE_INTERVAL)
    try:
        while True:
            try:
                count = await poll_once()
                if count:
                    logger.info("Cycle complete. integrations_polled=%d", count)
            except Exception as exc:
                logger.error("Poll cycle failed: %s", exc)
            # Correlation runs after polling, over already-stored generations, so
            # it never blocks or delays the fetch/store of new ElevenLabs data.
            # Its own failures are isolated from polling and from the loop.
            try:
                await correlate_once()
            except Exception as exc:
                logger.error("Correlation pass failed: %s", exc)
            await asyncio.sleep(settings.WAKE_INTERVAL)
    except asyncio.CancelledError:
        logger.info("ElevenLabs worker cancelled")
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(run_worker())
