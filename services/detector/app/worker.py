"""
services/detector/app/worker.py

Detect worker loop:
  1) find completed/stalled runs
  2) rebuild RunState from events
  3) run Tier 1 detectors
  4) store signals (shadow by default)
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
_SDK = os.path.join(_ROOT, "packages/sdk-py")
if _SDK not in sys.path:
    sys.path.insert(0, _SDK)

from dunetrace.detectors import run_detectors

from app.config import settings
from app.db import (
    LIVE_DETECTORS,
    close_pool,
    ensure_detector_schema,
    fetch_completed_runs,
    fetch_run_events,
    fetch_stalled_runs,
    init_pool,
    mark_run_processed,
    write_signals,
)
from app.run_builder import build_run_state

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("dunetrace.detector")


async def process_run(
    run_id: str,
    agent_id: str,
    agent_version: str,
    trigger: str,
) -> int:
    events = await fetch_run_events(run_id)
    if not events:
        await mark_run_processed(run_id, agent_id, agent_version, trigger, 0)
        return 0

    try:
        state = build_run_state(events)
        signals = run_detectors(state)
    except Exception as exc:
        logger.error("Run processing failed. run_id=%s err=%s", run_id, exc)
        await mark_run_processed(run_id, agent_id, agent_version, trigger, 0)
        return 0

    count = 0
    for signal in signals:
        is_live = signal.failure_type.value in LIVE_DETECTORS
        written = await write_signals([signal], shadow=not is_live)
        count += written

    await mark_run_processed(run_id, agent_id, agent_version, trigger, count)
    return count


async def poll_once() -> tuple[int, int]:
    completed = await fetch_completed_runs(limit=settings.BATCH_SIZE)
    stalled = await fetch_stalled_runs(
        stall_timeout_secs=settings.STALL_TIMEOUT_SECS,
        limit=settings.BATCH_SIZE,
    )
    runs = completed + stalled
    if not runs:
        return 0, 0

    total_signals = 0
    for r in runs:
        total_signals += await process_run(
            r["run_id"],
            r["agent_id"],
            r["agent_version"],
            r.get("trigger", "unknown"),
        )
    return len(runs), total_signals


async def run_worker() -> None:
    await init_pool()
    await ensure_detector_schema()
    logger.info("Detector worker started. poll_interval=%ss", settings.POLL_INTERVAL)
    try:
        while True:
            try:
                runs, signals = await poll_once()
                if runs:
                    logger.info("Cycle complete. runs=%d signals=%d", runs, signals)
            except Exception as exc:
                logger.error("Poll cycle failed: %s", exc)
            await asyncio.sleep(settings.POLL_INTERVAL)
    except asyncio.CancelledError:
        logger.info("Detector worker cancelled")
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(run_worker())
