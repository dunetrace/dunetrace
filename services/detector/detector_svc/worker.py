"""
services/detector/detector_svc/worker.py

Detect worker loop:
  1) find completed/stalled runs
  2) rebuild RunState from events
  3) run Tier 1 detectors
  4) store signals (shadow by default)
"""
from __future__ import annotations

import asyncio
import logging

from dunetrace.detectors import run_detectors
from dunetrace.models import FailureSignal, FailureType, Severity
from detector_svc.detectors import get_detectors

from detector_svc.config import settings
from detector_svc.db import (
    LIVE_DETECTORS,
    close_pool,
    ensure_detector_schema,
    fetch_completed_runs,
    fetch_run_events,
    fetch_stalled_runs,
    fetch_step_count_baseline,
    init_pool,
    mark_run_processed,
    write_signals,
)
from detector_svc.run_builder import build_run_state

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("dunetrace.detector")


def _injection_signal_from_events(events: list[dict], run_id: str, agent_id: str, agent_version: str):
    """
    Prompt injection is detected in the SDK on raw input before hashing.
    Evidence is embedded in the run.started payload as 'injection_signal'.
    Extract it here and materialise a FailureSignal.
    """
    for e in events:
        if e["event_type"] == "run.started":
            evidence = e.get("payload", {}).get("injection_signal")
            if evidence:
                return FailureSignal(
                    failure_type=FailureType.PROMPT_INJECTION_SIGNAL,
                    severity=Severity.CRITICAL,
                    run_id=run_id,
                    agent_id=agent_id,
                    agent_version=agent_version,
                    step_index=0,
                    confidence=0.85,
                    evidence=evidence,
                )
    return None


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
        state.baseline_p75_steps = await fetch_step_count_baseline(
            agent_id, agent_version, run_id
        )
        signals = run_detectors(state, detectors=get_detectors())
        inj = _injection_signal_from_events(events, run_id, agent_id, agent_version)
        if inj:
            signals.append(inj)
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
