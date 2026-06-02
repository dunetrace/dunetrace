"""
Polling worker that picks up completed runs, rebuilds their state from events,
runs all detectors, and stores any signals found.
"""

from __future__ import annotations

import asyncio
import logging

from dunetrace.detectors import run_detectors
from dunetrace.models import FailureSignal, FailureType, Severity
from dunetrace.risk_engine import RiskEngine
from detector_svc.detectors import get_detectors

from detector_svc.config import settings
from detector_svc.db import (
    LIVE_DETECTORS,
    close_pool,
    ensure_detector_schema,
    fetch_completed_runs,
    fetch_duration_baseline,
    fetch_latency_baseline,
    fetch_llm_tool_ratio_baseline,
    fetch_run_events,
    fetch_stalled_runs,
    fetch_step_count_baseline,
    fetch_token_growth_baseline,
    fetch_total_tokens_baseline,
    init_pool,
    mark_run_processed,
    write_signals,
    upsert_fired_issues,
    advance_clean_runs,
)
from detector_svc.run_builder import build_run_state

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("dunetrace.detector")


_COOCCURRENCE_MULTIPLIERS = {1: 1.0, 2: 1.15, 3: 1.30}


def _apply_cooccurrence_boost(signals: list[FailureSignal]) -> None:
    """Raise each signal's confidence when multiple independent signals co-fire.

    Co-occurring signals are strong evidence of a real failure — this reduces
    false positives without touching individual detector thresholds.
    Sets co_signal_count on every signal so the dashboard can show the badge.
    """
    n = len(signals)
    if n < 2:
        return
    multiplier = _COOCCURRENCE_MULTIPLIERS.get(n, 1.40)  # 4+ signals → ×1.40
    for sig in signals:
        sig.confidence = round(min(1.0, sig.confidence * multiplier), 4)
        sig.co_signal_count = n


def _apply_hard_override(signals: list[FailureSignal], risk) -> None:
    """If RiskEngine fired a hard rule, override every signal to CRITICAL/HIGH."""
    if not risk.severity:
        return
    from dunetrace.models import Severity

    sev = Severity(risk.severity)
    for sig in signals:
        sig.severity = sev
        sig.confidence = round(min(1.0, risk.confidence), 4)


def _injection_signal_from_events(
    events: list[dict], run_id: str, agent_id: str, agent_version: str
):
    """Extract prompt injection evidence from the run.started payload and build a FailureSignal. The SDK detects injection on raw input before hashing, so by the time we get here the evidence is already baked into the event."""
    for e in events:
        if e["event_type"] == "run.started":
            evidence = e.get("payload", {}).get("injection_signal")
            if evidence:
                matched = evidence.get("matched_pattern_count", 1)
                confidence = min(1.0, 0.5 + (matched - 1.0) * 0.4)
                return FailureSignal(
                    failure_type=FailureType.PROMPT_INJECTION_SIGNAL,
                    severity=Severity.CRITICAL,
                    run_id=run_id,
                    agent_id=agent_id,
                    agent_version=agent_version,
                    step_index=0,
                    confidence=confidence,
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
        (
            state.baseline_p75_steps,
            state.baseline_p75_latency_tool,
            state.baseline_p75_latency_llm,
            state.baseline_p75_token_growth,
            state.baseline_p75_llm_tool_ratio,
            state.baseline_p75_total_tokens,
            state.baseline_p75_duration_s,
        ) = await asyncio.gather(
            fetch_step_count_baseline(agent_id, agent_version, run_id),
            fetch_latency_baseline(agent_id, agent_version, run_id, "tool.called"),
            fetch_latency_baseline(agent_id, agent_version, run_id, "llm.called"),
            fetch_token_growth_baseline(agent_id, agent_version, run_id),
            fetch_llm_tool_ratio_baseline(agent_id, agent_version, run_id),
            fetch_total_tokens_baseline(agent_id, agent_version, run_id),
            fetch_duration_baseline(agent_id, agent_version, run_id),
        )
        signals = run_detectors(state, detectors=get_detectors(agent_id))
        inj = _injection_signal_from_events(events, run_id, agent_id, agent_version)
        if inj:
            signals.append(inj)

        risk = RiskEngine().evaluate(signals, state)
        logger.debug(
            "RiskEngine. run_id=%s confidence=%.2f active=%d severity=%s scores=%s",
            run_id,
            risk.confidence,
            risk.active_signals,
            risk.severity or "normal",
            risk.scores,
        )
        _apply_hard_override(signals, risk)
        _apply_cooccurrence_boost(signals)
    except Exception as exc:
        logger.error("Run processing failed. run_id=%s err=%s", run_id, exc)
        await mark_run_processed(run_id, agent_id, agent_version, trigger, 0)
        return 0

    count = 0
    for signal in signals:
        is_live = signal.failure_type.value in LIVE_DETECTORS
        written = await write_signals([signal], shadow=not is_live)
        count += written

    # Issue persistence: track open/resolved lifecycle per (agent_id, failure_type)
    fired_types = [s.failure_type.value for s in signals if s.failure_type.value in LIVE_DETECTORS]
    try:
        if fired_types:
            await upsert_fired_issues(agent_id, fired_types)
        await advance_clean_runs(agent_id, fired_types)
    except Exception as exc:
        logger.warning("Issue tracking failed for run_id=%s: %s", run_id, exc)

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

    semaphore = asyncio.Semaphore(settings.DETECTOR_CONCURRENCY)

    async def process_run_bounded(r):
        async with semaphore:
            return await process_run(
                r["run_id"], r["agent_id"], r["agent_version"], r.get("trigger", "unknown")
            )

    results = await asyncio.gather(*[process_run_bounded(r) for r in runs])
    return len(runs), sum(results)


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
