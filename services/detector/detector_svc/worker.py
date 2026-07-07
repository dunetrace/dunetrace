"""
Polling worker that picks up completed runs, rebuilds their state from events,
runs all detectors, and stores any signals found.
"""

from __future__ import annotations

import asyncio
import logging

from dunetrace.detectors import CUSTOM_DETECTOR_REGISTRY, HandoffContextLossDetector, run_detectors
from dunetrace.models import FailureSignal, FailureType, Severity
from dunetrace.risk_engine import RiskEngine
from detector_svc.detectors import get_detectors

from detector_svc.config import settings
from detector_svc.config_loader import load_custom_detector_budget
from detector_svc.custom_detector import evaluate_custom_detector
from detector_svc.custom_python_detectors import load_custom_detector_plugins
from detector_svc.db import (
    LIVE_DETECTORS,
    close_pool,
    ensure_detector_schema,
    fetch_completed_runs,
    fetch_custom_detectors,
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
    record_custom_detector_results,
    write_custom_signal,
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

# Loaded once at startup, like detector_svc/detectors.py's _CONFIG — restart the
# detector container to apply a detectors.yml change.
_CUSTOM_DETECTOR_BUDGET = load_custom_detector_budget()

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


def _accumulated_context_text(events: list[dict]) -> str:
    """A proxy for "everything this run currently knows": its own input_text
    plus every llm.responded/tool.responded output text seen so far. Used to
    compare a parent run's state against a child run's input at a handoff —
    see HandoffContextLossDetector's docstring."""
    parts = []
    for e in events:
        event_type = e.get("event_type")
        payload = e.get("payload") or {}
        if event_type == "run.started":
            input_text = payload.get("input_text")
            if input_text:
                parts.append(input_text)
        elif event_type in ("llm.responded", "tool.responded"):
            output = payload.get("output")
            if output:
                parts.append(output)
    return "\n".join(parts)


def _handoff_signal_from_events(
    events: list[dict],
    parent_events: list[dict],
    run_id: str,
    agent_id: str,
    agent_version: str,
    detector: HandoffContextLossDetector,
) -> FailureSignal | None:
    """Cross-run comparison for HANDOFF_CONTEXT_LOSS — see process_run()'s
    caller for how parent_events is fetched and filtered."""
    if not parent_events:
        return None
    child_started = next((e for e in events if e.get("event_type") == "run.started"), None)
    if child_started is None:
        return None
    child_input = (child_started.get("payload") or {}).get("input_text", "")
    # Only count what the parent knew up to the moment of handoff — later
    # parent activity (concurrent or after) must not leak into this signal.
    cutoff = child_started.get("timestamp", float("inf"))
    parent_context = _accumulated_context_text(
        [e for e in parent_events if e.get("timestamp", 0) <= cutoff]
    )
    return detector.evaluate_handoff(parent_context, child_input, run_id, agent_id, agent_version)


async def process_run(
    run_id: str,
    agent_id: str,
    agent_version: str,
    trigger: str,
    org_id: str,
) -> int:
    events = await fetch_run_events(run_id)
    if not events:
        await mark_run_processed(run_id, agent_id, agent_version, trigger, 0, org_id)
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
            fetch_step_count_baseline(org_id, agent_id, agent_version, run_id),
            fetch_latency_baseline(org_id, agent_id, agent_version, run_id, "tool.called"),
            fetch_latency_baseline(org_id, agent_id, agent_version, run_id, "llm.called"),
            fetch_token_growth_baseline(org_id, agent_id, agent_version, run_id),
            fetch_llm_tool_ratio_baseline(org_id, agent_id, agent_version, run_id),
            fetch_total_tokens_baseline(org_id, agent_id, agent_version, run_id),
            fetch_duration_baseline(org_id, agent_id, agent_version, run_id),
        )
        detectors = get_detectors(agent_id)
        signals = run_detectors(state, detectors=detectors)
        inj = _injection_signal_from_events(events, run_id, agent_id, agent_version)
        if inj:
            signals.append(inj)

        parent_run_id = next(
            (e.get("parent_run_id") for e in events if e.get("parent_run_id")), None
        )
        if parent_run_id:
            handoff_detector = next(
                (d for d in detectors if isinstance(d, HandoffContextLossDetector)), None
            )
            if handoff_detector is not None:
                parent_events = await fetch_run_events(parent_run_id)
                handoff = _handoff_signal_from_events(
                    events, parent_events, run_id, agent_id, agent_version, handoff_detector
                )
                if handoff:
                    signals.append(handoff)

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
    except Exception:
        logger.exception("Run processing failed. run_id=%s", run_id)
        await mark_run_processed(run_id, agent_id, agent_version, trigger, 0, org_id)
        return 0

    count = 0
    for signal in signals:
        plugin_name = (
            signal.evidence.get("detector_name")
            if signal.failure_type == FailureType.CUSTOM
            else None
        )
        if plugin_name:
            # A third-party Python-class custom detector (see
            # custom_python_detectors.py) — FailureType is a closed enum, so
            # it can't carry the plugin's own identity; evidence["detector_name"]
            # does instead, same convention JSON-config custom detectors already
            # use. Written via the same TEXT-failure_type path as those, not
            # write_signals() (which is enum-constrained and LIVE_DETECTORS-gated,
            # neither of which apply to a plugin the built-in allowlist has never
            # heard of).
            plugin_cls = next(
                (c for c in CUSTOM_DETECTOR_REGISTRY.values() if c.name == plugin_name), None
            )
            shadow = plugin_cls.SHADOW_BY_DEFAULT if plugin_cls else True
            await write_custom_signal(
                failure_type=plugin_name,
                severity=signal.severity.value,
                run_id=signal.run_id,
                agent_id=signal.agent_id,
                agent_version=signal.agent_version,
                step_index=signal.step_index,
                confidence=signal.confidence,
                evidence=signal.evidence,
                shadow=shadow,
                org_id=org_id,
            )
            count += 1
            continue

        is_live = signal.failure_type.value in LIVE_DETECTORS
        written = await write_signals([signal], shadow=not is_live, org_id=org_id)
        count += written

    # Issue persistence: track open/resolved lifecycle per (org_id, agent_id, failure_type)
    fired_types = [s.failure_type.value for s in signals if s.failure_type.value in LIVE_DETECTORS]
    try:
        if fired_types:
            await upsert_fired_issues(org_id, agent_id, fired_types)
        await advance_clean_runs(org_id, agent_id, fired_types)
    except Exception as exc:
        logger.warning("Issue tracking failed for run_id=%s: %s", run_id, exc)

    # Custom detectors — run after built-ins, tracked separately
    try:
        custom_defs = await fetch_custom_detectors(org_id, agent_id)
        if custom_defs:
            cdr_records = []
            for cd in custom_defs:
                result = evaluate_custom_detector(
                    cd["config"],
                    state,
                    evaluation_budget_ms=_CUSTOM_DETECTOR_BUDGET["evaluation_budget_ms"],
                    regex_timeout_ms=_CUSTOM_DETECTOR_BUDGET["regex_timeout_ms"],
                )
                fired = result is not None
                cdr_records.append(
                    {
                        "detector_id": cd["id"],
                        "run_id": run_id,
                        "agent_id": agent_id,
                        "fired": fired,
                    }
                )
                if fired:
                    await write_custom_signal(
                        failure_type=result["failure_type"],
                        severity=result["severity"],
                        run_id=run_id,
                        agent_id=agent_id,
                        agent_version=agent_version,
                        step_index=result["step_index"],
                        confidence=result["confidence"],
                        evidence=result["evidence"],
                        shadow=cd["shadow"],
                        org_id=org_id,
                    )
                    count += 1
            await record_custom_detector_results(cdr_records, org_id)
    except Exception as exc:
        logger.warning("Custom detector processing failed for run_id=%s: %s", run_id, exc)

    await mark_run_processed(run_id, agent_id, agent_version, trigger, count, org_id)
    return count


async def poll_once() -> tuple[int, int]:
    completed = await fetch_completed_runs(
        limit=settings.BATCH_SIZE,
        shard_count=settings.SHARD_COUNT,
        shard_index=settings.SHARD_INDEX,
    )
    stalled = await fetch_stalled_runs(
        stall_timeout_secs=settings.STALL_TIMEOUT_SECS,
        limit=settings.BATCH_SIZE,
        shard_count=settings.SHARD_COUNT,
        shard_index=settings.SHARD_INDEX,
    )
    runs = completed + stalled
    if not runs:
        return 0, 0

    semaphore = asyncio.Semaphore(settings.DETECTOR_CONCURRENCY)

    async def process_run_bounded(r):
        async with semaphore:
            return await process_run(
                r["run_id"],
                r["agent_id"],
                r["agent_version"],
                r.get("trigger", "unknown"),
                r["org_id"],
            )

    results = await asyncio.gather(*[process_run_bounded(r) for r in runs])
    return len(runs), sum(results)


async def run_worker() -> None:
    await init_pool()
    await ensure_detector_schema()
    loaded = load_custom_detector_plugins()
    if loaded:
        logger.info("Loaded %d custom detector plugin file(s).", loaded)
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
