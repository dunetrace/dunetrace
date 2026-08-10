"""
Polling worker that picks up completed runs, rebuilds their state from events,
runs all detectors, and stores any signals found.
"""

from __future__ import annotations

import asyncio
import logging

from dunetrace.detectors import (
    CUSTOM_DETECTOR_REGISTRY,
    DelegationLoopDetector,
    HandoffContextLossDetector,
    run_detectors,
)
from dunetrace.models import FailureSignal, FailureType, Severity
from dunetrace.risk_engine import RiskEngine
from detector_svc.detectors import get_detectors

from detector_svc.config import settings
from detector_svc.config_loader import load_custom_detector_budget
from detector_svc.custom_detector import evaluate_custom_detector
from detector_svc.custom_python_detectors import load_custom_detector_plugins
from detector_svc.db import (
    LIVE_DETECTORS,
    advance_watermark,
    close_pool,
    ensure_detector_schema,
    fetch_completed_runs,
    fetch_custom_detectors,
    fetch_duration_baseline,
    fetch_latency_baseline,
    fetch_llm_tool_ratio_baseline,
    fetch_run_events,
    fetch_run_lineage,
    fetch_run_state,
    fetch_stalled_runs,
    fetch_step_count_baseline,
    fetch_token_growth_baseline,
    fetch_total_tokens_baseline,
    get_watermark,
    prune_processed_runs,
    init_pool,
    MAX_PROCESSING_ATTEMPTS,
    clear_processing_failures,
    mark_run_processed,
    record_processing_failure,
    record_custom_detector_results,
    write_custom_signal,
    write_signals,
    upsert_fired_issues,
    advance_clean_runs,
    upsert_run_and_conversation,
    write_run_state_metrics,
)
from detector_svc.state_metrics import summarize_states
from detector_svc.run_builder import build_run_state
from detector_svc.run_graph import (
    RunNode,
    agent_sequence,
    build_agent_delegation_edges,
    build_ancestor_chain,
    find_cycle,
)

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("dunetrace.detector")

# Loaded once at startup, like detector_svc/detectors.py's _CONFIG — restart the
# detector container to apply a detectors.yml change.
_CUSTOM_DETECTOR_BUDGET = load_custom_detector_budget()

_COOCCURRENCE_MULTIPLIERS = {1: 1.0, 2: 1.15, 3: 1.30}


def _resolve_custom_detector_class(detector_name: str):
    """Find the class behind a TEXT-failure-type signal so its
    SHADOW_BY_DEFAULT can be honored. Searches BOTH sources of such signals:
    CUSTOM_DETECTOR_REGISTRY (customer Python-class plugins) and PACK_REGISTRY
    (first-party pack detectors). Pack detectors are deliberately kept out of
    CUSTOM_DETECTOR_REGISTRY (see BaseDetector.__init_subclass__), so a
    registry-only lookup would never find them and would silently fall back to
    shadow=True — coincidentally the voice pack's own default, but that would
    ignore a pack author who set SHADOW_BY_DEFAULT=False. Returns None if no
    class matches (the caller then defaults to shadow=True)."""
    for cls in CUSTOM_DETECTOR_REGISTRY.values():
        if cls.name == detector_name:
            return cls
    try:
        from dunetrace.packs import PACK_REGISTRY
    except Exception:
        return None
    for pack in PACK_REGISTRY.values():
        for cls in pack.detectors:
            if cls.name == detector_name:
                return cls
    return None


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


async def _delegation_signal_from_chain(
    run_id: str,
    agent_id: str,
    agent_version: str,
    parent_run_id: str,
    detector: DelegationLoopDetector,
) -> FailureSignal | None:
    """Cross-run graph evaluation for DELEGATION_LOOP. Walks this run's
    parent_run_id chain (one lightweight lineage fetch per hop), builds the
    agent-delegation graph, runs DFS cycle detection, and lets the detector
    decide whether it's a sustained loop. See run_graph.py."""
    start = RunNode(
        run_id=run_id,
        agent_id=agent_id,
        agent_version=agent_version,
        parent_run_id=parent_run_id,
    )
    chain = await build_ancestor_chain(start, fetch_run_lineage)
    if len(chain) < 2:
        return None
    cycle = find_cycle(build_agent_delegation_edges(chain))
    return detector.evaluate_delegation_cycle(
        cycle, agent_sequence(chain), run_id, agent_id, agent_version
    )


def _emit_otel_findings(
    run_id: str,
    org_id: str,
    signals: list,
    events: list,
    is_reprocess: bool,
) -> None:
    """Emit signal + policy spans for a processed run into its OTel trace.

    Best-effort and fully isolated: a no-op when OTel export is off (or the
    opentelemetry packages aren't installed in this service), and never raises
    into detection. Policy spans are emitted on first-process only, so a
    reprocess doesn't re-emit them (signals are already deduped by the caller).
    """
    try:
        import dunetrace.otel as _otel

        tracer = _otel.get_tracer()
        if tracer is None:
            return
        from dunetrace.integrations.otel import emit_run_findings

        cfg = _otel.active_config()
        policy_events = (
            [] if is_reprocess else [e for e in events if e.get("event_type") == "policy.triggered"]
        )
        emit_run_findings(
            tracer,
            run_id,
            signals=signals,
            policy_events=policy_events,
            org_id=org_id,
            capture_content=(cfg.capture_content if cfg else True),
        )
    except Exception as exc:
        logger.debug("OTel findings emission skipped for run_id=%s: %s", run_id, exc)


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

    # audit Finding 15: if this run was already processed and new events have since
    # arrived, re-detect ADDITIVELY — write only failure types not already recorded
    # for the run (so a benign run that later turns out to have looped gets its
    # signal, without duplicating existing signals or re-alerting them).
    prior = await fetch_run_state(run_id)
    is_reprocess = prior is not None
    existing_types: set = prior["signal_types"] if prior else set()

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
        detectors = await get_detectors(agent_id, org_id)
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

            delegation_detector = next(
                (d for d in detectors if isinstance(d, DelegationLoopDetector)), None
            )
            if delegation_detector is not None:
                delegation = await _delegation_signal_from_chain(
                    run_id, agent_id, agent_version, parent_run_id, delegation_detector
                )
                if delegation:
                    signals.append(delegation)

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
        logger.exception("Run processing failed. run_id=%s", run_id)
        # Deliberately NOT marked processed on the first failures. This block is
        # not pure computation — it holds seven baseline queries, a detector
        # lookup, a parent fetch and a lineage walk, all DB round-trips — so the
        # common failure here is transient. Marking it processed wrote
        # signal_count=0, a permanent "clean run" verdict nothing could revisit,
        # because a completed run never gains events and so never re-enters the
        # poll. Leaving it unmarked means the next poll retries it.
        attempts = await record_processing_failure(run_id, repr(exc))
        if attempts < MAX_PROCESSING_ATTEMPTS:
            logger.warning(
                "Run %s will be retried (attempt %d of %d)",
                run_id,
                attempts,
                MAX_PROCESSING_ATTEMPTS,
            )
            return 0
        # Budget spent: a deterministically-undetectable run must not be retried
        # on every poll forever. Record it as processed WITH the error, so it is
        # visible as failed rather than indistinguishable from clean.
        logger.error("Run %s failed detection %d times — recording as failed", run_id, attempts)
        await mark_run_processed(
            run_id,
            agent_id,
            agent_version,
            trigger,
            0,
            org_id,
            event_count=len(events),
            processing_error=repr(exc)[:2000],
        )
        await clear_processing_failures(run_id)
        return 0

    count = 0
    for signal in signals:
        plugin_name = (
            signal.evidence.get("detector_name")
            if signal.failure_type == FailureType.CUSTOM
            else None
        )
        # audit Finding 15: dedup writes by (run_id, failure_type). Skip a type
        # already recorded for this run — makes re-detection additive and
        # idempotent (no duplicate signals, no re-alert of an existing one).
        stored_type = plugin_name if plugin_name else signal.failure_type.value
        if stored_type in existing_types:
            continue
        if plugin_name:
            # A third-party Python-class custom detector (see
            # custom_python_detectors.py) — FailureType is a closed enum, so
            # it can't carry the plugin's own identity; evidence["detector_name"]
            # does instead, same convention JSON-config custom detectors already
            # use. Written via the same TEXT-failure_type path as those, not
            # write_signals() (which is enum-constrained and LIVE_DETECTORS-gated,
            # neither of which apply to a plugin the built-in allowlist has never
            # heard of).
            plugin_cls = _resolve_custom_detector_class(plugin_name)
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

    # OTel export (Phase 4): emit dunetrace.signal.* / dunetrace.policy.* child
    # spans into each run's trace so signals decided here show up in the
    # customer's OTel backend alongside the SDK-emitted run/LLM/tool spans. Only
    # LIVE, newly-fired signals (same filter issue tracking uses) so shadow
    # detectors and reprocess duplicates don't leak. No-op when OTel is disabled.
    live_new_signals = [
        s
        for s in signals
        if s.failure_type.value in LIVE_DETECTORS and s.failure_type.value not in existing_types
    ]
    _emit_otel_findings(run_id, org_id, live_new_signals, events, is_reprocess)

    # Issue persistence: track open/resolved lifecycle per (org_id, agent_id, failure_type).
    # audit Finding 15: on a reprocess, only NEW fired types matter, and the
    # consecutive-clean-runs counter must NOT be advanced again for a run it already
    # counted — so advance_clean_runs runs on first-process only.
    fired_types = [
        s.failure_type.value
        for s in signals
        if s.failure_type.value in LIVE_DETECTORS and s.failure_type.value not in existing_types
    ]
    try:
        if fired_types:
            await upsert_fired_issues(org_id, agent_id, fired_types)
        if not is_reprocess:
            await advance_clean_runs(org_id, agent_id, fired_types)
    except Exception as exc:
        logger.warning("Issue tracking failed for run_id=%s: %s", run_id, exc)

    # Conversation modeling (Phase 3.1): register this run in the runs
    # registry, and — when the SDK's dt.run(conversation_id=...) was set —
    # its owning conversation. Isolated in its own try/except, same as issue
    # tracking above, so a bug here never blocks built-in detection.
    try:
        conversation_external_id = next(
            (e.get("conversation_id") for e in events if e.get("conversation_id")), None
        )
        started_at = next(
            (e["timestamp"] for e in events if e["event_type"] == "run.started"),
            events[0]["timestamp"],
        )
        await upsert_run_and_conversation(
            run_id, org_id, agent_id, agent_version, started_at, conversation_external_id
        )
    except Exception as exc:
        logger.warning("Conversation registry update failed for run_id=%s: %s", run_id, exc)

    # Custom detectors — run after built-ins, tracked separately.
    # audit Finding 15: skip on reprocess — the custom signal is already deduped by
    # the type-filter above, and re-running would duplicate custom_detector_results.
    try:
        custom_defs = await fetch_custom_detectors(org_id, agent_id) if not is_reprocess else None
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

    # State metrics (Capability 3, Phase 3.3): precompute per-state time totals
    # for this run so api_svc can build cross-run analytics without re-reading
    # raw events. Own try/except — a bug here never blocks detection.
    try:
        summary = summarize_states(events)
        await write_run_state_metrics(
            run_id, org_id, agent_id, summary["run_started_ts"], summary["states"]
        )
    except Exception as exc:
        logger.warning("State metrics failed for run_id=%s: %s", run_id, exc)

    await mark_run_processed(
        run_id, agent_id, agent_version, trigger, count, org_id, event_count=len(events)
    )
    # A run that eventually succeeded leaves no failure history behind, so its
    # next failure (if any) starts with a full retry budget.
    await clear_processing_failures(run_id)
    return count


async def poll_once() -> tuple[int, int]:
    watermark = await get_watermark(settings.SHARD_INDEX, settings.SHARD_COUNT)
    completed = await fetch_completed_runs(
        limit=settings.BATCH_SIZE,
        shard_count=settings.SHARD_COUNT,
        shard_index=settings.SHARD_INDEX,
        watermark=watermark,
    )
    stalled = await fetch_stalled_runs(
        stall_timeout_secs=settings.STALL_TIMEOUT_SECS,
        limit=settings.BATCH_SIZE,
        shard_count=settings.SHARD_COUNT,
        shard_index=settings.SHARD_INDEX,
        watermark=watermark,
    )
    runs = completed + stalled

    # Only advance the watermark when neither query hit its LIMIT. A full batch
    # means there is more work behind it, and moving the window forward would
    # step over runs that were never processed. Draining first and advancing
    # later costs a few redundant scans; advancing early loses runs.
    drained = len(completed) < settings.BATCH_SIZE and len(stalled) < settings.BATCH_SIZE
    if drained:
        await advance_watermark(
            settings.SHARD_INDEX, settings.WATERMARK_GRACE_SECS, settings.SHARD_COUNT
        )
    elif watermark is None:
        logger.info(
            "Backlog present (completed=%d stalled=%d, batch=%d) — poll watermark "
            "stays unbounded until a cycle drains it.",
            len(completed),
            len(stalled),
            settings.BATCH_SIZE,
        )

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


_PRUNE_INTERVAL = 24 * 60 * 60  # once a day


async def _prune_loop() -> None:
    """Reclaim processed_runs rows whose events have already aged out.

    Only shard 0 prunes: the table isn't shard-partitioned, so every replica
    running this would just contend on the same rows for no extra throughput.

    Keeps deleting while a pass fills its batch, so a long-neglected table
    catches up over successive passes instead of shrinking by one batch a day.
    """
    while True:
        try:
            if settings.SHARD_INDEX == 0:
                total = 0
                while True:
                    deleted = await prune_processed_runs(batch_size=settings.PRUNE_BATCH_SIZE)
                    total += deleted
                    if deleted < settings.PRUNE_BATCH_SIZE:
                        break
                    # Yield between batches so polling isn't starved.
                    await asyncio.sleep(1)
                if total:
                    logger.info(
                        "Pruned %d processed_runs row(s) whose events had already "
                        "aged out of retention.",
                        total,
                    )
        except Exception:
            logger.exception("processed_runs prune failed")
        await asyncio.sleep(_PRUNE_INTERVAL)


async def run_worker() -> None:
    await init_pool()
    await ensure_detector_schema()
    loaded = load_custom_detector_plugins()
    if loaded:
        logger.info("Loaded %d custom detector plugin file(s).", loaded)
    # OTel export (Phase 4): reuse the SDK's config module so signal/policy spans
    # ship to the same OTLP endpoint. Opt-in via DUNETRACE_OTEL_* env; a no-op
    # here when unset. Never raises.
    try:
        import dunetrace.otel as _otel

        if _otel.init():
            logger.info("OTel export enabled for detector signals/policies.")
    except Exception as exc:
        logger.debug("OTel init skipped: %s", exc)
    logger.info(
        "Detector worker started. poll_interval=%ss shard=%d/%d watermark_grace=%ss",
        settings.POLL_INTERVAL,
        settings.SHARD_INDEX,
        settings.SHARD_COUNT,
        settings.WATERMARK_GRACE_SECS,
    )
    prune_task = asyncio.create_task(_prune_loop())
    try:
        while True:
            try:
                runs, signals = await poll_once()
                if runs:
                    logger.info("Cycle complete. runs=%d signals=%d", runs, signals)
            except Exception:
                # logger.exception, not logger.error("...: %s", exc): several
                # exceptions this loop can raise stringify to nothing —
                # asyncpg's connection errors among them — so "%s" produced a
                # bare "Poll cycle failed:" with no cause and no traceback. The
                # only way to attribute those was correlating timestamps against
                # an outage by hand.
                logger.exception("Poll cycle failed")
            await asyncio.sleep(settings.POLL_INTERVAL)
    except asyncio.CancelledError:
        logger.info("Detector worker cancelled")
    finally:
        prune_task.cancel()
        await close_pool()


if __name__ == "__main__":
    asyncio.run(run_worker())
