"""
Polling worker that picks up completed runs, rebuilds their state from events,
runs all detectors, and stores any signals found.
"""

from __future__ import annotations

import asyncio
import logging
import time

from dunetrace.detectors import (
    CUSTOM_DETECTOR_REGISTRY,
    DelegationLoopDetector,
    HandoffContextLossDetector,
    UngroundedDestinationDetector,
    run_detectors,
)
from typing import Optional

from dunetrace.models import FailureSignal, FailureType, RunState, Severity
from dunetrace.risk_engine import RiskEngine
from detector_svc.detectors import get_detectors

from detector_svc.config import settings
from detector_svc.config_loader import load_custom_detector_budget
from detector_svc.custom_detector import evaluate_custom_detector
from detector_svc.custom_python_detectors import load_custom_detector_plugins
from detector_svc.baseline_metrics import build_baseline_metrics
import detector_svc.db as _db
from detector_svc.db import (
    LIVE_DETECTORS,
    advance_watermark,
    close_pool,
    ensure_detector_schema,
    fetch_completed_runs,
    fetch_custom_detectors,
    count_agent_runs_capped,
    fetch_destination_baseline,
    fetch_duration_baseline,
    fetch_latency_baseline,
    fetch_memory_writes,
    fetch_llm_tool_ratio_baseline,
    fetch_run_events,
    fetch_run_lineage,
    fetch_run_state,
    fetch_stalled_runs,
    fetch_step_count_baseline,
    fetch_token_growth_baseline,
    fetch_total_tokens_baseline,
    get_watermark,
    prune_baseline_metrics,
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
    upsert_destination_baseline,
    upsert_run_and_conversation,
    write_baseline_metrics,
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
from dunetrace_schemas.metrics import (
    DEFAULT_LATENCY_BUCKETS,
    counter,
    db_ready,
    gauge,
    histogram,
    register_standard,
    set_schema_version,
    start_metrics_server,
)
from dunetrace_schemas.migrations import CURRENT_SCHEMA_VERSION, current_version

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("dunetrace.detector")

# Loaded once at startup, like detector_svc/detectors.py's _CONFIG — restart the
# detector container to apply a detectors.yml change.
_CUSTOM_DETECTOR_BUDGET = load_custom_detector_budget()

_COOCCURRENCE_MULTIPLIERS = {1: 1.0, 2: 1.15, 3: 1.30}

# ── Metrics ───────────────────────────────────────────────────────────────────
# Served on METRICS_PORT by dunetrace_schemas.metrics (no-ops when
# prometheus_client is absent, so nothing here can stop the worker).

register_standard("detector", settings.APP_VERSION)

BACKLOG_RUNS = gauge(
    "dunetrace_detector_backlog_runs",
    "Runs (completed + stalled) the last poll found waiting for detection.",
)
POLL_SATURATED = gauge(
    "dunetrace_detector_poll_saturated",
    "1 when the last poll's completed or stalled query hit BATCH_SIZE (more work "
    "behind it, watermark held), else 0.",
)
POLL_SECONDS = histogram(
    "dunetrace_detector_poll_seconds",
    "Wall-clock duration of one poll cycle (fetch + process every run found).",
    buckets=DEFAULT_LATENCY_BUCKETS,
)
RUNS_PROCESSED = counter(
    "dunetrace_detector_runs_processed_total",
    "Runs handed to process_run, by outcome: signals (>=1 signal written), clean, "
    "failed (retry budget spent, recorded with processing_error) or retry "
    "(left unmarked for the next poll).",
    ("result",),
)
EXCEPTIONS = counter(
    "dunetrace_detector_exceptions_total",
    "Exceptions caught by the worker, by site.",
    ("where",),
)
SIGNALS = counter(
    "dunetrace_detector_signals_total",
    "failure_signals rows written (built-in, JSON-config custom and plugin), "
    "by failure type and shadow flag.",
    ("failure_type", "shadow"),
)
# Pre-create the closed label sets so each series exists (at 0) from the first
# scrape — rate() over a series that only appears on its first increment
# misses that increment. failure_type is open-ended and is left lazy.
for _result in ("signals", "clean", "failed", "retry"):
    RUNS_PROCESSED.labels(result=_result)
for _where in (
    "poll",
    "process_run_detect",
    "process_run_write",
    "process_run_unhandled",
    "custom_detector",
):
    EXCEPTIONS.labels(where=_where)

# Readiness: the loop must have completed a poll recently. A wedged loop (an
# await that never returns, a CPU-bound detector) stops refreshing this, and
# /ready reports not-ready once it is older than POLL_STALE_FACTOR x
# POLL_INTERVAL. Monotonic seconds; set at startup so the window starts then.
POLL_STALE_FACTOR = 3
# A poll that is STILL RUNNING gets this much longer before it reads as wedged.
# Without it readiness was computed purely from the last COMPLETED cycle against
# a 15s window (3 x the 5s compose interval), while one cycle legitimately
# processes up to BATCH_SIZE completed plus BATCH_SIZE stalled runs, each ~15
# sequential round-trips, through a pool of 5. So the worker reported unhealthy
# for most of every busy cycle — exactly when a backlog existed — and an
# orchestrator acting on that restarted the busiest worker and handed it a
# larger backlog. The alerts worker already solved this by allowing an in-flight
# poll its claim timeout; this is the same idea with a bound of its own.
POLL_INFLIGHT_GRACE_SECS = 300.0
_last_poll_at: float | None = None
# Monotonic stamp of when the current (or last) poll STARTED. Distinguishing
# "in flight" from "finished" is the whole point: a slow cycle and a wedged loop
# are indistinguishable from completion times alone.
_poll_started_at: float | None = None


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


# A run the SDK's buffer shed events from is incomplete: a tool loop with its
# early calls missing looks clean, a run missing run.started has no tools list,
# and whatever DID fire was decided on partial data. Its signals are recorded —
# they are still worth looking at — but held in shadow, capped, and marked, so
# nothing downstream (alerts, issues, baselines, the risk engine) treats a
# verdict on partial data as a live one.
_INCOMPLETE_CONFIDENCE_CAP = 0.5
_INCOMPLETE_SEVERITY_CAP = Severity.MEDIUM
_SEVERITIES_ABOVE_CAP = frozenset({Severity.CRITICAL.value, Severity.HIGH.value})


# An implicit run (the SDK opened it because a patched LLM call had no run to
# attach to) is incomplete for the same reason: its boundary is a guess, so a
# verdict decided across it must not go live. Same shadow, cap and marker,
# with its own reason.
_REASON_SHED = "sdk_buffer_shed"
_REASON_IMPLICIT = "implicit_run"


def _incomplete_reason(state: RunState) -> Optional[str]:
    if state.dropped_events > 0:
        return _REASON_SHED
    # `is True`, not truthiness: a mocked or older RunState without the field
    # must read as an ordinary run, never as an incomplete one.
    if getattr(state, "implicit", False) is True:
        return _REASON_IMPLICIT
    return None


def _incomplete_marker(dropped_events: int, reason: str = _REASON_SHED) -> dict:
    return {"dropped_events": dropped_events, "reason": reason}


def _mark_incomplete(
    signal: FailureSignal, dropped_events: int, reason: str = _REASON_SHED
) -> None:
    """Cap a built-in / plugin signal decided on a shed or implicit run and mark its evidence."""
    if not isinstance(signal.evidence, dict):
        signal.evidence = {}
    signal.evidence["incomplete_data"] = _incomplete_marker(dropped_events, reason)
    signal.confidence = round(min(signal.confidence, _INCOMPLETE_CONFIDENCE_CAP), 4)
    if signal.severity.value in _SEVERITIES_ABOVE_CAP:
        signal.severity = _INCOMPLETE_SEVERITY_CAP


def _mark_incomplete_custom(result: dict, dropped_events: int, reason: str = _REASON_SHED) -> None:
    """Same treatment for a JSON-config custom detector's result dict."""
    evidence = dict(result.get("evidence") or {})
    evidence["incomplete_data"] = _incomplete_marker(dropped_events, reason)
    result["evidence"] = evidence
    try:
        confidence = float(result.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    result["confidence"] = round(min(confidence, _INCOMPLETE_CONFIDENCE_CAP), 4)
    if str(result.get("severity", "")).upper() in _SEVERITIES_ABOVE_CAP:
        result["severity"] = _INCOMPLETE_SEVERITY_CAP.value


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
    org_id: str,
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

    # The chain walker takes a one-argument fetcher; bind this run's org so
    # every hop stays inside it. An unscoped hop walks into another tenant's
    # delegation graph, because run_id is caller-supplied.
    async def _lineage(rid: str):
        return await fetch_run_lineage(rid, org_id)

    chain = await build_ancestor_chain(start, _lineage)
    if len(chain) < 2:
        return None
    cycle = find_cycle(build_agent_delegation_edges(chain))
    return detector.evaluate_delegation_cycle(
        cycle, agent_sequence(chain), run_id, agent_id, agent_version
    )


async def _apply_ungrounded_destination_cross_run(
    signals: list,
    state,
    detector,
    org_id: str,
    run_id: str,
    agent_id: str,
) -> None:
    """Cross-run enrichment for UNGROUNDED_DESTINATION (T6) plus novelty mode.

    Three things the in-run detector cannot do because it has no database:

    1. Escalate HIGH -> CRITICAL when the destination came from a PRIOR run's
       poisoned memory write for a key this run read back. `memory.read` carries
       only a key, so the value that re-entered the agent's context is simply not
       present in this run's events — without this step the flagship scenario
       (poisoned memory in run N, send in run N+2) tops out at HIGH.
    2. Fire novelty mode against the learned baseline, for closed-destination
       agents that opted in.
    3. Record this run's GROUNDED destinations as normal.

    Best-effort throughout: any failure here leaves the in-run verdict standing
    rather than losing the signal. Caller runs inside process_run's try block, so
    a raise would cost the whole run's detection.
    """
    if detector is None:
        return

    existing = next(
        (s for s in signals if s.failure_type == FailureType.UNGROUNDED_DESTINATION), None
    )

    # 1. Cross-run memory taint. Only worth a query when the in-run verdict found
    #    no taint — a CRITICAL signal needs no escalation.
    if existing is not None and existing.evidence.get("taint_source") is None:
        keys = sorted({m.key for m in state.memory_events if m.op == "read" and m.key})
        if keys:
            prior = await fetch_memory_writes(org_id, agent_id, keys, run_id)
            taint = detector.evaluate_cross_run_memory_taint(
                existing.evidence.get("destination", ""), prior
            )
            if taint:
                existing.severity = Severity.CRITICAL
                existing.confidence = 0.92
                existing.evidence["taint_source"] = taint

    novelty_on = "novelty" in (getattr(detector, "MODE", "") or "")
    if not novelty_on:
        return

    grounded, _ungrounded = detector.collect_destinations(state)

    # 2. Novelty — only against destinations that PASSED grounding, so it can
    #    never double-fire with the provenance verdict above.
    if existing is None and grounded:
        probe = [d for d, _t, _h in grounded]
        baseline = await fetch_destination_baseline(org_id, agent_id, probe)
        baseline_runs = await count_agent_runs_capped(org_id, agent_id, detector.MIN_BASELINE_RUNS)
        hit = detector.evaluate_novelty(grounded, baseline, baseline_runs)
        if hit:
            signals.append(
                FailureSignal(
                    failure_type=FailureType.UNGROUNDED_DESTINATION,
                    severity=Severity.MEDIUM,
                    run_id=run_id,
                    agent_id=agent_id,
                    agent_version=state.agent_version,
                    step_index=0,
                    confidence=0.55,
                    evidence={
                        "destination": hit["destination"],
                        "destination_type": hit["destination_type"],
                        "grounding_verdict": "grounded_but_novel",
                        "detection_mode": "novelty",
                        "baseline_size": len(baseline),
                        "baseline_runs": baseline_runs,
                        "taint_source": None,
                        "novelty_key": hit["novelty_key"],
                    },
                )
            )

    # 3. Baseline write — grounded destinations only. Writing ungrounded ones
    #    would teach the baseline that the first exfiltration is normal.
    if grounded:
        await upsert_destination_baseline(org_id, agent_id, [(d, t) for d, t, _h in grounded])


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


async def _handle_processing_failure(
    exc: Exception,
    *,
    run_id: str,
    agent_id: str,
    agent_version: str,
    trigger: str,
    org_id: str,
    event_count: int,
    stage: str,
) -> int:
    """The shared failure path for both halves of process_run. Always returns 0.

    A run is deliberately NOT marked processed on the first failures. Neither
    half is pure computation — the detect half holds seven baseline queries, a
    detector lookup, a parent fetch and a lineage walk; the write half is a
    sequence of inserts — so the common failure in both is transient. Marking
    such a run processed wrote signal_count=0, a permanent "clean run" verdict
    nothing could revisit, because a completed run never gains events and so
    never re-enters the poll. Leaving it unmarked means the next poll retries it,
    bounded by run_processing_failures.

    Once the budget is spent the run IS recorded, with processing_error set, so
    it reads as failed rather than as indistinguishable from clean.
    """
    logger.exception("Run processing failed in the %s half. run_id=%s", stage, run_id)
    EXCEPTIONS.labels(where=f"process_run_{stage}").inc()
    attempts = await record_processing_failure(run_id, org_id, repr(exc))
    if attempts < MAX_PROCESSING_ATTEMPTS:
        logger.warning(
            "Run %s will be retried (attempt %d of %d)",
            run_id,
            attempts,
            MAX_PROCESSING_ATTEMPTS,
        )
        RUNS_PROCESSED.labels(result="retry").inc()
        return 0
    logger.error("Run %s failed detection %d times — recording as failed", run_id, attempts)
    await mark_run_processed(
        run_id,
        agent_id,
        agent_version,
        trigger,
        0,
        org_id,
        event_count=event_count,
        processing_error=repr(exc)[:2000],
    )
    await clear_processing_failures(run_id, org_id)
    RUNS_PROCESSED.labels(result="failed").inc()
    return 0


async def process_run(
    run_id: str,
    agent_id: str,
    agent_version: str,
    trigger: str,
    org_id: str,
) -> int:
    events = await fetch_run_events(run_id, org_id)
    if not events:
        await mark_run_processed(run_id, agent_id, agent_version, trigger, 0, org_id)
        RUNS_PROCESSED.labels(result="clean").inc()
        return 0

    # audit Finding 15: if this run was already processed and new events have since
    # arrived, re-detect ADDITIVELY — write only failure types not already recorded
    # for the run (so a benign run that later turns out to have looped gets its
    # signal, without duplicating existing signals or re-alerting them).
    prior = await fetch_run_state(run_id, org_id)
    is_reprocess = prior["processed"]
    # Always the types already STORED for this run, even when processed_runs has
    # no row — see fetch_run_state. That is what makes a retry after a partial
    # write idempotent instead of duplicating every signal it already wrote.
    existing_types: set = prior["signal_types"]

    incomplete = False
    incomplete_reason: Optional[str] = None
    try:
        state = build_run_state(events)
        incomplete_reason = _incomplete_reason(state)
        incomplete = incomplete_reason is not None
        if incomplete_reason == _REASON_SHED:
            logger.info(
                "Run %s is incomplete: the SDK buffer shed %d of its events under overload. "
                "Its signals are held in shadow with capped confidence/severity and it is "
                "excluded from issue tracking and baselines. agent_id=%s",
                run_id,
                state.dropped_events,
                agent_id,
            )
        elif incomplete_reason == _REASON_IMPLICIT:
            logger.info(
                "Run %s was opened by the SDK on its own (no dt.run() or framework entry "
                "point), so its boundary is a guess. Its signals are held in shadow with "
                "capped confidence/severity and it is excluded from issue tracking and "
                "baselines. agent_id=%s",
                run_id,
                agent_id,
            )
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
                parent_events = await fetch_run_events(parent_run_id, org_id)
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
                    run_id, agent_id, agent_version, parent_run_id, delegation_detector, org_id
                )
                if delegation:
                    signals.append(delegation)

        ungrounded_detector = next(
            (d for d in detectors if isinstance(d, UngroundedDestinationDetector)), None
        )
        if ungrounded_detector is not None:
            try:
                await _apply_ungrounded_destination_cross_run(
                    signals, state, ungrounded_detector, org_id, run_id, agent_id
                )
            except Exception as exc:
                # Enrichment only. A failure here must leave the in-run HIGH
                # verdict standing, never cost the run its detection.
                logger.warning(
                    "UNGROUNDED_DESTINATION cross-run enrichment failed for run_id=%s: %s",
                    run_id,
                    exc,
                )

        # Shadow signals must not influence LIVE ones. A detector is in shadow
        # precisely because its precision is unvalidated, so letting it change
        # another signal's confidence, severity or co_signal_count means an
        # unvalidated detector is already affecting production output — which
        # is the one thing shadow mode exists to prevent. The shadow flag is
        # applied at WRITE time (LIVE_DETECTORS in db.py), which is far too late
        # for the three calls below; they run on the full list.
        #
        # Custom detectors are excluded here too: a plugin signal carries
        # FailureType.CUSTOM and its own shadow decision, so it is not
        # LIVE_DETECTORS-gated and cannot be assumed live.
        #
        # A shed (incomplete) run has NO live signals: every one of them is
        # written in shadow below, so none may feed the hard override or the
        # co-occurrence boost either.
        live_signals = (
            []
            if incomplete
            else [
                s
                for s in signals
                if s.failure_type != FailureType.CUSTOM and s.failure_type.value in LIVE_DETECTORS
            ]
        )

        risk = RiskEngine().evaluate(live_signals, state)
        logger.debug(
            "RiskEngine. run_id=%s confidence=%.2f active=%d severity=%s scores=%s",
            run_id,
            risk.confidence,
            risk.active_signals,
            risk.severity or "normal",
            risk.scores,
        )
        _apply_hard_override(live_signals, risk)
        _apply_cooccurrence_boost(live_signals)
    except Exception as exc:
        return await _handle_processing_failure(
            exc,
            run_id=run_id,
            agent_id=agent_id,
            agent_version=agent_version,
            trigger=trigger,
            org_id=org_id,
            event_count=len(events),
            stage="detect",
        )

    # The write half is guarded too. It used to sit OUTSIDE the try above,
    # so a transient failure here — a pool timeout on the third of three
    # write_signals calls — escaped process_run without recording a failure
    # and without marking the run processed. The next poll five seconds
    # later saw an unprocessed run and wrote the first two signals again,
    # every poll, for as long as the pressure lasted. fetch_run_state now
    # reads already-stored failure types regardless of processed_runs, so
    # the retry this schedules is idempotent.
    try:
        if incomplete:
            for signal in signals:
                _mark_incomplete(signal, state.dropped_events, incomplete_reason or _REASON_SHED)

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
                    shadow=shadow or incomplete,
                    org_id=org_id,
                )
                _count_signal(plugin_name, shadow or incomplete)
                count += 1
                continue

            is_live = signal.failure_type.value in LIVE_DETECTORS
            shadow = incomplete or not is_live
            written = await write_signals([signal], shadow=shadow, org_id=org_id)
            if written:
                _count_signal(signal.failure_type.value, shadow, written)
            count += written

        # OTel export (Phase 4): emit dunetrace.signal.* / dunetrace.policy.* child
        # spans into each run's trace so signals decided here show up in the
        # customer's OTel backend alongside the SDK-emitted run/LLM/tool spans. Only
        # LIVE, newly-fired signals (same filter issue tracking uses) so shadow
        # detectors and reprocess duplicates don't leak. No-op when OTel is disabled.
        live_new_signals = [
            s
            for s in signals
            if not incomplete
            and s.failure_type.value in LIVE_DETECTORS
            and s.failure_type.value not in existing_types
        ]
        _emit_otel_findings(run_id, org_id, live_new_signals, events, is_reprocess)

        # Issue persistence: track open/resolved lifecycle per (org_id, agent_id, failure_type).
        # audit Finding 15: on a reprocess, only NEW fired types matter, and the
        # consecutive-clean-runs counter must NOT be advanced again for a run it already
        # counted — so advance_clean_runs runs on first-process only.
        #
        # An incomplete run is "unknown", not clean and not fired: its signals are
        # all shadow (so fired_types is empty), and it must not advance the clean
        # counter either — five shed runs in a row would otherwise auto-resolve an
        # issue nobody verified.
        fired_types = [
            s.failure_type.value
            for s in signals
            if not incomplete
            and s.failure_type.value in LIVE_DETECTORS
            and s.failure_type.value not in existing_types
        ]
        try:
            if fired_types:
                await upsert_fired_issues(org_id, agent_id, fired_types)
            if not is_reprocess and not incomplete:
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
            custom_defs = (
                await fetch_custom_detectors(org_id, agent_id) if not is_reprocess else None
            )
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
                        if incomplete:
                            _mark_incomplete_custom(
                                result, state.dropped_events, incomplete_reason or _REASON_SHED
                            )
                        await write_custom_signal(
                            failure_type=result["failure_type"],
                            severity=result["severity"],
                            run_id=run_id,
                            agent_id=agent_id,
                            agent_version=agent_version,
                            step_index=result["step_index"],
                            confidence=result["confidence"],
                            evidence=result["evidence"],
                            shadow=cd["shadow"] or incomplete,
                            org_id=org_id,
                        )
                        _count_signal(result["failure_type"], cd["shadow"] or incomplete)
                        count += 1
                await record_custom_detector_results(cdr_records, org_id)
        except Exception as exc:
            EXCEPTIONS.labels(where="custom_detector").inc()
            logger.warning("Custom detector processing failed for run_id=%s: %s", run_id, exc)

        # Baseline metrics: store this run's scalars so the P75 baselines can be
        # computed from them later instead of re-derived from raw events, which are
        # pruned at EVENT_RETENTION_DAYS (see baseline_metrics.py). Written from the
        # same RunState the detectors just consumed, so a stored metric cannot drift
        # from the metric it will be compared against.
        #
        # `clean` mirrors the baseline's own definition: a run that fired a LIVE
        # signal is not a reference for normal. fired_types is already filtered to
        # LIVE_DETECTORS, and on a reprocess it holds only newly-fired types — so
        # `existing_types` is consulted too, or a re-detected run that had already
        # signalled would be recorded as clean.
        #
        # An incomplete run's step and token counts are truncated, so it is neither
        # a reference for normal nor evidence of a failure — it contributes nothing.
        try:
            if not incomplete:
                had_live_signal = bool(fired_types) or any(
                    ft in LIVE_DETECTORS for ft in existing_types
                )
                await write_baseline_metrics(
                    org_id,
                    run_id,
                    agent_id,
                    agent_version,
                    not had_live_signal,
                    build_baseline_metrics(state),
                )
        except Exception as exc:
            logger.warning("Baseline metrics failed for run_id=%s: %s", run_id, exc)

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
        await clear_processing_failures(run_id, org_id)
        RUNS_PROCESSED.labels(result="signals" if count else "clean").inc()
        return count
    except Exception as exc:
        return await _handle_processing_failure(
            exc,
            run_id=run_id,
            agent_id=agent_id,
            agent_version=agent_version,
            trigger=trigger,
            org_id=org_id,
            event_count=len(events),
            stage="write",
        )


async def poll_once() -> tuple[int, int]:
    global _poll_started_at
    started = time.monotonic()
    _poll_started_at = started
    try:
        return await _poll_once()
    finally:
        POLL_SECONDS.observe(time.monotonic() - started)


async def _poll_once() -> tuple[int, int]:
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
    # What this poll saw, not a table COUNT: a real backlog count would be one
    # more scan per poll of the same window, and "found BATCH_SIZE, watermark
    # held" already says the queue is deeper than one poll can drain.
    BACKLOG_RUNS.set(len(runs))
    POLL_SATURATED.set(0 if drained else 1)
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

    # return_exceptions=True, deliberately. Plain gather propagates the FIRST
    # exception immediately but does NOT cancel its siblings: they keep running
    # as detached tasks while _poll_cycle catches, skips the _last_poll_at
    # refresh, sleeps POLL_INTERVAL and starts a fresh poll — which re-selects
    # the very runs still in flight (they are still NOT EXISTS processed_runs)
    # and processes each a second time, concurrently with its own orphan. One
    # reproducibly-failing run also pinned /ready at 503 for a worker that was
    # otherwise healthy, because the liveness stamp only advances on a clean
    # cycle. Collecting the failures here keeps the batch whole: process_run
    # already records its own retry budget, so a run that raised past that is a
    # worker bug worth a loud line, not a reason to abandon 99 siblings.
    results = await asyncio.gather(*[process_run_bounded(r) for r in runs], return_exceptions=True)
    signals = 0
    for run, result in zip(runs, results):
        if isinstance(result, BaseException):
            EXCEPTIONS.labels(where="process_run_unhandled").inc()
            logger.error(
                "Run %s raised past process_run's own guards: %r",
                run["run_id"],
                result,
            )
            continue
        signals += result
    return len(runs), signals


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
                # Bounded by rank per agent, not by age — see
                # prune_baseline_metrics. Separate try/except so a failure here
                # can't stop processed_runs from being reclaimed.
                try:
                    trimmed = await prune_baseline_metrics()
                    if trimmed:
                        logger.info(
                            "Trimmed %d run_baseline_metrics row(s) beyond the "
                            "per-agent retention rank.",
                            trimmed,
                        )
                except Exception:
                    logger.exception("run_baseline_metrics prune failed")
        except Exception:
            logger.exception("processed_runs prune failed")
        await asyncio.sleep(_PRUNE_INTERVAL)


# The closed set of built-in failure types. Anything else — a JSON-config
# custom detector or a Python-class plugin — is counted as "custom".
_BUILTIN_FAILURE_TYPES = frozenset(ft.value for ft in FailureType)


def _count_signal(failure_type: str, shadow: bool, amount: int = 1) -> None:
    """Count one written signal, with the failure_type label bounded.

    A custom detector's name is free TEXT an org supplies through
    POST /v1/custom-detectors, and this worker is shared by every tenant — so
    passing it straight through made the label set unbounded and
    caller-controlled: 200 orgs x 5 detectors is 1000+ counter children (twice
    that with the shadow label), held in the registry for the life of the
    process, re-emitted on every scrape, with nothing bounding the name's
    length and deleting the detector never reclaiming the series.

    Per-detector counts are not lost: custom_detector_results records every
    evaluation per run, and custom_detectors.shadow_fire_count aggregates them
    — both org-scoped, both queryable, neither pinned in process memory.
    """
    label = failure_type if failure_type in _BUILTIN_FAILURE_TYPES else "custom"
    SIGNALS.labels(failure_type=label, shadow="true" if shadow else "false").inc(amount)


def _poll_age_seconds(now: float | None = None) -> float | None:
    """Seconds since the last successful poll; None before startup."""
    if _last_poll_at is None:
        return None
    return (time.monotonic() if now is None else now) - _last_poll_at


def _poll_stale_after_seconds() -> float:
    return POLL_STALE_FACTOR * float(settings.POLL_INTERVAL)


def _poll_is_in_flight() -> bool:
    """True while a poll has started and not yet refreshed the success stamp."""
    if _poll_started_at is None:
        return False
    return _last_poll_at is None or _poll_started_at > _last_poll_at


def poll_is_stale(now: float | None = None) -> bool:
    """True when the poll loop is wedged or was never started.

    A poll still IN FLIGHT is judged against POLL_INFLIGHT_GRACE_SECS instead of
    the completion window, so a long-but-healthy cycle is not mistaken for a
    dead one. A finished poll is judged against POLL_STALE_FACTOR x
    POLL_INTERVAL as before.
    """
    now = time.monotonic() if now is None else now
    if _poll_is_in_flight():
        started = _poll_started_at
        assert started is not None  # _poll_is_in_flight guarantees it
        return (now - started) > max(_poll_stale_after_seconds(), POLL_INFLIGHT_GRACE_SECS)
    age = _poll_age_seconds(now)
    return age is None or age > _poll_stale_after_seconds()


async def readiness_check() -> tuple[bool, dict]:
    """GET /ready: DB reachable at the required schema version AND the poll
    loop is alive. Runs on the worker's own event loop (scheduled there by the
    metrics thread), so a loop that is wedged never answers and the probe
    times out to 503 on its own."""
    ok, info = await db_ready(_db._pool, CURRENT_SCHEMA_VERSION)
    age = _poll_age_seconds()
    stale = poll_is_stale()
    in_flight = _poll_is_in_flight()
    info["poll"] = "stale" if stale else "ok"
    info["in_flight"] = in_flight
    info["last_poll_age_seconds"] = None if age is None else round(age, 1)
    info["poll_stale_after_seconds"] = (
        max(_poll_stale_after_seconds(), POLL_INFLIGHT_GRACE_SECS)
        if in_flight
        else _poll_stale_after_seconds()
    )
    return ok and not stale, info


async def _poll_cycle() -> None:
    """One iteration of the worker loop: poll, refresh the liveness stamp on
    success, count and log a failure. Never raises."""
    global _last_poll_at
    try:
        runs, signals = await poll_once()
        _last_poll_at = time.monotonic()
        if runs:
            logger.info("Cycle complete. runs=%d signals=%d", runs, signals)
    except Exception:
        EXCEPTIONS.labels(where="poll").inc()
        # logger.exception, not logger.error("...: %s", exc): several
        # exceptions this loop can raise stringify to nothing —
        # asyncpg's connection errors among them — so "%s" produced a
        # bare "Poll cycle failed:" with no cause and no traceback. The
        # only way to attribute those was correlating timestamps against
        # an outage by hand.
        logger.exception("Poll cycle failed")


async def _record_schema_version() -> None:
    """Point dunetrace_schema_version at what this worker actually observed.
    Best-effort: ensure_detector_schema already guaranteed >= CURRENT."""
    version = CURRENT_SCHEMA_VERSION
    try:
        if _db._pool is not None:
            async with _db._pool.acquire() as conn:
                version = await current_version(conn)
    except Exception as exc:
        logger.debug("schema version read for metrics failed: %s", exc)
    set_schema_version(version)


def start_metrics(loop: asyncio.AbstractEventLoop | None = None):
    """Start /metrics + /ready + /health on settings.METRICS_PORT (0 disables).
    Returns the server or None; a failure is logged, never raised — metrics are
    observability, not a dependency of detection."""
    global _last_poll_at
    if _last_poll_at is None:
        _last_poll_at = time.monotonic()  # the staleness window starts now
    try:
        return start_metrics_server(settings.METRICS_PORT, readiness_check, loop)
    except Exception:
        logger.exception("Metrics server failed to start; continuing without it")
        return None


async def run_worker() -> None:
    await init_pool()
    await ensure_detector_schema()
    await _record_schema_version()
    metrics_server = start_metrics(asyncio.get_running_loop())
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
            await _poll_cycle()
            await asyncio.sleep(settings.POLL_INTERVAL)
    except asyncio.CancelledError:
        logger.info("Detector worker cancelled")
    finally:
        prune_task.cancel()
        if metrics_server is not None:
            try:
                metrics_server.shutdown()
                metrics_server.server_close()
            except Exception:
                logger.debug("metrics server shutdown failed", exc_info=True)
        await close_pool()


if __name__ == "__main__":
    asyncio.run(run_worker())
