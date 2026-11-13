"""
Polling worker that picks up completed/errored runs, makes an adaptive
sampling decision (Phase 1.2), enforces per-agent monthly budgets, and — for
sampled runs — runs the configured semantic evaluators (Phase 1.3/1.4.1),
writes any findings to the shared failure_signals table, groups each finding
into its recurring pattern (Phase 1.4.2), applies any accumulated
false-positive feedback for that pattern before writing (Phase 1.4.3), gets a
second opinion from a different model before trusting a HIGH severity
finding for evaluators configured to require one (Phase 1.4.4), and enforces
the org's monthly billing quota alongside the existing per-agent budget
(Phase 1.5).

Trigger mechanism mirrors detector_svc/alerts_svc: poll the shared `events`
table on an interval, track what's already been handled in an owned table, no
message broker. See services/semantic/semantic_svc/db.py's module docstring
for why this reuses the existing pattern instead of introducing one (e.g.
Postgres LISTEN/NOTIFY) this codebase has never used.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

from dunetrace_schemas import metrics as _metrics
from dunetrace_schemas.migrations import CURRENT_SCHEMA_VERSION

import semantic_svc.db as _db
from semantic_svc.config import settings
from semantic_svc.config_loader import load_evaluator_config, load_sampling_rates
from semantic_svc.sampling import (
    MIN_CONVERSATION_RUNS,
    decide_conversation_sampling,
    decide_sampling,
)
from semantic_svc.evaluators.confusion_loop import ConfusionLoopEvaluator
from semantic_svc.evaluators.hallucination import HallucinationEvaluator
from semantic_svc.evaluators.off_topic_drift import OffTopicDriftEvaluator
from semantic_svc.evaluators.sycophancy_signal import SycophancySignalEvaluator
from semantic_svc.evaluators.task_completion import TaskCompletionEvaluator
from semantic_svc.evaluators.task_understanding_failure import (
    TaskUnderstandingFailureEvaluator,
)
from semantic_svc.evaluators.user_frustration import UserFrustrationEvaluator
from semantic_svc.evaluators.run_extract import build_evaluation_input
from semantic_svc.evaluators.conversation_extract import build_conversation_evaluation_input
from semantic_svc.grouping import root_cause_hash
from semantic_svc.db import (
    close_pool,
    consume_budget,
    consume_org_conversation_quota,
    consume_org_semantic_quota,
    ensure_semantic_schema,
    fetch_agent_semantic_config,
    fetch_conversation_run_ids,
    fetch_org_conversation_quota_settings,
    fetch_org_semantic_feedback_settings,
    fetch_org_semantic_quota_settings,
    fetch_run_conversation_id,
    fetch_run_events,
    fetch_signal_group_fp_count,
    fetch_unevaluated_runs,
    has_retrieval_event,
    has_structural_signal,
    init_pool,
    log_semantic_evaluation,
    mark_run_processed,
    record_signal_group_membership,
    write_quota_exceeded_signal,
    write_semantic_signal,
)

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("dunetrace.semantic")

# ── Metrics & readiness ───────────────────────────────────────────────────────
# Served by dunetrace_schemas.metrics' stdlib HTTP server on METRICS_PORT
# (/metrics, /ready, /health). run_worker() starts it in the enabled path only:
# a disabled worker logs one line and exits 0, and there is nothing to scrape.
_M_BACKLOG = _metrics.gauge(
    "dunetrace_semantic_backlog",
    "Unevaluated runs found by the most recent poll (capped at BATCH_SIZE).",
)
_M_PROCESSED = _metrics.counter(
    "dunetrace_semantic_processed_total",
    "Runs a sampling decision was recorded for, by outcome (sampled | skipped).",
    ("result",),
)
_M_FAILURES = _metrics.counter(
    "dunetrace_semantic_failures_total",
    "Errors the worker contained, by where they happened.",
    ("where",),
)
_M_EXTERNAL_SECONDS = _metrics.histogram(
    "dunetrace_semantic_external_call_seconds",
    "Wall-clock seconds per evaluator invocation (one LLM-backed DeepEval call).",
    ("provider",),
    buckets=_metrics.DEFAULT_LATENCY_BUCKETS,
)
_M_EXTERNAL_CALLS = _metrics.counter(
    "dunetrace_semantic_external_calls_total",
    "Evaluator invocations, by LLM provider and outcome (ok | error).",
    ("provider", "status"),
)
_M_POLL_SECONDS = _metrics.histogram(
    "dunetrace_semantic_poll_seconds",
    "Wall-clock seconds per poll cycle.",
    buckets=_metrics.DEFAULT_LATENCY_BUCKETS,
)

# /ready reports the poll loop stale once this many POLL_INTERVALs have passed
# without a successful cycle.
_STALE_AFTER_INTERVALS = 3
# A cycle that is STILL RUNNING gets this much longer before it reads as wedged.
# Freshness used to be computed purely from the last COMPLETED cycle against
# 3 x the poll interval, but one cycle can gather up to BATCH_SIZE runs and run four run-level plus three conversation-level LLM evaluators over each, which takes minutes —
# so the worker reported unhealthy for most of every busy cycle, which is
# precisely when it has work. An orchestrator acting on that restarts the
# busiest worker mid-batch. The grace is a bound, not an exemption: a genuinely
# wedged loop still goes stale, just later.
_INFLIGHT_GRACE_SECS = 900.0
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
    fresh, poll_info = _poll_freshness(_last_poll_ok_at, time.monotonic(), settings.POLL_INTERVAL)
    info.update(poll_info)
    return db_ok and fresh, info


async def _observed_evaluate(evaluator, payload, provider):
    """Run one evaluator off the event loop and observe the call.

    audit Finding 20: DeepEval's metric.measure() manages its own event loop;
    calling it directly inside this async worker loop corrupts the loop
    (IndexError: pop from an empty deque) and crash-loops the worker. A thread
    gives DeepEval its own loop, isolated from ours.

    Duration and outcome are recorded per LLM provider. The exception is
    re-raised: each call site decides what one failure costs (see the
    containment comments there), this only observes it.
    """
    label = str(provider or "unknown")
    status = "error"
    elapsed = 0.0

    def _call():
        # Timed INSIDE the worker thread, so the observation is the provider
        # call and nothing else. Stamping before `to_thread` also counted the
        # wait for a free thread in the default executor — and poll_once gathers
        # all BATCH_SIZE runs with no semaphore, so 100 runs x 4 evaluators is
        # 400 submissions against ~32 threads and the last one waits ~12 waves.
        # Its sample then read as a 60s provider latency when the provider
        # answered in 5, i.e. the panel said "OpenAI is slow" when the real
        # answer was "this worker is over-subscribed".
        nonlocal elapsed
        t0 = time.perf_counter()
        try:
            return evaluator.evaluate(payload)
        finally:
            elapsed = time.perf_counter() - t0

    try:
        result = await asyncio.to_thread(_call)
        status = "ok"
        return result
    finally:
        _M_EXTERNAL_SECONDS.labels(provider=label).observe(elapsed)
        _M_EXTERNAL_CALLS.labels(provider=label, status=status).inc()


# Loaded once at startup, like detector_svc/detectors.py's _CONFIG — restart
# the semantic worker container to apply a semantic-sampling.yml/
# semantic-evaluators.yml change.
_SAMPLING_RATES = load_sampling_rates()
_EVALUATOR_CONFIG = load_evaluator_config()

# Built lazily by run_worker(), never at import time: constructing a
# GPTModel/AnthropicModel requires a real API key, which may only exist in the
# actual runtime container — importing this module (e.g. for tests, or when
# SEMANTIC_WORKER_ENABLED=false) must never require one.
_evaluators: dict[str, object] = {}
# Phase 3.2 — built the same lazy way; empty until run_worker() builds it.
# _maybe_run_conversation_evaluator treats an empty dict as "conversation
# evaluation disabled." Multiple conversation-level evaluators (USER_FRUSTRATION,
# CONFUSION_LOOP, …) run in one pass over the same conversation window.
_conversation_evaluators: dict[str, object] = {}

# How many of a conversation's most recent runs UserFrustrationEvaluator
# reads — bounds both LLM context size and cost. MIN_CONVERSATION_RUNS
# (sampling.py) is the floor before evaluation is even considered; this is
# the ceiling on how much history a considered conversation actually gets.
_CONVERSATION_MAX_RUNS = 5
# Second-opinion instances (Phase 1.4.4) — only populated for evaluators with
# require_second_opinion: true in semantic-evaluators.yml, each built with a
# different provider/model than the primary. Empty dict means "no evaluator
# has one configured," not "not built yet" — checked via .get(name), so an
# unconfigured evaluator just never gets a second opinion, no error.
_second_opinion_evaluators: dict[str, object] = {}
# name -> the provider each second-opinion evaluator was built with, so its
# external-call metrics carry the right provider label (it is, by design,
# usually not the primary one). Populated beside _second_opinion_evaluators.
_second_opinion_providers: dict[str, str] = {}

_EVALUATOR_CLASSES = {
    HallucinationEvaluator.name: HallucinationEvaluator,
    TaskCompletionEvaluator.name: TaskCompletionEvaluator,
    TaskUnderstandingFailureEvaluator.name: TaskUnderstandingFailureEvaluator,
    OffTopicDriftEvaluator.name: OffTopicDriftEvaluator,
}

# Confidence -> severity. Only applied when an evaluator's EvalResult.fired is
# True — an evaluator that found no problem never produces a signal row at
# all, same as a structural detector that declines to fire.
_SEVERITY_THRESHOLDS = (
    (0.85, "CRITICAL"),
    (0.70, "HIGH"),
    (0.50, "MEDIUM"),
)

# Phase 1.4.3 — once a recurring pattern accumulates this many false_positive
# feedback marks, either its confidence is docked or (if the org has opted
# into auto_suppress) it stops being written at all. Same 3-strikes threshold
# as the existing structural mechanism (agent_detector_overrides).
_FP_SUPPRESS_THRESHOLD = 3
_FP_CONFIDENCE_PENALTY = 0.3


def _severity_for_confidence(confidence: float) -> str:
    for threshold, severity in _SEVERITY_THRESHOLDS:
        if confidence >= threshold:
            return severity
    return "LOW"


def _build_evaluators() -> dict[str, object]:
    provider = settings.SEMANTIC_LLM_PROVIDER
    return {
        HallucinationEvaluator.name: HallucinationEvaluator(
            provider, settings.HALLUCINATION_MODEL or None
        ),
        TaskCompletionEvaluator.name: TaskCompletionEvaluator(
            provider, settings.TASK_COMPLETION_MODEL or None
        ),
        TaskUnderstandingFailureEvaluator.name: TaskUnderstandingFailureEvaluator(
            provider, settings.TASK_UNDERSTANDING_FAILURE_MODEL or None
        ),
        OffTopicDriftEvaluator.name: OffTopicDriftEvaluator(
            provider, settings.OFF_TOPIC_DRIFT_MODEL or None
        ),
    }


def _build_conversation_evaluators() -> dict[str, object]:
    provider = settings.SEMANTIC_LLM_PROVIDER
    return {
        UserFrustrationEvaluator.name: UserFrustrationEvaluator(
            provider, settings.USER_FRUSTRATION_MODEL or None
        ),
        ConfusionLoopEvaluator.name: ConfusionLoopEvaluator(
            provider, settings.CONFUSION_LOOP_MODEL or None
        ),
        SycophancySignalEvaluator.name: SycophancySignalEvaluator(
            provider, settings.SYCOPHANCY_SIGNAL_MODEL or None
        ),
    }


# Second opinion for a mistral-primary deployment. Diversity has to come from
# the model rather than the vendor, because Mistral is the only European
# provider we support — see _resolve_second_opinion. The pair is picked so the
# second opinion is never the same model as the primary default
# (mistral-small-latest).
_MISTRAL_SECOND_OPINION_MODEL = "mistral-large-latest"
_MISTRAL_SECOND_OPINION_ALT = "mistral-medium-latest"


def _opposite_provider(provider: str) -> str:
    """The point of a second opinion is a genuinely different model — default
    to whichever provider ISN'T the primary one when semantic-evaluators.yml
    doesn't specify second_opinion_provider explicitly.

    mistral maps to itself: a customer on the European provider does not want
    the confirming call to leave it, so the second opinion differs by model
    instead. _resolve_second_opinion picks that model."""
    if provider == "mistral":
        return "mistral"
    return "anthropic" if provider == "openai" else "openai"


def _provider_has_key(provider: str) -> bool:
    if provider == "anthropic":
        return bool(settings.ANTHROPIC_API_KEY)
    if provider == "openai":
        return bool(settings.OPENAI_API_KEY)
    if provider == "mistral":
        return bool(settings.MISTRAL_API_KEY)
    return False


def _mistral_second_opinion_model(primary_model: str | None) -> str:
    """A different Mistral model from the primary one. Asking the identical
    model twice is not a second opinion."""
    resolved = (primary_model or "").strip().lower()
    if resolved.startswith("mistral-large"):
        return _MISTRAL_SECOND_OPINION_ALT
    return _MISTRAL_SECOND_OPINION_MODEL


def _resolve_second_opinion(name: str, primary_provider: str, cfg: dict) -> tuple[str, str | None]:
    """(provider, model) for one evaluator's second opinion.

    Enforces the residency boundary: when the primary provider is mistral, a
    second_opinion_provider naming a different vendor would send the full run
    text to that vendor on precisely the HIGH-confidence findings a customer
    cares most about — the opposite of why they selected a European provider.
    That configuration is suppressed in favour of an in-region Mistral model
    unless SEMANTIC_ALLOW_CROSS_PROVIDER_SECOND_OPINION is set.

    Nothing changes for openai/anthropic primaries: they keep crossing to each
    other, which is the existing behaviour and carries no residency promise.
    """
    configured = cfg.get("second_opinion_provider")
    provider = configured or _opposite_provider(primary_provider)
    # This evaluator's own primary model override, e.g. HALLUCINATION_MODEL.
    # Empty means the provider default, which _mistral_second_opinion_model
    # already treats as "not large".
    primary_model = getattr(settings, f"{name}_MODEL", "") or None

    if (
        primary_provider == "mistral"
        and provider != "mistral"
        and not settings.SEMANTIC_ALLOW_CROSS_PROVIDER_SECOND_OPINION
    ):
        in_region = _mistral_second_opinion_model(primary_model)
        logger.warning(
            "Second-opinion for %s: suppressed cross-provider second_opinion_provider "
            "%r because SEMANTIC_LLM_PROVIDER=mistral. The run text would have left "
            "the European provider on every HIGH-confidence finding. Using mistral/%s "
            "instead. Set SEMANTIC_ALLOW_CROSS_PROVIDER_SECOND_OPINION=true to allow "
            "it — see docs/integrations/mistral.md.",
            name,
            provider,
            in_region,
        )
        return "mistral", in_region

    if provider == "mistral" and not cfg.get("second_opinion_model"):
        # In-region by default rather than by override: still needs a model that
        # differs from the primary one.
        return provider, _mistral_second_opinion_model(primary_model)

    return provider, cfg.get("second_opinion_model")


def _build_second_opinion_evaluators() -> dict[str, object]:
    primary_provider = settings.SEMANTIC_LLM_PROVIDER
    built: dict[str, object] = {}
    for name, cfg in _EVALUATOR_CONFIG.items():
        if not cfg.get("require_second_opinion"):
            continue
        cls = _EVALUATOR_CLASSES.get(name)
        if cls is None:
            continue
        provider, second_opinion_model = _resolve_second_opinion(name, primary_provider, cfg)
        # audit Finding 19: a second opinion defaults to the OPPOSITE provider for
        # model diversity. If that provider has no API key configured (the common
        # single-provider deployment — e.g. OpenAI only), DeepEval's model
        # constructor raises and crash-loops the ENTIRE worker at startup. Degrade
        # gracefully: skip the second opinion for this evaluator with a warning
        # instead of taking the whole service down.
        if not _provider_has_key(provider):
            logger.warning(
                "Second-opinion for %s disabled: provider %r has no API key configured. "
                "Set its key (or set second_opinion_provider to your primary provider) "
                "to enable second-opinion evaluation.",
                name,
                provider,
            )
            continue
        built[name] = cls(provider, second_opinion_model)
        _second_opinion_providers[name] = provider
    return built


def _current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


async def _run_evaluators(
    run_id: str,
    agent_id: str,
    agent_version: str,
    org_id: str,
    evaluator_names: list[str] | None,
) -> int:
    """Runs the configured evaluators (or all of them, if the agent has no
    override) against the run's reconstructed content. For each evaluator
    that fires: applies any accumulated false-positive feedback for that
    recurring pattern (Phase 1.4.3), writes a failure_signals row (unless
    auto-suppressed), and groups it into its pattern (Phase 1.4.2). Returns
    the number of signals actually written (auto-suppressed findings don't
    count).
    """
    events = await fetch_run_events(org_id, run_id)
    run = build_evaluation_input(events)
    if run is None:
        return 0

    names = evaluator_names or list(_evaluators.keys())
    written = 0
    for name in names:
        evaluator = _evaluators.get(name)
        if evaluator is None:
            continue
        # Off the event loop and observed — see _observed_evaluate.
        #
        # Contained per evaluator: a provider error (rate limit, outage, a
        # response the schema parser rejects) must cost this one finding, not the
        # rest of the run's evaluators. Uncontained it propagates through
        # process_run into poll_once's asyncio.gather and takes the whole batch
        # down — and every evaluator already run for this run is billable, so the
        # retry pays for them a second time.
        try:
            result = await _observed_evaluate(evaluator, run, settings.SEMANTIC_LLM_PROVIDER)
        except Exception:
            _M_FAILURES.labels(where="evaluator").inc()
            logger.exception("Evaluator %s failed for run %s — skipping it", name, run_id)
            continue
        # Logged regardless of whether it fired — see semantic_evaluation_log's
        # schema comment for why failure_signals alone would undercount real
        # spend (it only ever stores fired findings).
        await log_semantic_evaluation(
            org_id,
            agent_id,
            result.evaluator,
            result.fired,
            result.prompt_tokens,
            result.completion_tokens,
            result.cost_usd,
        )
        if not result.fired:
            continue

        confidence = result.confidence
        rc_hash = root_cause_hash(result.reasoning)
        fp_count = await fetch_signal_group_fp_count(org_id, agent_id, result.evaluator, rc_hash)
        if fp_count >= _FP_SUPPRESS_THRESHOLD:
            feedback_settings = await fetch_org_semantic_feedback_settings(org_id)
            if feedback_settings["auto_suppress"]:
                logger.info(
                    "Suppressed by feedback (%d false positives) — evaluator=%s agent=%s",
                    fp_count,
                    result.evaluator,
                    agent_id,
                )
                continue
            confidence = max(0.0, confidence - _FP_CONFIDENCE_PENALTY)

        severity = _severity_for_confidence(confidence)
        second_opinion_evidence = None
        if severity == "HIGH":
            second_evaluator = _second_opinion_evaluators.get(name)
            if second_evaluator is not None:
                # A failed second opinion degrades to "no second opinion" — the
                # primary finding still stands at HIGH. Confirming a finding is
                # an enhancement; it must not be able to discard the finding it
                # was meant to confirm.
                try:
                    second_result = await _observed_evaluate(
                        second_evaluator,
                        run,
                        _second_opinion_providers.get(name, settings.SEMANTIC_LLM_PROVIDER),
                    )
                except Exception:
                    _M_FAILURES.labels(where="second_opinion").inc()
                    logger.exception(
                        "Second opinion for %s failed on run %s — keeping the "
                        "primary finding unconfirmed",
                        name,
                        run_id,
                    )
                    second_result = None
            if second_evaluator is not None and second_result is not None:
                await log_semantic_evaluation(
                    org_id,
                    agent_id,
                    second_result.evaluator,
                    second_result.fired,
                    second_result.prompt_tokens,
                    second_result.completion_tokens,
                    second_result.cost_usd,
                )
                agreed = second_result.fired
                if not agreed:
                    severity = "MEDIUM"
                second_opinion_evidence = {
                    "ran": True,
                    "agreed": agreed,
                    "reasoning": second_result.reasoning,
                    "prompt_tokens": second_result.prompt_tokens,
                    "completion_tokens": second_result.completion_tokens,
                    "cost_usd": second_result.cost_usd,
                }

        evidence = {
            "reasoning": result.reasoning,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "cost_usd": result.cost_usd,
        }
        if second_opinion_evidence is not None:
            evidence["second_opinion"] = second_opinion_evidence

        signal_id = await write_semantic_signal(
            evaluator=result.evaluator,
            severity=severity,
            run_id=run_id,
            agent_id=agent_id,
            agent_version=agent_version,
            confidence=confidence,
            evidence=evidence,
            org_id=org_id,
        )
        await record_signal_group_membership(
            org_id, agent_id, result.evaluator, result.reasoning, signal_id, run_id
        )
        written += 1
    return written


async def _maybe_run_conversation_evaluator(
    run_id: str, agent_id: str, agent_version: str, org_id: str
) -> int:
    """Independent of this run's own per-run sampling decision (see
    process_run) — a conversation can be selected for conversation-level
    monitoring regardless of whether any individual run within it happens to
    be sampled for per-run semantic evaluation. Returns signals written (one per
    conversation evaluator that fired).
    """
    if not _conversation_evaluators:
        return 0

    conversation_external_id = await fetch_run_conversation_id(org_id, run_id)
    if not conversation_external_id:
        return 0

    sibling_run_ids = await fetch_conversation_run_ids(
        org_id, agent_id, conversation_external_id, _CONVERSATION_MAX_RUNS
    )
    sampled, _reason = decide_conversation_sampling(
        conversation_external_id, len(sibling_run_ids), rates=_SAMPLING_RATES
    )
    if not sampled:
        return 0

    month = _current_month()
    quota_settings = await fetch_org_conversation_quota_settings(org_id)
    allowed = await consume_org_conversation_quota(
        org_id, month, quota_settings["quota"], quota_settings["allow_overage"]
    )
    if not allowed:
        logger.info(
            "Conversation evaluation quota exhausted — org=%s month=%s quota=%d",
            org_id,
            month,
            quota_settings["quota"],
        )
        return 0

    runs_events = [(rid, await fetch_run_events(org_id, rid)) for rid in sibling_run_ids]
    conversation_input = build_conversation_evaluation_input(runs_events)
    if conversation_input is None:
        return 0

    # One quota unit was consumed for this conversation-evaluation pass above; the
    # pass now runs every registered conversation evaluator over the same window
    # (each is its own LLM call, cost-logged individually). A pass writes one
    # signal per evaluator that fired.
    signals_written = 0
    for evaluator in _conversation_evaluators.values():
        # Contained for the same reason as the run-level loop above.
        try:
            result = await _observed_evaluate(
                evaluator, conversation_input, settings.SEMANTIC_LLM_PROVIDER
            )
        except Exception:
            _M_FAILURES.labels(where="conversation_evaluator").inc()
            logger.exception(
                "Conversation evaluator %s failed for conversation %s — skipping it",
                getattr(evaluator, "name", evaluator),
                conversation_external_id,
            )
            continue
        await log_semantic_evaluation(
            org_id,
            agent_id,
            result.evaluator,
            result.fired,
            result.prompt_tokens,
            result.completion_tokens,
            result.cost_usd,
        )
        if not result.fired:
            continue

        await write_semantic_signal(
            evaluator=result.evaluator,
            severity=_severity_for_confidence(result.confidence),
            run_id=run_id,  # the triggering/most-recent run in the window
            agent_id=agent_id,
            agent_version=agent_version,
            confidence=result.confidence,
            evidence={
                "reasoning": result.reasoning,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "cost_usd": result.cost_usd,
                "conversation_id": conversation_external_id,
                "run_ids_considered": sibling_run_ids,
            },
            org_id=org_id,
        )
        signals_written += 1
    return signals_written


async def process_run(
    run_id: str, agent_id: str, agent_version: str, org_id: str
) -> tuple[bool, int]:
    """Returns (sampled, signals_written)."""
    structural, retrieval, agent_config = await asyncio.gather(
        has_structural_signal(org_id, run_id),
        has_retrieval_event(org_id, run_id),
        fetch_agent_semantic_config(org_id, agent_id),
    )

    sampled, reason, _rate = decide_sampling(
        run_id, structural, retrieval, agent_config, rates=_SAMPLING_RATES
    )

    if sampled:
        month = _current_month()

        # Org-level billing quota (Phase 1.5) — the mandatory ceiling across
        # ALL of this org's agents combined. Checked before the optional
        # per-agent budget below, since it's the outer limit.
        org_settings = await fetch_org_semantic_quota_settings(org_id)
        org_allowed = await consume_org_semantic_quota(
            org_id, month, org_settings["quota"], org_settings["allow_overage"]
        )
        if not org_allowed:
            sampled = False
            reason = "org_quota_exceeded"
            logger.info(
                "Org semantic quota exhausted — org=%s month=%s quota=%d",
                org_id,
                month,
                org_settings["quota"],
            )
            await write_quota_exceeded_signal(
                org_id, agent_id, agent_version, run_id, month, "org_quota", org_settings["quota"]
            )

    budget_monthly = agent_config.get("budget_monthly") if agent_config else None
    if sampled and budget_monthly is not None:
        month = _current_month()
        allowed = await consume_budget(org_id, agent_id, month, budget_monthly)
        if not allowed:
            sampled = False
            reason = "budget_exceeded"
            logger.info(
                "Semantic budget exhausted — org=%s agent=%s month=%s budget=%d",
                org_id,
                agent_id,
                month,
                budget_monthly,
            )
            await write_quota_exceeded_signal(
                org_id, agent_id, agent_version, run_id, month, "agent_budget", budget_monthly
            )

    signals_written = 0
    if sampled:
        evaluator_names = agent_config.get("evaluators") if agent_config else None
        signals_written = await _run_evaluators(
            run_id, agent_id, agent_version, org_id, evaluator_names
        )

    # Phase 3.2 — isolated in its own try/except, same reasoning as
    # detector_svc's issue-tracking/custom-detector steps: a bug in
    # conversation-level evaluation must never block this run's own
    # sampling decision from being recorded.
    try:
        signals_written += await _maybe_run_conversation_evaluator(
            run_id, agent_id, agent_version, org_id
        )
    except Exception as exc:
        _M_FAILURES.labels(where="conversation").inc()
        logger.warning("Conversation evaluation failed for run_id=%s: %s", run_id, exc)

    await mark_run_processed(run_id, agent_id, agent_version, org_id, sampled, reason)
    return sampled, signals_written


async def poll_once() -> tuple[int, int, int]:
    """Returns (runs_seen, runs_sampled, signals_written)."""
    runs = await fetch_unevaluated_runs(limit=settings.BATCH_SIZE)
    _M_BACKLOG.set(len(runs))
    if not runs:
        return 0, 0, 0

    results = await asyncio.gather(
        *[process_run(r["run_id"], r["agent_id"], r["agent_version"], r["org_id"]) for r in runs]
    )
    sampled_count = sum(1 for sampled, _ in results if sampled)
    signals_count = sum(count for _, count in results)
    _M_PROCESSED.labels(result="sampled").inc(sampled_count)
    _M_PROCESSED.labels(result="skipped").inc(len(runs) - sampled_count)
    return len(runs), sampled_count, signals_count


async def run_worker() -> None:
    if not settings.SEMANTIC_WORKER_ENABLED:
        logger.info(
            "Semantic worker disabled (SEMANTIC_WORKER_ENABLED=false) — exiting "
            "without opening a DB connection."
        )
        return

    # Observability comes up first so a slow DB start reads as a 503 /ready
    # rather than a connection refused. start_metrics_server never raises —
    # metrics are observability, not a dependency; METRICS_PORT=0 disables it.
    _metrics.register_standard("semantic", settings.APP_VERSION)
    metrics_server = _metrics.start_metrics_server(
        settings.METRICS_PORT, _ready_check, asyncio.get_running_loop()
    )

    global _evaluators, _second_opinion_evaluators, _conversation_evaluators
    _evaluators = _build_evaluators()
    _second_opinion_evaluators = _build_second_opinion_evaluators()
    _conversation_evaluators = _build_conversation_evaluators()

    await init_pool()
    await ensure_semantic_schema()
    # ensure_semantic_schema applied, then required, CURRENT_SCHEMA_VERSION —
    # so that is the version this process is now running against.
    _metrics.set_schema_version(CURRENT_SCHEMA_VERSION)
    _mark_poll_ok()  # seed /ready freshness: the first cycle gets the full grace window
    logger.info(
        "Semantic worker started. poll_interval=%ss metrics_port=%s",
        settings.POLL_INTERVAL,
        settings.METRICS_PORT,
    )
    try:
        while True:
            started = time.perf_counter()
            _mark_poll_started()
            try:
                runs, sampled, signals = await poll_once()
                _mark_poll_ok()
                if runs:
                    logger.info(
                        "Cycle complete. runs=%d sampled=%d signals=%d", runs, sampled, signals
                    )
            except Exception:
                _M_FAILURES.labels(where="poll").inc()
                logger.exception("Poll cycle failed")
            _M_POLL_SECONDS.observe(time.perf_counter() - started)
            await asyncio.sleep(settings.POLL_INTERVAL)
    except asyncio.CancelledError:
        logger.info("Semantic worker cancelled")
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
