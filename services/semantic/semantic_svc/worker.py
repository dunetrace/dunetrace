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
from datetime import datetime, timezone

from semantic_svc.config import settings
from semantic_svc.config_loader import load_evaluator_config, load_sampling_rates
from semantic_svc.sampling import (
    MIN_CONVERSATION_RUNS,
    decide_conversation_sampling,
    decide_sampling,
)
from semantic_svc.evaluators.hallucination import HallucinationEvaluator
from semantic_svc.evaluators.task_completion import TaskCompletionEvaluator
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
# Phase 3.2 — built the same lazy way. None until run_worker() builds it;
# _maybe_run_conversation_evaluator treats None as "conversation evaluation
# disabled," same as an unconfigured entry in _evaluators.
_conversation_evaluator: object = None

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

_EVALUATOR_CLASSES = {
    HallucinationEvaluator.name: HallucinationEvaluator,
    TaskCompletionEvaluator.name: TaskCompletionEvaluator,
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
    }


def _build_conversation_evaluator() -> UserFrustrationEvaluator:
    provider = settings.SEMANTIC_LLM_PROVIDER
    return UserFrustrationEvaluator(provider, settings.USER_FRUSTRATION_MODEL or None)


def _opposite_provider(provider: str) -> str:
    """The point of a second opinion is a genuinely different model — default
    to whichever provider ISN'T the primary one when semantic-evaluators.yml
    doesn't specify second_opinion_provider explicitly."""
    return "anthropic" if provider == "openai" else "openai"


def _provider_has_key(provider: str) -> bool:
    if provider == "anthropic":
        return bool(settings.ANTHROPIC_API_KEY)
    if provider == "openai":
        return bool(settings.OPENAI_API_KEY)
    return False


def _build_second_opinion_evaluators() -> dict[str, object]:
    primary_provider = settings.SEMANTIC_LLM_PROVIDER
    built: dict[str, object] = {}
    for name, cfg in _EVALUATOR_CONFIG.items():
        if not cfg.get("require_second_opinion"):
            continue
        cls = _EVALUATOR_CLASSES.get(name)
        if cls is None:
            continue
        provider = cfg.get("second_opinion_provider") or _opposite_provider(primary_provider)
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
        built[name] = cls(provider, cfg.get("second_opinion_model"))
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
    events = await fetch_run_events(run_id)
    run = build_evaluation_input(events)
    if run is None:
        return 0

    names = evaluator_names or list(_evaluators.keys())
    written = 0
    for name in names:
        evaluator = _evaluators.get(name)
        if evaluator is None:
            continue
        # audit Finding 20: DeepEval's metric.measure() manages its own event
        # loop; calling it directly inside this async worker loop corrupts the
        # loop (IndexError: pop from an empty deque) and crash-loops the worker.
        # Run it in a thread so DeepEval gets its own loop, isolated from ours.
        result = await asyncio.to_thread(evaluator.evaluate, run)
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
                # audit Finding 20: isolate DeepEval's sync loop (see above).
                second_result = await asyncio.to_thread(second_evaluator.evaluate, run)
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
    be sampled for per-run semantic evaluation. Returns signals written (0 or 1).
    """
    if _conversation_evaluator is None:
        return 0

    conversation_external_id = await fetch_run_conversation_id(run_id)
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

    runs_events = [(rid, await fetch_run_events(rid)) for rid in sibling_run_ids]
    conversation_input = build_conversation_evaluation_input(runs_events)
    if conversation_input is None:
        return 0

    # audit Finding 20: isolate DeepEval's sync loop (see _run_evaluators).
    result = await asyncio.to_thread(_conversation_evaluator.evaluate, conversation_input)
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
        return 0

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
    return 1


async def process_run(
    run_id: str, agent_id: str, agent_version: str, org_id: str
) -> tuple[bool, int]:
    """Returns (sampled, signals_written)."""
    structural, retrieval, agent_config = await asyncio.gather(
        has_structural_signal(run_id),
        has_retrieval_event(run_id),
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
        logger.warning("Conversation evaluation failed for run_id=%s: %s", run_id, exc)

    await mark_run_processed(run_id, agent_id, agent_version, org_id, sampled, reason)
    return sampled, signals_written


async def poll_once() -> tuple[int, int, int]:
    """Returns (runs_seen, runs_sampled, signals_written)."""
    runs = await fetch_unevaluated_runs(limit=settings.BATCH_SIZE)
    if not runs:
        return 0, 0, 0

    results = await asyncio.gather(
        *[process_run(r["run_id"], r["agent_id"], r["agent_version"], r["org_id"]) for r in runs]
    )
    sampled_count = sum(1 for sampled, _ in results if sampled)
    signals_count = sum(count for _, count in results)
    return len(runs), sampled_count, signals_count


async def run_worker() -> None:
    if not settings.SEMANTIC_WORKER_ENABLED:
        logger.info(
            "Semantic worker disabled (SEMANTIC_WORKER_ENABLED=false) — exiting "
            "without opening a DB connection."
        )
        return

    global _evaluators, _second_opinion_evaluators, _conversation_evaluator
    _evaluators = _build_evaluators()
    _second_opinion_evaluators = _build_second_opinion_evaluators()
    _conversation_evaluator = _build_conversation_evaluator()

    await init_pool()
    await ensure_semantic_schema()
    logger.info("Semantic worker started. poll_interval=%ss", settings.POLL_INTERVAL)
    try:
        while True:
            try:
                runs, sampled, signals = await poll_once()
                if runs:
                    logger.info(
                        "Cycle complete. runs=%d sampled=%d signals=%d", runs, sampled, signals
                    )
            except Exception as exc:
                logger.error("Poll cycle failed: %s", exc)
            await asyncio.sleep(settings.POLL_INTERVAL)
    except asyncio.CancelledError:
        logger.info("Semantic worker cancelled")
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(run_worker())
