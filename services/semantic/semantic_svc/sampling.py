"""
Adaptive sampling engine (Phase 1.2). Pure decision logic — no DB, no I/O — so
it's fully unit-testable and reusable by worker.py without a running stack.

Budget enforcement is deliberately NOT here: it requires reading and
atomically incrementing usage counters, which is inherently stateful/async.
worker.py calls decide_sampling() first (this module), then separately checks
budget via semantic_svc.db.consume_budget() and downgrades the decision if the
agent's monthly budget is exhausted. See worker.py::process_run.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace


def stable_bucket(key: str) -> int:
    """Deterministic bucket in [0, 100) for a string key.

    Not Python's builtin hash(): that's randomized per-process for str
    (PYTHONHASHSEED), so the same run_id would land in a different bucket
    after every worker restart. sha256 is stable across restarts, processes,
    and Python versions — required for "same run_id always gets the same
    sampling decision."
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 100


@dataclass(frozen=True)
class SamplingRates:
    structural_signal_rate: float = 1.0
    semantic_critical_rate: float = 1.0
    retrieval_rate: float = 0.20
    baseline_rate: float = 0.05
    # Phase 3.2 — independent of the per-run rates above. Bucketed on the
    # conversation's own external_id, not run_id: once a conversation lands
    # in the sampled bucket, it stays sampled for its whole lifetime (every
    # new run re-triggers a fresh conversation-level re-evaluation); once
    # out, it's out. Conversation-level evaluation carries its own quota
    # (see db.py's org_conversation_evaluation_usage) precisely because it's
    # a separate, more expensive kind of spend from per-run sampling.
    conversation_evaluation_rate: float = 0.50


DEFAULT_RATES = SamplingRates()


def decide_sampling(
    run_id: str,
    has_structural_signal: bool,
    has_retrieval_event: bool,
    agent_config: dict | None,
    rates: SamplingRates = DEFAULT_RATES,
) -> tuple[bool, str, float]:
    """Returns (sampled, reason, rate_used).

    Precedence (highest first). The two "always evaluate" cases are quality
    gates (something already broke, or this agent is flagged high-stakes) —
    they win regardless of any per-agent rate override. The per-agent
    override, when set, replaces the *general population* rate (retrieval vs.
    baseline) for everything else:

      1. structural signal fired on this run     -> structural_signal_rate
      2. agent flagged semantic_critical=true    -> semantic_critical_rate
      3. per-agent sample_rate override           -> agent_config["sample_rate"]
      4. retrieval event occurred                 -> retrieval_rate
      5. baseline                                 -> baseline_rate

    Determinism: stable_bucket(run_id) is the same every time, so re-deciding
    the same run_id (e.g. a re-poll after a config change) never flips the
    outcome unless the rate itself changed.
    """
    if has_structural_signal:
        rate, reason = rates.structural_signal_rate, "structural_signal"
    elif agent_config and agent_config.get("semantic_critical"):
        rate, reason = rates.semantic_critical_rate, "semantic_critical"
    elif agent_config and agent_config.get("sample_rate") is not None:
        rate, reason = float(agent_config["sample_rate"]), "agent_override"
    elif has_retrieval_event:
        rate, reason = rates.retrieval_rate, "retrieval"
    else:
        rate, reason = rates.baseline_rate, "baseline"

    sampled = stable_bucket(run_id) < rate * 100
    return sampled, reason, rate


def with_overrides(rates: SamplingRates, **overrides: float) -> SamplingRates:
    """Convenience for config_loader — apply only the keys actually present in
    a parsed config file, leaving the rest at rates' current values."""
    return replace(rates, **overrides)


# Minimum sibling runs before a conversation is even considered for
# conversation-level evaluation — a single run (or two) isn't enough turns
# for "frustration" to be a meaningful, distinguishable signal from one bad
# reply. Tunable; not exposed via semantic-sampling.yml (unlike the rates
# above) since it's a evaluation-validity floor, not a cost/sampling knob.
MIN_CONVERSATION_RUNS = 3


def decide_conversation_sampling(
    conversation_external_id: str,
    run_count: int,
    rates: SamplingRates = DEFAULT_RATES,
) -> tuple[bool, str]:
    """Returns (sampled, reason). Same stable_bucket() determinism as
    decide_sampling() — the same conversation_external_id always gets the
    same decision, so a conversation doesn't flip in and out of monitoring
    as it grows (barring a rate change)."""
    if run_count < MIN_CONVERSATION_RUNS:
        return False, "too_few_runs"

    sampled = stable_bucket(conversation_external_id) < rates.conversation_evaluation_rate * 100
    return sampled, "conversation_sample_rate" if sampled else "not_sampled"
