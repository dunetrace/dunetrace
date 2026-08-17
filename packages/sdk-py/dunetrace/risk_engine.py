"""
RiskEngine: post-processes detector signals to produce an aggregate run-level
risk score with multi-signal boosting, hard rules, and time escalation.

Why this exists over raw detector confidences
---------------------------------------------
Individual detectors fire with hardcoded confidence floats (e.g. 0.95 for
TOOL_LOOP, 0.70 for GOAL_ABANDONMENT). Those floats are per-signal precision
estimates tuned in isolation. They don't account for two things that change
the real probability of a genuine failure:

  1. Co-occurrence: if TOOL_LOOP + RETRY_STORM + CONTEXT_BLOAT all fire in the
     same run, the joint probability is much higher than any single signal.
     Multi-signal boosting captures this without calibration data.

  2. Extremity: a tool called 8× with identical args is qualitatively different
     from 3×. Hard rules encode these deterministic cut-offs so they can't be
     suppressed by a low base score.

  3. Slow runaways: some failures accumulate over many steps and never trigger a
     single detector at a high threshold. Time escalation catches these by
     amplifying confidence when latency is already excessive.

Integration
-----------
Call after run_detectors() in the detector worker::

    from dunetrace.detectors import run_detectors
    from dunetrace.risk_engine import RiskEngine

    signals   = run_detectors(state)
    risk      = RiskEngine().evaluate(signals, state)
    # risk.confidence gates alert dispatch; risk.scores explain the alert

The RiskScore does NOT replace FailureSignals — individual signals still carry
detector-level evidence (which tool looped, which step failed, etc.). The score
is a run-level envelope used by the alert and dashboard layers.
"""

from __future__ import annotations

from collections import Counter
from typing import List

from dunetrace.models import EventType, FailureSignal, RiskScore, RunState


class RiskEngine:
    """
    Produces a run-level RiskScore from detector signals and RunState.

    All thresholds are UPPERCASE class attributes — override per agent category::

        engine = RiskEngine(HARD_LOOP_CALLS=6, BOOST_MULTIPLIERS={2: 1.3, 3: 1.5})

    Unknown keys raise TypeError immediately (same convention as BaseDetector).
    """

    # ── Feature score constants ─────────────────────────────────────────────────
    LOOP_WINDOW = 5  # sliding window for loop score
    STAGNATION_STEPS = 4  # tail LLM-only steps that map to score 1.0
    TOKEN_MAX_GROWTH = 3.0  # token growth ratio that maps to score 1.0
    RETRY_MAX_FAILS = 5  # consecutive failures that map to score 1.0
    LATENCY_OVERAGE_CAP = 4.0  # (5× threshold - 1) maps to latency score 1.0

    # ── Hard rule constants ─────────────────────────────────────────────────────
    HARD_LOOP_CALLS = 8  # same-tool call count across full run
    HARD_SIMILARITY_THRESH = 0.9  # fraction with identical args
    HARD_FAILURE_STREAK = 5  # consecutive tool failures

    # ── Scoring constants ───────────────────────────────────────────────────────
    # Boost multipliers keyed by active feature count (capped at 3)
    BOOST_MULTIPLIERS = {0: 1.0, 1: 1.0, 2: 1.2, 3: 1.4}
    # Feature score threshold to count as "active"
    ACTIVE_THRESHOLD = 0.6
    # Latency escalation weight: confidence *= (1 + latency_score * weight)
    LATENCY_ESCALATION_WEIGHT = 0.5

    def __init__(self, **overrides: object) -> None:
        tunable = {k for k in vars(type(self)) if k.isupper()}
        unknown = set(overrides) - tunable
        if unknown:
            raise TypeError(
                f"RiskEngine: unknown parameter(s) {sorted(unknown)}. Tunable: {sorted(tunable)}"
            )
        for k, v in overrides.items():
            setattr(self, k, v)

    # ── Public API ──────────────────────────────────────────────────────────────

    def evaluate(self, signals: List[FailureSignal], run_state: RunState) -> RiskScore:
        """
        Evaluate aggregate run risk.

        :param signals:   FailureSignals from run_detectors() for this run.
        :param run_state: Reconstructed RunState.
        :returns: RiskScore. Hard rules return early with deterministic confidence;
                  normal runs go through feature scoring + boosting + escalation.
        """
        scores = {
            "loop": self._loop_score(run_state),
            "stagnation": self._stagnation_score(run_state),
            "token": self._token_score(run_state),
            "retry": self._retry_score(run_state),
            "latency": self._latency_score(run_state),
        }

        # Hard rules — deterministic, skip continuous scoring
        for override in self._check_hard_rules(run_state):
            return override

        # Base: strongest single feature
        base_confidence = max(scores.values()) if scores else 0.0

        # Multi-signal boost: co-firing features reduce false positives
        active = sum(1 for s in scores.values() if s > self.ACTIVE_THRESHOLD)
        multiplier = self.BOOST_MULTIPLIERS.get(min(active, 3), 1.4)
        confidence = min(1.0, base_confidence * multiplier)

        # Time escalation: amplify for slow runaways
        time_mult = 1.0 + (scores["latency"] * self.LATENCY_ESCALATION_WEIGHT)
        confidence = min(1.0, confidence * time_mult)

        return RiskScore(
            confidence=confidence,
            active_signals=active,
            scores=scores,
        )

    # ── Hard rules ──────────────────────────────────────────────────────────────

    def _check_hard_rules(self, state: RunState) -> List[RiskScore]:
        """
        Deterministic overrides for extreme conditions.
        First match wins when the caller iterates.
        No calibration needed — the pattern is structurally unambiguous.
        """
        overrides: List[RiskScore] = []

        # Extreme tool loop: same tool called many times with nearly identical args
        max_calls, similarity = self._loop_stats(state)
        if max_calls >= self.HARD_LOOP_CALLS and similarity >= self.HARD_SIMILARITY_THRESH:
            overrides.append(
                RiskScore(
                    confidence=0.98,
                    active_signals=1,
                    scores={"loop": 1.0},
                    severity="CRITICAL",
                )
            )

        # Extreme failure streak: agent cannot recover from dependency failure
        consec = self._consecutive_failures(state)
        if consec >= self.HARD_FAILURE_STREAK:
            overrides.append(
                RiskScore(
                    confidence=0.95,
                    active_signals=1,
                    scores={"retry": 1.0},
                    severity="HIGH",
                )
            )

        return overrides

    # ── Feature scores ──────────────────────────────────────────────────────────

    def _loop_score(self, state: RunState) -> float:
        """
        Fraction of the last LOOP_WINDOW tool calls dominated by a single tool.
        0.0 when fewer than 2 tool calls; 1.0 when all calls are the same tool.
        """
        if len(state.tool_calls) < 2:
            return 0.0
        recent = state.tool_calls[-self.LOOP_WINDOW :]
        counts = Counter(tc.tool_name for tc in recent)
        max_count = max(counts.values())
        return min(1.0, max_count / len(recent))

    def _stagnation_score(self, state: RunState) -> float:
        """
        Fraction of the last STAGNATION_STEPS events that are LLM-only.
        Only meaningful when the agent has made at least one tool call (otherwise
        a tool-free agent would always score high).
        """
        if not state.tool_calls or not state.events:
            return 0.0
        tail = state.events[-self.STAGNATION_STEPS :]
        llm_only = sum(1 for e in tail if e.event_type.value.startswith("llm."))
        return min(1.0, llm_only / self.STAGNATION_STEPS)

    def _token_score(self, state: RunState) -> float:
        """
        Prompt token growth from first to last LLM call, normalised so
        TOKEN_MAX_GROWTH maps to 1.0. Returns 0.0 on insufficient data.
        """
        calls = [c for c in state.llm_calls if c.prompt_tokens and c.prompt_tokens > 0]
        if len(calls) < 2:
            return 0.0
        first: int = calls[0].prompt_tokens or 0
        last: int = calls[-1].prompt_tokens or 0
        if first < 10 or last < 100:
            return 0.0
        growth = last / first
        return min(1.0, (growth - 1.0) / (self.TOKEN_MAX_GROWTH - 1.0))

    def _retry_score(self, state: RunState) -> float:
        """Consecutive tail failures normalised so RETRY_MAX_FAILS maps to 1.0."""
        return min(1.0, self._consecutive_failures(state) / self.RETRY_MAX_FAILS)

    def _latency_score(self, state: RunState) -> float:
        """
        Worst step overage, normalised so LATENCY_OVERAGE_CAP times the threshold
        maps to 1.0. 0.0 when no step exceeds its threshold.

        Thresholds mirror SlowStepDetector: tool.called→15s, llm.called→30s, else→60s.
        """
        if not state.step_durations_ms or not state.events:
            return 0.0

        _THRESHOLDS = [
            ("tool.called", 15_000),
            ("llm.called", 30_000),
            ("", 60_000),
        ]
        agent_events = [e for e in state.events if e.event_type is not EventType.EXTERNAL_SIGNAL]
        step_event_type = {e.step_index: e.event_type.value for e in agent_events}

        worst_ratio = 0.0
        for step_idx, duration_ms in state.step_durations_ms.items():
            et = step_event_type.get(step_idx, "")
            threshold = next(
                (ms for prefix, ms in _THRESHOLDS if not prefix or et.startswith(prefix)),
                60_000,
            )
            if duration_ms > threshold:
                worst_ratio = max(worst_ratio, duration_ms / threshold)

        if worst_ratio <= 1.0:
            return 0.0
        return min(1.0, (worst_ratio - 1.0) / self.LATENCY_OVERAGE_CAP)

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _consecutive_failures(self, state: RunState) -> int:
        """Count of consecutive tool failures at the tail of the tool call list."""
        count = 0
        for tc in reversed(state.tool_calls):
            if tc.success is False:
                count += 1
            else:
                break
        return count

    def _loop_stats(self, state: RunState) -> tuple[int, float]:
        """
        (max_call_count_for_any_tool, arg_similarity) across ALL tool calls.
        arg_similarity = fraction of the dominant tool's calls sharing the most
        common args value (1.0 = all identical, 0.0 = all different).
        """
        if not state.tool_calls:
            return 0, 0.0

        by_tool: dict[str, list] = {}
        for tc in state.tool_calls:
            by_tool.setdefault(tc.tool_name, []).append(tc)

        max_calls = 0
        similarity = 0.0
        for calls in by_tool.values():
            if len(calls) > max_calls:
                max_calls = len(calls)
                counts = Counter(tc.args for tc in calls)
                most_common = counts.most_common(1)[0][1]
                similarity = most_common / len(calls)

        return max_calls, similarity
