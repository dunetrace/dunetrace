"""
Tier 1 structural detectors. Each takes a RunState and returns an optional signal.
No LLM calls, no external dependencies, deterministic, <1ms per check.

New detectors default to shadow mode. Don't flip them live until precision >80%
is confirmed on real data.

Tuning
------
Detectors accept keyword overrides for their UPPERCASE class attributes:

    ToolLoopDetector(THRESHOLD=2)

Each detector's docstring lists its tunable parameters. Unknown keys raise
TypeError immediately — no silent fallbacks:

    ToolLoopDetector(THREHOLD=2)  # TypeError: unknown parameter 'THREHOLD'

The defaults in TIER1_DETECTORS are conservative starting points.
Tuned values belong in the detector service config, not here.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional

from dunetrace.models import EventType, FailureSignal, FailureType, RunState, Severity


def _scale_confidence(ratio: float) -> float:
    """Confidence as a function of how far the observation exceeds its trigger threshold.

    ratio = observed / threshold.  At ratio=1.0 (barely triggers): 0.5.
    Reaches 1.0 when ratio ≥ 3.25 (2.25× beyond the trigger point).
    Applied to count/ratio detectors; binary detectors keep their static values.
    """
    return min(1.0, 0.5 + (ratio - 1.0) * 0.4)


# ── Base ──────────────────────────────────────────────────────────────────────


class BaseDetector:
    name: str = "base"

    def __init__(self, **overrides: object) -> None:
        """Accept keyword overrides for UPPERCASE class attributes. Unknown keys raise TypeError at startup, not at runtime.

        Example:
            ToolLoopDetector(THRESHOLD=2)          # ok
            ToolLoopDetector(THREHOLD=2)            # TypeError
        """
        tunable: set[str] = set()
        for cls in type(self).__mro__:
            if cls is object:
                break
            tunable.update(k for k in vars(cls) if k.isupper())

        unknown = set(overrides) - tunable
        if unknown:
            raise TypeError(
                f"{type(self).__name__}: unknown parameter(s) {sorted(unknown)}. "
                f"Tunable: {sorted(tunable) if tunable else 'none'}"
            )
        for k, v in overrides.items():
            setattr(self, k, v)

    def check(self, state: RunState) -> Optional[FailureSignal]:
        raise NotImplementedError


# ── TOOL_LOOP ─────────────────────────────────────────────────────────────────


class ToolLoopDetector(BaseDetector):
    """
    Same tool called >= THRESHOLD times within a WINDOW of steps. High confidence — the
    pattern is structurally unambiguous.

    Tunable: WINDOW (default 5) — sliding window width. Increase for agents that legitimately
    burst the same tool (e.g. paginated search). THRESHOLD (default 3) — repetitions needed
    to fire; lower values increase sensitivity and false-positive rate.
    """

    name = "TOOL_LOOP"
    WINDOW = 5
    THRESHOLD = 3

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if len(state.tool_calls) < self.WINDOW:
            return None

        window = state.tool_calls[-self.WINDOW :]
        counts = Counter(c.tool_name for c in window)

        for tool, count in counts.items():
            if count >= self.THRESHOLD:
                all_calls = [c for c in state.tool_calls if c.tool_name == tool]
                args_hashes = [c.args_hash for c in all_calls]
                unique_hashes = len(set(args_hashes))
                calls_with_result = [c for c in all_calls if c.success is not None]
                success_rate = (
                    sum(1 for c in calls_with_result if c.success)
                    / len(calls_with_result)
                    if calls_with_result
                    else None
                )
                first_step = all_calls[0].step_index
                last_step = all_calls[-1].step_index
                loop_steps = set(range(first_step + 1, last_step + 1))
                wasted_tokens = (
                    sum(
                        lc.prompt_tokens
                        for lc in state.llm_calls
                        if lc.step_index in loop_steps and lc.prompt_tokens is not None
                    )
                    or None
                )
                return FailureSignal(
                    failure_type=FailureType.TOOL_LOOP,
                    severity=Severity.HIGH,
                    run_id=state.run_id,
                    agent_id=state.agent_id,
                    agent_version=state.agent_version,
                    step_index=window[-1].step_index,
                    confidence=_scale_confidence(len(all_calls) / self.THRESHOLD),
                    evidence={
                        "tool": tool,
                        "count": len(all_calls),
                        "window": self.WINDOW,
                        "first_step": first_step,
                        "last_step": last_step,
                        "step_indices": [c.step_index for c in all_calls],
                        "args_hashes": args_hashes,
                        "args_identical": unique_hashes == 1,
                        "args_similar": unique_hashes <= 2,
                        "success_rate": success_rate,
                        "wasted_tokens": wasted_tokens,
                    },
                )
        return None


# ── TOOL_THRASHING ─────────────────────────────────────────────────────────────


class ToolThrashingDetector(BaseDetector):
    """
    Agent oscillates between exactly two tools — [A, B, A, B, A, B] — within WINDOW steps,
    usually because it can't reconcile conflicting tool outputs.

    Tunable: WINDOW (default 6) — must be even for a clean alternating-pair check. Larger
    values require the oscillation to be sustained longer before firing.
    """

    name = "TOOL_THRASHING"
    WINDOW = 6

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if len(state.tool_calls) < self.WINDOW:
            return None

        names = [c.tool_name for c in state.tool_calls[-self.WINDOW :]]
        unique = set(names)

        if len(unique) == 2:
            alternating = all(names[i] != names[i + 1] for i in range(len(names) - 1))
            if alternating:
                tools = list(unique)
                return FailureSignal(
                    failure_type=FailureType.TOOL_THRASHING,
                    severity=Severity.HIGH,
                    run_id=state.run_id,
                    agent_id=state.agent_id,
                    agent_version=state.agent_version,
                    step_index=state.current_step,
                    confidence=0.90,
                    evidence={
                        "tool_a": tools[0],
                        "tool_b": tools[1],
                        "pattern": names,
                        "count": len(names),
                    },
                )
        return None


# ── TOOL_AVOIDANCE ─────────────────────────────────────────────────────────────


class ToolAvoidanceDetector(BaseDetector):
    """
    Agent produced a final answer without calling any tools, despite tools being available.
    Lower confidence (0.75) — some queries legitimately don't need tools, so validate on
    real data before treating this as live.

    MIN_LLM_CALLS (default 2) guards against short 1-step runs where the agent never had a
    real chance to decide about tool use — those inflate the false-positive rate a lot.
    Raise if your agent routinely answers in 1–2 turns without tools by design.
    """

    name = "TOOL_AVOIDANCE"
    MIN_LLM_CALLS = 2

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if state.exit_reason != "final_answer":
            return None
        if not state.available_tools:
            return None
        if state.tool_calls:
            return None
        if len(state.llm_calls) < self.MIN_LLM_CALLS:
            return None

        return FailureSignal(
            failure_type=FailureType.TOOL_AVOIDANCE,
            severity=Severity.MEDIUM,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=state.current_step,
            confidence=0.75,
            evidence={
                "available_tools": state.available_tools,
                "tool_calls_made": 0,
                "llm_calls": len(state.llm_calls),
            },
        )


# ── GOAL_ABANDONMENT ──────────────────────────────────────────────────────────


class GoalAbandonmentDetector(BaseDetector):
    """
    Agent used tools, then stopped mid-run without a final answer — STALL_STEPS consecutive
    LLM events with no tool calls after at least one tool was called.

    Tunable: STALL_STEPS (default 4). Increase for agents that do multi-step reasoning
    between tool calls; decrease to catch abandonment faster.
    """

    name = "GOAL_ABANDONMENT"
    STALL_STEPS = 4

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if state.exit_reason is not None:
            return None
        if not state.tool_calls:
            return None

        recent = state.events[-self.STALL_STEPS :]
        if len(recent) < self.STALL_STEPS:
            return None

        all_llm = all(e.event_type.value.startswith("llm.") for e in recent)
        if all_llm:
            steps_since_last_tool = state.current_step - state.tool_calls[-1].step_index
            return FailureSignal(
                failure_type=FailureType.GOAL_ABANDONMENT,
                severity=Severity.MEDIUM,
                run_id=state.run_id,
                agent_id=state.agent_id,
                agent_version=state.agent_version,
                step_index=state.current_step,
                confidence=_scale_confidence(steps_since_last_tool / self.STALL_STEPS),
                evidence={
                    "stall_steps": self.STALL_STEPS,
                    "last_tool_step": state.tool_calls[-1].step_index,
                    "last_tool_used": state.tool_calls[-1].tool_name,
                    "current_step": state.current_step,
                    "steps_since_last_tool": steps_since_last_tool,
                    "stall_event_sequence": [e.event_type.value for e in recent],
                },
            )
        return None


# ── PROMPT_INJECTION_SIGNAL ───────────────────────────────────────────────────

_INJECTION_PATTERNS_COMPILED = [
    (label, re.compile(p, re.IGNORECASE))
    for label, p in [
        (
            "ignore_instructions",
            r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?",
        ),
        (
            "disregard_instructions",
            r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
        ),
        (
            "forget_instructions",
            r"forget\s+(all\s+)?(previous|prior|above)\s+instructions?",
        ),
        ("you_are_now", r"you\s+are\s+now\s+"),
        ("new_role", r"your\s+new\s+(role|persona|identity|instructions?)\s+(is|are)"),
        ("act_as", r"act\s+as\s+(if\s+you\s+are\s+)?(a|an|the)\s+"),
        ("pretend", r"pretend\s+(you\s+are|to\s+be)\s+"),
        (
            "do_not_follow",
            r"do\s+not\s+follow\s+(your\s+)?(previous|prior|original)\s+",
        ),
        ("system_colon", r"system\s*:\s*you\s+are"),
        ("system_tag", r"\[system\]"),
        ("im_start", r"<\|im_start\|>"),
        ("system_pipe", r"<\|system\|>"),
        ("hash_system", r"###\s*system"),
        ("jailbreak", r"jailbreak"),
        ("dan_mode", r"dan\s+mode"),
        ("developer_mode", r"developer\s+mode\s+(enabled|on)"),
        ("override_safety", r"override\s+(safety|guidelines|restrictions)"),
        ("bypass_safety", r"bypass\s+(safety|restrictions|filters)"),
    ]
]


class PromptInjectionDetector(BaseDetector):
    """
    Pattern-matches user input against known injection signatures, before any LLM call.
    No tunable parameters — extend by adding entries to _INJECTION_PATTERNS_COMPILED.
    """

    name = "PROMPT_INJECTION_SIGNAL"

    def check_input(self, input_text: str, state: RunState) -> Optional[FailureSignal]:
        matched = [
            label
            for label, pattern in _INJECTION_PATTERNS_COMPILED
            if pattern.search(input_text)
        ]
        if not matched:
            return None

        return FailureSignal(
            failure_type=FailureType.PROMPT_INJECTION_SIGNAL,
            severity=Severity.CRITICAL,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=0,
            confidence=_scale_confidence(len(matched)),
            evidence={
                "matched_pattern_count": len(matched),
                "matched_patterns": matched[:5],
                "input_length": len(input_text),
            },
        )

    def check(self, state: RunState) -> Optional[FailureSignal]:
        return None


# ── RAG_EMPTY_RETRIEVAL ───────────────────────────────────────────────────────


class RagEmptyRetrievalDetector(BaseDetector):
    """
    Retrieval returned zero results or a below-threshold score, but the agent answered anyway —
    drawing from training memory instead of retrieved context.

    Tunable: MIN_SCORE (default 0.3) — raise for stricter RAG quality, lower if your retrieval
    system uses a compressed score range. MIN_RESULTS (default 1) — raise if the agent needs
    multiple grounding documents before answering.
    """

    name = "RAG_EMPTY_RETRIEVAL"
    MIN_SCORE = 0.3
    MIN_RESULTS = 1

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if state.exit_reason != "final_answer":
            return None
        if not state.retrievals:
            return None

        bad_retrievals = [
            r
            for r in state.retrievals
            if r.result_count < self.MIN_RESULTS
            or (r.top_score is not None and r.top_score < self.MIN_SCORE)
        ]

        if not bad_retrievals:
            return None

        worst = min(bad_retrievals, key=lambda r: r.result_count)
        return FailureSignal(
            failure_type=FailureType.RAG_EMPTY_RETRIEVAL,
            severity=Severity.MEDIUM,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=state.current_step,
            confidence=0.88,
            evidence={
                "index_name": worst.index_name,
                "result_count": worst.result_count,
                "top_score": worst.top_score,
                "bad_retrievals": len(bad_retrievals),
            },
        )


# ── LLM_TRUNCATION_LOOP ───────────────────────────────────────────────────────


class LlmTruncationLoopDetector(BaseDetector):
    """
    finish_reason="length" fires THRESHOLD or more times in a run. One truncation is
    recoverable; multiple means the agent isn't handling incomplete responses — it keeps
    calling the LLM with a context that truncates every time (tool outputs appended without
    summarising, context already bloated, etc.). HIGH severity because truncated outputs
    break downstream logic: partial JSON, cut-off plans, incomplete code.

    Tunable: THRESHOLD (default 2). Set to 1 for zero tolerance; raise for models where
    a single truncation is expected and handled by the agent.
    """

    name = "LLM_TRUNCATION_LOOP"
    THRESHOLD = 2

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if len(state.llm_calls) < self.THRESHOLD:
            return None

        truncated = [c for c in state.llm_calls if c.finish_reason == "length"]

        if len(truncated) < self.THRESHOLD:
            return None

        return FailureSignal(
            failure_type=FailureType.LLM_TRUNCATION_LOOP,
            severity=Severity.HIGH,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=state.current_step,
            confidence=_scale_confidence(len(truncated) / self.THRESHOLD),
            evidence={
                "truncation_count": len(truncated),
                "total_llm_calls": len(state.llm_calls),
                "first_truncation_step": truncated[0].step_index,
                "last_truncation_step": truncated[-1].step_index,
                "token_counts_at_truncation": [
                    c.prompt_tokens for c in truncated if c.prompt_tokens is not None
                ],
                "models": sorted({c.model for c in truncated if c.model}),
            },
        )


# ── CONTEXT_BLOAT ─────────────────────────────────────────────────────────────


class ContextBloatDetector(BaseDetector):
    """
    prompt_tokens grew by GROWTH_FACTOR or more from first to last LLM call. The agent
    is accumulating tool outputs, history, or retrieved docs without pruning. Left unchecked:
    context window overflow, attention dilution, and escalating per-call cost. MEDIUM severity
    — bloat is a leading indicator; the run may still succeed. Pairs with LLM_TRUNCATION_LOOP
    which fires when bloat causes actual truncation.

    Tunable: MIN_CALLS (default 3) — minimum LLM calls needed before checking, prevents
    false positives on short runs. GROWTH_FACTOR (default 3.0) — last/first token ratio;
    lower for stricter cost control, raise for agents that intentionally accumulate context.
    MIN_LAST_TOKENS (default 2000) — suppresses false positives where proportional growth
    on a tiny context isn't actually a problem.
    """

    name = "CONTEXT_BLOAT"
    MIN_CALLS = 3
    GROWTH_FACTOR = 3.0
    MIN_LAST_TOKENS = 2000
    INFLATION_FACTOR = 2.0  # multiplier over P75 baseline when history is available

    def check(self, state: RunState) -> Optional[FailureSignal]:
        calls_with_tokens = [
            c
            for c in state.llm_calls
            if c.prompt_tokens is not None and c.prompt_tokens > 0
        ]

        if len(calls_with_tokens) < self.MIN_CALLS:
            return None

        first_tokens = calls_with_tokens[0].prompt_tokens
        last_tokens = calls_with_tokens[-1].prompt_tokens

        if first_tokens < 10:
            return None

        if last_tokens < self.MIN_LAST_TOKENS:
            return None

        growth = last_tokens / first_tokens

        if state.baseline_p75_token_growth is not None:
            effective_threshold = (
                state.baseline_p75_token_growth * self.INFLATION_FACTOR
            )
        else:
            effective_threshold = self.GROWTH_FACTOR

        if growth < effective_threshold:
            return None

        evidence: dict = {
            "first_tokens": first_tokens,
            "last_tokens": last_tokens,
            "growth_factor": round(growth, 2),
            "threshold_factor": round(effective_threshold, 2),
            "llm_call_count": len(calls_with_tokens),
            "first_call_step": calls_with_tokens[0].step_index,
            "last_call_step": calls_with_tokens[-1].step_index,
            "token_growth_sequence": [
                {"step": c.step_index, "tokens": c.prompt_tokens}
                for c in calls_with_tokens
            ],
        }
        if state.baseline_p75_token_growth is not None:
            evidence["baseline_p75"] = round(state.baseline_p75_token_growth, 2)

        return FailureSignal(
            failure_type=FailureType.CONTEXT_BLOAT,
            severity=Severity.MEDIUM,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=state.current_step,
            confidence=_scale_confidence(growth / effective_threshold),
            evidence=evidence,
        )


# ── SLOW_STEP ──────────────────────────────────────────────────────────────────


class SlowStepDetector(BaseDetector):
    """
    Any single step takes longer than a type-specific threshold. Thresholds are set by what
    precedes the gap: tool.called (15s — hung API), llm.called (30s — provider latency spike),
    everything else (60s). A single slow step is meaningful on its own — a tool hanging 45s is
    a problem once. Severity scales: 2–5× threshold → MEDIUM, >5× → HIGH.

    Tunable: THRESHOLDS is an ordered list of (event_type_prefix, threshold_ms, label). First
    matching prefix wins; the empty-string entry is a catch-all and must stay last.
    Default: [("tool.called", 15_000, ...), ("llm.called", 30_000, ...), ("", 60_000, ...)].
    """

    name = "SLOW_STEP"

    THRESHOLDS = [
        ("tool.called", 15_000, "tool execution"),
        ("llm.called", 30_000, "LLM call"),
        ("", 60_000, "step"),
    ]
    INFLATION_FACTOR = 2.0  # multiplier over P75 baseline when history is available

    def _threshold_for(
        self, event_type: str, state: Optional[RunState] = None
    ) -> tuple[int, str]:
        for prefix, static_ms, label in self.THRESHOLDS:
            if not prefix or event_type.startswith(prefix):
                if state is not None:
                    if (
                        prefix == "tool.called"
                        and state.baseline_p75_latency_tool is not None
                    ):
                        return (
                            int(
                                state.baseline_p75_latency_tool * self.INFLATION_FACTOR
                            ),
                            label,
                        )
                    if (
                        prefix == "llm.called"
                        and state.baseline_p75_latency_llm is not None
                    ):
                        return (
                            int(state.baseline_p75_latency_llm * self.INFLATION_FACTOR),
                            label,
                        )
                return static_ms, label
        return 60_000, "step"

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if not state.step_durations_ms or not state.events:
            return None

        worst_step_idx = None
        worst_duration = 0
        worst_threshold = 1
        worst_label = "step"
        worst_event_type = ""

        # Exclude external.signal events — they share the step_index of the
        # agent event they annotate and must not overwrite its event_type or
        # timestamp in the lookup dicts.
        agent_events = [
            e for e in state.events if e.event_type is not EventType.EXTERNAL_SIGNAL
        ]
        step_event_type = {e.step_index: e.event_type.value for e in agent_events}
        step_timestamp = {e.step_index: e.timestamp for e in agent_events}

        for step_idx, duration_ms in state.step_durations_ms.items():
            event_type = step_event_type.get(step_idx, "")
            threshold_ms, label = self._threshold_for(event_type, state)

            if duration_ms > threshold_ms:
                ratio = duration_ms / threshold_ms
                if ratio > (worst_duration / max(worst_threshold, 1)):
                    worst_step_idx = step_idx
                    worst_duration = duration_ms
                    worst_threshold = threshold_ms
                    worst_label = label
                    worst_event_type = event_type

        if worst_step_idx is None:
            return None

        ratio = worst_duration / worst_threshold
        severity = Severity.HIGH if ratio >= 5 else Severity.MEDIUM

        evidence: dict = {
            "step_index": worst_step_idx,
            "duration_ms": worst_duration,
            "threshold_ms": worst_threshold,
            "event_type": worst_event_type,
            "step_label": worst_label,
            "ratio": round(ratio, 1),
            "all_slow_steps": {
                k: v
                for k, v in state.step_durations_ms.items()
                if v > self._threshold_for(step_event_type.get(k, ""), state)[0]
            },
        }

        # Include the raw P75 baseline so dashboards can show what normal looks like.
        if (
            worst_event_type.startswith("tool.called")
            and state.baseline_p75_latency_tool is not None
        ):
            evidence["baseline_p75"] = round(state.baseline_p75_latency_tool, 1)
        elif (
            worst_event_type.startswith("llm.called")
            and state.baseline_p75_latency_llm is not None
        ):
            evidence["baseline_p75"] = round(state.baseline_p75_latency_llm, 1)

        # Correlate with external signals that occurred during the slow step.
        # A signal is coincident when its timestamp falls within
        # [step_start, step_start + duration].
        if state.external_signals:
            step_start = step_timestamp.get(worst_step_idx, 0.0)
            step_end = step_start + worst_duration / 1000.0
            coincident = [
                {
                    k: v
                    for k, v in [
                        ("signal_name", sig.signal_name),
                        ("source", sig.source),
                        ("meta", sig.meta or None),
                    ]
                    if v
                }
                for sig in state.external_signals
                if step_start <= sig.timestamp <= step_end
            ]
            if coincident:
                evidence["coincident_signals"] = coincident

        return FailureSignal(
            failure_type=FailureType.SLOW_STEP,
            severity=severity,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=worst_step_idx,
            confidence=_scale_confidence(ratio),
            evidence=evidence,
        )


# ── RETRY_STORM ───────────────────────────────────────────────────────────────


class RetryStormDetector(BaseDetector):
    """
    Same tool called THRESHOLD or more times in a row, all returning success=False.
    Unlike TOOL_LOOP, args_hash may differ — the agent is genuinely retrying — but the tool
    keeps failing. Indicates a broken dependency (API down, rejecting every request) that
    the agent can't detect and back off from. HIGH severity — each failure burns an LLM turn
    to re-plan, and the agent will almost always exhaust max_iterations with nothing to show.

    Evidence: args_identical (True if no variation in args), reason_identical (True if the same
    error every time), failure_reason_hash (common error hash when reason_identical).

    Tunable: THRESHOLD (default 3). Lower to catch dependency failures faster; raise for agents
    with built-in retry logic where 2 failures before escalation are expected.
    """

    name = "RETRY_STORM"
    THRESHOLD = 3

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if len(state.tool_calls) < self.THRESHOLD:
            return None

        best_tool = None
        best_count = 0
        best_streak: list = []

        i = len(state.tool_calls) - 1
        while i >= 0:
            tc = state.tool_calls[i]
            if tc.success is not False:
                i -= 1
                continue

            tool = tc.tool_name
            j = i
            streak = []
            while (
                j >= 0
                and state.tool_calls[j].tool_name == tool
                and state.tool_calls[j].success is False
            ):
                streak.append(state.tool_calls[j])
                j -= 1

            if len(streak) >= self.THRESHOLD and len(streak) > best_count:
                best_count = len(streak)
                best_tool = tool
                best_streak = streak  # ordered newest-first

            i = j - 1

        if best_tool is None:
            return None

        # Analyse the streak for args identity and failure reason identity.
        # streak is newest-first; reverse for chronological ordering in evidence.
        best_streak.reverse()
        args_hashes = [tc.args_hash for tc in best_streak]
        error_hashes = [tc.error_hash for tc in best_streak]

        # Self-correction check: if the tool subsequently succeeded after the streak,
        # the agent recovered and this is CoT/retry behaviour, not a storm.
        last_fail_step = best_streak[-1].step_index
        recovered = any(
            tc.tool_name == best_tool
            and tc.success is True
            and tc.step_index > last_fail_step
            for tc in state.tool_calls
        )
        if recovered:
            return None

        args_identical = len(set(args_hashes)) == 1
        all_have_reason = all(h is not None for h in error_hashes)
        reason_identical = all_have_reason and len(set(error_hashes)) == 1
        failure_reason_hash = error_hashes[0] if reason_identical else None

        return FailureSignal(
            failure_type=FailureType.RETRY_STORM,
            severity=Severity.HIGH,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=state.current_step,
            confidence=_scale_confidence(best_count / self.THRESHOLD),
            evidence={
                "tool": best_tool,
                "consecutive_fails": best_count,
                "threshold": self.THRESHOLD,
                "first_fail_step": best_streak[0].step_index,
                "step_indices": [tc.step_index for tc in best_streak],
                "args_identical": args_identical,
                "error_hashes": error_hashes,
                "failure_reason_hash": failure_reason_hash,
                "reason_identical": reason_identical,
            },
        )


# ── EMPTY_LLM_RESPONSE ─────────────────────────────────────────────────────────


class EmptyLlmResponseDetector(BaseDetector):
    """
    output_length == 0 with finish_reason == "stop" — the model returned nothing.
    Most frameworks don't handle this gracefully; the agent typically crashes, loops, or
    silently produces a blank answer. High precision — a legitimate zero-length stop
    response is effectively impossible in normal operation. No tunable parameters.
    """

    name = "EMPTY_LLM_RESPONSE"

    def check(self, state: RunState) -> Optional[FailureSignal]:
        empty = [
            c
            for c in state.llm_calls
            if c.finish_reason == "stop" and getattr(c, "output_length", None) == 0
        ]
        if not empty:
            return None

        first = empty[0]
        return FailureSignal(
            failure_type=FailureType.EMPTY_LLM_RESPONSE,
            severity=Severity.HIGH,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=first.step_index,
            confidence=0.95,
            evidence={
                "occurrences": len(empty),
                "first_step": first.step_index,
                "finish_reason": "stop",
            },
        )


# ── STEP_COUNT_INFLATION ───────────────────────────────────────────────────────


class StepCountInflationDetector(BaseDetector):
    """
    Current run exceeded INFLATION_FACTOR × the P75 step count for this (agent_id,
    agent_version) over the last 50 successful runs. Skips silently when baseline is absent
    — needs at least 10 historical runs to be meaningful.

    Tunable: INFLATION_FACTOR (default 2.0). Lower to catch moderate inflation earlier;
    raise for research agents with high step variance (2.5–3.0) or lower for coding agents
    with tight, predictable step counts (1.5).
    """

    name = "STEP_COUNT_INFLATION"
    INFLATION_FACTOR = 2.0

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if state.baseline_p75_steps is None:
            return None

        if state.current_step <= state.baseline_p75_steps * self.INFLATION_FACTOR:
            return None

        ratio = state.current_step / state.baseline_p75_steps
        # confidence anchored to how far above the effective threshold (baseline × factor)
        effective_threshold = state.baseline_p75_steps * self.INFLATION_FACTOR

        return FailureSignal(
            failure_type=FailureType.STEP_COUNT_INFLATION,
            severity=Severity.MEDIUM,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=state.current_step,
            confidence=_scale_confidence(state.current_step / effective_threshold),
            evidence={
                "current_steps": state.current_step,
                "baseline_p75": round(state.baseline_p75_steps, 1),
                "inflation_ratio": round(ratio, 2),
                "threshold_factor": self.INFLATION_FACTOR,
            },
        )


# ── CASCADING_TOOL_FAILURE ─────────────────────────────────────────────────────


class CascadingToolFailureDetector(BaseDetector):
    """
    THRESHOLD or more consecutive failures across at least 2 distinct tools. Unlike RETRY_STORM
    (same tool) or TOOL_THRASHING (alternation pattern), this fires when multiple tools are all
    broken — usually a shared upstream dependency (DB, API gateway) that every tool depends on.
    HIGH severity — the agent can't make progress regardless of which tool it switches to.

    Tunable: THRESHOLD (default 3). Raise for agents that handle partial dependency failures
    gracefully where 2 consecutive failures before recovery are expected.
    """

    name = "CASCADING_TOOL_FAILURE"
    THRESHOLD = 3

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if len(state.tool_calls) < self.THRESHOLD:
            return None

        failed_run = []
        for tc in reversed(state.tool_calls):
            if tc.success is not False:
                break
            failed_run.append(tc)

        if len(failed_run) < self.THRESHOLD:
            return None

        distinct_tools = {tc.tool_name for tc in failed_run}
        if len(distinct_tools) < 2:
            return None

        first_fail_step = failed_run[-1].step_index

        return FailureSignal(
            failure_type=FailureType.CASCADING_TOOL_FAILURE,
            severity=Severity.HIGH,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=state.current_step,
            confidence=_scale_confidence(len(failed_run) / self.THRESHOLD),
            evidence={
                "consecutive_failures": len(failed_run),
                "distinct_tools": sorted(distinct_tools),
                "threshold": self.THRESHOLD,
                "first_fail_step": first_fail_step,
            },
        )


# ── FIRST_STEP_FAILURE ─────────────────────────────────────────────────────────


class FirstStepFailureDetector(BaseDetector):
    """
    Error, empty LLM output, or tool failure at step <= MAX_STEP. Early failures point to
    different root causes than mid-run failures — malformed input, prompt syntax errors, policy
    refusals, missing params, or auth failures on the first tool call. Debug the setup, not the
    loop. MEDIUM severity per occurrence; high recurrence is handled upstream by alert rate logic.

    Tunable: MAX_STEP (default 2). Raise for agents with a longer init sequence (auth + warmup
    before the first real tool call).
    """

    name = "FIRST_STEP_FAILURE"
    MAX_STEP = 2

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if state.exit_reason == "error" and state.current_step <= self.MAX_STEP:
            return FailureSignal(
                failure_type=FailureType.FIRST_STEP_FAILURE,
                severity=Severity.MEDIUM,
                run_id=state.run_id,
                agent_id=state.agent_id,
                agent_version=state.agent_version,
                step_index=state.current_step,
                confidence=0.90,
                evidence={
                    "trigger": "run_errored",
                    "failed_step": state.current_step,
                    "max_step": self.MAX_STEP,
                },
            )

        early_empty = [
            c
            for c in state.llm_calls
            if c.step_index <= self.MAX_STEP
            and getattr(c, "output_length", None) == 0
            and c.finish_reason == "stop"
        ]
        if early_empty:
            return FailureSignal(
                failure_type=FailureType.FIRST_STEP_FAILURE,
                severity=Severity.MEDIUM,
                run_id=state.run_id,
                agent_id=state.agent_id,
                agent_version=state.agent_version,
                step_index=early_empty[0].step_index,
                confidence=0.88,
                evidence={
                    "trigger": "empty_llm_response",
                    "failed_step": early_empty[0].step_index,
                    "max_step": self.MAX_STEP,
                },
            )

        early_fail = [
            tc
            for tc in state.tool_calls
            if tc.step_index <= self.MAX_STEP and tc.success is False
        ]
        if early_fail:
            return FailureSignal(
                failure_type=FailureType.FIRST_STEP_FAILURE,
                severity=Severity.MEDIUM,
                run_id=state.run_id,
                agent_id=state.agent_id,
                agent_version=state.agent_version,
                step_index=early_fail[0].step_index,
                confidence=0.85,
                evidence={
                    "trigger": "tool_failure",
                    "failed_step": early_fail[0].step_index,
                    "tool": early_fail[0].tool_name,
                    "max_step": self.MAX_STEP,
                },
            )

        return None


# ── REASONING_SPIN ─────────────────────────────────────────────────────────────


class ReasoningSpinDetector(BaseDetector):
    """
    The LLM:tool call ratio is extremely skewed — the agent spent most of its steps
    deliberating rather than acting. A healthy agent alternates think→act→observe;
    a spinning one does think→think→think→(minimal action). Different from TOOL_AVOIDANCE
    (zero tool calls) and GOAL_ABANDONMENT (stopped mid-run) — here the agent did complete
    with tool use, but the ratio is way off. Only fires at final_answer. MEDIUM severity —
    the run may have finished, but efficiency is poor and harder variants will hit step limits.

    Tunable: MIN_LLM_CALLS (default 5) prevents false positives on short runs.
    RATIO_THRESHOLD (default 4.0) — LLM calls / tool calls. Raise for agents with intentional
    multi-step chain-of-thought designs where high ratios are expected.
    """

    name = "REASONING_STALL"
    MIN_LLM_CALLS = 5
    RATIO_THRESHOLD = 4.0
    INFLATION_FACTOR = 2.0  # multiplier over P75 baseline when history is available

    def check(self, state: RunState) -> Optional[FailureSignal]:
        # Skip errored runs — FIRST_STEP_FAILURE and RETRY_STORM cover those.
        if state.exit_reason == "error":
            return None

        llm_count = len(state.llm_calls)
        tool_count = len(state.tool_calls)

        if llm_count < self.MIN_LLM_CALLS:
            return None

        ratio = llm_count / max(tool_count, 1)

        if state.baseline_p75_llm_tool_ratio is not None:
            effective_threshold = (
                state.baseline_p75_llm_tool_ratio * self.INFLATION_FACTOR
            )
        else:
            effective_threshold = self.RATIO_THRESHOLD

        if ratio < effective_threshold:
            return None

        # A run that ended with a final answer is inefficient (MEDIUM).
        # A run that stalled without converging shows the ratio caused failure (HIGH).
        severity = (
            Severity.MEDIUM if state.exit_reason == "final_answer" else Severity.HIGH
        )

        action_events = [
            e
            for e in state.events
            if e.event_type.value.startswith("llm.called")
            or e.event_type.value.startswith("tool.called")
        ]
        event_sequence = [
            "llm" if e.event_type.value.startswith("llm.") else "tool"
            for e in action_events
        ]

        evidence: dict = {
            "llm_calls": llm_count,
            "tool_calls": tool_count,
            "ratio": round(ratio, 2),
            "threshold": round(effective_threshold, 2),
            "exit_reason": state.exit_reason,
            "event_sequence": event_sequence,
        }
        if state.baseline_p75_llm_tool_ratio is not None:
            evidence["baseline_p75"] = round(state.baseline_p75_llm_tool_ratio, 2)

        return FailureSignal(
            failure_type=FailureType.REASONING_STALL,
            severity=severity,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=state.current_step,
            confidence=_scale_confidence(ratio / effective_threshold),
            evidence=evidence,
        )


# ── COST_SPIKE ────────────────────────────────────────────────────────────────


class CostSpikeDetector(BaseDetector):
    """
    Total token consumption (prompt + completion) for this run is unusually high
    compared to the agent's own P75 baseline.

    Catches runs that burned far more tokens than typical — due to runaway loops,
    context explosion, or a model swap that wasn't accounted for in budgeting.

    Tunable: INFLATION_FACTOR (default 3.0) — how many times over P75 before firing.
    STATIC_THRESHOLD_TOKENS (default 50 000) — fallback when no baseline is available.
    MIN_LLM_CALLS (default 1) — skip runs with no LLM activity.
    """

    name = "COST_SPIKE"
    INFLATION_FACTOR = 3.0
    STATIC_THRESHOLD_TOKENS = 50_000
    MIN_LLM_CALLS = 1

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if len(state.llm_calls) < self.MIN_LLM_CALLS:
            return None

        total_tokens = sum(
            (c.prompt_tokens or 0) + (c.completion_tokens or 0) for c in state.llm_calls
        )
        if total_tokens == 0:
            return None

        if state.baseline_p75_total_tokens is not None:
            threshold = state.baseline_p75_total_tokens * self.INFLATION_FACTOR
        else:
            threshold = float(self.STATIC_THRESHOLD_TOKENS)

        if total_tokens < threshold:
            return None

        ratio = total_tokens / max(threshold, 1)
        evidence: dict = {
            "total_tokens": total_tokens,
            "threshold": int(threshold),
            "inflation_ratio": round(ratio, 2),
            "llm_calls": len(state.llm_calls),
        }
        if state.baseline_p75_total_tokens is not None:
            evidence["baseline_p75"] = int(state.baseline_p75_total_tokens)

        return FailureSignal(
            failure_type=FailureType.COST_SPIKE,
            severity=Severity.MEDIUM,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=state.current_step,
            confidence=_scale_confidence(ratio),
            evidence=evidence,
        )


# ── SESSION_LATENCY ───────────────────────────────────────────────────────────


class SessionLatencyDetector(BaseDetector):
    """
    Total wall-clock run duration is unusually high compared to the agent's own P75 baseline.

    Catches slow runs caused by hanging tools, large context windows, or infrastructure issues
    that inflate per-run latency without necessarily triggering a step-level SLOW_STEP signal.

    Tunable: INFLATION_FACTOR (default 3.0) — how many times over P75 before firing.
    STATIC_THRESHOLD_SECS (default 300) — fallback threshold when no baseline exists (5 min).
    MIN_EVENTS (default 2) — need at least two events to compute a duration.
    """

    name = "SESSION_LATENCY"
    INFLATION_FACTOR = 3.0
    STATIC_THRESHOLD_SECS = 300
    MIN_EVENTS = 2

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if len(state.events) < self.MIN_EVENTS:
            return None

        timestamps = [e.timestamp for e in state.events]
        duration_s = max(timestamps) - min(timestamps)

        if duration_s <= 0:
            return None

        if state.baseline_p75_duration_s is not None:
            threshold = state.baseline_p75_duration_s * self.INFLATION_FACTOR
        else:
            threshold = float(self.STATIC_THRESHOLD_SECS)

        if duration_s < threshold:
            return None

        ratio = duration_s / max(threshold, 1)
        evidence: dict = {
            "duration_s": round(duration_s, 1),
            "threshold_s": round(threshold, 1),
            "inflation_ratio": round(ratio, 2),
        }
        if state.baseline_p75_duration_s is not None:
            evidence["baseline_p75_s"] = round(state.baseline_p75_duration_s, 1)

        return FailureSignal(
            failure_type=FailureType.SESSION_LATENCY,
            severity=Severity.MEDIUM,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=state.current_step,
            confidence=_scale_confidence(ratio),
            evidence=evidence,
        )


# ── Registry ──────────────────────────────────────────────────────────────────

TIER1_DETECTORS: List[BaseDetector] = [
    ToolLoopDetector(),
    ToolThrashingDetector(),
    ToolAvoidanceDetector(),
    GoalAbandonmentDetector(),
    RagEmptyRetrievalDetector(),
    LlmTruncationLoopDetector(),
    ContextBloatDetector(),
    SlowStepDetector(),
    RetryStormDetector(),
    EmptyLlmResponseDetector(),
    StepCountInflationDetector(),
    CascadingToolFailureDetector(),
    FirstStepFailureDetector(),
    ReasoningSpinDetector(),
    CostSpikeDetector(),
    SessionLatencyDetector(),
    # PromptInjectionDetector is handled separately (needs raw input)
]

PROMPT_INJECTION_DETECTOR = PromptInjectionDetector()


def run_detectors(
    state: RunState,
    detectors: Optional[List[BaseDetector]] = None,
) -> List[FailureSignal]:
    """Run all detectors against a run state. Pass a custom list to use production-tuned
    parameters; defaults to TIER1_DETECTORS if omitted. Returns one FailureSignal per
    triggered detector, or an empty list if nothing fired."""
    active = detectors if detectors is not None else TIER1_DETECTORS
    signals = []
    for detector in active:
        signal = detector.check(state)
        if signal:
            signals.append(signal)
    return signals
