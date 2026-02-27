"""
dunetrace/detectors.py

Tier 1 structural detectors. All are:
- Pure functions (RunState → Optional[FailureSignal])
- Zero LLM calls
- Zero external dependencies
- Deterministic
- <1ms per check

Every detector ships in shadow mode first. Do not graduate to live
until precision > 80% is validated on real customer data.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from dunetrace.models import FailureSignal, FailureType, RunState, Severity


# ── Base ──────────────────────────────────────────────────────────────────────

class BaseDetector:
    name: str = "base"

    def check(self, state: RunState) -> Optional[FailureSignal]:
        raise NotImplementedError


# ── TOOL_LOOP ─────────────────────────────────────────────────────────────────

class ToolLoopDetector(BaseDetector):
    """
    Same tool called >= THRESHOLD times within a WINDOW of steps.

    Invariant: No tool should dominate a sliding window without progress.
    High confidence — this pattern is structurally unambiguous.
    """
    name      = "TOOL_LOOP"
    WINDOW    = 5
    THRESHOLD = 3

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if len(state.tool_calls) < self.WINDOW:
            return None

        window = state.tool_calls[-self.WINDOW:]
        counts = Counter(c.tool_name for c in window)

        for tool, count in counts.items():
            if count >= self.THRESHOLD:
                return FailureSignal(
                    failure_type=FailureType.TOOL_LOOP,
                    severity=Severity.HIGH,
                    run_id=state.run_id,
                    agent_id=state.agent_id,
                    agent_version=state.agent_version,
                    step_index=state.current_step,
                    confidence=0.95,
                    evidence={
                        "tool":   tool,
                        "count":  count,
                        "window": self.WINDOW,
                    },
                )
        return None


# ── TOOL_THRASHING ─────────────────────────────────────────────────────────────

class ToolThrashingDetector(BaseDetector):
    """
    Agent oscillates between exactly two tools without making progress.
    Pattern: [A, B, A, B, A, B] within WINDOW steps.

    Indicates the agent cannot reconcile conflicting tool outputs.
    """
    name   = "TOOL_THRASHING"
    WINDOW = 6

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if len(state.tool_calls) < self.WINDOW:
            return None

        names = [c.tool_name for c in state.tool_calls[-self.WINDOW:]]
        unique = set(names)

        if len(unique) == 2:
            # Check strictly alternating
            alternating = all(
                names[i] != names[i + 1]
                for i in range(len(names) - 1)
            )
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
                        "tool_a":  tools[0],
                        "tool_b":  tools[1],
                        "pattern": names,
                        "count":   len(names),
                    },
                )
        return None


# ── TOOL_AVOIDANCE ─────────────────────────────────────────────────────────────

class ToolAvoidanceDetector(BaseDetector):
    """
    Agent produced a final answer without calling any tools,
    despite tools being available.

    Lower confidence (0.75) because some queries legitimately don't
    need tools. Precision validation on real data is critical here.
    """
    name = "TOOL_AVOIDANCE"

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if state.exit_reason != "final_answer":
            return None
        if not state.available_tools:
            return None
        if state.tool_calls:
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
            },
        )


# ── GOAL_ABANDONMENT ──────────────────────────────────────────────────────────

class GoalAbandonmentDetector(BaseDetector):
    """
    Agent was using tools, then stopped calling tools mid-run
    without producing a final answer.

    STALL_STEPS consecutive LLM events with no tool calls,
    after at least one tool call had been made.
    """
    name        = "GOAL_ABANDONMENT"
    STALL_STEPS = 4

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if state.exit_reason is not None:
            return None  # run has completed normally
        if not state.tool_calls:
            return None  # never used tools — different problem

        recent = state.events[-self.STALL_STEPS:]
        if len(recent) < self.STALL_STEPS:
            return None

        all_llm = all(
            e.event_type.value.startswith("llm.")
            for e in recent
        )
        if all_llm:
            return FailureSignal(
                failure_type=FailureType.GOAL_ABANDONMENT,
                severity=Severity.MEDIUM,
                run_id=state.run_id,
                agent_id=state.agent_id,
                agent_version=state.agent_version,
                step_index=state.current_step,
                confidence=0.70,
                evidence={
                    "stall_steps":       self.STALL_STEPS,
                    "last_tool_step":    state.tool_calls[-1].step_index,
                    "current_step":      state.current_step,
                },
            )
        return None


# ── PROMPT_INJECTION_SIGNAL ───────────────────────────────────────────────────

_INJECTION_PATTERNS_COMPILED = [
    (label, re.compile(p, re.IGNORECASE))
    for label, p in [
        ("ignore_instructions",   r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?"),
        ("disregard_instructions",r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?"),
        ("forget_instructions",   r"forget\s+(all\s+)?(previous|prior|above)\s+instructions?"),
        ("you_are_now",           r"you\s+are\s+now\s+"),
        ("new_role",              r"your\s+new\s+(role|persona|identity|instructions?)\s+(is|are)"),
        ("act_as",                r"act\s+as\s+(if\s+you\s+are\s+)?(a|an|the)\s+"),
        ("pretend",               r"pretend\s+(you\s+are|to\s+be)\s+"),
        ("do_not_follow",         r"do\s+not\s+follow\s+(your\s+)?(previous|prior|original)\s+"),
        ("system_colon",          r"system\s*:\s*you\s+are"),
        ("system_tag",            r"\[system\]"),
        ("im_start",              r"<\|im_start\|>"),
        ("system_pipe",           r"<\|system\|>"),
        ("hash_system",           r"###\s*system"),
        ("jailbreak",             r"jailbreak"),
        ("dan_mode",              r"dan\s+mode"),
        ("developer_mode",        r"developer\s+mode\s+(enabled|on)"),
        ("override_safety",       r"override\s+(safety|guidelines|restrictions)"),
        ("bypass_safety",         r"bypass\s+(safety|restrictions|filters)"),
    ]
]


class PromptInjectionDetector(BaseDetector):
    """
    Pattern-matches user input against known prompt injection signatures.
    Fires on input receipt — before any LLM call.
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
            confidence=0.85,
            evidence={
                "matched_pattern_count": len(matched),
                "matched_patterns":      matched[:5],
                "input_length":          len(input_text),
            },
        )

    def check(self, state: RunState) -> Optional[FailureSignal]:
        return None


# ── RAG_EMPTY_RETRIEVAL ───────────────────────────────────────────────────────

class RagEmptyRetrievalDetector(BaseDetector):
    """
    Retrieval returned zero results or below-threshold relevance score,
    but the agent produced a final answer anyway.

    The agent answered from memory (or hallucinated) on a query
    that was supposed to be grounded in retrieved data.
    """
    name          = "RAG_EMPTY_RETRIEVAL"
    MIN_SCORE     = 0.3   # below this = "effectively empty"
    MIN_RESULTS   = 1

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if state.exit_reason != "final_answer":
            return None
        if not state.retrievals:
            return None  # no retrieval was attempted

        # Check if any retrieval returned bad results
        bad_retrievals = [
            r for r in state.retrievals
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
                "index_name":    worst.index_name,
                "result_count":  worst.result_count,
                "top_score":     worst.top_score,
                "bad_retrievals": len(bad_retrievals),
            },
        )


# ── LLM_TRUNCATION_LOOP ───────────────────────────────────────────────────────

class LlmTruncationLoopDetector(BaseDetector):
    """
    finish_reason="length" fires THRESHOLD or more times within a run.

    "length" means the model hit its output token limit and stopped
    mid-generation — the response is incomplete. One occurrence is
    recoverable. Multiple occurrences means the agent is not handling
    truncated responses: it keeps calling the LLM with a context that
    produces truncated output every time, typically because:
      - Tool outputs are being appended to context without summarising
      - A summarisation step is present but not reducing context enough
      - The agent is in a loop and context has bloated past the safe zone

    HIGH severity — truncated outputs frequently cause downstream failures
    (broken JSON, incomplete plans, cut-off code).
    """
    name      = "LLM_TRUNCATION_LOOP"
    THRESHOLD = 2   # 2+ truncations in one run = systematic problem

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if len(state.llm_calls) < self.THRESHOLD:
            return None

        truncated = [
            c for c in state.llm_calls
            if c.finish_reason == "length"
        ]

        if len(truncated) < self.THRESHOLD:
            return None

        return FailureSignal(
            failure_type=FailureType.LLM_TRUNCATION_LOOP,
            severity=Severity.HIGH,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=state.current_step,
            confidence=0.90,
            evidence={
                "truncation_count":    len(truncated),
                "total_llm_calls":     len(state.llm_calls),
                "first_truncation_step": truncated[0].step_index,
                "last_truncation_step":  truncated[-1].step_index,
            },
        )


# ── CONTEXT_BLOAT ─────────────────────────────────────────────────────────────

class ContextBloatDetector(BaseDetector):
    """
    prompt_tokens grows by GROWTH_FACTOR or more from the first to the
    last LLM call within a run (minimum MIN_CALLS calls required).

    Context bloat means the agent is accumulating context (tool outputs,
    conversation history, retrieved docs) without pruning or summarising.
    Left unchecked, this leads to:
      - Context window overflow (API error or silent truncation)
      - LLM performance degradation from attention dilution
      - Escalating cost on every subsequent call in the run

    MEDIUM severity because bloat is a leading indicator, not a confirmed
    failure — the run may still succeed. Pairs naturally with
    LLM_TRUNCATION_LOOP which fires when bloat causes actual truncation.
    """
    name          = "CONTEXT_BLOAT"
    MIN_CALLS     = 3       # need at least 3 data points for a trend
    GROWTH_FACTOR = 3.0     # prompt_tokens grew 3x from call[0] to call[-1]

    def check(self, state: RunState) -> Optional[FailureSignal]:
        # Only consider calls where prompt_tokens was reported
        calls_with_tokens = [
            c for c in state.llm_calls
            if c.prompt_tokens is not None and c.prompt_tokens > 0
        ]

        if len(calls_with_tokens) < self.MIN_CALLS:
            return None

        first_tokens = calls_with_tokens[0].prompt_tokens
        last_tokens  = calls_with_tokens[-1].prompt_tokens

        # Guard against degenerate first value
        if first_tokens < 10:
            return None

        growth = last_tokens / first_tokens

        if growth < self.GROWTH_FACTOR:
            return None

        return FailureSignal(
            failure_type=FailureType.CONTEXT_BLOAT,
            severity=Severity.MEDIUM,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=state.current_step,
            confidence=0.80,
            evidence={
                "first_tokens":    first_tokens,
                "last_tokens":     last_tokens,
                "growth_factor":   round(growth, 2),
                "llm_call_count":  len(calls_with_tokens),
                "first_call_step": calls_with_tokens[0].step_index,
                "last_call_step":  calls_with_tokens[-1].step_index,
            },
        )


# ── SLOW_STEP ──────────────────────────────────────────────────────────────────

class SlowStepDetector(BaseDetector):
    """
    Any single step transition takes longer than a type-specific threshold.

    Thresholds are set by what precedes the gap:
      - After tool.called:   tool execution — slow means hung API or timeout
      - After llm.called:    LLM inference — slow means provider latency spike
      - After other events:  agent overhead — slow means something unexpected

    A single slow step is meaningful on its own (unlike loops that need N
    repetitions). A tool hanging for 45 seconds is a problem regardless of
    whether it happens once or ten times.

    Severity scales with how far the duration exceeds the threshold:
      - 2–5× threshold  → MEDIUM
      - >5× threshold   → HIGH
    """
    name = "SLOW_STEP"

    # (event_type_prefix, threshold_ms, label)
    THRESHOLDS = [
        ("tool.called",  15_000, "tool execution"),
        ("llm.called",   30_000, "LLM call"),
        ("",             60_000, "step"),          # catch-all
    ]

    def _threshold_for(self, event_type: str) -> tuple[int, str]:
        for prefix, ms, label in self.THRESHOLDS:
            if not prefix or event_type.startswith(prefix):
                return ms, label
        return 60_000, "step"

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if not state.step_durations_ms or not state.events:
            return None

        worst_step_idx = None
        worst_duration = 0
        worst_threshold = 1
        worst_label = "step"
        worst_event_type = ""

        # Build a map: step_index → event_type so we know what caused the gap
        step_event_type = {e.step_index: e.event_type.value for e in state.events}

        for step_idx, duration_ms in state.step_durations_ms.items():
            event_type = step_event_type.get(step_idx, "")
            threshold_ms, label = self._threshold_for(event_type)

            if duration_ms > threshold_ms:
                # Track the worst offender only — one signal per run
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

        return FailureSignal(
            failure_type=FailureType.SLOW_STEP,
            severity=severity,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=worst_step_idx,
            confidence=0.92,
            evidence={
                "step_index":    worst_step_idx,
                "duration_ms":   worst_duration,
                "threshold_ms":  worst_threshold,
                "event_type":    worst_event_type,
                "step_label":    worst_label,
                "ratio":         round(ratio, 1),
                "all_slow_steps": {
                    k: v for k, v in state.step_durations_ms.items()
                    if v > self._threshold_for(step_event_type.get(k, ""))[0]
                },
            },
        )


# ── Registry ──────────────────────────────────────────────────────────────────

TIER1_DETECTORS = [
    ToolLoopDetector(),
    ToolThrashingDetector(),
    ToolAvoidanceDetector(),
    GoalAbandonmentDetector(),
    RagEmptyRetrievalDetector(),
    LlmTruncationLoopDetector(),
    ContextBloatDetector(),
    SlowStepDetector(),
    # PromptInjectionDetector is handled separately (needs raw input)
]

PROMPT_INJECTION_DETECTOR = PromptInjectionDetector()


def run_detectors(state: RunState) -> list[FailureSignal]:
    """Run all Tier 1 detectors against the current run state."""
    signals = []
    for detector in TIER1_DETECTORS:
        signal = detector.check(state)
        if signal:
            signals.append(signal)
    return signals