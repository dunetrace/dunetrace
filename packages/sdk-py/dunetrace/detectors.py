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

Tuning
------
All detectors accept keyword overrides for their UPPERCASE class attributes:

    ToolLoopDetector(THRESHOLD=2)

Each detector's docstring lists its tunable parameters and what they control.
Unknown parameter names raise TypeError immediately — there is no silent fallback:

    ToolLoopDetector(THREHOLD=2)  # raises TypeError: unknown parameter 'THREHOLD'

The public defaults in TIER1_DETECTORS are conservative starting points.
Tuned values derived from real customer data belong in the private
detector service configuration, not here.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional

from dunetrace.models import FailureSignal, FailureType, RunState, Severity


# ── Base ──────────────────────────────────────────────────────────────────────

class BaseDetector:
    name: str = "base"

    def __init__(self, **overrides: object) -> None:
        """Instantiate with optional threshold overrides.

        Only UPPERCASE class attributes are tunable. Passing an unknown key
        raises TypeError immediately so typos fail at startup, not at runtime.

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
    Same tool called >= THRESHOLD times within a WINDOW of steps.

    Invariant: No tool should dominate a sliding window without progress.
    High confidence — this pattern is structurally unambiguous.

    Tunable parameters:
        WINDOW    (int, default 5)  — sliding window width in tool calls.
                  Increase for agents that legitimately call the same tool
                  in bursts (e.g. paginated search). Decrease to catch
                  tighter loops faster.
        THRESHOLD (int, default 3)  — minimum repetitions within WINDOW
                  to trigger. Must be <= WINDOW. Lower values increase
                  sensitivity and false-positive rate.
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

    Tunable parameters:
        WINDOW (int, default 6)  — number of recent tool calls to inspect.
               Must be even for a clean alternating-pair pattern. Larger
               values require the oscillation to be sustained longer before
               firing; smaller values fire on shorter bursts.
    """
    name   = "TOOL_THRASHING"
    WINDOW = 6

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if len(state.tool_calls) < self.WINDOW:
            return None

        names = [c.tool_name for c in state.tool_calls[-self.WINDOW:]]
        unique = set(names)

        if len(unique) == 2:
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

    MIN_LLM_CALLS guards against short runs (e.g. 1-step "I know the answer"
    responses) where the agent barely started and had no meaningful opportunity
    to reason about whether a tool was needed. Firing on those inflates
    false-positive rate significantly.

    Tunable parameters:
        MIN_LLM_CALLS (int, default 2)  — minimum number of LLM calls the
                      run must contain before this detector fires. Prevents
                      false positives on trivially short runs. Raise if your
                      agent category routinely answers in 1–2 LLM turns
                      without tools by design.
    """
    name          = "TOOL_AVOIDANCE"
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
                "llm_calls":       len(state.llm_calls),
            },
        )


# ── GOAL_ABANDONMENT ──────────────────────────────────────────────────────────

class GoalAbandonmentDetector(BaseDetector):
    """
    Agent was using tools, then stopped calling tools mid-run
    without producing a final answer.

    STALL_STEPS consecutive LLM events with no tool calls,
    after at least one tool call had been made.

    Tunable parameters:
        STALL_STEPS (int, default 4)  — number of consecutive LLM-only events
                    required to trigger. Increase for agents that legitimately
                    do multi-step reasoning between tool calls. Decrease to
                    catch abandonment faster.
    """
    name        = "GOAL_ABANDONMENT"
    STALL_STEPS = 4

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if state.exit_reason is not None:
            return None
        if not state.tool_calls:
            return None

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
        ("ignore_instructions",    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?"),
        ("disregard_instructions", r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?"),
        ("forget_instructions",    r"forget\s+(all\s+)?(previous|prior|above)\s+instructions?"),
        ("you_are_now",            r"you\s+are\s+now\s+"),
        ("new_role",               r"your\s+new\s+(role|persona|identity|instructions?)\s+(is|are)"),
        ("act_as",                 r"act\s+as\s+(if\s+you\s+are\s+)?(a|an|the)\s+"),
        ("pretend",                r"pretend\s+(you\s+are|to\s+be)\s+"),
        ("do_not_follow",          r"do\s+not\s+follow\s+(your\s+)?(previous|prior|original)\s+"),
        ("system_colon",           r"system\s*:\s*you\s+are"),
        ("system_tag",             r"\[system\]"),
        ("im_start",               r"<\|im_start\|>"),
        ("system_pipe",            r"<\|system\|>"),
        ("hash_system",            r"###\s*system"),
        ("jailbreak",              r"jailbreak"),
        ("dan_mode",               r"dan\s+mode"),
        ("developer_mode",         r"developer\s+mode\s+(enabled|on)"),
        ("override_safety",        r"override\s+(safety|guidelines|restrictions)"),
        ("bypass_safety",          r"bypass\s+(safety|restrictions|filters)"),
    ]
]


class PromptInjectionDetector(BaseDetector):
    """
    Pattern-matches user input against known prompt injection signatures.
    Fires on input receipt — before any LLM call.

    No tunable parameters. The pattern list (_INJECTION_PATTERNS_COMPILED)
    is a module-level constant; extend it by adding entries there.
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

    Tunable parameters:
        MIN_SCORE   (float, default 0.3)  — relevance score below which a
                    retrieval is considered "effectively empty". Raise for
                    stricter RAG quality requirements; lower if your retrieval
                    system uses a compressed score range.
        MIN_RESULTS (int, default 1)      — minimum result count required
                    for a retrieval to be considered non-empty. Raise if
                    your agent requires multiple grounding documents.
    """
    name        = "RAG_EMPTY_RETRIEVAL"
    MIN_SCORE   = 0.3
    MIN_RESULTS = 1

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if state.exit_reason != "final_answer":
            return None
        if not state.retrievals:
            return None

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
                "index_name":     worst.index_name,
                "result_count":   worst.result_count,
                "top_score":      worst.top_score,
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

    Tunable parameters:
        THRESHOLD (int, default 2)  — number of truncated LLM responses
                  required to trigger. Default of 2 means "more than one
                  truncation = systematic problem". Set to 1 for zero-tolerance
                  environments; raise for models with known token limit issues
                  where a single truncation is expected and handled.
    """
    name      = "LLM_TRUNCATION_LOOP"
    THRESHOLD = 2

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
                "truncation_count":      len(truncated),
                "total_llm_calls":       len(state.llm_calls),
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

    Tunable parameters:
        MIN_CALLS       (int, default 3)    — minimum LLM calls with token
                        data required before checking for a trend. Prevents
                        false positives on short runs.
        GROWTH_FACTOR   (float, default 3.0) — ratio of last/first prompt
                        tokens that triggers the signal. 3.0 = context
                        tripled. Lower for stricter cost control; raise for
                        agents designed to accumulate context intentionally
                        (e.g. long-horizon coding agents).
        MIN_LAST_TOKENS (int, default 2000) — minimum final prompt token
                        count required to trigger. Suppresses false positives
                        on small-context agents where proportional growth
                        poses no real truncation or cost risk.
    """
    name            = "CONTEXT_BLOAT"
    MIN_CALLS       = 3
    GROWTH_FACTOR   = 3.0
    MIN_LAST_TOKENS = 2000

    def check(self, state: RunState) -> Optional[FailureSignal]:
        calls_with_tokens = [
            c for c in state.llm_calls
            if c.prompt_tokens is not None and c.prompt_tokens > 0
        ]

        if len(calls_with_tokens) < self.MIN_CALLS:
            return None

        first_tokens = calls_with_tokens[0].prompt_tokens
        last_tokens  = calls_with_tokens[-1].prompt_tokens

        if first_tokens < 10:
            return None

        if last_tokens < self.MIN_LAST_TOKENS:
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
      - After tool.called:  tool execution — slow means hung API or timeout
      - After llm.called:   LLM inference — slow means provider latency spike
      - After other events: agent overhead — slow means something unexpected

    A single slow step is meaningful on its own (unlike loops that need N
    repetitions). A tool hanging for 45 seconds is a problem regardless of
    whether it happens once or ten times.

    Severity scales with how far the duration exceeds the threshold:
      - 2–5× threshold  → MEDIUM
      - >5× threshold   → HIGH

    Tunable parameters:
        THRESHOLDS (list of (prefix, ms, label), default shown below)  —
                   ordered list of (event_type_prefix, threshold_ms, label).
                   First matching prefix wins. The empty-string entry is a
                   catch-all and should remain last.

                   Default:
                       [("tool.called", 15_000, "tool execution"),
                        ("llm.called",  30_000, "LLM call"),
                        ("",            60_000, "step")]

                   Tune per-category: web-search agents hitting slow external
                   APIs may warrant a higher tool threshold; latency-sensitive
                   applications may want lower LLM thresholds.
    """
    name = "SLOW_STEP"

    THRESHOLDS = [
        ("tool.called", 15_000, "tool execution"),
        ("llm.called",  30_000, "LLM call"),
        ("",            60_000, "step"),
    ]

    def _threshold_for(self, event_type: str) -> tuple[int, str]:
        for prefix, ms, label in self.THRESHOLDS:
            if not prefix or event_type.startswith(prefix):
                return ms, label
        return 60_000, "step"

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if not state.step_durations_ms or not state.events:
            return None

        worst_step_idx  = None
        worst_duration  = 0
        worst_threshold = 1
        worst_label     = "step"
        worst_event_type = ""

        step_event_type = {e.step_index: e.event_type.value for e in state.events}

        for step_idx, duration_ms in state.step_durations_ms.items():
            event_type = step_event_type.get(step_idx, "")
            threshold_ms, label = self._threshold_for(event_type)

            if duration_ms > threshold_ms:
                ratio = duration_ms / threshold_ms
                if ratio > (worst_duration / max(worst_threshold, 1)):
                    worst_step_idx   = step_idx
                    worst_duration   = duration_ms
                    worst_threshold  = threshold_ms
                    worst_label      = label
                    worst_event_type = event_type

        if worst_step_idx is None:
            return None

        ratio    = worst_duration / worst_threshold
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
                "step_index":   worst_step_idx,
                "duration_ms":  worst_duration,
                "threshold_ms": worst_threshold,
                "event_type":   worst_event_type,
                "step_label":   worst_label,
                "ratio":        round(ratio, 1),
                "all_slow_steps": {
                    k: v for k, v in state.step_durations_ms.items()
                    if v > self._threshold_for(step_event_type.get(k, ""))[0]
                },
            },
        )


# ── RETRY_STORM ───────────────────────────────────────────────────────────────

class RetryStormDetector(BaseDetector):
    """
    Same tool called THRESHOLD or more times in a row where every preceding
    tool.responded reported success=False.

    Distinct from TOOL_LOOP: args_hash may differ (genuine retry with varied
    inputs) but the tool keeps failing. Indicates a broken dependency — an
    API that is down, returning errors, or rejecting every request — that the
    agent cannot detect and back off from.

    HIGH severity: every failure burns tokens (LLM re-plans after each failed
    call) and the agent will almost always reach max_iterations without
    producing a useful answer.

    Tunable parameters:
        THRESHOLD (int, default 3)  — consecutive failures on the same tool
                  required to trigger. Lower values catch dependency failures
                  faster; raise for agents that implement their own retry
                  logic and where 2 retries are expected before escalating.
    """
    name      = "RETRY_STORM"
    THRESHOLD = 3

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if len(state.tool_calls) < self.THRESHOLD:
            return None

        best_tool  = None
        best_count = 0
        best_first = 0

        i = len(state.tool_calls) - 1
        while i >= 0:
            tc = state.tool_calls[i]
            if tc.success is not False:
                i -= 1
                continue

            tool  = tc.tool_name
            count = 0
            j     = i
            while j >= 0 and state.tool_calls[j].tool_name == tool and state.tool_calls[j].success is False:
                count += 1
                j -= 1

            if count >= self.THRESHOLD and count > best_count:
                best_count = count
                best_tool  = tool
                best_first = state.tool_calls[j + 1].step_index

            i = j - 1

        if best_tool is None:
            return None

        return FailureSignal(
            failure_type=FailureType.RETRY_STORM,
            severity=Severity.HIGH,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=state.current_step,
            confidence=0.92,
            evidence={
                "tool":              best_tool,
                "consecutive_fails": best_count,
                "threshold":         self.THRESHOLD,
                "first_fail_step":   best_first,
            },
        )


# ── EMPTY_LLM_RESPONSE ─────────────────────────────────────────────────────────

class EmptyLlmResponseDetector(BaseDetector):
    """
    output_length == 0 on an llm.responded event with finish_reason == "stop".

    The model was asked something and returned nothing. Most agent frameworks
    don't handle empty responses gracefully — the agent typically crashes,
    loops, or silently produces a blank final answer. High precision because
    a legitimate zero-length stop response is effectively impossible in normal
    operation.

    No tunable parameters — the condition is binary and has no meaningful
    threshold to adjust.
    """
    name = "EMPTY_LLM_RESPONSE"

    def check(self, state: RunState) -> Optional[FailureSignal]:
        empty = [
            c for c in state.llm_calls
            if c.finish_reason == "stop"
            and getattr(c, "output_length", None) == 0
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
                "occurrences":   len(empty),
                "first_step":    first.step_index,
                "finish_reason": "stop",
            },
        )


# ── STEP_COUNT_INFLATION ───────────────────────────────────────────────────────

class StepCountInflationDetector(BaseDetector):
    """
    Current run used more than INFLATION_FACTOR × the P75 step count for
    this (agent_id, agent_version) over the last 50 historical runs.

    Requires state.baseline_p75_steps to be populated by the worker before
    this detector runs. Returns None (skip silently) when baseline is absent
    — the detector needs at least 10 historical runs to be meaningful.

    Fires once per run (highest severity), scoped to the final step_index.

    Tunable parameters:
        INFLATION_FACTOR (float, default 2.0)  — multiplier applied to the
                         P75 baseline. A run is flagged when:
                             current_steps > baseline_p75 × INFLATION_FACTOR
                         Lower values catch moderate inflation earlier;
                         higher values reserve the signal for severe cases.
                         Tune per-category: research agents with high step
                         variance may need 2.5–3.0 to avoid noise; coding
                         agents with stable step counts may warrant 1.5.
    """
    name             = "STEP_COUNT_INFLATION"
    INFLATION_FACTOR = 2.0

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if state.baseline_p75_steps is None:
            return None

        if state.current_step <= state.baseline_p75_steps * self.INFLATION_FACTOR:
            return None

        ratio = state.current_step / state.baseline_p75_steps

        return FailureSignal(
            failure_type=FailureType.STEP_COUNT_INFLATION,
            severity=Severity.MEDIUM,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=state.current_step,
            confidence=0.82,
            evidence={
                "current_steps":    state.current_step,
                "baseline_p75":     round(state.baseline_p75_steps, 1),
                "inflation_ratio":  round(ratio, 2),
                "threshold_factor": self.INFLATION_FACTOR,
            },
        )


# ── CASCADING_TOOL_FAILURE ─────────────────────────────────────────────────────

class CascadingToolFailureDetector(BaseDetector):
    """
    THRESHOLD or more consecutive tool calls all returned success=False,
    across at least 2 distinct tools.

    Distinguishes from RETRY_STORM (same tool) and TOOL_THRASHING (alternation
    pattern regardless of success). Cascade = multiple tools all broken in the
    same run, often because a shared upstream dependency (a database, an API
    gateway) has failed and every tool that depends on it returns an error.

    HIGH severity: the agent cannot make progress regardless of which tool it
    switches to, and will burn all remaining iterations before giving up.

    Tunable parameters:
        THRESHOLD (int, default 3)  — minimum consecutive cross-tool failures
                  required to trigger. Raise for agents that handle partial
                  dependency failures gracefully and where 2 failures before
                  recovery are expected.
    """
    name      = "CASCADING_TOOL_FAILURE"
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
            confidence=0.88,
            evidence={
                "consecutive_failures": len(failed_run),
                "distinct_tools":       sorted(distinct_tools),
                "threshold":            self.THRESHOLD,
                "first_fail_step":      first_fail_step,
            },
        )


# ── FIRST_STEP_FAILURE ─────────────────────────────────────────────────────────

class FirstStepFailureDetector(BaseDetector):
    """
    Error, empty LLM output, or tool failure at step <= MAX_STEP.

    Early failures have a completely different root cause and remediation
    profile vs mid-run failures:
      - Not a logic or tool problem — it's the entrypoint
      - Most likely: malformed input, prompt syntax error, policy refusal,
        missing required parameter, or auth failure on the first tool call
      - Debugging target: the run setup, not the agent loop

    MEDIUM severity because it's a single data point — one bad input doesn't
    mean the agent is broken. HIGH if it happens repeatedly (handled by alerts
    rate logic).

    Tunable parameters:
        MAX_STEP (int, default 2)  — step index at or below which a failure
                 is classified as "first step". Steps 0, 1, 2 = the setup
                 phase. Raise for agents with a longer initialisation sequence
                 (e.g. agents that authenticate and warm up before the first
                 real tool call).
    """
    name     = "FIRST_STEP_FAILURE"
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
                    "trigger":     "run_errored",
                    "failed_step": state.current_step,
                    "max_step":    self.MAX_STEP,
                },
            )

        early_empty = [
            c for c in state.llm_calls
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
                    "trigger":     "empty_llm_response",
                    "failed_step": early_empty[0].step_index,
                    "max_step":    self.MAX_STEP,
                },
            )

        early_fail = [
            tc for tc in state.tool_calls
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
                    "trigger":     "tool_failure",
                    "failed_step": early_fail[0].step_index,
                    "tool":        early_fail[0].tool_name,
                    "max_step":    self.MAX_STEP,
                },
            )

        return None


# ── REASONING_SPIN ─────────────────────────────────────────────────────────────

class ReasoningSpinDetector(BaseDetector):
    """
    The agent made far more LLM calls than tool calls within a completed run,
    indicating it spent most of its iterations reasoning/planning rather than
    taking actions that advance state.

    A healthy agent alternates: think → act → observe → think → act.
    A spinning agent does: think → think → think → think → (minimal action).

    This is different from TOOL_AVOIDANCE (zero tool calls) and
    GOAL_ABANDONMENT (tools used then stopped mid-run). REASONING_SPIN fires
    when the agent did use tools but the LLM:tool ratio is extremely skewed,
    meaning the agent spent the bulk of its iterations deliberating without
    making meaningful progress through tool use.

    Fires only at final_answer to avoid false positives on in-progress runs
    where the agent is legitimately planning before a burst of tool calls.

    MEDIUM severity — the run may have completed, but efficiency is poor and
    the agent is likely to hit step limits on harder variants of the same task.

    Tunable parameters:
        MIN_LLM_CALLS   (int, default 5)    — minimum LLM calls before checking.
                        Prevents false positives on very short runs.
        RATIO_THRESHOLD (float, default 4.0) — LLM calls / tool calls at or above
                        which the signal fires. 4.0 means 4 LLM calls per 1 tool
                        call. Raise for agents with multi-step chain-of-thought
                        designs where high LLM:tool ratios are intentional.
    """
    name            = "REASONING_SPIN"
    MIN_LLM_CALLS   = 5
    RATIO_THRESHOLD = 4.0

    def check(self, state: RunState) -> Optional[FailureSignal]:
        if state.exit_reason != "final_answer":
            return None

        llm_count  = len(state.llm_calls)
        tool_count = len(state.tool_calls)

        if llm_count < self.MIN_LLM_CALLS:
            return None

        ratio = llm_count / max(tool_count, 1)

        if ratio < self.RATIO_THRESHOLD:
            return None

        return FailureSignal(
            failure_type=FailureType.REASONING_STALL,
            severity=Severity.MEDIUM,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=state.current_step,
            confidence=0.70,
            evidence={
                "llm_calls":      llm_count,
                "tool_calls":     tool_count,
                "ratio":          round(ratio, 2),
                "threshold":      self.RATIO_THRESHOLD,
            },
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
    # PromptInjectionDetector is handled separately (needs raw input)
]

PROMPT_INJECTION_DETECTOR = PromptInjectionDetector()


def run_detectors(
    state: RunState,
    detectors: Optional[List[BaseDetector]] = None,
) -> List[FailureSignal]:
    """Run Tier 1 detectors against the current run state.

    Args:
        state:     The reconstructed run state to check.
        detectors: Detector list to use. Defaults to TIER1_DETECTORS (public
                   conservative defaults). Pass a custom list to use
                   production-tuned parameters from the private detector
                   configuration without modifying this file.

    Returns:
        List of FailureSignal, one per triggered detector. Empty list if
        no failures detected.
    """
    active = detectors if detectors is not None else TIER1_DETECTORS
    signals = []
    for detector in active:
        signal = detector.check(state)
        if signal:
            signals.append(signal)
    return signals
