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

import ast
import json
import logging
import re
import time
from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional, Tuple
from urllib.parse import urlsplit as _urlsplit

from dunetrace.models import (
    AgentEvent,
    EventType,
    FailureSignal,
    FailureType,
    LlmCall,
    RunState,
    Severity,
    ToolCall,
)
from dunetrace.policies import compute_run_cost

logger = logging.getLogger("dunetrace.detectors")


def _scale_confidence(ratio: float) -> float:
    """Confidence as a function of how far the observation exceeds its trigger threshold.

    ratio = observed / threshold.  At ratio=1.0 (barely triggers): 0.5.
    Reaches 1.0 when ratio ≥ 3.25 (2.25× beyond the trigger point).
    Applied to count/ratio detectors; binary detectors keep their static values.
    """
    return min(1.0, 0.5 + (ratio - 1.0) * 0.4)


def _is_unmeasurable(call: "LlmCall") -> bool:
    """True when this call's response object could not be read at all.

    `instrumentation_degraded` is set by the auto-instrumentation extractors
    (auto.py) when a response is not the shape they expect — a raw/streaming
    envelope, a version skew, a wrapper like with_raw_response.create(). Such a
    call carries no trustworthy finish_reason and no trustworthy output text, so
    every detector that keys on either must exclude it rather than read the
    fabricated defaults that used to be substituted.
    """
    return getattr(call, "instrumentation_degraded", None) is not None


# The shape a call takes when instrumentation is silently broken: nothing was
# read, but the call demonstrably happened (latency was measured). Kept here as
# one definition because InstrumentationDegradedDetector and the vendor-side
# fleet query (services/api/api_svc/instrumentation_health.py) must agree on it
# — they are the in-run and cross-run views of the same condition.
def _matches_degraded_fingerprint(call: "LlmCall") -> bool:
    return (
        (call.output_length or 0) == 0
        and (call.completion_tokens or 0) == 0
        and (call.latency_ms or 0) > 0
        and (call.finish_reason is None or call.finish_reason == "stop")
    )


# ── Base ──────────────────────────────────────────────────────────────────────

# Third-party detector classes register here automatically (see
# BaseDetector.__init_subclass__ below) — the plugin surface detector_svc's
# custom_python_detectors.py loads from disk and merges into get_detectors().
# Keyed by class name; a name collision is last-write-wins (matches how
# accidentally defining two classes with the same name in one module would
# already behave — no additional surprise introduced here).
CUSTOM_DETECTOR_REGISTRY: dict[str, type["BaseDetector"]] = {}


class BaseDetector:
    name: str = "base"

    # None = detector computes its own severity (static or dynamic, per subclass).
    # Set to a Severity to force every signal this detector emits to that level —
    # tunable the same way as any other UPPERCASE attribute, including from
    # detectors.yml (see detector_svc/config_loader.py).
    SEVERITY: Optional[Severity] = None

    # Soft performance budget in nanoseconds. run_detectors() logs a warning (does
    # not raise, does not drop the signal) when a single on_run_completion() call
    # exceeds this. Default matches this module's design goal of <1ms/check.
    MAX_COST_NS: int = 1_000_000

    # Metadata for third-party subclasses only (built-ins defined in this module
    # don't need these — they're wired into detector_svc's own _DETECTOR_CLASSES
    # dict directly, and their shadow/live status is decided by the curated
    # LIVE_DETECTORS allowlist, not by a per-class default). CATEGORY groups
    # plugin detectors for reporting/UI purposes, distinct from a built-in
    # detector's identity. SHADOW_BY_DEFAULT mirrors JSON-config custom
    # detectors always starting in shadow mode — a plugin author can set this
    # False if they've already validated precision and want to go live
    # immediately, though the safe default is True.
    CATEGORY: str = "custom"
    SHADOW_BY_DEFAULT: bool = True

    # None = built-in Tier 1 detector, always runs for every org (every
    # detector in this module leaves this at the default). A non-None value
    # (e.g. "voice") marks this as belonging to a first-party detector pack
    # (packages/sdk-py/dunetrace/packs/) — detector_svc only evaluates it for
    # an org that has activated that pack (see detector_svc/packs.py). This is
    # a distinct concern from CUSTOM_DETECTOR_REGISTRY/SHADOW_BY_DEFAULT above:
    # packs are Dunetrace-owned feature modules a customer opts into wholesale,
    # not user-defined detector logic dropped in via a plugin file.
    pack: Optional[str] = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Only third-party subclasses register — built-in Tier 1 detectors are
        # defined in this same module and are wired into detector_svc's
        # _DETECTOR_CLASSES dict explicitly, not via this registry. Checking
        # __module__ (rather than requiring plugin authors to opt in with a
        # flag) means zero extra ceremony for the common case: subclass
        # BaseDetector anywhere outside dunetrace.detectors and it just works.
        #
        # Pack detectors (cls.pack is not None) are the one deliberate
        # exception: they live outside this module too (packages/sdk-py/
        # dunetrace/packs/voice.py etc.), which would otherwise also match
        # this condition. A pack detector registers explicitly via
        # register_pack() (see dunetrace/packs/base.py) instead, and must
        # NOT also land in CUSTOM_DETECTOR_REGISTRY — that registry's
        # detectors run unconditionally for every org
        # (detector_svc/detectors.py::_build_plugin_detectors), which would
        # silently defeat per-org pack activation for exactly the classes
        # that most need it gated. Caught via a real failing test
        # (test_tenant_isolation_org_a_activation_does_not_affect_org_b)
        # where a fake pack detector leaked into every org through this path
        # before this check existed.
        if cls.__module__ != BaseDetector.__module__ and cls.pack is None:
            CUSTOM_DETECTOR_REGISTRY[cls.__name__] = cls

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

    def on_event(self, event: AgentEvent, state: RunState) -> Optional[FailureSignal]:
        """Called incrementally as each event arrives during a run. Default no-op.

        Not currently invoked by any built-in call site — both the server-side
        detector worker and the SDK's OTel path call on_run_completion() once, at
        run end. This is an extension point for future streaming/incremental
        detectors; override it only if a detector genuinely needs to react
        mid-run rather than to the accumulated RunState.
        """
        return None

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        """Called once when a run completes, with the full accumulated RunState.

        This is what every Tier 1 detector implements. Must return either None or
        a FailureSignal with a populated `evidence: dict` — evidence is what
        the native root-cause LLM analysis reads to explain *why* the signal
        fired. run_detectors() logs a warning if evidence isn't a dict, but does
        not drop the signal.
        """
        raise NotImplementedError


# ── OVERSIZED_TOOL_ARGUMENTS ──────────────────────────────────────────────────


class OversizedToolArgumentsDetector(BaseDetector):
    """
    An agent that stuffs a very large payload into a tool call — dumping an entire
    document or conversation into a single `args` string — is a common cost and
    latency footgun, and often a sign the agent is using a tool as a scratchpad
    instead of reasoning.

    Tunable: MAX_ARG_LENGTH (default 10_000) — fire if any tool call's args exceeds
    this character limit.
    """

    name = "OVERSIZED_TOOL_ARGUMENTS"
    SEVERITY = Severity.MEDIUM
    MAX_ARG_LENGTH = 10_000

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        for tc in state.tool_calls:
            # args_length is the pre-truncation length, set by transports that
            # cap the stored string. The OTLP path truncates at 8192 chars —
            # below this detector's default threshold — so measuring len(args)
            # alone would make this detector unreachable for OTel-ingested runs
            # and would understate the payload wherever it did fire.
            arg_length = tc.args_length if tc.args_length is not None else len(tc.args or "")
            if arg_length > self.MAX_ARG_LENGTH:
                return FailureSignal(
                    failure_type=FailureType.OVERSIZED_TOOL_ARGUMENTS,
                    severity=self.SEVERITY,
                    run_id=state.run_id,
                    agent_id=state.agent_id,
                    agent_version=state.agent_version,
                    step_index=tc.step_index,
                    confidence=0.9,
                    evidence={
                        "step_index": tc.step_index,
                        "tool_name": tc.tool_name,
                        "arg_length": arg_length,
                        "threshold": self.MAX_ARG_LENGTH,
                    },
                )
        return None


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
    SEVERITY = Severity.HIGH
    WINDOW = 5
    THRESHOLD = 3

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        if len(state.tool_calls) < self.WINDOW:
            return None

        window = state.tool_calls[-self.WINDOW :]
        counts = Counter(c.tool_name for c in window)

        for tool, count in counts.items():
            if count >= self.THRESHOLD:
                all_calls = [c for c in state.tool_calls if c.tool_name == tool]
                args_list = [c.args for c in all_calls]
                unique_args = len(set(args_list))
                calls_with_result = [c for c in all_calls if c.success is not None]
                success_rate = (
                    sum(1 for c in calls_with_result if c.success) / len(calls_with_result)
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
                    severity=self.SEVERITY,
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
                        "args": args_list,
                        "args_identical": unique_args == 1,
                        "args_similar": unique_args <= 2,
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
    SEVERITY = Severity.HIGH
    WINDOW = 6

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
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
                    severity=self.SEVERITY,
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
    SEVERITY = Severity.MEDIUM
    MIN_LLM_CALLS = 2

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
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
            severity=self.SEVERITY,
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
    SEVERITY = Severity.MEDIUM
    STALL_STEPS = 4

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
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
                severity=self.SEVERITY,
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
    Extend the signature set by adding entries to _INJECTION_PATTERNS_COMPILED.

    Unlike every other detector in this module, this one runs **in-path** — the
    SDK calls it inside ``dt.run()`` before the agent does any work (see
    client.py), because the signal has to exist by the time ``run.started`` is
    emitted. That makes its cost the caller's latency, so the scan is bounded:
    see SCAN_HEAD_CHARS/SCAN_TAIL_CHARS.

    SCAN_HEAD_CHARS / SCAN_TAIL_CHARS (default 16384 each) set how much text is
    scanned from each end; an input at or below their sum is scanned whole, with
    no gap. The 18 patterns cost roughly 0.3µs per character scanned, so the
    defaults cap the scan at ~10ms — against an LLM call of 500ms or more, i.e.
    ~2%. Lower them if you run latency-critical agents (voice, realtime) on large
    inputs and would rather trade edge coverage for a tighter run-start; raise
    them if inputs routinely exceed 32K characters and you want full coverage.

    Unlike every other detector's tunables these are *not* settable from
    detectors.yml — that file configures the detector worker, and this scan runs
    in the SDK, in the customer's process, which never reads it. Set them on this
    class (or a subclass) instead.
    """

    name = "PROMPT_INJECTION_SIGNAL"
    SEVERITY = Severity.CRITICAL

    # Bound the in-path scan. Cost is linear in characters scanned (~0.3µs/char
    # across the 18 patterns), so an unbounded scan makes dt.run()'s latency a
    # function of input size — a 1 MB input (a document, a stuffed RAG context)
    # cost ~340ms of synchronous, blocking time before the agent ran at all.
    #
    # The sizing goal is coverage, not a fixed microsecond budget: the cost that
    # matters is the fraction of a real agent step, and an LLM call is 500ms+.
    # 16K+16K caps the scan at ~10ms (~2% of one LLM call) while scanning any
    # input up to 32K characters *in full* — which is most of them — so the
    # head/tail split only degrades coverage for genuinely large inputs.
    #
    # Beyond that, head and tail are scanned because injections cluster at the
    # edges: a prefix override ("ignore all previous instructions...") or a
    # payload appended after legitimate content. Text between the two windows is
    # not scanned. This is a detection *signal*, not a security boundary.
    SCAN_HEAD_CHARS: int = 16_384
    SCAN_TAIL_CHARS: int = 16_384

    def _scan_windows(self, input_text: str) -> tuple:
        """Return the slices of input_text to scan, and whether it was truncated.

        Windows are scanned as separate strings rather than concatenated, so no
        pattern can match across the join between head and tail and produce a
        match that isn't in the real input.
        """
        head_n, tail_n = self.SCAN_HEAD_CHARS, self.SCAN_TAIL_CHARS
        if len(input_text) <= head_n + tail_n:
            return (input_text,), False
        return (input_text[:head_n], input_text[-tail_n:]), True

    def check_input(self, input_text: str, state: RunState) -> Optional[FailureSignal]:
        windows, truncated = self._scan_windows(input_text)
        matched = [
            label
            for label, pattern in _INJECTION_PATTERNS_COMPILED
            if any(pattern.search(w) for w in windows)
        ]
        if not matched:
            return None

        evidence = {
            "matched_pattern_count": len(matched),
            "matched_patterns": matched[:5],
            "input_length": len(input_text),
        }
        if truncated:
            # Tells a reader of this signal that absence of a pattern is not
            # proof of absence in the full input.
            evidence["scanned_chars"] = self.SCAN_HEAD_CHARS + self.SCAN_TAIL_CHARS
            evidence["scan_truncated"] = True

        return FailureSignal(
            failure_type=FailureType.PROMPT_INJECTION_SIGNAL,
            severity=self.SEVERITY,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=0,
            confidence=_scale_confidence(len(matched)),
            evidence=evidence,
        )

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
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
    SEVERITY = Severity.MEDIUM
    MIN_SCORE = 0.3
    MIN_RESULTS = 1

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
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
            severity=self.SEVERITY,
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


# ── EXCESSIVE_RETRIEVAL ───────────────────────────────────────────────────────


class ExcessiveRetrievalDetector(BaseDetector):
    """A run issues an unusually high volume of retrieval calls.

    This is a volume signal, not a duplication signal: it fires on the total
    number of retrievals in a run regardless of whether the queries repeat.
    High retrieval volume is itself a cost/efficiency concern — an agent that
    needs many searches to answer is often failing to ground efficiently.

    Tunable: MAX_RETRIEVALS (default 8) — fire at or above this many retrievals.
    """

    name = "EXCESSIVE_RETRIEVAL"
    SEVERITY = Severity.MEDIUM
    MAX_RETRIEVALS = 8

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        retrieval_count = len(state.retrievals)
        if retrieval_count < self.MAX_RETRIEVALS:
            return None

        first_step = min(retrieval.step_index for retrieval in state.retrievals)
        last_step = max(retrieval.step_index for retrieval in state.retrievals)
        # Scale confidence with how far the run runs past the threshold, so a
        # 40-retrieval run ranks above one that just tips over the line.
        confidence = min(0.95, 0.8 + 0.01 * (retrieval_count - self.MAX_RETRIEVALS))
        return FailureSignal(
            failure_type=FailureType.EXCESSIVE_RETRIEVAL,
            severity=self.SEVERITY,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=last_step,
            confidence=confidence,
            evidence={
                "retrieval_count": retrieval_count,
                "threshold": self.MAX_RETRIEVALS,
                "indexes": sorted({retrieval.index_name for retrieval in state.retrievals}),
                "first_step": first_step,
                "last_step": last_step,
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
    SEVERITY = Severity.HIGH
    THRESHOLD = 2

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        if len(state.llm_calls) < self.THRESHOLD:
            return None

        # None (unreadable response) is deliberately not "length": we cannot
        # claim a truncation we never observed. This detector keys solely on
        # finish_reason with no corroborating evidence, so an unreadable call
        # contributes nothing here and is reported by
        # INSTRUMENTATION_DEGRADED instead.
        truncated = [
            c for c in state.llm_calls if c.finish_reason == "length" and not _is_unmeasurable(c)
        ]

        if len(truncated) < self.THRESHOLD:
            return None

        return FailureSignal(
            failure_type=FailureType.LLM_TRUNCATION_LOOP,
            severity=self.SEVERITY,
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


# ── SILENT_TRUNCATION ─────────────────────────────────────────────────────────


class SilentTruncationDetector(BaseDetector):
    """
    A single LLM response was truncated (finish_reason "length" for OpenAI,
    "max_tokens" for Anthropic) and the agent proceeded without recovering — no
    retry, no continuation call. One truncation the agent silently builds on:
    partial JSON, a cut-off plan, half a code block, passed downstream as if it
    were complete.

    Distinct from LLM_TRUNCATION_LOOP, which fires on REPEATED truncation
    (THRESHOLD or more). This detector owns the single-occurrence case and steps
    aside at or above LOOP_THRESHOLD so the two never double-fire on one run.

    Fires HIGH when the truncated response was the run's final output (nothing
    ran after it to catch the problem), MEDIUM when the agent did more work
    afterward but never retried the truncated call.

    "Recovered" is defined narrowly and cheaply: any LLM call after the truncated
    one is treated as a retry/continuation (the agent noticed and re-asked), so
    the detector does not fire. Absence of any later LLM call means the truncated
    output was consumed as-is.

    Tunable:
      MIN_OUTPUT_LENGTH — ignore truncations whose output is shorter than this
        (default 1). A zero-length "truncated" response is EMPTY_LLM_RESPONSE's
        concern, not this detector's.
      LOOP_THRESHOLD — the truncation count at/above which LLM_TRUNCATION_LOOP
        takes over (default 2, kept in sync with that detector's THRESHOLD).
    """

    name = "SILENT_TRUNCATION"
    SEVERITY = None  # computed per-signal: MEDIUM or HIGH
    MAX_COST_NS = 50_000

    TRUNCATION_REASONS = ("length", "max_tokens")
    MIN_OUTPUT_LENGTH = 1
    LOOP_THRESHOLD = 2

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        llm_calls = state.llm_calls
        if not llm_calls:
            return None

        # `None in TRUNCATION_REASONS` is False, which is the correct branch —
        # made explicit here so it reads as a decision rather than an accident.
        # Same reasoning as LLM_TRUNCATION_LOOP: finish_reason is this
        # detector's only evidence, so an unreadable one is not evidence.
        all_truncated = [
            c
            for c in llm_calls
            if c.finish_reason in self.TRUNCATION_REASONS and not _is_unmeasurable(c)
        ]
        if not all_truncated:
            return None
        # Repeated truncation is LLM_TRUNCATION_LOOP's job — don't double-fire.
        if len(all_truncated) >= self.LOOP_THRESHOLD:
            return None

        t = all_truncated[0]
        # An empty "truncated" response is a different failure (EMPTY_LLM_RESPONSE).
        if t.output_length is not None and t.output_length < self.MIN_OUTPUT_LENGTH:
            return None
        # A later LLM call means the agent re-asked (retry/continuation) — recovered.
        if any(c.step_index > t.step_index for c in llm_calls):
            return None

        later_tool_steps = [
            tc.step_index for tc in state.tool_calls if tc.step_index > t.step_index
        ]
        is_final = not later_tool_steps  # nothing ran after the truncated output

        return FailureSignal(
            failure_type=FailureType.SILENT_TRUNCATION,
            severity=Severity.HIGH if is_final else Severity.MEDIUM,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=t.step_index,
            confidence=0.85 if is_final else 0.7,
            evidence={
                "truncated_step": t.step_index,
                "finish_reason": t.finish_reason,
                "output_length": t.output_length,
                "model": t.model,
                "recovered": False,
                "was_final_output": is_final,
                "subsequent_tool_steps": later_tool_steps,
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
    SEVERITY = Severity.MEDIUM
    MIN_CALLS = 3
    GROWTH_FACTOR = 3.0
    MIN_LAST_TOKENS = 2000
    INFLATION_FACTOR = 2.0  # multiplier over P75 baseline when history is available

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        calls_with_tokens = [
            c for c in state.llm_calls if c.prompt_tokens is not None and c.prompt_tokens > 0
        ]

        if len(calls_with_tokens) < self.MIN_CALLS:
            return None

        first_tokens: int = calls_with_tokens[0].prompt_tokens or 0
        last_tokens: int = calls_with_tokens[-1].prompt_tokens or 0

        if first_tokens < 10:
            return None

        if last_tokens < self.MIN_LAST_TOKENS:
            return None

        growth = last_tokens / first_tokens

        if state.baseline_p75_token_growth is not None:
            effective_threshold = state.baseline_p75_token_growth * self.INFLATION_FACTOR
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
                {"step": c.step_index, "tokens": c.prompt_tokens} for c in calls_with_tokens
            ],
        }
        if state.baseline_p75_token_growth is not None:
            evidence["baseline_p75"] = round(state.baseline_p75_token_growth, 2)

        return FailureSignal(
            failure_type=FailureType.CONTEXT_BLOAT,
            severity=self.SEVERITY,
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
    MIN_THRESHOLD_MS (default 500) — floor on the baseline-derived threshold. Without it, an
    agent whose historical calls are all near-instant (e.g. mocked tools in a demo agent) learns
    a P75 baseline of a few ms, and any real call — even a fast, unremarkable 300ms one — reads
    as a huge multiple of that baseline and fires. The floor keeps the threshold meaningful in
    absolute terms regardless of how degenerate the learned baseline is.
    """

    name = "SLOW_STEP"
    # Dynamic by default (see on_run_completion) — set SEVERITY explicitly to
    # force a fixed level regardless of ratio.
    SEVERITY = None

    THRESHOLDS = [
        ("tool.called", 15_000, "tool execution"),
        ("llm.called", 30_000, "LLM call"),
        ("", 60_000, "step"),
    ]
    INFLATION_FACTOR = 2.0  # multiplier over P75 baseline when history is available
    MIN_THRESHOLD_MS = 500  # floor for baseline-derived thresholds — see class docstring

    def _threshold_for(self, event_type: str, state: Optional[RunState] = None, tool_name: Optional[str] = None) -> tuple[int, str]:
        for prefix, static_ms, label in self.THRESHOLDS:
            if not prefix or event_type.startswith(prefix):
                if state is not None:
                    if prefix == "tool.called":
                        # Check for per-tool baseline first
                        if tool_name and state.baseline_p75_latency_by_tool and tool_name in state.baseline_p75_latency_by_tool:
                            return (
                                max(1, int(state.baseline_p75_latency_by_tool[tool_name] * self.INFLATION_FACTOR)),
                                label,
                            )
                        # Fallback to global tool baseline
                        if state.baseline_p75_latency_tool is not None:
                            return (
                                int(state.baseline_p75_latency_tool * self.INFLATION_FACTOR),
                                label,
                            )
                    if prefix == "llm.called" and state.baseline_p75_latency_llm is not None:
                        return (
                            max(
                                int(state.baseline_p75_latency_llm * self.INFLATION_FACTOR),
                                self.MIN_THRESHOLD_MS,
                            ),
                            label,
                        )
                return static_ms, label
        return 60_000, "step"

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
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
        agent_events = [e for e in state.events if e.event_type is not EventType.EXTERNAL_SIGNAL]
        # First-write-wins: the initiating event (tool.called, llm.called) at each
        # step determines the threshold, not the responding or completing event.
        step_event_type: dict[int, str] = {}
        step_timestamp: dict[int, float] = {}
        step_tool_name: dict[int, str] = {}
        for e in agent_events:
            if e.step_index not in step_event_type:
                step_event_type[e.step_index] = e.event_type.value
                step_timestamp[e.step_index] = e.timestamp
                if e.event_type.value == "tool.called":
                    step_tool_name[e.step_index] = e.payload.get("tool_name", "")

        for step_idx, duration_ms in state.step_durations_ms.items():
            event_type = step_event_type.get(step_idx, "")
            tool_name = step_tool_name.get(step_idx)
            threshold_ms, label = self._threshold_for(event_type, state, tool_name)

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
        severity = (
            self.SEVERITY
            if self.SEVERITY is not None
            else (Severity.HIGH if ratio >= 5 else Severity.MEDIUM)
        )

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
                if v > self._threshold_for(step_event_type.get(k, ""), state, step_tool_name.get(k))[0]
            },
        }

        # Include the raw P75 baseline so dashboards can show what normal looks like.
        if worst_event_type.startswith("tool.called"):
            worst_tool_name = step_tool_name.get(worst_step_idx)
            if worst_tool_name and state.baseline_p75_latency_by_tool and worst_tool_name in state.baseline_p75_latency_by_tool:
                evidence["baseline_p75"] = round(state.baseline_p75_latency_by_tool[worst_tool_name], 1)
            elif state.baseline_p75_latency_tool is not None:
                evidence["baseline_p75"] = round(state.baseline_p75_latency_tool, 1)
        elif (
            worst_event_type.startswith("llm.called") and state.baseline_p75_latency_llm is not None
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
    Unlike TOOL_LOOP, args may differ — the agent is genuinely retrying — but the tool
    keeps failing. Indicates a broken dependency (API down, rejecting every request) that
    the agent can't detect and back off from. HIGH severity — each failure burns an LLM turn
    to re-plan, and the agent will almost always exhaust max_iterations with nothing to show.

    Evidence: args_identical (True if no variation in args), reason_identical (True if the same
    error every time), failure_reason (common error text when reason_identical).

    Tunable: THRESHOLD (default 3). Lower to catch dependency failures faster; raise for agents
    with built-in retry logic where 2 failures before escalation are expected.
    """

    name = "RETRY_STORM"
    SEVERITY = Severity.HIGH
    THRESHOLD = 3

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
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
        args_list = [tc.args for tc in best_streak]
        errors = [tc.error for tc in best_streak]

        # Self-correction check: if the tool subsequently succeeded after the streak,
        # the agent recovered and this is CoT/retry behaviour, not a storm.
        last_fail_step = best_streak[-1].step_index
        recovered = any(
            tc.tool_name == best_tool and tc.success is True and tc.step_index > last_fail_step
            for tc in state.tool_calls
        )
        if recovered:
            return None

        args_identical = len(set(args_list)) == 1
        all_have_reason = all(e is not None for e in errors)
        reason_identical = all_have_reason and len(set(errors)) == 1
        failure_reason = errors[0] if reason_identical else None

        return FailureSignal(
            failure_type=FailureType.RETRY_STORM,
            severity=self.SEVERITY,
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
                "errors": errors,
                "failure_reason": failure_reason,
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
    SEVERITY = Severity.HIGH

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        # `finish_reason is None` means the response was never read (see
        # _is_unmeasurable) and must NOT be treated as a "stop". This is the
        # exact false positive that motivated provenance: a LangGraph agent
        # whose with_raw_response.create() returned a LegacyAPIResponse produced
        # a fabricated ("", "stop") pair on 100% of runs, which is byte-for-byte
        # this detector's trigger. The `== "stop"` comparison already excludes
        # None; the explicit degraded check below covers the partial case where
        # finish_reason was readable but the content was not.
        empty = [
            c
            for c in state.llm_calls
            if c.finish_reason == "stop"
            and getattr(c, "output_length", None) == 0
            and not _is_unmeasurable(c)
        ]
        if not empty:
            return None

        first = empty[0]
        return FailureSignal(
            failure_type=FailureType.EMPTY_LLM_RESPONSE,
            severity=self.SEVERITY,
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
    — needs at least 20 historical runs to be meaningful.

    Tunable: INFLATION_FACTOR (default 2.0). Lower to catch moderate inflation earlier;
    raise for research agents with high step variance (2.5–3.0) or lower for coding agents
    with tight, predictable step counts (1.5).
    """

    name = "STEP_COUNT_INFLATION"
    SEVERITY = Severity.MEDIUM
    INFLATION_FACTOR = 2.0

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        if state.baseline_p75_steps is None:
            return None

        if state.current_step <= state.baseline_p75_steps * self.INFLATION_FACTOR:
            return None

        ratio = state.current_step / state.baseline_p75_steps
        # confidence anchored to how far above the effective threshold (baseline × factor)
        effective_threshold = state.baseline_p75_steps * self.INFLATION_FACTOR

        return FailureSignal(
            failure_type=FailureType.STEP_COUNT_INFLATION,
            severity=self.SEVERITY,
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
    SEVERITY = Severity.HIGH
    THRESHOLD = 3

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
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
            severity=self.SEVERITY,
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
    SEVERITY = Severity.MEDIUM
    MAX_STEP = 2

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        if state.exit_reason == "error" and state.current_step <= self.MAX_STEP:
            return FailureSignal(
                failure_type=FailureType.FIRST_STEP_FAILURE,
                severity=self.SEVERITY,
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

        # Same ("", "stop") fabrication risk as EMPTY_LLM_RESPONSE, and the
        # same resolution: an unreadable response is not an empty one.
        early_empty = [
            c
            for c in state.llm_calls
            if c.step_index <= self.MAX_STEP
            and getattr(c, "output_length", None) == 0
            and c.finish_reason == "stop"
            and not _is_unmeasurable(c)
        ]
        if early_empty:
            return FailureSignal(
                failure_type=FailureType.FIRST_STEP_FAILURE,
                severity=self.SEVERITY,
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
            tc for tc in state.tool_calls if tc.step_index <= self.MAX_STEP and tc.success is False
        ]
        if early_fail:
            return FailureSignal(
                failure_type=FailureType.FIRST_STEP_FAILURE,
                severity=self.SEVERITY,
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
    MIN_THRESHOLD_RATIO (default 2.0) — floor on the baseline-derived threshold. Without it, an
    agent whose historical runs are consistently tool-heavy (e.g. baseline ratio 0.1) learns an
    effective threshold of 0.2 after INFLATION_FACTOR — so almost any run with even one or two
    LLM calls per tool call reads as a huge multiple of that baseline and fires, even though
    that ratio is unremarkable in absolute terms. See SlowStepDetector.MIN_THRESHOLD_MS for the
    same rationale applied to a different detector.
    """

    name = "REASONING_STALL"
    # Dynamic by default (see on_run_completion) — set SEVERITY explicitly to
    # force a fixed level regardless of exit_reason.
    SEVERITY = None
    MIN_LLM_CALLS = 5
    RATIO_THRESHOLD = 4.0
    INFLATION_FACTOR = 2.0  # multiplier over P75 baseline when history is available
    MIN_THRESHOLD_RATIO = 2.0  # floor for baseline-derived threshold — see class docstring

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        # Skip errored runs — FIRST_STEP_FAILURE and RETRY_STORM cover those.
        if state.exit_reason == "error":
            return None

        llm_count = len(state.llm_calls)
        tool_count = len(state.tool_calls)

        if llm_count < self.MIN_LLM_CALLS:
            return None

        ratio = llm_count / max(tool_count, 1)

        if state.baseline_p75_llm_tool_ratio is not None:
            effective_threshold = max(
                state.baseline_p75_llm_tool_ratio * self.INFLATION_FACTOR,
                self.MIN_THRESHOLD_RATIO,
            )
        else:
            effective_threshold = self.RATIO_THRESHOLD

        if ratio < effective_threshold:
            return None

        # A run that ended with a final answer is inefficient (MEDIUM).
        # A run that stalled without converging shows the ratio caused failure (HIGH).
        severity = (
            self.SEVERITY
            if self.SEVERITY is not None
            else (Severity.MEDIUM if state.exit_reason == "final_answer" else Severity.HIGH)
        )

        action_events = [
            e
            for e in state.events
            if e.event_type.value.startswith("llm.called")
            or e.event_type.value.startswith("tool.called")
        ]
        event_sequence = [
            "llm" if e.event_type.value.startswith("llm.") else "tool" for e in action_events
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
    MIN_THRESHOLD_TOKENS (default 2000) — floor on the baseline-derived threshold. Without it,
    an agent whose historical runs are all tiny (e.g. short mocked/demo runs) learns a
    baseline of a few hundred tokens, and any real run — even one using an unremarkable few
    thousand tokens — reads as a huge multiple of that baseline and fires. Matches
    ContextBloatDetector.MIN_LAST_TOKENS in both value and rationale.
    """

    name = "COST_SPIKE"
    SEVERITY = Severity.MEDIUM
    INFLATION_FACTOR = 3.0
    STATIC_THRESHOLD_TOKENS = 50_000
    MIN_LLM_CALLS = 1
    MIN_THRESHOLD_TOKENS = 2000  # floor for baseline-derived threshold — see class docstring

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        if len(state.llm_calls) < self.MIN_LLM_CALLS:
            return None

        total_tokens = sum(
            (c.prompt_tokens or 0) + (c.completion_tokens or 0) + (c.reasoning_tokens or 0)
            for c in state.llm_calls
        )
        if total_tokens == 0:
            return None

        if state.baseline_p75_total_tokens is not None:
            threshold = max(
                state.baseline_p75_total_tokens * self.INFLATION_FACTOR,
                self.MIN_THRESHOLD_TOKENS,
            )
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
            severity=self.SEVERITY,
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
    MIN_THRESHOLD_S (default 5.0) — floor on the baseline-derived threshold. Without it, an
    agent whose historical runs are all near-instant (e.g. short degenerate runs, or events
    batched with near-identical timestamps) learns a P75 baseline of a fraction of a second,
    and any real run — even one taking well under a second — reads as a huge multiple of that
    baseline and fires with a nonsensical "0s" duration/threshold in the alert text.
    """

    name = "SESSION_LATENCY"
    SEVERITY = Severity.MEDIUM
    INFLATION_FACTOR = 3.0
    STATIC_THRESHOLD_SECS = 300
    MIN_EVENTS = 2
    MIN_THRESHOLD_S = 5.0  # floor for baseline-derived threshold — see class docstring

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        if len(state.events) < self.MIN_EVENTS:
            return None

        timestamps = [e.timestamp for e in state.events]
        duration_s = max(timestamps) - min(timestamps)

        if duration_s <= 0:
            return None

        if state.baseline_p75_duration_s is not None:
            threshold = max(
                state.baseline_p75_duration_s * self.INFLATION_FACTOR, self.MIN_THRESHOLD_S
            )
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
            severity=self.SEVERITY,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=state.current_step,
            confidence=_scale_confidence(ratio),
            evidence=evidence,
        )


# ── PREMATURE_TERMINATION ───────────────────────────────────────────────────────


class PrematureTerminationDetector(BaseDetector):
    """
    A tool call failed, then the agent's next LLM response claims success anyway,
    without acknowledging the failure. The flagship "silent degradation" detector —
    the agent didn't just fail to notice an error, it actively told the user the
    opposite of what happened.

    Data-availability note: `tool_responded`'s `success` (bool) and `error` (str,
    only ever populated when success=False — see run_context.py::tool_responded)
    is the primary "tool had a problem" signal. The wire format can also carry
    the tool's raw response body via `tool_responded(..., output=...)`, but only
    when the calling code passes it — most auto-instrumentation doesn't (see
    docs/detectors.md's Tool Argument Fabrication section for which integrations
    do). When that text is present, `ERROR_MARKERS` is also checked against it, so
    a tool that self-reports `success=True` but whose own body reads like an
    error (a real pattern: HTTP 200 with `{"error": "not found"}`) still counts as
    a problem. There is still no HTTP status code or tool schema (nullability)
    in the wire format either way.

    Fires once per run, on the first (failed tool call, completion claim) pair
    found, matching every other Tier 1 detector's one-signal-per-run contract.

    Severity: HIGH, or CRITICAL if the completion claim is also the last
    llm.responded event in the run (no further attempt to notice or fix it).

    Tunable: COMPLETION_TERMS, ERROR_ACKNOWLEDGMENT_TERMS, ERROR_MARKERS (word
    lists, case-insensitive substring match unless CASE_SENSITIVE=True).
    MIN_MESSAGE_LENGTH (default 20) — skip LLM outputs too short to carry
    real claim-of-success context (e.g. a bare "Done.").
    """

    name = "PREMATURE_TERMINATION"
    SEVERITY = None  # computed per-signal: HIGH or CRITICAL

    ERROR_MARKERS = [
        "error",
        "exception",
        "traceback",
        "failed",
        "not found",
        "unavailable",
        "timeout",
    ]
    COMPLETION_TERMS = [
        "scheduled",
        "booked",
        "completed",
        "successfully",
        "done",
        "sent",
        "created",
        "updated",
        "confirmed",
        "processed",
        "finished",
        "saved",
        "deleted",
        "resolved",
    ]
    ERROR_ACKNOWLEDGMENT_TERMS = [
        "however",
        "unable",
        "couldn't",
        "failed",
        "error",
        "issue",
        "problem",
        "unfortunately",
        "sorry",
    ]
    CASE_SENSITIVE = False
    MIN_MESSAGE_LENGTH = 20

    def _contains_any(self, text: str, terms: list) -> Optional[str]:
        haystack = text if self.CASE_SENSITIVE else text.lower()
        for term in terms:
            needle = term if self.CASE_SENSITIVE else term.lower()
            if needle in haystack:
                return term
        return None

    def _tool_problem_text(self, tc: ToolCall) -> Optional[str]:
        """Text to run ERROR_MARKERS against for this call, or None if it shows
        no sign of failure. success=False is the primary gate; a call that
        self-reports success but whose raw output body (when instrumented)
        contains an error marker is also treated as a problem."""
        if tc.success is False:
            return tc.error or ""
        if tc.output and self._contains_any(tc.output, self.ERROR_MARKERS):
            return tc.output
        return None

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        problem_calls = [
            (tc, text)
            for tc in state.tool_calls
            for text in [self._tool_problem_text(tc)]
            if text is not None
        ]
        if not problem_calls:
            return None

        # Absence of output text has three distinct causes and only one of them
        # is "the agent said nothing":
        #
        #   1. instrumentation_degraded present -> the response was never read.
        #      INSTRUMENTATION_DEGRADED owns this run; returning None here is
        #      correct but its silence is explained by that signal rather than
        #      being mistaken for a clean verdict.
        #   2. output_length > 0 with no text -> DUNETRACE_OMIT_LLM_OUTPUT_TEXT.
        #      A deliberate operator choice, not a fault. Nothing to report.
        #   3. genuinely empty responses -> EMPTY_LLM_RESPONSE's job.
        #
        # All three end in `return None` here, but they are not the same None,
        # and conflating them is what let a broken pipeline read as a healthy
        # agent for the entire duration of the incident this comment documents.
        llm_responses = [
            e
            for e in state.events
            if e.event_type == EventType.LLM_RESPONDED and e.payload.get("output")
        ]
        if not llm_responses:
            if any(_is_unmeasurable(c) for c in state.llm_calls):
                logger.debug(
                    "%s: no readable completion text on run %s — instrumentation "
                    "degraded, see INSTRUMENTATION_DEGRADED",
                    self.name,
                    state.run_id,
                )
            return None
        last_llm_step = max(e.step_index for e in llm_responses)

        for tc, problem_text in problem_calls:
            claim = next((e for e in llm_responses if e.step_index > tc.step_index), None)
            if claim is None:
                continue

            output = claim.payload["output"]
            if len(output) < self.MIN_MESSAGE_LENGTH:
                continue

            matched_completion = self._contains_any(output, self.COMPLETION_TERMS)
            if not matched_completion:
                continue

            matched_ack = self._contains_any(output, self.ERROR_ACKNOWLEDGMENT_TERMS)
            if matched_ack:
                continue

            matched_error_marker = self._contains_any(problem_text, self.ERROR_MARKERS)
            is_final = claim.step_index >= last_llm_step

            return FailureSignal(
                failure_type=FailureType.PREMATURE_TERMINATION,
                severity=Severity.CRITICAL if is_final else Severity.HIGH,
                run_id=state.run_id,
                agent_id=state.agent_id,
                agent_version=state.agent_version,
                step_index=claim.step_index,
                confidence=0.85 if is_final else 0.75,
                evidence={
                    "failed_tool": tc.tool_name,
                    "failed_tool_step": tc.step_index,
                    "tool_error": tc.error,
                    "matched_error_marker": matched_error_marker,
                    "failure_source": "declared" if tc.success is False else "output_text",
                    "claim_step": claim.step_index,
                    "matched_completion_term": matched_completion,
                    "is_final_message": is_final,
                    "output_snippet": output[:200],
                },
            )

        return None


# ── UNREAD_TOOL_ERROR ────────────────────────────────────────────────────────────


class UnreadToolErrorDetector(BaseDetector):
    """
    A tool call failed, and the agent's very next action either ignores it
    entirely (proceeds straight to another tool call) or responds without
    acknowledging anything went wrong. The leading indicator for
    PREMATURE_TERMINATION — this fires on absence of acknowledgment alone,
    with no requirement that the agent go on to positively claim success.
    A run that fires PREMATURE_TERMINATION usually fires this one too; a run
    can fire this one without ever firing PREMATURE_TERMINATION (the agent
    silently moves on without claiming anything either way).

    Same data-availability note as PrematureTerminationDetector: `success is
    False` is the primary "tool_result has error markers" gate. When the
    caller also passes raw output text to `tool_responded()`, ERROR_MARKERS is
    additionally checked against it, so a call that self-reports success but
    whose own body reads like an error also counts. Still no HTTP status or
    tool schema info in the wire format either way.

    Counts every failed tool call in the run whose next action (by step
    index) is either another tool call, or an LLM response containing no
    error-acknowledgment term. Fires once per run if that count is >= 1 (one
    signal, matching every other Tier 1 detector's contract) — MEDIUM by
    default, HIGH if the count is >= 2 (chained silent errors).

    Tunable: ERROR_ACKNOWLEDGMENT_TERMS (shared word list with
    PrematureTerminationDetector — override both together if you tune one).
    """

    name = "UNREAD_TOOL_ERROR"
    SEVERITY = None  # computed per-signal: MEDIUM or HIGH

    ERROR_MARKERS = [
        "error",
        "exception",
        "traceback",
        "failed",
        "not found",
        "unavailable",
        "timeout",
    ]
    ERROR_ACKNOWLEDGMENT_TERMS = [
        "however",
        "unable",
        "couldn't",
        "failed",
        "error",
        "issue",
        "problem",
        "unfortunately",
        "sorry",
    ]
    CASE_SENSITIVE = False

    def _contains_any(self, text: str, terms: list) -> Optional[str]:
        haystack = text if self.CASE_SENSITIVE else text.lower()
        for term in terms:
            needle = term if self.CASE_SENSITIVE else term.lower()
            if needle in haystack:
                return term
        return None

    def _tool_problem_text(self, tc: ToolCall) -> Optional[str]:
        """Text to run ERROR_MARKERS against for this call, or None if it shows
        no sign of failure. success=False is the primary gate; a call that
        self-reports success but whose raw output body (when instrumented)
        contains an error marker is also treated as a problem."""
        if tc.success is False:
            return tc.error or ""
        if tc.output and self._contains_any(tc.output, self.ERROR_MARKERS):
            return tc.output
        return None

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        problem_calls = [
            (tc, text)
            for tc in state.tool_calls
            for text in [self._tool_problem_text(tc)]
            if text is not None
        ]
        if not problem_calls:
            return None

        next_actionable = [
            e
            for e in state.events
            if e.event_type in (EventType.TOOL_CALLED, EventType.LLM_RESPONDED)
        ]

        unread: list = []
        for tc, problem_text in problem_calls:
            nxt = next((e for e in next_actionable if e.step_index > tc.step_index), None)
            if nxt is None:
                continue  # run ended right after the failure — no next action to judge

            if nxt.event_type == EventType.TOOL_CALLED:
                unread.append((tc, nxt, "tool_call", problem_text))
                continue

            output = nxt.payload.get("output", "")
            if self._contains_any(output, self.ERROR_ACKNOWLEDGMENT_TERMS):
                continue  # agent addressed it — not unread

            unread.append((tc, nxt, "llm_response", problem_text))

        if not unread:
            return None

        tc, nxt, next_action_type, problem_text = unread[0]
        severity = Severity.HIGH if len(unread) >= 2 else Severity.MEDIUM
        matched_error_marker = self._contains_any(problem_text, self.ERROR_MARKERS)

        return FailureSignal(
            failure_type=FailureType.UNREAD_TOOL_ERROR,
            severity=severity,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=nxt.step_index,
            confidence=0.80 if len(unread) >= 2 else 0.65,
            evidence={
                "failed_tool": tc.tool_name,
                "failed_tool_step": tc.step_index,
                "tool_error": tc.error,
                "matched_error_marker": matched_error_marker,
                "failure_source": "declared" if tc.success is False else "output_text",
                "next_action_step": nxt.step_index,
                "next_action_type": next_action_type,
                "unread_count": len(unread),
            },
        )


# ── TOOL_ARGUMENT_FABRICATION ─────────────────────────────────────────────────


class ToolArgumentFabricationDetector(BaseDetector):
    """
    A tool call's arguments reference a specific entity (a UUID, an email
    address, a file path, an integer ID, a quoted string) that never appears
    anywhere the agent could plausibly have gotten it from: the user's input,
    the system prompt, or the raw output of any tool call earlier in the run.
    Provenance-only — this does NOT check whether the value is *correct*, only
    whether the agent could have had it from context. A correct-but-ungrounded
    guess still fires; the point is catching invention, not verification.

    Data-availability note: system_prompt and a tool call's raw output text are
    both optional, instrumentation-dependent fields (see
    dunetrace.client.Dunetrace.run's system_prompt param and
    run_context.py::tool_responded's output param) — most auto-instrumentation
    doesn't populate them yet (dt.tool() and the CrewAI integration do; the
    generic httpx/requests patches deliberately don't, since reading a
    response body there risks consuming a stream the caller still needs).
    Consequences:

    - Missing system_prompt: an entity sourced only from the system prompt
      (e.g. a fixed account ID baked into the agent's instructions) can read
      as fabricated. Not fully compensable — documented, not solved.
    - Missing tool output for an EARLIER call in the run: from that point on,
      an entity could legitimately have come from that invisible result, so
      this detector stops evaluating further calls in the run entirely rather
      than risk false-firing on the single most common real pattern (chaining
      an ID from one tool's result into the next tool's arguments).

    Small integers (1-100) are excluded from extraction outright — they
    recur constantly by coincidence (page numbers, counts, retries) and a
    substring check against them is too easy to satisfy by accident either
    way. Common recurring words (user, admin, dates, etc.) are allowlisted.

    Args parsing: `tc.args` is a string (the wire format never carries
    structured args — see ToolCall.args), but in practice it's almost always
    `str(a_dict)` (this SDK's own tool_called()) or `JSON.stringify(args)`
    (the TS SDK) — both parseable. Entities are extracted from parsed dict/
    list VALUES only, never keys, and only string values with no whitespace
    are treated as identifier candidates outright (a free-text value like a
    search query is exactly as "unverifiable" as an identifier under a naive
    reading of the spec, but flagging every query/message argument would
    swamp real signal — see the false-positive-averse mandate this detector
    was built under). When args_text isn't parseable as either format, this
    falls back to scanning the raw text directly (quoted substrings, still
    filtered to no-whitespace tokens) — a real but rare degradation, since an
    unparsed fallback can't distinguish a dict key from a value.

    Fires once per run, on the first tool call with a fabricated entity,
    matching every other Tier 1 detector's one-signal-per-run contract.

    Severity: HIGH by default, CRITICAL if the tool name matches a
    destructive-sounding pattern (delete_*, remove_*, drop_*, transfer_*,
    send_*, pay_*) — a fabricated argument to one of these is a materially
    bigger blast radius than to a read-only lookup.

    Tunable: ALLOWLIST, DESTRUCTIVE_TOOL_PATTERNS, SMALL_INT_MIN/MAX,
    CASE_SENSITIVE.
    """

    name = "TOOL_ARGUMENT_FABRICATION"
    SEVERITY = None  # computed per-signal: HIGH or CRITICAL

    UUID_RE = re.compile(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    )
    EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
    URL_RE = re.compile(r"\bhttps?://[^\s\"'<>]+")
    FILE_PATH_RE = re.compile(r"\b[\w.\-]+(?:/[\w.\-]+)+\b")
    QUOTED_STRING_RE = re.compile(r"\"([^\"]{3,})\"|'([^']{3,})'")
    INTEGER_ID_RE = re.compile(r"\b\d+\b")

    ALLOWLIST = [
        "user",
        "admin",
        "system",
        "root",
        "guest",
        "test",
        "hello",
        "world",
        "today",
        "tomorrow",
        "yesterday",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]
    DESTRUCTIVE_TOOL_PATTERNS = [
        "delete_",
        "remove_",
        "drop_",
        "transfer_",
        "send_",
        "pay_",
    ]
    SMALL_INT_MIN = 1
    SMALL_INT_MAX = 100
    CASE_SENSITIVE = False
    # audit Finding 21: characters that make a token "identifier-shaped".
    ID_SEPARATORS = "_-./:@#"
    # A pure-alphabetic value with no separator/digit is only treated as an
    # identifier when it's at least this long (an opaque token). Below it, short
    # unstructured alpha values — airport codes ("CDG"), category names, rephrased
    # query terms — are legitimately agent-generated and must NOT be flagged.
    MIN_OPAQUE_LEN = 12

    def _try_parse_args(self, args_text: str):
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(args_text)
            except Exception:
                continue
        return None

    def _collect_leaf_texts(self, value) -> List[Tuple[str, bool]]:
        """Walk a parsed args structure to its leaves. Returns (text,
        is_string_value) pairs — is_string_value gates the whole-value
        identifier check; numeric leaves go through INTEGER_ID_RE's
        small-int-aware path instead of bypassing it."""
        leaves: List[Tuple[str, bool]] = []
        if isinstance(value, dict):
            for v in value.values():
                leaves += self._collect_leaf_texts(v)
        elif isinstance(value, (list, tuple, set)):
            for v in value:
                leaves += self._collect_leaf_texts(v)
        elif isinstance(value, str):
            leaves.append((value, True))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            leaves.append((str(value), False))
        return leaves

    def _looks_like_identifier(self, s: str) -> bool:
        # audit Finding 21: a whitespace-free string alone is NOT enough — that
        # flagged legitimately agent-generated canonical values (e.g. an airport
        # code "CDG" derived from "Paris") as fabricated, firing HIGH false
        # positives. Require actual identifier structure: an ID separator, an
        # embedded digit, or (for pure-alpha tokens) opaque length. Dedicated
        # UUID/email/URL/path regexes in _extract_entities still catch those
        # shapes independently.
        if len(s) < 3 or any(ch.isspace() for ch in s):
            return False
        if any(ch in self.ID_SEPARATORS for ch in s) or any(ch.isdigit() for ch in s):
            return True
        return len(s) >= self.MIN_OPAQUE_LEN

    def _extract_entities(self, args_text: str) -> List[str]:
        parsed = self._try_parse_args(args_text)
        leaves = self._collect_leaf_texts(parsed) if parsed is not None else [(args_text, False)]

        candidates: List[str] = []
        for text, is_string_value in leaves:
            candidates += self.UUID_RE.findall(text)
            candidates += self.EMAIL_RE.findall(text)
            candidates += self.URL_RE.findall(text)
            candidates += self.FILE_PATH_RE.findall(text)
            for m in self.INTEGER_ID_RE.finditer(text):
                n = int(m.group(0))
                if self.SMALL_INT_MIN <= n <= self.SMALL_INT_MAX:
                    continue  # coincidental small integers — see class docstring
                candidates.append(m.group(0))

            if parsed is not None:
                if is_string_value and self._looks_like_identifier(text):
                    candidates.append(text)
            else:
                for m in self.QUOTED_STRING_RE.finditer(text):
                    val = m.group(1) or m.group(2)
                    if val and self._looks_like_identifier(val):
                        candidates.append(val)

        seen = set()
        entities = []
        for c in candidates:
            key = c if self.CASE_SENSITIVE else c.lower()
            if key in seen:
                continue
            seen.add(key)
            entities.append(c)
        return entities

    def _is_allowlisted(self, entity: str) -> bool:
        return entity.lower() in self.ALLOWLIST

    def _is_destructive_tool(self, tool_name: str) -> bool:
        name = tool_name if self.CASE_SENSITIVE else tool_name.lower()
        return any(name.startswith(p) for p in self.DESTRUCTIVE_TOOL_PATTERNS)

    def _in_corpus(self, entity: str, corpus: str) -> bool:
        needle = entity if self.CASE_SENSITIVE else entity.lower()
        haystack = corpus if self.CASE_SENSITIVE else corpus.lower()
        return needle in haystack

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        if not state.tool_calls:
            return None

        static_parts = []
        if state.input_text:
            static_parts.append(state.input_text)
        if state.system_prompt:
            static_parts.append(state.system_prompt)
        static_corpus = "\n".join(static_parts)

        prior_outputs: List[str] = []
        visibility_complete = True

        for tc in state.tool_calls:
            if visibility_complete:
                corpus = static_corpus
                if prior_outputs:
                    corpus = corpus + "\n" + "\n".join(prior_outputs)

                fabricated = None
                for entity in self._extract_entities(tc.args):
                    if self._is_allowlisted(entity):
                        continue
                    if self._in_corpus(entity, corpus):
                        continue
                    fabricated = entity
                    break

                if fabricated is not None:
                    destructive = self._is_destructive_tool(tc.tool_name)
                    return FailureSignal(
                        failure_type=FailureType.TOOL_ARGUMENT_FABRICATION,
                        severity=Severity.CRITICAL if destructive else Severity.HIGH,
                        run_id=state.run_id,
                        agent_id=state.agent_id,
                        agent_version=state.agent_version,
                        step_index=tc.step_index,
                        confidence=0.85 if destructive else 0.7,
                        evidence={
                            "tool_name": tc.tool_name,
                            "tool_step": tc.step_index,
                            "fabricated_entity": fabricated,
                            "is_destructive_tool": destructive,
                            "args_snippet": tc.args[:200],
                        },
                    )

            # This call's own result becomes valid grounding for LATER calls.
            # If it isn't recorded, we can no longer rule out that a later
            # entity came from it — stop evaluating rather than risk a false
            # positive on ID-chaining, the single most common legitimate
            # pattern this detector would otherwise misfire on.
            if tc.output:
                prior_outputs.append(tc.output)
            elif tc.success is not None:
                visibility_complete = False

        return None


# ── RETRIEVED_CONTENT_INJECTION ───────────────────────────────────────────────


class RetrievedContentInjectionDetector(BaseDetector):
    """
    Content the agent pulled in from a retrieval or tool call — search
    results, a fetched web page, an MCP response — contains text that reads
    as an instruction directed at the agent itself, rather than data about
    the world. Indirect prompt injection: the attacker never talks to the
    agent directly, they plant the instruction somewhere the agent will read
    it back to itself. Distinct from `PROMPT_INJECTION_SIGNAL`, which checks
    the user's own input at run-start — this one checks content the agent
    retrieves *during* the run, a completely different attack surface with a
    completely different author (a third party controlling a web page or
    document, not the end user).

    Data-availability note: both content sources this detector reads are
    optional, instrumentation-dependent fields. `RetrievalResult.content`
    exists only when the caller passes `content=` to `retrieval_responded()`
    — the RAG retrieval pipeline transmits no document text at all otherwise
    (only `result_count`/`top_score`). `ToolCall.output` is the same
    optional field `TOOL_ARGUMENT_FABRICATION` and the wire-format extension
    for `PREMATURE_TERMINATION`/`UNREAD_TOOL_ERROR` rely on. A run whose
    instrumentation surfaces neither simply gives this detector nothing to
    check — it does not fire on absence of data, only on a positive pattern
    match in data that is present.

    Fires once per run, on the first retrieval or tool result whose content
    matches an injection marker — matching every other Tier 1 detector's
    one-signal-per-run contract.

    Severity: HIGH on any match. CRITICAL if, after the matching content was
    read, the agent goes on to call a tool it had never called before in
    this run AND whose name doesn't appear anywhere in the user's own input
    — the strongest available proxy for "the injected instruction actually
    changed what the agent did," since we have no ground truth for intent,
    only for whether the *shape* of tool use changed in a way the user's own
    request doesn't explain.

    Tunable: INJECTION_PHRASES, CASE_SENSITIVE, DETECT_BEHAVIOR_DEVIATION.
    """

    name = "RETRIEVED_CONTENT_INJECTION"
    SEVERITY = None  # computed per-signal: HIGH or CRITICAL

    INJECTION_PHRASES = [
        "ignore previous instructions",
        "ignore all previous instructions",
        "disregard previous instructions",
        "disregard all previous instructions",
        "your new task is",
        "you must",
        "you should now",
        "you are now",
    ]
    ROLE_MARKER_RE = re.compile(r"(?:^|\n)\s*(system|assistant)\s*:", re.IGNORECASE)
    DELIMITER_RE = re.compile(r"\[/?INST\]|<<SYS>>|<</SYS>>", re.IGNORECASE)
    BASE64_BLOCK_RE = re.compile(r"[A-Za-z0-9+/]{100,}={0,2}")
    CASE_SENSITIVE = False
    DETECT_BEHAVIOR_DEVIATION = True

    def _match_injection_marker(self, text: str) -> Optional[str]:
        haystack = text if self.CASE_SENSITIVE else text.lower()
        for phrase in self.INJECTION_PHRASES:
            needle = phrase if self.CASE_SENSITIVE else phrase.lower()
            if needle in haystack:
                return phrase
        if self.ROLE_MARKER_RE.search(text):
            return "embedded role marker"
        if self.DELIMITER_RE.search(text):
            return "instruction delimiter"
        if self.BASE64_BLOCK_RE.search(text):
            return "long base64 block"
        return None

    def _is_behavior_deviation(self, state: RunState, after_step: int) -> bool:
        if not self.DETECT_BEHAVIOR_DEVIATION:
            return False
        prior_tools = {tc.tool_name for tc in state.tool_calls if tc.step_index <= after_step}
        input_text = (state.input_text or "").lower()
        for tc in state.tool_calls:
            if tc.step_index <= after_step:
                continue
            if tc.tool_name in prior_tools:
                continue
            if tc.tool_name.lower() in input_text:
                continue
            return True
        return False

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        candidates: List[Tuple[int, str, str, str]] = []  # (step, source_type, source_name, text)
        for r in state.retrievals:
            if r.content:
                candidates.append((r.step_index, "retrieval", r.index_name, r.content))
        for tc in state.tool_calls:
            if tc.output:
                candidates.append((tc.step_index, "tool", tc.tool_name, tc.output))
        candidates.sort(key=lambda c: c[0])

        for step_index, source_type, source_name, text in candidates:
            marker = self._match_injection_marker(text)
            if marker is None:
                continue

            deviated = self._is_behavior_deviation(state, step_index)
            return FailureSignal(
                failure_type=FailureType.RETRIEVED_CONTENT_INJECTION,
                severity=Severity.CRITICAL if deviated else Severity.HIGH,
                run_id=state.run_id,
                agent_id=state.agent_id,
                agent_version=state.agent_version,
                step_index=step_index,
                confidence=0.85 if deviated else 0.7,
                evidence={
                    "source_type": source_type,
                    "source_name": source_name,
                    "source_step": step_index,
                    "matched_marker": marker,
                    "behavior_deviation": deviated,
                    "content_snippet": text[:200],
                },
            )

        return None


# ── AGENT_HANDOFF_FAILURE ────────────────────────────────────────────────────


class AgentHandoffFailureDetector(BaseDetector):
    """
    A handoff tool completed with no useful response, or reported failure.

    Handoff tools are identified by convention: names ending in `_agent`, or
    names starting with `delegate_`, `handoff_`, or `transfer_to_` (the OpenAI
    Swarm / Agents SDK handoff convention). The prefix is `transfer_to_`, not a
    bare `transfer_`, so a `transfer_funds`-style money-transfer tool isn't
    misread as a handoff. EXCLUDED_TOOL_NAMES is a stop-list for names that match
    a pattern but are known non-handoffs (e.g. `user_agent`), extendable via
    detectors.yml as real traffic surfaces more collisions.

    Distinct from `HANDOFF_CONTEXT_LOSS`: that one compares two runs linked by
    `parent_run_id` for lost context, whereas this fires within a single run when
    a handoff *tool call* itself fails or returns an empty/insufficient payload.

    Tunable: MIN_OUTPUT_LENGTH, HANDOFF_PATTERNS, EXCLUDED_TOOL_NAMES.
    KNOWN_EMPTY_RESPONSES catches terse strings that look successful but carry no
    useful handoff payload.
    """

    name = "AGENT_HANDOFF_FAILURE"
    SEVERITY = Severity.HIGH
    MAX_COST_NS = 50_000

    MIN_OUTPUT_LENGTH = 10
    HANDOFF_PATTERNS = ("_agent", "delegate_", "handoff_", "transfer_to_")
    # Names that match a pattern above but are known NOT to be handoffs
    # (convention collisions). Extend as real traffic surfaces more.
    EXCLUDED_TOOL_NAMES = frozenset({"user_agent"})
    KNOWN_EMPTY_RESPONSES = frozenset(
        {
            "",
            "done",
            "ok",
            "complete",
            "finished",
            "n/a",
            "null",
            "none",
            "success",
        }
    )

    def _is_handoff_tool(self, call: ToolCall) -> bool:
        name = call.tool_name.lower()
        if name in self.EXCLUDED_TOOL_NAMES:
            return False
        return any(
            name.endswith(pattern) if pattern.startswith("_") else name.startswith(pattern)
            for pattern in self.HANDOFF_PATTERNS
        )

    def _observed_output_length(self, call: ToolCall) -> Optional[int]:
        if call.output_length is not None:
            return call.output_length
        if call.output is not None:
            return len(call.output)
        return None

    def _known_empty_response(self, call: ToolCall) -> Optional[str]:
        if call.output is None:
            return None
        normalized = call.output.strip().lower()
        if normalized in self.KNOWN_EMPTY_RESPONSES:
            return normalized
        return None

    def _is_empty_response(self, call: ToolCall) -> bool:
        if self._known_empty_response(call) is not None:
            return True
        output_length = self._observed_output_length(call)
        return output_length is not None and output_length < self.MIN_OUTPUT_LENGTH

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        for call in state.tool_calls:
            if not self._is_handoff_tool(call):
                continue

            reason = None
            if call.success is False:
                reason = "tool_failed"
            elif call.success is True and self._is_empty_response(call):
                reason = (
                    "known_empty_response"
                    if self._known_empty_response(call) is not None
                    else "short_output"
                )

            if reason is None:
                continue

            output_length = self._observed_output_length(call)
            known_empty_response = self._known_empty_response(call)
            return FailureSignal(
                failure_type=FailureType.AGENT_HANDOFF_FAILURE,
                severity=self.SEVERITY,
                run_id=state.run_id,
                agent_id=state.agent_id,
                agent_version=state.agent_version,
                step_index=call.step_index,
                confidence=0.9 if call.success is False else 0.85,
                evidence={
                    "tool_name": call.tool_name,
                    "step_index": call.step_index,
                    "output_length": output_length,
                    "success": call.success,
                    "reason": reason,
                    "known_empty_response": known_empty_response,
                    "min_output_length": self.MIN_OUTPUT_LENGTH,
                },
            )

        return None


# ── HANDOFF_CONTEXT_LOSS ──────────────────────────────────────────────────────


class HandoffContextLossDetector(BaseDetector):
    """
    Multi-agent state loss: when one agent run hands off to another (agent A
    invokes agent B as a distinct `dt.run()`, linked via `parent_run_id`),
    a meaningful chunk of what A had learned doesn't make it into B's input.

    This detector is NOT evaluated via `on_run_completion` like every other
    Tier 1 detector — `on_run_completion(state: RunState)` only ever sees
    one run, and comparing a handoff fundamentally requires two. Changing
    that shared contract to thread a second RunState through was explicitly
    out of scope (the six-detector brief's own hard constraint: preserve the
    existing detector base class API, don't touch it for one detector's
    benefit). Instead this follows the exact precedent `PromptInjectionDetector`
    already set: `on_run_completion` always returns None here too, and the
    real logic lives in `evaluate_handoff()`, called explicitly from
    `services/detector/detector_svc/worker.py::process_run()` — the one
    place that already has both sides' data. Still registered normally in
    `_DETECTOR_CLASSES` so `SIZE_DROP_THRESHOLD`/`ENTITY_LOSS_THRESHOLD`
    get the same per-agent-category YAML tuning every other detector gets.

    How the worker gets the parent's data: `parent_run_id` is already a
    real, plumbed-through field (SDK → events table → OTel span attribute)
    but was never queried anywhere before this. Since `parent_run_id` IS the
    parent run's own `run_id`, the worker fetches it via the same, already-
    indexed `fetch_run_events()` every run uses — no new event type, no new
    database index. Parent events are filtered to those at or before the
    child's own `run.started` timestamp, so parent activity that happens
    later (concurrently or after the handoff) can't leak into "what A knew
    at the moment of handoff." "A's output state" is approximated as A's
    `input_text` plus every recorded `llm.responded`/`tool.responded`
    output up to that point — everything text-visible that A had accumulated.
    "B's input state" is simply B's own `input_text`.

    Known, disclosed limitations (matching the original spec):
    - Only fires when `parent_run_id` is actually set. No auto-instrumentation
      in this repo sets it today for LangGraph or CrewAI hierarchical
      multi-agent crews — that requires framework-specific hooks recognizing
      a handoff and threading the parent's run_id through, which is its own
      scope, not built here. A single-agent run, or a multi-agent run whose
      integration doesn't set `parent_run_id`, silently never fires — that's
      expected, not a bug.
    - No entity-loss detection at all if the parent's output text was never
      instrumented either (same optional/instrumentation-dependent caveat
      as `TOOL_ARGUMENT_FABRICATION` and `RETRIEVED_CONTENT_INJECTION`).

    Fires HIGH when both: the child's input is more than `SIZE_DROP_THRESHOLD`
    (default 0.5, i.e. 50%) smaller than the parent's accumulated context,
    AND at least `ENTITY_LOSS_THRESHOLD` (default 1) entities (UUIDs, emails,
    URLs, integer IDs of 3+ digits) present in the parent's context are
    missing from the child's input entirely.

    Tunable: SIZE_DROP_THRESHOLD, ENTITY_LOSS_THRESHOLD.
    """

    name = "HANDOFF_CONTEXT_LOSS"
    SEVERITY = Severity.HIGH

    SIZE_DROP_THRESHOLD = 0.5
    ENTITY_LOSS_THRESHOLD = 1

    UUID_RE = re.compile(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    )
    EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
    URL_RE = re.compile(r"\bhttps?://[^\s\"'<>]+")
    INTEGER_ID_RE = re.compile(r"\b\d{3,}\b")  # 3+ digits — shorter numbers too coincidental

    def _extract_entities(self, text: str) -> List[str]:
        candidates: List[str] = []
        candidates += self.UUID_RE.findall(text)
        candidates += self.EMAIL_RE.findall(text)
        candidates += self.URL_RE.findall(text)
        candidates += self.INTEGER_ID_RE.findall(text)
        seen = set()
        entities = []
        for c in candidates:
            key = c.lower()
            if key in seen:
                continue
            seen.add(key)
            entities.append(c)
        return entities

    def evaluate_handoff(
        self,
        parent_context: str,
        child_input: str,
        run_id: str,
        agent_id: str,
        agent_version: str,
    ) -> Optional[FailureSignal]:
        if not parent_context:
            return None
        size_before = len(parent_context)
        size_after = len(child_input or "")
        if size_before == 0:
            return None

        drop_ratio = (size_before - size_after) / size_before
        if drop_ratio <= self.SIZE_DROP_THRESHOLD:
            return None

        entities_before = self._extract_entities(parent_context)
        child_lower = (child_input or "").lower()
        missing = [e for e in entities_before if e.lower() not in child_lower]
        if len(missing) < self.ENTITY_LOSS_THRESHOLD:
            return None

        return FailureSignal(
            failure_type=FailureType.HANDOFF_CONTEXT_LOSS,
            severity=self.SEVERITY,
            run_id=run_id,
            agent_id=agent_id,
            agent_version=agent_version,
            step_index=0,
            confidence=min(1.0, 0.5 + len(missing) * 0.15),
            evidence={
                "parent_context_length": size_before,
                "child_input_length": size_after,
                "size_drop_ratio": round(drop_ratio, 2),
                "missing_entities": missing[:5],
                "missing_entity_count": len(missing),
            },
        )

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        return None


# ── RUNAWAY_ITERATION ─────────────────────────────────────────────────────────


class RunawayIterationDetector(BaseDetector):
    """
    A run crosses its step count or cost budget with no sign it's actually
    concluding — the agent just keeps going. Distinct from
    STEP_COUNT_INFLATION (which compares against this agent's own learned
    baseline) and COST_SPIKE (token count vs. baseline): this one uses fixed,
    absolute ceilings, and specifically requires the *absence* of any
    completion signal in the agent's own recent output — a run that legitimately
    needed 80 steps and said so along the way isn't runaway, it's just a big
    job that finished on its own terms.

    Cost is computed via dunetrace.policies.compute_run_cost(state.llm_calls)
    — the same USD-estimation logic runtime policies already use for the
    cost_usd trigger. No wire-format gap here: prompt/completion token counts
    and model names have always been part of llm.responded's payload and
    RunState.llm_calls, both client- and server-side.

    Strongest completion signal available is structural, not textual:
    state.exit_reason == "final_answer" (set by an explicit run.final_answer()
    call) means the agent itself declared the run done — that overrides the
    text-pattern check entirely and this detector never fires on such a run,
    regardless of step count or cost. Absent that, it falls back to scanning
    the last LOOKBACK_MESSAGES llm.responded outputs for completion language
    (final answer markers, "task complete", etc.) — the same kind of raw-text
    scan PREMATURE_TERMINATION/UNREAD_TOOL_ERROR already do — but that text is
    OPTIONAL, not guaranteed. It is absent when the caller sets
    DUNETRACE_OMIT_LLM_OUTPUT_TEXT (a deliberate bandwidth/privacy choice) and
    when instrumentation could not read the response at all. Absence therefore
    means "no completion signal was observable", never "no completion signal was
    given" — reading it the latter way makes this detector fire on runs it
    cannot actually assess, which is why an unmeasurable run raises
    INSTRUMENTATION_DEGRADED instead.

    Fires HIGH when either the step or cost ceiling is crossed with no
    completion signal; CRITICAL when both are crossed simultaneously.

    Tunable: STEP_THRESHOLD, COST_THRESHOLD_USD, COMPLETION_PATTERNS,
    LOOKBACK_MESSAGES, CASE_SENSITIVE.
    """

    name = "RUNAWAY_ITERATION"
    SEVERITY = None  # computed per-signal: HIGH or CRITICAL

    STEP_THRESHOLD = 50
    COST_THRESHOLD_USD = 1.0
    LOOKBACK_MESSAGES = 3
    COMPLETION_PATTERNS = [
        "final answer",
        "task complete",
        "task is complete",
        "i'm done",
        "that completes",
        "in conclusion",
        "to summarize",
        "here is the final",
        "all done",
        "completed successfully",
    ]
    CASE_SENSITIVE = False

    def _contains_completion_pattern(self, text: str) -> bool:
        haystack = text if self.CASE_SENSITIVE else text.lower()
        for pattern in self.COMPLETION_PATTERNS:
            needle = pattern if self.CASE_SENSITIVE else pattern.lower()
            if needle in haystack:
                return True
        return False

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        step_exceeded = state.current_step > self.STEP_THRESHOLD
        cost = compute_run_cost(state.llm_calls)
        cost_exceeded = cost > self.COST_THRESHOLD_USD
        if not step_exceeded and not cost_exceeded:
            return None

        if state.exit_reason == "final_answer":
            return None  # the agent itself declared the run done

        llm_outputs = [
            e
            for e in state.events
            if e.event_type == EventType.LLM_RESPONDED and e.payload.get("output")
        ]

        # This detector fails in the OPPOSITE direction from EMPTY_LLM_RESPONSE:
        # it fires on the ABSENCE of completion language, so unreadable text
        # produces a false positive rather than a false negative. A run whose
        # text we never had is a run whose completion signal we cannot assess,
        # and "could not assess" must not be reported as "no completion signal
        # was given" — that is the false docstring this class used to carry.
        #
        # Two ways the text can be missing without the agent doing anything
        # wrong: instrumentation could not read the response, or the operator
        # set DUNETRACE_OMIT_LLM_OUTPUT_TEXT. Both leave llm.responded events
        # present but output-free, so both are caught here.
        responded = [e for e in state.events if e.event_type == EventType.LLM_RESPONDED]
        if responded and not llm_outputs:
            logger.debug(
                "%s: %d llm.responded event(s) on run %s carry no output text — "
                "cannot assess completion language, so not firing. Either "
                "DUNETRACE_OMIT_LLM_OUTPUT_TEXT is set or instrumentation is "
                "degraded (see INSTRUMENTATION_DEGRADED).",
                self.name,
                len(responded),
                state.run_id,
            )
            return None

        recent = llm_outputs[-self.LOOKBACK_MESSAGES :]
        if any(self._contains_completion_pattern(e.payload["output"]) for e in recent):
            return None  # a completion signal exists — not runaway

        both_exceeded = step_exceeded and cost_exceeded
        return FailureSignal(
            failure_type=FailureType.RUNAWAY_ITERATION,
            severity=Severity.CRITICAL if both_exceeded else Severity.HIGH,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=state.current_step,
            confidence=0.9 if both_exceeded else 0.75,
            evidence={
                "step_count": state.current_step,
                "step_threshold": self.STEP_THRESHOLD,
                "step_exceeded": step_exceeded,
                "estimated_cost_usd": round(cost, 4),
                "cost_threshold_usd": self.COST_THRESHOLD_USD,
                "cost_exceeded": cost_exceeded,
                "recent_messages_checked": len(recent),
            },
        )


# ── MODEL_FALLBACK_DRIFT ──────────────────────────────────────────────────────


class ModelFallbackDriftDetector(BaseDetector):
    """
    Within one run the agent's LLM model changed to a less capable one (gpt-4o →
    gpt-4o-mini, claude-sonnet → claude-haiku). Usually a silent fallback under
    rate limiting or an SDK's automatic-retry-on-a-cheaper-model behavior; the
    run keeps going on a weaker model and quality drops with nothing surfacing
    it. Fires on the first downgrade in the run.

    Only compares calls WITHIN a single run, which is a single agent — a
    multi-agent system runs each agent as its own `dt.run()`, so "different
    agents use different models by design" never reaches this detector (those
    are separate runs, each judged on its own). An upgrade (mini → 4o) or a
    same-tier switch (gpt-4o → claude-3-5-sonnet) is not a downgrade and does
    not fire. Unknown models (not in the tier map) are skipped rather than
    guessed at.

    Evidence records whether a rate-limit external signal preceded the switch,
    which is the usual cause worth acting on.

    Tunable: MODEL_TIERS — the model→capability-tier map (higher = more capable).
    Override in detectors.yml to add models or re-rank tiers.
    """

    name = "MODEL_FALLBACK_DRIFT"
    SEVERITY = Severity.MEDIUM
    MAX_COST_NS = 50_000

    # Matched by longest substring, so versioned names (gpt-4o-2024-08-06,
    # claude-3-5-sonnet-20241022) resolve to their family. Higher tier = more
    # capable; a strictly lower tier on a later call is a downgrade.
    MODEL_TIERS: dict[str, int] = {
        # OpenAI
        "gpt-3.5": 1,
        "gpt-4o-mini": 2,
        "gpt-4o": 3,
        "gpt-4-turbo": 3,
        "gpt-4": 3,
        "o1-mini": 2,
        "o3-mini": 2,
        "o1": 3,
        "o3": 3,
        "gpt-5-mini": 3,
        "gpt-5": 4,
        # Anthropic
        "claude-3-haiku": 2,
        "claude-3-5-haiku": 2,
        "claude-haiku": 2,
        "claude-3-sonnet": 3,
        "claude-3-5-sonnet": 3,
        "claude-sonnet-4": 3,
        "claude-sonnet": 3,
        "claude-3-opus": 4,
        "claude-opus-4": 4,
        "claude-opus": 4,
    }

    def _tier(self, model: str) -> Optional[int]:
        if not model:
            return None
        m = model.lower()
        best_key: Optional[str] = None
        for key in self.MODEL_TIERS:
            if key in m and (best_key is None or len(key) > len(best_key)):
                best_key = key
        return self.MODEL_TIERS[best_key] if best_key is not None else None

    def _preceded_by_rate_limit(self, state: RunState, step: int) -> bool:
        return any(
            "rate" in (s.signal_name or "").lower() and s.step_index <= step
            for s in state.external_signals
        )

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        calls = [c for c in state.llm_calls if c.model]
        if len(calls) < 2:
            return None

        for prev, cur in zip(calls, calls[1:]):
            if cur.model == prev.model:
                continue
            prev_tier, cur_tier = self._tier(prev.model), self._tier(cur.model)
            if prev_tier is None or cur_tier is None:
                continue  # unknown model on either side — can't judge a downgrade
            if cur_tier >= prev_tier:
                continue  # upgrade or same tier
            return FailureSignal(
                failure_type=FailureType.MODEL_FALLBACK_DRIFT,
                severity=self.SEVERITY,
                run_id=state.run_id,
                agent_id=state.agent_id,
                agent_version=state.agent_version,
                step_index=cur.step_index,
                confidence=0.8,
                evidence={
                    "from_model": prev.model,
                    "to_model": cur.model,
                    "from_tier": prev_tier,
                    "to_tier": cur_tier,
                    "tier_delta": prev_tier - cur_tier,
                    "downgrade_step": cur.step_index,
                    "preceded_by_rate_limit": self._preceded_by_rate_limit(state, cur.step_index),
                },
            )
        return None


# ── MEMORY_POISONING ──────────────────────────────────────────────────────────


class MemoryPoisonedDetector(BaseDetector):
    """
    An injection / override directive was persisted into the agent's own memory
    (a conversation buffer, scratchpad, or long-term store) — content that will
    steer the agent when it reads that memory back on a later step or turn.

    This is a third, distinct injection surface from the two Dunetrace already
    covers, differing in *when* and *from where* the hostile text enters:

    - `PROMPT_INJECTION_SIGNAL` checks the user's own input at run-start.
    - `RETRIEVED_CONTENT_INJECTION` checks content pulled in from a retrieval or
      tool result *during* the run (read once, acted on immediately).
    - `MEMORY_POISONING` checks what the agent *writes to memory* — the danger
      is persistence: a directive planted in memory survives across steps and
      across turns, and re-steers the agent every time that memory is loaded,
      long after the step that wrote it. The classic attack is content from an
      untrusted channel (a retrieved document, a tool response, an external
      feed) being summarized/persisted into memory verbatim, injection and all.

    Reads the typed `state.memory_events` view (built from `memory.*` events —
    see run_context.py / run_builder.py). A run that never touches the memory
    channel gives this detector nothing to check; it fires only on a positive
    marker match in a written value, never on absence of memory activity.

    Marker vocabulary is deliberately narrower than `PROMPT_INJECTION_SIGNAL`'s:
    it keeps the unambiguous *override* signatures (ignore/disregard/forget
    instructions, override/bypass safety, embedded role markers, instruction
    delimiters, jailbreak/DAN) but drops the role-play ones (`act as`,
    `pretend`, `you are now`, `your new role is`) — those legitimately appear in
    user personalization that agents routinely persist ("act as my travel
    planner", "you are now my coding assistant"), and including them tanks
    precision on benign stored preferences. See scripts/calibration/
    memory_poisoning_calibration.md.

    Fires once per run, on the first written value that matches a marker.

    Severity: HIGH on any match. CRITICAL when the write is higher-confidence
    poisoning — either its `source` is an attacker-controllable channel
    (`retrieval` / `tool_output` / `external`), or the poisoned key is
    subsequently *read* in the same run (the poisoned memory was actually
    loaded back, not just written and left dormant).

    Tunable: POISON_PHRASES, CASE_SENSITIVE, REQUIRE_UNTRUSTED_SOURCE.
    """

    name = "MEMORY_POISONING"
    SEVERITY = None  # computed per-signal: HIGH or CRITICAL

    # Substring markers (case-insensitive by default). Override-style directives
    # only — no role-play phrases (see class docstring / calibration).
    POISON_PHRASES = [
        "ignore previous instructions",
        "ignore all previous instructions",
        "ignore prior instructions",
        "ignore the above instructions",
        "ignore earlier instructions",
        "disregard previous instructions",
        "disregard all previous instructions",
        "disregard prior instructions",
        "disregard the above",
        "forget previous instructions",
        "forget all previous instructions",
        "forget everything above",
        "do not follow your previous",
        "do not follow the previous",
        "do not follow your original",
        "override safety",
        "override your safety",
        "override the safety",
        "bypass safety",
        "bypass your restrictions",
        "bypass all restrictions",
        "developer mode enabled",
        "jailbreak",
        "dan mode",
    ]
    # Structural markers matched by regex, independent of POISON_PHRASES.
    ROLE_MARKER_RE = re.compile(r"(?:^|\n)\s*(system|assistant)\s*:", re.IGNORECASE)
    DELIMITER_RE = re.compile(
        r"\[/?INST\]|<<SYS>>|<</SYS>>|<\|im_start\|>|<\|system\|>|###\s*system",
        re.IGNORECASE,
    )
    CASE_SENSITIVE = False
    # When True, only fire on writes whose source is a known untrusted channel
    # (retrieval/tool_output/external). Off by default so that source-less
    # framework-auto-captured writes (unknown provenance) still fire.
    REQUIRE_UNTRUSTED_SOURCE = False

    _UNTRUSTED_SOURCES = frozenset({"retrieval", "tool_output", "external"})

    def _match_marker(self, text: str) -> Optional[str]:
        haystack = text if self.CASE_SENSITIVE else text.lower()
        for phrase in self.POISON_PHRASES:
            needle = phrase if self.CASE_SENSITIVE else phrase.lower()
            if needle in haystack:
                return phrase
        if self.ROLE_MARKER_RE.search(text):
            return "embedded_role_marker"
        if self.DELIMITER_RE.search(text):
            return "instruction_delimiter"
        return None

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        writes = [m for m in state.memory_events if m.op == "written" and m.value]
        if not writes:
            return None

        # Keys read at or after their write step — the poisoned content was
        # actually loaded back, not just written and left dormant.
        reads = [m for m in state.memory_events if m.op == "read"]

        matched = []
        for w in writes:
            value = w.value or ""  # writes are pre-filtered to a truthy value; narrows Optional
            marker = self._match_marker(value)
            if marker is None:
                continue
            untrusted = w.source in self._UNTRUSTED_SOURCES
            if self.REQUIRE_UNTRUSTED_SOURCE and not untrusted:
                continue
            matched.append((w, marker, untrusted, value))

        if not matched:
            return None

        w, marker, untrusted, value = matched[0]
        consumed = any(r.key == w.key and r.step_index >= w.step_index for r in reads)
        critical = untrusted or consumed
        return FailureSignal(
            failure_type=FailureType.MEMORY_POISONING,
            severity=Severity.CRITICAL if critical else Severity.HIGH,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=w.step_index,
            confidence=0.85 if critical else 0.7,
            evidence={
                "memory_key": w.key,
                "source": w.source,
                "matched_marker": marker,
                "untrusted_source": untrusted,
                "consumed": consumed,
                "write_step": w.step_index,
                "poisoned_write_count": len(matched),
                "value_snippet": value[:200],
            },
        )


# ── DELEGATION_LOOP ───────────────────────────────────────────────────────────


class DelegationLoopDetector(BaseDetector):
    """
    Two or more agents delegate to each other in a cycle — agent A hands off to
    B, B hands back to A, A hands to B again — and the loop keeps going around
    instead of converging. A multi-agent analogue of `TOOL_LOOP`: no single run
    is misbehaving, but the *system* of runs is stuck in a mutual-delegation
    spin, burning tokens and never terminating.

    Like `HANDOFF_CONTEXT_LOSS`, this can't run via `on_run_completion(state)` —
    that contract only ever sees one run, and a delegation cycle is a property of
    the run *graph*. So it follows the same precedent `PROMPT_INJECTION_SIGNAL`
    and `HANDOFF_CONTEXT_LOSS` set: `on_run_completion` returns None here, and the
    real logic runs from `services/detector/detector_svc/worker.py::process_run()`
    via `evaluate_delegation_cycle()`, the one place with cross-run graph access.

    How the graph is built: a run's `parent_run_id` (auto-threaded by the SDK
    when one `dt.run()` opens inside another — see client.py) links a child run
    to the run that spawned it. The worker walks that chain to the root
    (`run_graph.build_ancestor_chain`), derives the directed agent-delegation
    graph from it, and runs three-colour DFS cycle detection
    (`run_graph.find_cycle`). The *run* graph is a forest and can never cycle;
    the cycle is in the *agent* dimension (A → B → A).

    Fires only when the loop is sustained: at least `MIN_LOOP_RUNS` (default 5)
    runs in the chain participate in a detected agent cycle. 5 is the calibrated
    boundary that separates a runaway loop from a legitimate iterative supervisor
    exchange (supervisor delegates, worker returns, supervisor delegates again,
    then finishes — 4 runs, which must NOT fire); see
    scripts/calibration/delegation_loop_calibration.md.

    Severity: HIGH normally; CRITICAL once `CRITICAL_LOOP_RUNS` (default 7) runs
    are caught in the loop — a runaway that isn't self-terminating.

    Disclosed limitation: only fires when `parent_run_id` is set along the chain.
    Auto-threading (Phase 2.1) covers nested `dt.run()` calls on the same task or
    an asyncio child task; a sub-agent dispatched to a bare thread, or a
    framework that collapses a whole crew into a single run, produces no
    multi-run graph to walk.

    Tunable: MIN_LOOP_RUNS, CRITICAL_LOOP_RUNS.
    """

    name = "DELEGATION_LOOP"
    SEVERITY = None  # computed per-signal: HIGH or CRITICAL

    MIN_LOOP_RUNS = 5
    CRITICAL_LOOP_RUNS = 7

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        return None

    def evaluate_delegation_cycle(
        self,
        cycle: Optional[List[str]],
        agent_sequence: List[str],
        run_id: str,
        agent_id: str,
        agent_version: str,
    ) -> Optional[FailureSignal]:
        """Decide whether a detected agent cycle is a sustained delegation loop.

        ``cycle`` is the DFS result (e.g. ``['A', 'B', 'A']``) or None if the
        agent graph was acyclic. ``agent_sequence`` is the full root-first agent
        order along the chain, with repetition preserved, so we can measure how
        many runs are caught in the loop.
        """
        if not cycle:
            return None
        cycle_agents = set(cycle)
        loop_run_count = sum(1 for a in agent_sequence if a in cycle_agents)
        if loop_run_count < self.MIN_LOOP_RUNS:
            return None  # a single hand-back, not a sustained loop

        critical = loop_run_count >= self.CRITICAL_LOOP_RUNS
        return FailureSignal(
            failure_type=FailureType.DELEGATION_LOOP,
            severity=Severity.CRITICAL if critical else Severity.HIGH,
            run_id=run_id,
            agent_id=agent_id,
            agent_version=agent_version,
            step_index=0,  # a cross-run signal — no single step owns it
            confidence=min(0.9, 0.6 + 0.05 * (loop_run_count - self.MIN_LOOP_RUNS + 1)),
            evidence={
                "cycle": cycle,
                "cycle_agents": sorted(cycle_agents),
                "cycle_length": len(cycle_agents),
                "loop_run_count": loop_run_count,
                "delegation_chain": agent_sequence,
                "min_loop_runs": self.MIN_LOOP_RUNS,
            },
        )


# ── UNGROUNDED_DESTINATION ────────────────────────────────────────────────────
#
# Shared extraction helpers. These deliberately do NOT replace
# ToolArgumentFabricationDetector's own _try_parse_args/_collect_leaf_texts, even
# though the parsing half overlaps: that detector is calibrated and live, and
# re-pointing it at shared code — however behaviour-preserving it looks — is a
# regression risk taken for style. The duplication is ~20 lines and bounded.

# Multi-label public suffixes. The real Public Suffix List has ~9,000 entries and
# needs periodic refreshes; the core SDK has zero required dependencies, so it
# cannot carry one. This bundled subset covers the exfiltration-relevant tail:
# country-code second-levels, and the free shared-hosting providers an attacker
# can stand a destination up on in minutes.
#
# Incompleteness degrades safely and cannot cause a missed detection, because
# GROUNDING never consults this table — grounding compares full hostnames. It is
# used only for (a) allowlist parenting and (b) novelty-baseline keys, where a
# missing entry just makes the allowlist less generous and the baseline key more
# specific. Both fail toward firing, not toward silence.
_MULTI_LABEL_SUFFIXES = frozenset(
    {
        "co.uk",
        "org.uk",
        "ac.uk",
        "gov.uk",
        "me.uk",
        "net.uk",
        "com.au",
        "net.au",
        "org.au",
        "edu.au",
        "gov.au",
        "co.jp",
        "or.jp",
        "ne.jp",
        "ac.jp",
        "go.jp",
        "co.nz",
        "co.za",
        "co.in",
        "co.kr",
        "com.br",
        "com.mx",
        "com.cn",
        "com.sg",
        "com.hk",
        "com.tw",
        "com.tr",
        "com.ar",
        "com.pl",
        "github.io",
        "gitlab.io",
        "pages.dev",
        "workers.dev",
        "web.app",
        "firebaseapp.com",
        "vercel.app",
        "netlify.app",
        "herokuapp.com",
        "azurewebsites.net",
        "blob.core.windows.net",
        "s3.amazonaws.com",
        "r2.dev",
        "trycloudflare.com",
        "ngrok.io",
        "ngrok-free.app",
        "glitch.me",
        "repl.co",
        "replit.dev",
        "surge.sh",
        "onrender.com",
        "fly.dev",
        "cloudfunctions.net",
        "run.app",
        "appspot.com",
    }
)

# Last labels accepted for bare-domain extraction. Only consulted when
# bare_domain is explicitly enabled (it is off by default) — a bare-domain regex
# with no TLD gate matches "report.pdf", "v1.2.3" and "obj.method".
_COMMON_TLDS = frozenset(
    {
        "com",
        "net",
        "org",
        "io",
        "co",
        "ai",
        "app",
        "dev",
        "cloud",
        "sh",
        "me",
        "info",
        "biz",
        "xyz",
        "top",
        "site",
        "online",
        "test",
        "example",
        "uk",
        "de",
        "fr",
        "nl",
        "eu",
        "us",
        "ca",
        "au",
        "jp",
        "cn",
        "in",
        "br",
        "ru",
        "ch",
        "se",
        "no",
        "fi",
        "it",
        "es",
        "pl",
        "tr",
        "za",
        "mx",
        "kr",
    }
)

_DEST_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]*[\w-]\b")
_DEST_URL_RE = re.compile(r"\bhttps?://[^\s\"'<>\\)\]}]+", re.IGNORECASE)
_DEST_BARE_DOMAIN_RE = re.compile(
    r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}\b", re.IGNORECASE
)

# Memory write sources an attacker can reach. Same vocabulary MEMORY_POISONING
# uses (see MemoryPoisonedDetector._UNTRUSTED_SOURCES) — kept as its own constant
# so neither detector's calibration can silently move the other's.
_UNTRUSTED_MEMORY_SOURCES = frozenset({"retrieval", "tool_output", "external"})


def _normalize_host(host: str) -> str:
    """Lowercase, strip userinfo/port/trailing dot, and IDNA-normalize a hostname.

    IDNA folding is stdlib (`str.encode("idna")`) and collapses a homograph-
    adjacent evasion class for free. It raises on plenty of real-world input
    (over-long labels, empty labels), so it is strictly best-effort.
    """
    h = host.strip().strip(".").lower()
    if "@" in h:  # userinfo
        h = h.rsplit("@", 1)[1]
    if h.startswith("["):  # IPv6 literal
        return h.split("]", 1)[0] + "]"
    if ":" in h:
        h = h.split(":", 1)[0]
    # str.isascii() is a C-level flag check; the equivalent
    # `any(ord(c) > 127 for c in h)` is a Python-level loop per host and showed
    # up in profiling on args carrying hundreds of candidates.
    if not h.isascii():
        try:
            h = h.encode("idna").decode("ascii")
        except Exception:
            pass
    return h


def _registrable_domain(host: str) -> str:
    """eTLD+1 for a normalized host, using the bundled suffix table.

    Allowlist parenting and novelty keys only — never grounding. See the
    _MULTI_LABEL_SUFFIXES comment for why that distinction is load-bearing.
    """
    labels = host.split(".")
    if len(labels) < 3:
        return host
    if ".".join(labels[-2:]) in _MULTI_LABEL_SUFFIXES:
        return ".".join(labels[-3:])
    if ".".join(labels[-3:]) in _MULTI_LABEL_SUFFIXES:
        return ".".join(labels[-4:]) if len(labels) >= 4 else host
    return ".".join(labels[-2:])


def _host_in_allowlist(host: str, allowlist) -> bool:
    """Label-aware suffix match: `corp.com` covers `mail.corp.com` but NOT
    `evilcorp.com`. A plain endswith() would let an attacker register the latter
    and inherit the allowlist entry."""
    h = host.lower()
    for entry in allowlist or ():
        e = str(entry).strip().lower().lstrip(".")
        if not e:
            continue
        if h == e or h.endswith("." + e):
            return True
    return False


def _parse_tool_args(args_text: str):
    """JSON first, then Python literal — the Python SDK emits str(dict) and the
    TS SDK emits JSON.stringify. Returns None when neither parses."""
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(args_text)
        except Exception:
            continue
    return None


def _walk_arg_strings(value, max_depth: int = 12, max_nodes: int = 5000) -> List[Tuple[str, str]]:
    """Yield (path, text) for every string leaf of a parsed args structure.

    Values only, never keys — a key named "email" is schema, not data. Non-string
    leaves are skipped rather than stringified: an int can never be a
    destination, and stringifying it only adds noise for the regexes to chew on.

    Bounded by depth AND node count. This runs inside the user's agent process
    on the in-path path, so a deeply nested or merely enormous payload must
    degrade to partial extraction, never to a RecursionError or a hang.
    """
    out: List[Tuple[str, str]] = []
    remaining = [max_nodes]

    def walk(node, path: str, depth: int) -> None:
        if remaining[0] <= 0 or depth > max_depth:
            return
        remaining[0] -= 1
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else str(k), depth + 1)
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]", depth + 1)
        elif isinstance(node, (set, frozenset)):
            # Sets have no stable iteration order; sort so arg_path and the
            # chosen candidate are deterministic across runs and processes.
            for i, v in enumerate(sorted(node, key=repr)):
                walk(v, f"{path}[{i}]", depth + 1)
        elif isinstance(node, str):
            out.append((path or "<root>", node))

    walk(value, "", 0)
    return out


class UngroundedDestinationDetector(BaseDetector):
    """
    A tool call sends data to a destination — an email address, a URL host, a
    bare domain — that does not appear anywhere in the run's own trusted input
    surface. The actuation signature of agent data exfiltration: after a
    successful injection, the agent emails `attacker@evil.test`, an address the
    legitimate task never mentioned.

    PROVENANCE-BASED, NOT NOVELTY-BASED, and that is the whole design. An
    open-destination agent (a support agent emailing an ever-new customer every
    run) makes per-destination history worthless — every legitimate run contains
    a destination never seen before. So the default mode never asks "have I seen
    this destination before"; it asks "did THIS RUN's own trusted inputs contain
    this string". A support agent whose CRM lookup returned the customer's
    address is silent by construction, forever, no matter how many new customers
    it writes to.

    Three checks, in order:

    1. EXTRACT destination candidates from each tool call's args, recursing
       through nested structures (values only, never keys), bounded by depth and
       node count.
    2. GROUNDING. Is the destination present in the run's trusted-input surface?
       Grounded → silent. The surface is enumerated in _partition_surfaces();
       the deliberate exclusion is LLM output and the agent's own earlier tool
       args, because grounding on the model's own text is circular — an injected
       model would emit the destination and thereby ground it.
    3. TAINT. Does the ungrounded destination appear in attacker-controllable
       content: an untrusted-source memory value, or a content block carrying an
       injection marker?

    CONDITIONAL DEMOTION resolves the conflict at the heart of this. Tool output
    and retrieval content are simultaneously the legitimate way a destination
    enters a run (a CRM lookup returns the customer's email) and the attacker's
    primary injection channel (a poisoned document carries theirs). A block that
    matches an injection marker is therefore REMOVED from the grounded surface
    and MOVED to the taint surface. Clean blocks ground normally. The same rule
    applies to the user's own input when the run carries a PROMPT_INJECTION
    signal. One rule, three channels.

    Severity: HIGH when ungrounded. CRITICAL when ungrounded AND tainted — the
    difference between "this address came from nowhere I can see" and "this
    address came from content an attacker controls".

    Alert wording is deliberately "verify this destination", never "you are
    breached". The commonest false positive is a destination that arrived
    through instrumentation this run didn't capture, and an alert that overstates
    it teaches operators to ignore the detector.

    Cross-run taint (T6): a destination planted in run N's memory and read back
    in run N+2 is ungrounded in N+2 but has no in-run taint source, because
    `memory.read` carries only a key — the SDK never sees the value that came
    back, so there is nothing to reconstruct. evaluate_cross_run_memory_taint()
    closes that, fed by the worker from prior `memory.written` events for the
    same agent and key. Deferred and NOT implemented: taint from an injection
    signal on an ancestor run. Ancestor signals are derived, not ingested, so
    whether the parent's signals exist when the child is processed depends on
    poll ordering — which would make this detector's severity nondeterministic.
    Sequential runs sharing a memory store are also not in a parent_run_id chain
    at all, so an ancestor walk would not have covered the memory case anyway.

    In-path use: this detector is in TIER1_DETECTORS, so a trigger="signal"
    policy can stop a send BEFORE it executes (RunContext.tool_called appends the
    call and evaluates policies before the tool body runs). Two attributes exist
    for that path and matter only there: TOOL_NAME_SCOPE limits which tools are
    scanned, and MAX_SCAN_NS is a hard wall-clock abort that returns None rather
    than adding latency to the agent. Degradation is prevent → detect, never
    detect → nothing: the server-side instance runs unscoped with a far larger
    budget and still catches whatever the in-path pass skipped.

    Novelty mode (MODE="provenance+novelty") is a secondary, per-agent opt-in for
    CLOSED-destination agents (billing, internal reporting) where history
    genuinely defines normal. It evaluates only destinations that PASSED
    grounding, so it can never double-fire with provenance mode, and it needs a
    server-supplied baseline — the in-path instance has no database and stays
    provenance-only regardless of config.

    Known limit: taint matching is verbatim-string. A destination the agent
    ASSEMBLES from fragments ("email the user at the domain in field X") never
    appears whole in any taint source, so it stays HIGH rather than escalating to
    CRITICAL. Encoded destinations are the same class. Out of scope — that is
    what approval policies on send-class tools are for.

    Tunable: CANDIDATE_TYPES, ALLOWLISTED_DOMAINS, MODE, MIN_BASELINE_RUNS,
    MAX_CANDIDATES_PER_RUN, MAX_DEPTH, MAX_NODES, MAX_SCAN_NS, TOOL_NAME_SCOPE,
    SEND_TOOL_PATTERNS, CASE_SENSITIVE.
    """

    name = "UNGROUNDED_DESTINATION"
    SEVERITY = None  # computed per-signal: MEDIUM | HIGH | CRITICAL

    CANDIDATE_TYPES = ["email", "url"]  # + "bare_domain" (off: FP-prone, see module notes)
    ALLOWLISTED_DOMAINS: List[str] = []
    MODE = "provenance"  # "provenance" | "provenance+novelty"
    MIN_BASELINE_RUNS = 20  # matches detector_svc/db.py::_MIN_BASELINE_RUNS
    MAX_CANDIDATES_PER_RUN = 50
    MAX_DEPTH = 12
    MAX_NODES = 5000
    # Hard wall-clock abort. Generous by default (server-side, off the agent's
    # critical path); the TIER1 instance is constructed with the 1ms in-path
    # budget. Exceeding it returns None — see the fail-open note in the docstring.
    MAX_SCAN_NS = 50_000_000
    # Total trusted+taint content this detector will scan. Above it the run is
    # skipped entirely (fail open) rather than evaluated against a truncated
    # corpus — see the guard in on_run_completion for why truncation is unsafe.
    # The TIER1 instance is constructed with a much tighter in-path value.
    MAX_SURFACE_CHARS = 20_000_000
    # Longest tool-args string parsed structurally. Above it the args are scanned
    # as raw text, truncated to this length — see the guard's comment for why
    # truncating args is safe in a way truncating the surface is not.
    MAX_ARGS_CHARS = 1_000_000
    # None = scan every tool. A list of substrings limits scanning to
    # destination-bearing tools, which is all that matters for prevention.
    TOOL_NAME_SCOPE: Optional[List[str]] = None
    SEND_TOOL_PATTERNS = [
        "send",
        "email",
        "mail",
        "message",
        "post",
        "publish",
        "upload",
        "webhook",
        "http",
        "request",
        "fetch",
        "notify",
        "share",
    ]
    CASE_SENSITIVE = False

    # Injection markers for conditional demotion. Intentionally the union of
    # RETRIEVED_CONTENT_INJECTION's and MEMORY_POISONING's unambiguous override
    # signatures — demotion should be at least as sensitive as either detector,
    # since a block either of them would flag must not be allowed to ground.
    DEMOTION_PHRASES = [
        "ignore previous instructions",
        "ignore all previous instructions",
        "ignore prior instructions",
        "ignore the above instructions",
        "disregard previous instructions",
        "disregard all previous instructions",
        "disregard prior instructions",
        "disregard the above",
        "forget previous instructions",
        "forget all previous instructions",
        "forget everything above",
        "your new task is",
        "you should now",
        "override safety",
        "bypass safety",
        "bypass your restrictions",
        "jailbreak",
        "dan mode",
    ]
    _DEMOTION_ROLE_RE = re.compile(r"(?:^|\n)\s*(system|assistant)\s*:", re.IGNORECASE)
    _DEMOTION_DELIM_RE = re.compile(
        r"\[/?INST\]|<<SYS>>|<</SYS>>|<\|im_start\|>|<\|system\|>|###\s*system",
        re.IGNORECASE,
    )

    # ── Extraction ───────────────────────────────────────────────────────────

    # Cheap necessary conditions for the two structural regexes. A substring scan
    # of a pre-lowered string is memchr-fast (~37µs per 100KB); the equivalent
    # IGNORECASE regex is ~2.2ms. Since a regex match is impossible unless its
    # anchor text is present, gating on these is semantics-preserving and turns
    # the dominant cost into a rounding error on the overwhelmingly common
    # no-marker path. (An alternation regex over the phrase list was measured at
    # ~25ms per 100KB — 40x slower than the plain loop. Do not "optimize" this
    # back into one pattern.)
    _ROLE_HINTS = ("system", "assistant")
    _DELIM_HINTS = ("[inst", "<<sys", "</sys", "<|im", "<|system", "###")

    def _phrase_hints(self):
        """First token of each phrase — a necessary condition for that phrase, so
        their union is a necessary condition for the whole list. DERIVED, not
        hardcoded: DEMOTION_PHRASES is a tunable, and a hardcoded hint set would
        silently stop matching an operator's added phrases."""
        key = tuple(self.DEMOTION_PHRASES or ())
        cached = getattr(self, "_phrase_hints_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        hints = tuple({(p.split()[0].lower() if p.split() else p.lower()) for p in key})
        self._phrase_hints_cache = (key, hints)
        return hints

    def _demotion_marker(self, text: str) -> Optional[str]:
        low = text if self.CASE_SENSITIVE else text.lower()

        if any(h in low for h in self._phrase_hints()):
            for phrase in self.DEMOTION_PHRASES:
                needle = phrase if self.CASE_SENSITIVE else phrase.lower()
                if needle in low:
                    return phrase

        if any(h in low for h in self._ROLE_HINTS) and self._DEMOTION_ROLE_RE.search(text):
            return "embedded_role_marker"
        if any(h in low for h in self._DELIM_HINTS) and self._DEMOTION_DELIM_RE.search(text):
            return "instruction_delimiter"
        return None

    def _destinations_in(self, text: str) -> List[Tuple[str, str, str]]:
        """(destination, destination_type, host) for every candidate in `text`.

        Emails keep the full address as the match key — `a+x@corp.com` and
        `a@corp.com` route to the same mailbox but are different destinations to
        a reviewer, and collapsing them would let a tagged address inherit an
        untagged one's grounding.
        """
        found: List[Tuple[str, str, str]] = []
        types = set(self.CANDIDATE_TYPES or ())

        if "email" in types:
            for m in _DEST_EMAIL_RE.findall(text):
                addr = m.lower()
                found.append((addr, "email", _normalize_host(addr.rsplit("@", 1)[1])))

        if "url" in types:
            for raw in _DEST_URL_RE.findall(text):
                try:
                    host = _urlsplit(raw.rstrip(".,;")).hostname or ""
                except Exception:
                    host = ""
                if not host:
                    continue
                h = _normalize_host(host)
                if h:
                    found.append((h, "url", h))

        if "bare_domain" in types:
            # Emails and URLs already consumed their hosts; strip them so the
            # same host isn't reported twice under two types.
            residue = _DEST_EMAIL_RE.sub(" ", text)
            residue = _DEST_URL_RE.sub(" ", residue)
            for raw in _DEST_BARE_DOMAIN_RE.findall(residue):
                h = _normalize_host(raw)
                if not h or "." not in h:
                    continue
                if h.rsplit(".", 1)[1] not in _COMMON_TLDS:
                    continue
                found.append((h, "bare_domain", h))

        return found

    def _in_scope(self, tool_name: str) -> bool:
        if self.TOOL_NAME_SCOPE is None:
            return True
        name = tool_name if self.CASE_SENSITIVE else tool_name.lower()
        return any(str(p).lower() in name for p in self.TOOL_NAME_SCOPE)

    def _is_send_tool(self, tool_name: str) -> bool:
        name = tool_name if self.CASE_SENSITIVE else tool_name.lower()
        return any(str(p).lower() in name for p in self.SEND_TOOL_PATTERNS)

    # ── Grounded / taint surfaces ────────────────────────────────────────────

    def _input_is_injected(self, state: RunState) -> Optional[str]:
        """True when the run's own input carries an injection signal.

        Two sources because the two paths differ: server-side the SDK's
        precomputed evidence rides on the run.started payload; in-path it may not
        be attached yet, so the marker vocabulary is also applied directly.
        """
        for e in state.events:
            if e.event_type == EventType.RUN_STARTED and (e.payload or {}).get("injection_signal"):
                return "injection_signal"
        if state.input_text:
            return self._demotion_marker(state.input_text)
        return None

    def _partition_surfaces(self, state: RunState):
        """Split run content into (grounded_texts, taint_blocks, surfaces_used).

        Grounded — and why each is trusted:
          input_text     the task the principal actually asked for; this IS intent
          system_prompt  operator-authored config (fixed archive addresses etc.)
          tool output    a value the SYSTEM returned; the entry that keeps
                         open-destination agents silent
          retrieval      same argument: corpus data the system supplied
          memory value   written earlier from a channel the attacker can't reach

        Excluded, deliberately:
          llm output     circular — an injected model would ground itself
          earlier args   circular for the same reason
          untrusted mem  that's the taint set
          tool names     labels, not destinations
        """
        grounded: List[str] = []
        taint: List[dict] = []
        used: List[str] = []

        injected_input = self._input_is_injected(state)
        if state.input_text:
            if injected_input:
                taint.append(
                    {
                        "kind": "user_input",
                        "text": state.input_text,
                        "step_index": 0,
                        "matched_marker": injected_input,
                    }
                )
            else:
                grounded.append(state.input_text)
                used.append("input_text")

        if state.system_prompt:
            grounded.append(state.system_prompt)
            used.append("system_prompt")

        for tc in state.tool_calls:
            if not tc.output:
                continue
            marker = self._demotion_marker(tc.output)
            if marker:
                taint.append(
                    {
                        "kind": "tool_output",
                        "text": tc.output,
                        "tool_name": tc.tool_name,
                        "step_index": tc.step_index,
                        "matched_marker": marker,
                    }
                )
            else:
                grounded.append(tc.output)
                if "tool_output" not in used:
                    used.append("tool_output")

        for r in state.retrievals:
            if not r.content:
                continue
            marker = self._demotion_marker(r.content)
            if marker:
                taint.append(
                    {
                        "kind": "retrieval",
                        "text": r.content,
                        "index_name": r.index_name,
                        "step_index": r.step_index,
                        "matched_marker": marker,
                    }
                )
            else:
                grounded.append(r.content)
                if "retrieval" not in used:
                    used.append("retrieval")

        reads = [m for m in state.memory_events if m.op == "read"]
        for m in state.memory_events:
            if m.op != "written" or not m.value:
                continue
            if m.source in _UNTRUSTED_MEMORY_SOURCES:
                taint.append(
                    {
                        "kind": "memory_write",
                        "text": m.value,
                        "memory_key": m.key,
                        "memory_source": m.source,
                        "step_index": m.step_index,
                        "read_back": any(
                            r.key == m.key and r.step_index >= m.step_index for r in reads
                        ),
                        "matched_marker": self._demotion_marker(m.value),
                    }
                )
            else:
                grounded.append(m.value)
                if "memory" not in used:
                    used.append("memory")

        return grounded, taint, used

    def _contains(self, needle: str, haystack: str) -> bool:
        if self.CASE_SENSITIVE:
            return needle in haystack
        return needle.lower() in haystack.lower()

    # ── Cross-run taint (T6), fed by the worker ──────────────────────────────

    def evaluate_cross_run_memory_taint(self, destination: str, prior_writes: List[dict]):
        """Taint from a PRIOR run's memory write, for a key this run read back.

        `prior_writes` is supplied by the detector worker (see
        detector_svc/db.py::fetch_memory_writes) as dicts with key/value/source/
        run_id/step_index. Pure and injected, so it unit-tests without a database
        — the same shape DelegationLoopDetector.evaluate_delegation_cycle uses.

        Kept off on_run_completion deliberately: the in-path instance has no
        database, and a detector that silently needs one would be a trap.
        """
        for w in prior_writes or ():
            value = w.get("value") or ""
            if w.get("source") not in _UNTRUSTED_MEMORY_SOURCES:
                continue
            if not self._contains(destination, value):
                continue
            return {
                "kind": "memory_write",
                "memory_key": w.get("key"),
                "memory_source": w.get("source"),
                "read_back": True,  # the worker only queries keys this run read
                "cross_run": True,
                "origin_run_id": w.get("run_id"),
                "step_index": w.get("step_index"),
                "matched_marker": self._demotion_marker(value),
                "signal_id": None,  # reserved for deferred ancestor-signal taint
            }
        return None

    def collect_destinations(self, state: RunState):
        """(grounded, ungrounded) destination tuples for this run, as
        [(destination, destination_type, host), ...].

        Server-side helper for the novelty pass and the baseline write. A pure
        function of RunState rather than state stashed on the instance during
        on_run_completion(): detector instances are reused across runs, so a
        stashed list would leak one run's destinations into the next — and into
        another agent's, once the worker is sharded.
        """
        grounded_texts, _taint, _used = self._partition_surfaces(state)
        corpus = "\n".join(grounded_texts)
        corpus_cmp = corpus if self.CASE_SENSITIVE else corpus.lower()

        grounded: List[Tuple[str, str, str]] = []
        ungrounded: List[Tuple[str, str, str]] = []
        seen: set = set()
        for tc in state.tool_calls:
            if not self._in_scope(tc.tool_name):
                continue
            parsed = _parse_tool_args(tc.args)
            leaves = (
                _walk_arg_strings(parsed, self.MAX_DEPTH, self.MAX_NODES)
                if parsed is not None
                else [("<raw>", tc.args)]
            )
            for _path, text in leaves:
                for dest, dtype, host in self._destinations_in(text):
                    if dest in seen:
                        continue
                    seen.add(dest)
                    if _host_in_allowlist(host, self.ALLOWLISTED_DOMAINS):
                        continue
                    dest_cmp = dest if self.CASE_SENSITIVE else dest.lower()
                    (grounded if dest_cmp in corpus_cmp else ungrounded).append((dest, dtype, host))
        return grounded, ungrounded

    # ── Novelty mode, fed by the worker ──────────────────────────────────────

    def evaluate_novelty(self, grounded_destinations, baseline, baseline_runs: int):
        """Secondary mode for CLOSED-destination agents.

        Evaluates only destinations that PASSED grounding, so it can never
        double-fire with the provenance verdict. Returns None below
        MIN_BASELINE_RUNS — an immature baseline makes every destination look
        novel, which is exactly the alert storm that would get this switched off.
        """
        if not self.MODE or "novelty" not in self.MODE:
            return None
        if baseline_runs < self.MIN_BASELINE_RUNS:
            return None
        known = set(baseline or ())
        for dest, dtype, host in grounded_destinations or ():
            key = dest if dtype == "email" else _registrable_domain(host)
            if key in known:
                continue
            return {"destination": dest, "destination_type": dtype, "novelty_key": key}
        return None

    # ── Main ─────────────────────────────────────────────────────────────────

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        if not state.tool_calls:
            return None
        started_ns = time.perf_counter_ns()

        # Surface size guard. In-path this runs inside the agent process, and a
        # run carrying megabytes of captured tool output would cost real latency.
        # It aborts rather than scanning a TRUNCATED corpus: a short corpus makes
        # grounded destinations read as ungrounded, and a false positive in-path
        # can block a legitimate send via a stop policy. Fail open, not loud.
        surface_chars = (
            len(state.input_text or "")
            + len(state.system_prompt or "")
            + sum(len(tc.output or "") for tc in state.tool_calls)
            + sum(len(r.content or "") for r in state.retrievals)
            + sum(len(m.value or "") for m in state.memory_events)
        )
        if surface_chars > self.MAX_SURFACE_CHARS:
            return None

        grounded_texts, taint_blocks, surfaces_used = self._partition_surfaces(state)
        grounded_corpus = "\n".join(grounded_texts)
        # Case-fold the corpus and every taint block ONCE. Doing it inside
        # _contains() re-allocated the entire corpus per candidate, which turned
        # a 400-candidate run against a 500KB corpus into 200MB of transient
        # string churn.
        grounded_cmp = grounded_corpus if self.CASE_SENSITIVE else grounded_corpus.lower()
        for block in taint_blocks:
            raw = block.get("text") or ""
            block["_cmp"] = raw if self.CASE_SENSITIVE else raw.lower()

        # Fabrication stops the whole run when a tool output is missing, because
        # its claim is "no source exists" and an unseen output falsifies that.
        # This detector must not: with tool output rarely instrumented, aborting
        # would blind it on most real traffic — the traffic exfiltration happens
        # in. Missing outputs become a confidence penalty instead, and never
        # weaken taint, which is independent positive evidence.
        visibility_complete = not any(
            tc.success is not None and not tc.output for tc in state.tool_calls
        )

        best: Optional[Dict[str, Any]] = None
        candidate_count = 0
        ungrounded_count = 0
        truncated = False
        grounded_seen: List[Tuple[str, str, str]] = []

        for tc in state.tool_calls:
            if not self._in_scope(tc.tool_name):
                continue
            if time.perf_counter_ns() - started_ns > self.MAX_SCAN_NS:
                # Fail open. The server-side instance runs unscoped with a much
                # larger budget, so an in-path abort costs prevention, not
                # detection. Silent by design: this path is the user's agent
                # process, and neither an exception nor a log line belongs there.
                return None

            if len(tc.args) > self.MAX_ARGS_CHARS:
                # Truncating ARGS can only cause a missed detection, never a
                # false one — the exact opposite of truncating the grounded
                # corpus, which is why that one aborts instead. In-path this is
                # the right trade: prevention degrades to server-side detection.
                # (ast.literal_eval alone costs ~900µs on a 10KB arg, most of the
                # in-path budget, so this guard has to come before the parse.)
                leaves = [("<truncated>", tc.args[: self.MAX_ARGS_CHARS])]
            else:
                parsed = _parse_tool_args(tc.args)
                leaves = (
                    _walk_arg_strings(parsed, self.MAX_DEPTH, self.MAX_NODES)
                    if parsed is not None
                    else [("<raw>", tc.args)]
                )
            # Parsing a large structure is itself a real cost, so re-check before
            # walking it — otherwise a single 400-element arg blows the whole
            # in-path budget between two per-tool checks.
            if time.perf_counter_ns() - started_ns > self.MAX_SCAN_NS:
                return None

            for leaf_i, (path, text) in enumerate(leaves):
                if (leaf_i & 63) == 0 and time.perf_counter_ns() - started_ns > self.MAX_SCAN_NS:
                    return None
                for cand_i, (dest, dtype, host) in enumerate(self._destinations_in(text)):
                    # One leaf can hold hundreds of candidates (a 400-recipient
                    # send), so the budget has to be checked here too — the
                    # per-leaf check never fires when there is only one leaf.
                    if (
                        cand_i & 31
                    ) == 0 and time.perf_counter_ns() - started_ns > self.MAX_SCAN_NS:
                        return None
                    candidate_count += 1
                    if candidate_count > self.MAX_CANDIDATES_PER_RUN:
                        truncated = True
                        break
                    if _host_in_allowlist(host, self.ALLOWLISTED_DOMAINS):
                        continue
                    dest_cmp = dest if self.CASE_SENSITIVE else dest.lower()
                    if dest_cmp in grounded_cmp:
                        grounded_seen.append((dest, dtype, host))
                        continue

                    ungrounded_count += 1
                    taint = None
                    for block in taint_blocks:
                        if dest_cmp in block["_cmp"]:
                            taint = {k: v for k, v in block.items() if k not in ("text", "_cmp")}
                            break

                    severity = Severity.CRITICAL if taint else Severity.HIGH
                    confidence = 0.85 if taint else 0.65
                    if taint and taint.get("read_back"):
                        confidence = 0.92
                    if not visibility_complete:
                        confidence -= 0.15
                        if not taint:
                            severity = Severity.HIGH
                    if self._is_send_tool(tc.tool_name):
                        confidence += 0.05
                    confidence = round(max(0.30, min(0.95, confidence)), 4)

                    rank = (2 if severity == Severity.CRITICAL else 1, confidence)
                    if best is None or rank > best["rank"]:
                        best = {
                            "rank": rank,
                            "severity": severity,
                            "confidence": confidence,
                            "destination": dest,
                            "destination_type": dtype,
                            "destination_host": host,
                            "tool_name": tc.tool_name,
                            "tool_step": tc.step_index,
                            "arg_path": path,
                            "taint": taint,
                            "args_snippet": tc.args[:200],
                            "args_truncated": bool(
                                tc.args_length is not None and tc.args_length > len(tc.args)
                            ),
                        }
                if candidate_count > self.MAX_CANDIDATES_PER_RUN:
                    break
            if candidate_count > self.MAX_CANDIDATES_PER_RUN:
                break

        if best is None:
            return None

        return FailureSignal(
            failure_type=FailureType.UNGROUNDED_DESTINATION,
            severity=best["severity"],
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=best["tool_step"],
            confidence=best["confidence"],
            evidence={
                "destination": best["destination"],
                "destination_type": best["destination_type"],
                "destination_host": best["destination_host"],
                "tool_name": best["tool_name"],
                "tool_step": best["tool_step"],
                "arg_path": best["arg_path"],
                "grounding_verdict": "ungrounded",
                "grounded_surfaces": surfaces_used,
                "output_visibility": "complete" if visibility_complete else "partial",
                "taint_source": best["taint"],
                "detection_mode": "provenance",
                "baseline_size": None,
                "baseline_runs": None,
                "candidate_count": candidate_count,
                "ungrounded_count": ungrounded_count,
                "candidates_truncated": truncated,
                "args_truncated": best["args_truncated"],
                "args_snippet": best["args_snippet"],
            },
        )


# ── INSTRUMENTATION_DEGRADED ──────────────────────────────────────────────────

# Detectors whose only evidence is completion text or finish_reason. When a run
# is unmeasurable these do not fire, and their silence must not be read as a
# clean verdict — naming them in the signal is what turns "no findings" into
# "these specific checks could not run".
_TEXT_DEPENDENT_DETECTORS = (
    "EMPTY_LLM_RESPONSE",
    "PREMATURE_TERMINATION",
    "UNREAD_TOOL_ERROR",
    "LLM_TRUNCATION_LOOP",
    "SILENT_TRUNCATION",
    "RUNAWAY_ITERATION",
    "FIRST_STEP_FAILURE",
)


class InstrumentationDegradedDetector(BaseDetector):
    """The SDK could not measure this run's LLM calls.

    This is not a statement about the agent. It is a statement about the
    telemetry, and it exists because the detector layer previously could not
    distinguish "I looked and found nothing wrong" from "I could not look".
    Those two produce identical output — no signal — and the second one silently
    disables a third of the battery.

    Fires on either of two conditions:

    1. Any call carries an ``instrumentation_degraded`` marker, meaning an
       extractor in auto.py could not read the provider's response object and
       said so instead of substituting a plausible default.

    2. Every call matches the degraded fingerprint — zero output, zero
       completion tokens, non-zero latency. A call that measurably took time and
       measurably produced nothing, repeated across an entire run, is a broken
       pipeline rather than a model that answered nothing every single turn.
       Requires MIN_CALLS so a one-call run with a genuinely empty response is
       left to EMPTY_LLM_RESPONSE, which is the right detector for that.

    Deliberately does NOT fire on a run whose text is merely absent because
    DUNETRACE_OMIT_LLM_OUTPUT_TEXT is set: that omission is a deliberate
    operator choice, output_length is still transmitted, and calling it
    degraded would report every privacy-conscious deployment as broken.

    Tunable: MIN_CALLS (default 2).
    """

    name = "INSTRUMENTATION_DEGRADED"
    SEVERITY = Severity.MEDIUM
    MIN_CALLS = 2

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        calls = state.llm_calls
        if not calls:
            return None

        marked = [c for c in calls if _is_unmeasurable(c)]
        # The `is not None` filter is redundant at runtime — _is_unmeasurable
        # already guarantees it — but it is what narrows Optional[str] to str
        # for the type checker, which cannot see through that helper.
        shapes = sorted(
            {c.instrumentation_degraded for c in marked if c.instrumentation_degraded is not None}
        )

        if marked:
            reason = "unreadable_response_shape"
            affected = marked
        elif len(calls) >= self.MIN_CALLS and all(_matches_degraded_fingerprint(c) for c in calls):
            # No marker, but every call is structurally blank. This catches a
            # break upstream of our extractors — a framework that hands us an
            # already-emptied response — where nothing had a chance to mark it.
            reason = "all_calls_structurally_blank"
            affected = list(calls)
        else:
            return None

        return FailureSignal(
            failure_type=FailureType.INSTRUMENTATION_DEGRADED,
            severity=self.SEVERITY,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=affected[0].step_index,
            confidence=0.95 if marked else 0.8,
            evidence={
                "reason": reason,
                # The shape that defeated extraction, e.g.
                # "openai_response_shape:LegacyAPIResponse". Empty for the
                # fingerprint path, where nothing identified itself.
                "unreadable_shapes": shapes,
                "affected_calls": len(affected),
                "total_llm_calls": len(calls),
                "providers": sorted({c.provider for c in affected if c.provider}),
                "models": sorted({c.model for c in affected if c.model}),
                "first_step": affected[0].step_index,
                # What this run's silence does NOT mean.
                "suppressed_detectors": list(_TEXT_DEPENDENT_DETECTORS),
                "unmeasurable": [
                    "llm.responded.output",
                    "llm.responded.finish_reason",
                ],
            },
        )


# ── Registry ──────────────────────────────────────────────────────────────────

TIER1_DETECTORS: List[BaseDetector] = [
    InstrumentationDegradedDetector(),
    OversizedToolArgumentsDetector(),
    ToolLoopDetector(),
    ToolThrashingDetector(),
    ToolAvoidanceDetector(),
    GoalAbandonmentDetector(),
    RagEmptyRetrievalDetector(),
    ExcessiveRetrievalDetector(),
    LlmTruncationLoopDetector(),
    SilentTruncationDetector(),
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
    PrematureTerminationDetector(),
    UnreadToolErrorDetector(),
    ToolArgumentFabricationDetector(),
    RetrievedContentInjectionDetector(),
    AgentHandoffFailureDetector(),
    RunawayIterationDetector(),
    ModelFallbackDriftDetector(),
    MemoryPoisonedDetector(),
    # In-path instance, deliberately NOT the class defaults. Benchmarking against
    # realistic payloads (100KB+ tool outputs, 8-level nesting, 400 candidates in
    # one arg) put unscoped scanning 6–9x over MAX_COST_NS, and the cost is
    # surface-side — case-folding and marker-scanning captured tool output —
    # so limiting which TOOLS are scanned barely moves it. What bounds it is
    # bounding the INPUTS, plus a hard abort. All three fail open: the run is
    # skipped, the server-side instance (class defaults, no caps) still catches
    # it, and the user's agent never pays more than the budget.
    UngroundedDestinationDetector(
        MAX_SCAN_NS=1_000_000,  # == BaseDetector.MAX_COST_NS
        MAX_SURFACE_CHARS=30_000,
        MAX_ARGS_CHARS=4_000,
        TOOL_NAME_SCOPE=UngroundedDestinationDetector.SEND_TOOL_PATTERNS,
    ),
    # PromptInjectionDetector and HandoffContextLossDetector are handled
    # separately — the former needs raw input, the latter needs a second
    # run's data, which no detector in this list ever gets (see their
    # docstrings)
]

PROMPT_INJECTION_DETECTOR = PromptInjectionDetector()


# ── Cost budget tracking ────────────────────────────────────────────────────────
#
# Soft enforcement of BaseDetector.MAX_COST_NS. A single detector call exceeding
# its budget already logs a warning (see run_detectors() below); this adds three
# things on top, scoped as a stopgap rather than a full metrics pipeline:
#   1. that warning is rate-limited (once per detector per minute) instead of
#      firing on every single over-budget call
#   2. a rolling P50/P95/P99 (last _COST_WINDOW_S) is tracked per detector name
#      and logged alongside the rate-limited warning
#   3. a detector whose P99 stays over budget for the full window is downgraded:
#      run_detectors(..., context="runtime") skips it, so a detector that grew
#      from O(1) to O(n) stops silently degrading the SDK's per-step hot path
#      (see run_context.py's signal-trigger policy check). context="analytics"
#      (the default — the server-side detector worker, replay, and the OTel
#      integration's once-per-run call all use this) never skips a downgraded
#      detector; the run is still fully evaluated there.
#
# time.monotonic()'s epoch is unspecified (often time-since-boot on Linux), so
# "not yet warned"/"not yet downgraded" is tracked with None, never a 0.0
# sentinel — comparing now - 0.0 >= threshold silently fails on a freshly
# booted host where monotonic() itself is still small.

_COST_WINDOW_S = 300.0  # 5 minutes
_COST_SAMPLE_CAP = 1000  # bounds memory regardless of call frequency
_COST_WARNING_RATE_LIMIT_S = 60.0
_COST_MIN_SAMPLES_FOR_DOWNGRADE = 5  # avoid downgrading off one or two slow calls


class _DetectorCostTracker:
    """Per-detector-name rolling cost history. One instance per name, shared
    across every run_detectors() call in this process (not per RunState)."""

    __slots__ = ("samples", "last_warning_at", "downgraded_at")

    def __init__(self) -> None:
        self.samples: Deque[tuple[float, int]] = deque(maxlen=_COST_SAMPLE_CAP)
        self.last_warning_at: Optional[float] = None
        self.downgraded_at: Optional[float] = None  # None = not downgraded

    def record(self, elapsed_ns: int) -> None:
        self.samples.append((time.monotonic(), elapsed_ns))

    def percentiles(self, window_s: float = _COST_WINDOW_S):
        """Return (p50, p95, p99, count) in ns over the last window_s, or None if empty."""
        cutoff = time.monotonic() - window_s
        recent = sorted(ns for ts, ns in self.samples if ts >= cutoff)
        if not recent:
            return None
        n = len(recent)

        def pct(p: float) -> int:
            return recent[min(n - 1, int(n * p))]

        return pct(0.50), pct(0.95), pct(0.99), n


_cost_trackers: dict[str, _DetectorCostTracker] = {}


def _get_cost_tracker(name: str) -> _DetectorCostTracker:
    tracker = _cost_trackers.get(name)
    if tracker is None:
        tracker = _DetectorCostTracker()
        _cost_trackers[name] = tracker
    return tracker


def reset_cost_downgrade(name: str) -> None:
    """Manually re-enable a detector auto-downgraded to analytics-only after its
    P99 stayed over MAX_COST_NS for a full _COST_WINDOW_S. See run_detectors()."""
    tracker = _cost_trackers.get(name)
    if tracker is not None:
        tracker.downgraded_at = None


def _record_cost_and_maybe_warn(detector: BaseDetector, elapsed_ns: int, run_id: str) -> None:
    tracker = _get_cost_tracker(detector.name)
    tracker.record(elapsed_ns)
    if elapsed_ns <= detector.MAX_COST_NS:
        return

    now = time.monotonic()

    # audit Finding 29: evaluate the downgrade decision on EVERY over-budget call,
    # BEFORE (and independent of) the rate-limited warning below. Previously this
    # was nested inside the once-per-minute warning block, so after the very first
    # warning (fired when n=1, too few samples to downgrade) the rate-limit
    # short-circuited the check for ~60s — meaning a detector that blew its budget
    # on the hot path was NOT downgraded promptly. Percentiles are only computed
    # once enough samples exist and while not yet downgraded, so this adds no cost
    # to well-behaved detectors or after a downgrade has fired.
    if tracker.downgraded_at is None and len(tracker.samples) >= _COST_MIN_SAMPLES_FOR_DOWNGRADE:
        stats = tracker.percentiles()
        if stats is not None:
            p50, p95, p99, n = stats
            if p99 > detector.MAX_COST_NS and n >= _COST_MIN_SAMPLES_FOR_DOWNGRADE:
                tracker.downgraded_at = now
                logger.warning(
                    "Detector %s P99 (%dns) has exceeded its cost budget (%dns) for the "
                    "last %ds — downgrading to analytics-only. It will be skipped by "
                    "run_detectors(context='runtime') until manually re-enabled via "
                    "dunetrace.detectors.reset_cost_downgrade(%r).",
                    detector.name,
                    p99,
                    detector.MAX_COST_NS,
                    int(_COST_WINDOW_S),
                    detector.name,
                )

    if (
        tracker.last_warning_at is not None
        and now - tracker.last_warning_at < _COST_WARNING_RATE_LIMIT_S
    ):
        return
    tracker.last_warning_at = now

    logger.warning(
        "Detector %s exceeded its cost budget: %dns > %dns (run_id=%s)",
        detector.name,
        elapsed_ns,
        detector.MAX_COST_NS,
        run_id,
    )

    stats = tracker.percentiles()
    if stats is not None:
        p50, p95, p99, n = stats
        logger.info(
            "Detector %s cost stats (last %ds, n=%d): p50=%dns p95=%dns p99=%dns budget=%dns",
            detector.name,
            int(_COST_WINDOW_S),
            n,
            p50,
            p95,
            p99,
            detector.MAX_COST_NS,
        )


def run_detectors(
    state: RunState,
    detectors: Optional[List[BaseDetector]] = None,
    context: str = "analytics",
) -> List[FailureSignal]:
    """Run all detectors against a run state. Pass a custom list to use production-tuned
    parameters; defaults to TIER1_DETECTORS if omitted. Returns one FailureSignal per
    triggered detector, or an empty list if nothing fired.

    context distinguishes the SDK's per-step runtime hot path ("runtime" — see
    run_context.py, only reachable when a trigger="signal" policy is configured)
    from every other call site ("analytics" — the server-side detector worker,
    on-demand replay, and the OTel integration's once-per-run call). A detector
    auto-downgraded for exceeding its cost budget (see _record_cost_and_maybe_warn)
    is skipped only under context="runtime" — analytics calls always evaluate
    every detector, since they aren't the repeated-per-step path this protects.

    Each detector's on_run_completion() is timed against its MAX_COST_NS budget —
    exceeding it logs a rate-limited warning but never raises or drops the signal.
    Signals with non-dict evidence are similarly logged (not dropped) — evidence is
    what root-cause consumers downstream read, so a malformed shape there is worth
    surfacing even though it isn't this function's job to fix.
    """
    active = detectors if detectors is not None else TIER1_DETECTORS
    signals = []
    for detector in active:
        # context="policy" deliberately does NOT honour the cost downgrade.
        # A trigger="signal" policy is a safety control — the product's promise
        # that a tool loop is stopped before it burns the budget — and a safety
        # control must not switch itself off because it was slow. Cost is shed
        # by scope instead: run_context passes only the detectors the active
        # signal policies actually reference. "runtime" keeps the old
        # shed-on-cost behaviour for any non-enforcement hot path.
        if context == "runtime":
            tracker = _cost_trackers.get(detector.name)
            if tracker is not None and tracker.downgraded_at is not None:
                continue

        t0 = time.perf_counter_ns()
        try:
            signal = detector.on_run_completion(state)
        except Exception:
            # One detector must not cost the run its other 30. Failure isolation
            # was already applied at plugin *construction*
            # (detector_svc/detectors.py::_build_plugin_detectors), which made
            # this gap easy to miss: a registered plugin or pack detector that
            # raises on an unusual RunState — an empty tool_calls list, a None
            # field it didn't expect — used to abort the whole battery and
            # record the run as clean. ERROR, not debug: a raising detector is a
            # bug worth seeing, it just isn't worth losing detection over.
            logger.error(
                "Detector %s raised on run %s — skipping it for this run",
                detector.name,
                state.run_id,
                exc_info=True,
            )
            continue
        elapsed_ns = time.perf_counter_ns() - t0
        _record_cost_and_maybe_warn(detector, elapsed_ns, state.run_id)

        if signal:
            if not isinstance(signal.evidence, dict):
                logger.warning(
                    "Detector %s returned a signal with non-dict evidence (%s) — "
                    "root-cause consumers expect evidence: dict.",
                    detector.name,
                    type(signal.evidence).__name__,
                )
            signals.append(signal)
    return signals
