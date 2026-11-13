"""
Single source of truth for the three wire-format enums: EventType, Severity and
FailureType.

This module is deliberately *not* an enum. It is plain data — three ordered lists
of ``(NAME, value, comment)`` tuples — so a script can read it with zero
dependencies and emit the two real ``Enum`` modules from it:

  - ``packages/sdk-py/dunetrace/_enums.py``            (re-exported by dunetrace.models)
  - ``packages/schemas-py/dunetrace_schemas/enums.py``

The SDK must not import ``dunetrace_schemas`` (zero-dependency by design) and the
schemas package must not import the SDK, so the two enum modules cannot share an
import. They are generated from here instead: edit THIS file, then run

    python scripts/gen_enums.py

``python scripts/gen_enums.py --check`` (CI, docs-consistency job) fails when
either generated module is stale. ``packages/schemas-py/tests/test_sdk_parity.py``
remains as the backstop that asserts the two enums agree at import time.

``comment`` is either ``None`` or a string emitted as ``#`` lines immediately
*before* the member — use it to introduce a group of members (embedded newlines
become separate comment lines). Member order is preserved in the output.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

EnumMember = Tuple[str, str, Optional[str]]

EVENT_TYPE_MEMBERS: List[EnumMember] = [
    ("RUN_STARTED", "run.started", None),
    ("RUN_COMPLETED", "run.completed", None),
    ("RUN_ERRORED", "run.errored", None),
    ("LLM_CALLED", "llm.called", None),
    ("LLM_RESPONDED", "llm.responded", None),
    ("TOOL_CALLED", "tool.called", None),
    ("TOOL_RESPONDED", "tool.responded", None),
    ("RETRIEVAL_CALLED", "retrieval.called", None),
    ("RETRIEVAL_RESPONDED", "retrieval.responded", None),
    ("EXTERNAL_SIGNAL", "external.signal", None),
    ("POLICY_TRIGGERED", "policy.triggered", None),
    (
        "POLICY_EVALUATED",
        "policy.evaluated",
        "Policy evaluation observability (rate-limited): one record per policy\n"
        "evaluation, shipped via the normal transport but routed by ingest into the\n"
        "policy_evaluations table rather than stored as a run event.",
    ),
    (
        "TRANSCRIPTION_RECEIVED",
        "transcription.received",
        'Voice-agent events (detector pack "voice"). Additive: emitted only by the\n'
        "optional voice-specific RunContext helpers; no built-in detector reads\n"
        "them until the voice pack is active.",
    ),
    ("TTS_GENERATED", "tts.generated", None),
    ("VOICE_ACTIVITY_DETECTED", "voice_activity.detected", None),
    ("TURN_TAKING", "turn_taking.changed", None),
    ("RECORDING_AVAILABLE", "recording.available", None),
    (
        "APPROVAL_REQUESTED",
        "approval.requested",
        "Human-in-the-loop approval events (Capability 2). Emitted around a\n"
        "require_approval policy gate.",
    ),
    ("APPROVAL_GRANTED", "approval.granted", None),
    ("APPROVAL_DENIED", "approval.denied", None),
    ("APPROVAL_TIMEOUT", "approval.timeout", None),
    (
        "MEMORY_WRITTEN",
        "memory.written",
        "Agent memory channel (Capability 1). Content written to / read from / cleared\n"
        "in agent memory, so a run's persisted state is observable.",
    ),
    ("MEMORY_READ", "memory.read", None),
    ("MEMORY_CLEARED", "memory.cleared", None),
]

SEVERITY_MEMBERS: List[EnumMember] = [
    ("CRITICAL", "CRITICAL", None),
    ("HIGH", "HIGH", None),
    ("MEDIUM", "MEDIUM", None),
    ("LOW", "LOW", None),
]

FAILURE_TYPE_MEMBERS: List[EnumMember] = [
    ("TOOL_LOOP", "TOOL_LOOP", None),
    ("TOOL_THRASHING", "TOOL_THRASHING", None),
    ("SCATTERSHOT_TOOL_USE", "SCATTERSHOT_TOOL_USE", None),
    ("TOOL_AVOIDANCE", "TOOL_AVOIDANCE", None),
    ("GOAL_ABANDONMENT", "GOAL_ABANDONMENT", None),
    ("PROMPT_INJECTION_SIGNAL", "PROMPT_INJECTION_SIGNAL", None),
    ("RAG_EMPTY_RETRIEVAL", "RAG_EMPTY_RETRIEVAL", None),
    ("EXCESSIVE_RETRIEVAL", "EXCESSIVE_RETRIEVAL", None),
    ("LLM_TRUNCATION_LOOP", "LLM_TRUNCATION_LOOP", None),
    ("SILENT_TRUNCATION", "SILENT_TRUNCATION", None),
    ("CONTEXT_BLOAT", "CONTEXT_BLOAT", None),
    ("SLOW_STEP", "SLOW_STEP", None),
    ("RETRY_STORM", "RETRY_STORM", None),
    ("EMPTY_LLM_RESPONSE", "EMPTY_LLM_RESPONSE", None),
    ("STEP_COUNT_INFLATION", "STEP_COUNT_INFLATION", None),
    ("CASCADING_TOOL_FAILURE", "CASCADING_TOOL_FAILURE", None),
    ("FIRST_STEP_FAILURE", "FIRST_STEP_FAILURE", None),
    ("USER_DISSATISFACTION", "USER_DISSATISFACTION", None),
    ("INTENT_MISALIGNMENT", "INTENT_MISALIGNMENT", None),
    ("REASONING_STALL", "REASONING_STALL", None),
    # Historical: the wire value keeps its _PROXY suffix; the member name does not.
    ("CONFIDENT_HALLUCINATION", "CONFIDENT_HALLUCINATION_PROXY", None),
    ("POLICY_VIOLATION", "POLICY_VIOLATION", None),
    ("COST_SPIKE", "COST_SPIKE", None),
    ("SESSION_LATENCY", "SESSION_LATENCY", None),
    ("PREMATURE_TERMINATION", "PREMATURE_TERMINATION", None),
    ("UNREAD_TOOL_ERROR", "UNREAD_TOOL_ERROR", None),
    ("TOOL_ARGUMENT_FABRICATION", "TOOL_ARGUMENT_FABRICATION", None),
    ("RETRIEVED_CONTENT_INJECTION", "RETRIEVED_CONTENT_INJECTION", None),
    ("HANDOFF_CONTEXT_LOSS", "HANDOFF_CONTEXT_LOSS", None),
    ("AGENT_HANDOFF_FAILURE", "AGENT_HANDOFF_FAILURE", None),
    ("RUNAWAY_ITERATION", "RUNAWAY_ITERATION", None),
    ("MODEL_FALLBACK_DRIFT", "MODEL_FALLBACK_DRIFT", None),
    ("MEMORY_POISONING", "MEMORY_POISONING", None),
    ("DELEGATION_LOOP", "DELEGATION_LOOP", None),
    ("OVERSIZED_TOOL_ARGUMENTS", "OVERSIZED_TOOL_ARGUMENTS", None),
    ("UNGROUNDED_DESTINATION", "UNGROUNDED_DESTINATION", None),
    ("UNRESOLVED_AMBIGUITY", "UNRESOLVED_AMBIGUITY", None),
    (
        "INSTRUMENTATION_DEGRADED",
        "INSTRUMENTATION_DEGRADED",
        "Not an agent failure: the SDK could not measure the run. Kept in the same\n"
        "enum so it travels the existing signal pipeline, but it describes the\n"
        "telemetry, not the agent.",
    ),
    ("CUSTOM", "CUSTOM", "Sentinel for user-defined custom detectors."),
]

# (class name, members) in the order the generated modules define them.
ENUMS: List[Tuple[str, List[EnumMember]]] = [
    ("EventType", EVENT_TYPE_MEMBERS),
    ("Severity", SEVERITY_MEMBERS),
    ("FailureType", FAILURE_TYPE_MEMBERS),
]
