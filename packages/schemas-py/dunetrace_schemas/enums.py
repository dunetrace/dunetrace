"""
Canonical enums for the Dunetrace wire format.

These mirror `dunetrace.models` (the SDK) value-for-value but are defined
independently — this package has no import dependency on the SDK, and the SDK
has no import dependency on this package. `tests/test_sdk_parity.py` asserts
the two stay in sync; that test is the enforcement mechanism, not an import.
"""

from __future__ import annotations

from enum import Enum


class EventType(str, Enum):
    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_ERRORED = "run.errored"
    LLM_CALLED = "llm.called"
    LLM_RESPONDED = "llm.responded"
    TOOL_CALLED = "tool.called"
    TOOL_RESPONDED = "tool.responded"
    RETRIEVAL_CALLED = "retrieval.called"
    RETRIEVAL_RESPONDED = "retrieval.responded"
    EXTERNAL_SIGNAL = "external.signal"
    POLICY_TRIGGERED = "policy.triggered"
    # Policy evaluation observability record (rate-limited). Routed by ingest into
    # policy_evaluations, not stored as a run event. Kept value-for-value in sync
    # with dunetrace.models.EventType — see test_sdk_parity.py.
    POLICY_EVALUATED = "policy.evaluated"
    # Voice-agent events (detector pack "voice"). Kept value-for-value in sync
    # with dunetrace.models.EventType — see test_sdk_parity.py.
    TRANSCRIPTION_RECEIVED = "transcription.received"
    TTS_GENERATED = "tts.generated"
    VOICE_ACTIVITY_DETECTED = "voice_activity.detected"
    TURN_TAKING = "turn_taking.changed"
    # Human-in-the-loop approval events (Capability 2). Kept value-for-value in
    # sync with dunetrace.models.EventType — see test_sdk_parity.py.
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_GRANTED = "approval.granted"
    APPROVAL_DENIED = "approval.denied"
    APPROVAL_TIMEOUT = "approval.timeout"
    # Agent memory channel (Capability 1). Kept value-for-value in sync with
    # dunetrace.models.EventType — see test_sdk_parity.py.
    MEMORY_WRITTEN = "memory.written"
    MEMORY_READ = "memory.read"
    MEMORY_CLEARED = "memory.cleared"


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class FailureType(str, Enum):
    TOOL_LOOP = "TOOL_LOOP"
    TOOL_THRASHING = "TOOL_THRASHING"
    TOOL_AVOIDANCE = "TOOL_AVOIDANCE"
    GOAL_ABANDONMENT = "GOAL_ABANDONMENT"
    PROMPT_INJECTION_SIGNAL = "PROMPT_INJECTION_SIGNAL"
    RAG_EMPTY_RETRIEVAL = "RAG_EMPTY_RETRIEVAL"
    EXCESSIVE_RETRIEVAL = "EXCESSIVE_RETRIEVAL"
    LLM_TRUNCATION_LOOP = "LLM_TRUNCATION_LOOP"
    SILENT_TRUNCATION = "SILENT_TRUNCATION"
    CONTEXT_BLOAT = "CONTEXT_BLOAT"
    SLOW_STEP = "SLOW_STEP"
    RETRY_STORM = "RETRY_STORM"
    EMPTY_LLM_RESPONSE = "EMPTY_LLM_RESPONSE"
    STEP_COUNT_INFLATION = "STEP_COUNT_INFLATION"
    CASCADING_TOOL_FAILURE = "CASCADING_TOOL_FAILURE"
    FIRST_STEP_FAILURE = "FIRST_STEP_FAILURE"
    USER_DISSATISFACTION = "USER_DISSATISFACTION"
    INTENT_MISALIGNMENT = "INTENT_MISALIGNMENT"
    REASONING_STALL = "REASONING_STALL"
    CONFIDENT_HALLUCINATION = "CONFIDENT_HALLUCINATION_PROXY"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    COST_SPIKE = "COST_SPIKE"
    SESSION_LATENCY = "SESSION_LATENCY"
    PREMATURE_TERMINATION = "PREMATURE_TERMINATION"
    UNREAD_TOOL_ERROR = "UNREAD_TOOL_ERROR"
    TOOL_ARGUMENT_FABRICATION = "TOOL_ARGUMENT_FABRICATION"
    RETRIEVED_CONTENT_INJECTION = "RETRIEVED_CONTENT_INJECTION"
    HANDOFF_CONTEXT_LOSS = "HANDOFF_CONTEXT_LOSS"
    AGENT_HANDOFF_FAILURE = "AGENT_HANDOFF_FAILURE"
    RUNAWAY_ITERATION = "RUNAWAY_ITERATION"
    MODEL_FALLBACK_DRIFT = "MODEL_FALLBACK_DRIFT"
    MEMORY_POISONING = "MEMORY_POISONING"
    DELEGATION_LOOP = "DELEGATION_LOOP"
    CUSTOM = "CUSTOM"  # sentinel for user-defined custom detectors
