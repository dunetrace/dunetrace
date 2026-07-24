"""
Phase 4: the detector worker emits signal/policy spans into a run's OTel trace.

Tests _emit_otel_findings in isolation with an in-memory tracer (no DB, no real
OTLP endpoint). Verifies it is a no-op when OTel is off, emits signal + policy
spans when on, and skips policy re-emission on reprocess.
"""

from __future__ import annotations

import time
import unittest
import uuid

import pytest

try:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from dunetrace.integrations.otel import _trace_id
    from dunetrace.models import FailureSignal, FailureType, Severity

    import detector_svc.worker as worker

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

if not _OTEL_AVAILABLE:
    raise unittest.SkipTest("opentelemetry not installed — skipping detector OTel tests")


def _tracer():
    mem = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(mem))
    return provider.get_tracer("dunetrace"), mem


def _signal(run_id: str) -> FailureSignal:
    return FailureSignal(
        failure_type=FailureType.TOOL_LOOP,
        severity=Severity.HIGH,
        run_id=run_id,
        agent_id="agent-1",
        agent_version="v1",
        step_index=3,
        confidence=0.9,
        evidence={"repeats": 3},
    )


def _policy_event(run_id: str) -> dict:
    return {
        "event_type": "policy.triggered",
        "run_id": run_id,
        "timestamp": time.time(),
        "payload": {
            "policy_name": "stopper",
            "action_type": "stop",
            "trigger": "cost_usd",
            "value": 1.0,
        },
    }


class _Cfg:
    capture_content = True


def test_noop_when_otel_disabled(monkeypatch):
    monkeypatch.setattr("dunetrace.otel.get_tracer", lambda name="dunetrace": None)
    run_id = str(uuid.uuid4())
    # Must not raise even though a tracer is unavailable.
    worker._emit_otel_findings(run_id, "org_1", [_signal(run_id)], [_policy_event(run_id)], False)


def test_emits_signal_and_policy_spans(monkeypatch):
    tracer, mem = _tracer()
    monkeypatch.setattr("dunetrace.otel.get_tracer", lambda name="dunetrace": tracer)
    monkeypatch.setattr("dunetrace.otel.active_config", lambda: _Cfg())

    run_id = str(uuid.uuid4())
    worker._emit_otel_findings(
        run_id, "org_1", [_signal(run_id)], [_policy_event(run_id)], is_reprocess=False
    )

    spans = mem.get_finished_spans()
    names = {s.name for s in spans}
    assert "dunetrace.signal.TOOL_LOOP" in names
    assert "dunetrace.policy.stop" in names
    # All findings land in the run's own trace.
    assert all(s.context.trace_id == _trace_id(run_id) for s in spans)


def test_reprocess_skips_policy_but_keeps_signals(monkeypatch):
    tracer, mem = _tracer()
    monkeypatch.setattr("dunetrace.otel.get_tracer", lambda name="dunetrace": tracer)
    monkeypatch.setattr("dunetrace.otel.active_config", lambda: _Cfg())

    run_id = str(uuid.uuid4())
    worker._emit_otel_findings(
        run_id, "org_1", [_signal(run_id)], [_policy_event(run_id)], is_reprocess=True
    )

    names = {s.name for s in mem.get_finished_spans()}
    assert "dunetrace.signal.TOOL_LOOP" in names
    assert "dunetrace.policy.stop" not in names  # policies not re-emitted on reprocess
