"""
Regression suite for the instrumentation-provenance work.

The incident: a LangGraph + ChatOpenAI agent using dt.auto_instrument(["openai"])
emitted llm.responded events reading {"output": "", "output_length": 0,
"completion_tokens": 0, "finish_reason": "stop"} on every turn, including turns
where the model had produced a real multi-hundred-character answer.

Cause: langchain_openai calls client.with_raw_response.create(), so
Completions.create returned a LegacyAPIResponse rather than a ChatCompletion.
The extractors hit their fallback branches and substituted ("", "stop") — which
is byte-for-byte EmptyLlmResponseDetector's trigger condition. A shape mismatch
in instrumentation became a HIGH-severity behavioural alert on 100% of runs, and
nothing logged, warned, or degraded.

Every test here fails against the pre-provenance code.
"""

from __future__ import annotations

import json
import sqlite3
import unittest

from dunetrace.detectors import (
    EmptyLlmResponseDetector,
    InstrumentationDegradedDetector,
    LlmTruncationLoopDetector,
    RunawayIterationDetector,
    run_detectors,
)
from dunetrace.models import AgentEvent, EventType, FailureType, LlmCall, RunState


class _LegacyAPIResponse:
    """The real shape from the incident.

    openai's LegacyAPIResponse wraps the raw HTTP response and exposes .parse();
    it has no .choices and no .usage. Reproduced structurally rather than
    imported so the test does not depend on an openai version that still ships
    the class.
    """

    def __init__(self) -> None:
        self.http_response = object()

    def parse(self):  # pragma: no cover - never called; that IS the bug
        raise AssertionError("the buggy path never called .parse()")


class _RealChatCompletion:
    """A genuinely empty completion: shape read fine, model returned nothing."""

    def __init__(self) -> None:
        msg = type("M", (), {"content": ""})()
        choice = type("C", (), {"finish_reason": "stop", "message": msg})()
        self.choices = [choice]
        self.usage = type("U", (), {"prompt_tokens": 812, "completion_tokens": 0})()


def _run_with(calls, events=None):
    state = RunState(run_id="r1", agent_id="a1", agent_version="v1")
    state.llm_calls.extend(calls)
    state.events.extend(events or [])
    return state


# ── Acceptance 1: the regression ─────────────────────────────────────────────


class TestUnreadableShapeDoesNotFabricate(unittest.TestCase):
    def setUp(self):
        from dunetrace import auto

        auto._WARNED_SHAPES.clear()

    def test_extractors_return_none_not_a_plausible_default(self):
        from dunetrace.auto import _openai_content, _openai_finish_reason

        resp = _LegacyAPIResponse()
        # Pre-change these returned "stop" and "" — the fabricated pair.
        self.assertIsNone(_openai_finish_reason(resp))
        self.assertIsNone(_openai_content(resp))

    def test_emit_records_degraded_marker_and_omits_finish_reason(self):
        from dunetrace.auto import _emit_openai_response

        emitted = {}

        class _Run:
            def llm_responded(self, **kw):
                emitted.update(kw)

        _emit_openai_response(_Run(), _LegacyAPIResponse(), 0.0)
        self.assertIsNone(emitted["finish_reason"])
        self.assertIsNone(emitted["output"])
        self.assertEqual(
            emitted["instrumentation_degraded"],
            "openai_response_shape:_LegacyAPIResponse",
        )

    def test_empty_llm_response_does_not_fire_on_an_unreadable_shape(self):
        """THE regression test. This is the false positive that fired 16/16."""
        calls = [
            LlmCall(
                model="gpt-4o",
                prompt_tokens=800,
                finish_reason=None,
                latency_ms=900,
                step_index=i,
                timestamp=float(i),
                output_length=0,
                completion_tokens=0,
                instrumentation_degraded="openai_response_shape:LegacyAPIResponse",
            )
            for i in range(1, 4)
        ]
        self.assertIsNone(EmptyLlmResponseDetector().on_run_completion(_run_with(calls)))

    def test_one_warning_per_process_per_shape(self):
        from dunetrace import auto

        with self.assertLogs("dunetrace.auto", level="WARNING") as first:
            auto._degraded_marker("openai", _LegacyAPIResponse())
        self.assertIn("_LegacyAPIResponse", first.output[0])
        # Second occurrence of the same shape must not log again.
        with self.assertLogs("dunetrace.auto", level="WARNING") as second:
            auto._degraded_marker("openai", _LegacyAPIResponse())
            auto.logger.warning("sentinel")
        self.assertEqual(len(second.output), 1)
        self.assertIn("sentinel", second.output[0])


# ── Acceptance 2: exactly one INSTRUMENTATION_DEGRADED naming the shape ──────


class TestInstrumentationDegradedSignal(unittest.TestCase):
    def _degraded_run(self):
        calls = [
            LlmCall(
                model="gpt-4o",
                prompt_tokens=800,
                finish_reason=None,
                latency_ms=900,
                step_index=i,
                timestamp=float(i),
                output_length=0,
                completion_tokens=0,
                provider="openai",
                instrumentation_degraded="openai_response_shape:LegacyAPIResponse",
            )
            for i in range(1, 4)
        ]
        return _run_with(calls)

    def test_exactly_one_signal_naming_the_unreadable_shape(self):
        signals = run_detectors(self._degraded_run())
        degraded = [s for s in signals if s.failure_type == FailureType.INSTRUMENTATION_DEGRADED]
        self.assertEqual(len(degraded), 1)
        self.assertIn(
            "openai_response_shape:LegacyAPIResponse",
            degraded[0].evidence["unreadable_shapes"],
        )

    def test_signal_names_what_was_unmeasurable_and_what_it_suppressed(self):
        sig = InstrumentationDegradedDetector().on_run_completion(self._degraded_run())
        self.assertIn("llm.responded.output", sig.evidence["unmeasurable"])
        self.assertIn("llm.responded.finish_reason", sig.evidence["unmeasurable"])
        self.assertIn("EMPTY_LLM_RESPONSE", sig.evidence["suppressed_detectors"])
        self.assertEqual(sig.evidence["providers"], ["openai"])

    def test_empty_llm_response_is_replaced_not_merely_silenced(self):
        """The whole point: the run still produces a finding, but one that
        names the telemetry rather than blaming the agent."""
        types = {s.failure_type for s in run_detectors(self._degraded_run())}
        self.assertIn(FailureType.INSTRUMENTATION_DEGRADED, types)
        self.assertNotIn(FailureType.EMPTY_LLM_RESPONSE, types)

    def test_healthy_run_produces_no_degraded_signal(self):
        calls = [
            LlmCall(
                model="gpt-4o",
                prompt_tokens=800,
                finish_reason="stop",
                latency_ms=900,
                step_index=i,
                timestamp=float(i),
                output_length=420,
                completion_tokens=105,
            )
            for i in range(1, 4)
        ]
        self.assertIsNone(InstrumentationDegradedDetector().on_run_completion(_run_with(calls)))


# ── Acceptance 3: the real failure must still be caught ─────────────────────


class TestGenuinelyEmptyResponseStillFires(unittest.TestCase):
    def test_real_empty_completion_still_fires_empty_llm_response(self):
        """The false positive must not have been 'fixed' by disabling the
        detector. A real ChatCompletion with real usage and an empty message is
        still EMPTY_LLM_RESPONSE."""
        from dunetrace.auto import _emit_openai_response

        emitted = {}

        class _Run:
            def llm_responded(self, **kw):
                emitted.update(kw)

        _emit_openai_response(_Run(), _RealChatCompletion(), 0.0)
        self.assertEqual(emitted["finish_reason"], "stop")
        self.assertEqual(emitted["output"], "")
        self.assertIsNone(emitted["instrumentation_degraded"])

        call = LlmCall(
            model="gpt-4o",
            prompt_tokens=812,
            finish_reason="stop",
            latency_ms=700,
            step_index=1,
            timestamp=1.0,
            output_length=0,
            completion_tokens=0,
        )
        sig = EmptyLlmResponseDetector().on_run_completion(_run_with([call]))
        self.assertIsNotNone(sig, "a genuinely empty response must still be reported")
        self.assertEqual(sig.failure_type, FailureType.EMPTY_LLM_RESPONSE)


# ── Acceptance 4: estimated vs exact prompt tokens ──────────────────────────


class TestPromptTokenProvenance(unittest.TestCase):
    def _client(self):
        from dunetrace import Dunetrace

        c = Dunetrace(api_key="k")
        c._ship = lambda batch: None
        return c

    def test_true_when_usage_was_absent(self):
        c = self._client()
        with c.run("a1") as run:
            run.llm_called("gpt-4o", prompt_tokens=200, prompt_tokens_estimated=True)
            run.llm_responded(completion_tokens=10, output="hi", output_length=2)
            self.assertTrue(run.state.llm_calls[0].prompt_tokens_estimated)
        c.shutdown(timeout=1)

    def test_false_once_usage_supplied_an_exact_count(self):
        c = self._client()
        with c.run("a1") as run:
            run.llm_called("gpt-4o", prompt_tokens=200, prompt_tokens_estimated=True)
            run.llm_responded(completion_tokens=10, prompt_tokens=812, output="hi", output_length=2)
            self.assertFalse(run.state.llm_calls[0].prompt_tokens_estimated)
            self.assertEqual(run.state.llm_calls[0].prompt_tokens, 812)
        c.shutdown(timeout=1)

    def test_survives_the_server_side_rebuild(self):
        """In-process state is not what production detection runs on."""
        from dunetrace.run_builder import build_run_state

        base = {"run_id": "r1", "agent_id": "a1", "agent_version": "v1"}
        events = [
            {
                **base,
                "event_type": "llm.called",
                "step_index": 1,
                "timestamp": 1.0,
                "payload": {
                    "model": "gpt-4o",
                    "prompt_tokens": 200,
                    "call_id": 0,
                    "prompt_tokens_estimated": True,
                },
            },
            {
                **base,
                "event_type": "llm.responded",
                "step_index": 1,
                "timestamp": 2.0,
                "payload": {
                    "completion_tokens": 5,
                    "latency_ms": 900,
                    "output_length": 0,
                    "call_id": 0,
                    "instrumentation_degraded": "openai_response_shape:LegacyAPIResponse",
                },
            },
        ]
        state = build_run_state(events)
        self.assertTrue(state.llm_calls[0].prompt_tokens_estimated)
        self.assertEqual(
            state.llm_calls[0].instrumentation_degraded,
            "openai_response_shape:LegacyAPIResponse",
        )
        self.assertIsNone(state.llm_calls[0].finish_reason)
        self.assertIsNone(EmptyLlmResponseDetector().on_run_completion(state))


# ── Acceptance 5: the fleet query ───────────────────────────────────────────


class TestFleetQuery(unittest.TestCase):
    def _db(self):
        con = sqlite3.connect(":memory:")
        con.execute(
            "CREATE TABLE events (org_id TEXT, agent_id TEXT, run_id TEXT, "
            "event_type TEXT, step_index INT, received_at TEXT, payload TEXT)"
        )
        return con

    def _add(self, con, agent, n, *, blank):
        for i in range(n):
            con.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?,?)",
                (
                    "o1",
                    agent,
                    f"{agent}-run{i}",
                    "llm.called",
                    i,
                    "2026-08-20",
                    json.dumps({"provider": "openai", "prompt_tokens": 800}),
                ),
            )
            payload = (
                {
                    "output_length": 0,
                    "finish_reason": "stop",
                    "completion_tokens": 0,
                    "latency_ms": 900,
                }
                if blank
                else {
                    "output_length": 420,
                    "finish_reason": "stop",
                    "completion_tokens": 105,
                    "latency_ms": 900,
                }
            )
            con.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?,?)",
                (
                    "o1",
                    agent,
                    f"{agent}-run{i}",
                    "llm.responded",
                    i,
                    "2026-08-20",
                    json.dumps(payload),
                ),
            )

    def test_broken_agent_scores_one_and_healthy_scores_zero(self):
        import sys, os

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../services/api"))
        from api_svc.instrumentation_health import (
            BROKEN_INSTRUMENTATION_THRESHOLD,
            blank_response_rate_sql,
        )

        con = self._db()
        self._add(con, "broken-agent", 25, blank=True)
        self._add(con, "healthy-agent", 25, blank=False)
        rows = {
            r[1]: r
            for r in con.execute(blank_response_rate_sql("sqlite", min_calls=20, since_days=None))
        }
        self.assertAlmostEqual(rows["broken-agent"][5], 1.0)
        self.assertAlmostEqual(rows["healthy-agent"][5], 0.0)
        self.assertGreater(rows["broken-agent"][5], BROKEN_INSTRUMENTATION_THRESHOLD)
        self.assertLess(rows["healthy-agent"][5], BROKEN_INSTRUMENTATION_THRESHOLD)
        # provider is joined from llm.called, where it actually lives.
        self.assertEqual(rows["broken-agent"][2], "openai")

    def test_current_sdk_omits_finish_reason_and_is_still_caught(self):
        """Post-provenance the SDK omits finish_reason rather than sending
        'stop', so the query must catch NULL as well as the legacy value."""
        import sys, os

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../services/api"))
        from api_svc.instrumentation_health import blank_response_rate_sql

        con = self._db()
        for i in range(25):
            con.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?,?)",
                (
                    "o1",
                    "new-sdk",
                    f"r{i}",
                    "llm.responded",
                    i,
                    "2026-08-20",
                    json.dumps(
                        {
                            "output_length": 0,
                            "completion_tokens": 0,
                            "latency_ms": 900,
                            "instrumentation_degraded": "openai_response_shape:X",
                        }
                    ),
                ),
            )
        rows = list(con.execute(blank_response_rate_sql("sqlite", min_calls=20, since_days=None)))
        self.assertAlmostEqual(rows[0][5], 1.0)


# ── Direction-of-failure guards ─────────────────────────────────────────────


class TestDetectorsThatFailTheOtherWay(unittest.TestCase):
    def test_runaway_iteration_does_not_fire_when_text_is_unreadable(self):
        """RUNAWAY_ITERATION fires on the ABSENCE of completion language, so
        unreadable text is a false POSITIVE here — the opposite direction from
        EMPTY_LLM_RESPONSE."""
        state = RunState(run_id="r1", agent_id="a1", agent_version="v1")
        state.current_step = 80
        for i in range(1, 4):
            state.llm_calls.append(
                LlmCall(
                    model="gpt-4o",
                    prompt_tokens=100,
                    finish_reason=None,
                    latency_ms=900,
                    step_index=i,
                    timestamp=float(i),
                    output_length=0,
                    completion_tokens=0,
                    instrumentation_degraded="openai_response_shape:LegacyAPIResponse",
                )
            )
            state.events.append(
                AgentEvent(
                    event_type=EventType.LLM_RESPONDED,
                    run_id="r1",
                    agent_id="a1",
                    agent_version="v1",
                    step_index=i,
                    timestamp=float(i),
                    payload={"output_length": 0, "completion_tokens": 0},
                )
            )
        self.assertIsNone(RunawayIterationDetector().on_run_completion(state))

    def test_truncation_loop_does_not_infer_length_from_an_unreadable_call(self):
        calls = [
            LlmCall(
                model="gpt-4o",
                prompt_tokens=100,
                finish_reason=None,
                latency_ms=900,
                step_index=i,
                timestamp=float(i),
                output_length=0,
                completion_tokens=0,
                instrumentation_degraded="openai_response_shape:LegacyAPIResponse",
            )
            for i in range(1, 4)
        ]
        self.assertIsNone(LlmTruncationLoopDetector().on_run_completion(_run_with(calls)))


# ── Acceptance 6: healthy runs unchanged apart from additive fields ─────────


class TestWireFormatCompatibility(unittest.TestCase):
    def _emit_and_capture(self, **kw):
        from dunetrace import Dunetrace
        from dunetrace.models import EventType

        c = Dunetrace(api_key="k")
        c._ship = lambda batch: None
        with c.run("a1") as run:
            run.llm_called("gpt-4o", prompt_tokens=100)
            run.llm_responded(**kw)
            events = list(run.state.events)
        c.shutdown(timeout=1)
        called = [e for e in events if e.event_type == EventType.LLM_CALLED][-1]
        responded = [e for e in events if e.event_type == EventType.LLM_RESPONDED][-1]
        return called.payload, responded.payload

    def test_manual_caller_payloads_are_byte_identical(self):
        """A manual caller passing no new arguments must emit exactly the keys
        it always did — no nulls, no new keys."""
        called, responded = self._emit_and_capture(
            completion_tokens=10,
            latency_ms=50,
            finish_reason="stop",
            output="hello",
            output_length=5,
        )
        self.assertEqual(set(called), {"model", "prompt_tokens", "call_id"})
        self.assertEqual(
            set(responded),
            {
                "completion_tokens",
                "latency_ms",
                "finish_reason",
                "output_length",
                "output",
                "call_id",
            },
        )

    def test_degraded_call_omits_rather_than_nulls(self):
        _called, responded = self._emit_and_capture(
            completion_tokens=0,
            latency_ms=900,
            finish_reason=None,
            output=None,
            output_length=0,
            instrumentation_degraded="openai_response_shape:X",
        )
        self.assertNotIn("finish_reason", responded, "null must never go on this wire")
        self.assertNotIn("output", responded)
        self.assertEqual(responded["instrumentation_degraded"], "openai_response_shape:X")

    def test_run_started_carries_the_sdk_fingerprint(self):
        from dunetrace.auto import instrumentation_fingerprint

        fp = instrumentation_fingerprint()
        self.assertIn("sdk_version", fp)
        self.assertIsInstance(fp["instrumented"], dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
