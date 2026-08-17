"""
Phase 4: integration tests against REAL emitter output.

The fixtures in fixtures/emitter_captures/ are OTLP spans captured from the
actual instrumentation of three emitters (OTel-contrib openai-v2, Traceloop,
OpenLIT) driving a real openai SDK call through a mocked HTTP transport. Only
the model's HTTP response was stubbed; the span attributes are each emitter's
genuine output.

These tests run the captures through the real translator (otlp_to_events) and,
where the detector service is importable, through the real run reconstruction
and detector suite, so the whole OTLP -> events -> detectors pipeline is
exercised on real-world span shapes.

Capture provenance (see fixtures/emitter_captures): opentelemetry-
instrumentation-openai-v2, opentelemetry-instrumentation-openai (Traceloop),
openlit, against openai 1.x/2.x.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))

from ingest_svc.otel import otlp_to_events

# The real run reconstruction lives in the detector service. Import it if the
# service is on-disk (it is, in the repo); otherwise skip the detector tests and
# keep the translation tests, which are the primary check.
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "detector")))
try:
    from detector_svc.run_builder import build_run_state
    from dunetrace.detectors import run_detectors

    _DETECTORS_AVAILABLE = True
except Exception:
    _DETECTORS_AVAILABLE = False

_FIXTURES = os.path.join(_HERE, "fixtures", "emitter_captures")
_EMITTERS = ["otel-contrib", "traceloop", "openlit"]


def _events(emitter: str) -> list[dict]:
    with open(os.path.join(_FIXTURES, f"{emitter}.json")) as f:
        payload = json.load(f)
    return otlp_to_events(payload["resourceSpans"])


def _one(events: list[dict], event_type: str) -> dict:
    return next(e for e in events if e["event_type"] == event_type)


# ── Translation of real emitter spans ────────────────────────────────────────────


class TestRealEmitterTranslation:
    def test_all_emitters_produce_a_run_and_an_llm_call(self):
        for emitter in _EMITTERS:
            events = _events(emitter)
            types = [e["event_type"] for e in events]
            assert "run.started" in types, emitter
            # openlit's span carries ERROR status (an openlit-side issue with the
            # stubbed response), so its run correctly translates to run.errored.
            assert ("run.completed" in types) or ("run.errored" in types), emitter
            assert "llm.called" in types, emitter
            assert "llm.responded" in types, emitter
            assert _one(events, "llm.called")["payload"]["model"] == "gpt-4o", emitter

    def test_otel_contrib_tokens(self):
        # Official OTel GenAI instrumentation: current input/output_tokens convention.
        events = _events("otel-contrib")
        assert _one(events, "llm.called")["payload"]["prompt_tokens"] == 24
        resp = _one(events, "llm.responded")["payload"]
        assert resp["completion_tokens"] == 8
        assert resp["finish_reason"] == "stop"

    def test_traceloop_tokens_and_output_messages(self):
        # Traceloop emits the assistant reply under gen_ai.output.messages
        # (structured), the gap Phase 4 discovered and fixed.
        events = _events("traceloop")
        assert _one(events, "llm.called")["payload"]["prompt_tokens"] == 24
        resp = _one(events, "llm.responded")["payload"]
        assert resp["completion_tokens"] == 8
        assert "Paris" in resp.get("output", "")

    def test_openlit_model_translates(self):
        # OpenLIT hit an internal error extracting tokens against the stubbed
        # response, so tokens are absent; the model still translates cleanly.
        # Documented as an openlit-side issue, not a translation gap.
        events = _events("openlit")
        assert _one(events, "llm.called")["payload"]["model"] == "gpt-4o"


# ── Detectors run on the translated events ───────────────────────────────────────


@pytest.mark.skipif(not _DETECTORS_AVAILABLE, reason="detector service not importable")
class TestDetectorsRunOnTranslatedEvents:
    @pytest.mark.parametrize("emitter", _EMITTERS)
    def test_pipeline_runs_end_to_end(self, emitter):
        events = _events(emitter)
        state = build_run_state(events)  # the real production reconstruction
        signals = run_detectors(state)  # must execute without error
        assert isinstance(signals, list)
        # The translated LLM call reached the reconstructed run state.
        assert len(state.llm_calls) == 1
        assert state.llm_calls[0].model == "gpt-4o"
