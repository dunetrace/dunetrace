"""
Push a full agent run through the Dunetrace SDK with OTel export on, exercising
every span type so the harness backends have something real to render:

    run -> LLM -> tool (plain) -> tool (HTTP-shaped) -> retrieval -> voice

It also emits the server-side findings (signal + policy spans) the detector
service would emit, so a complete run's trace is visible end to end.

Run the harness first (otel/docker-compose.yml), then:

    python otel/exercise_agent.py

Writes otel/last_run.json (run_id + trace_id + expected span names) for verify.py.

Env (sensible defaults for the harness):
    DUNETRACE_OTEL_ENDPOINT   default http://localhost:4317
    DUNETRACE_OTEL_PROTOCOL   default grpc
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Point the SDK at the local collector before it initializes.
os.environ.setdefault("DUNETRACE_OTEL_ENABLED", "1")
os.environ.setdefault("DUNETRACE_OTEL_ENDPOINT", "http://localhost:4317")
os.environ.setdefault("DUNETRACE_OTEL_PROTOCOL", "grpc")
os.environ.setdefault("DUNETRACE_OTEL_SERVICE_NAME", "dunetrace")
os.environ.setdefault("OTEL_EXPORTER_OTLP_INSECURE", "true")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "sdk-py"))

from dunetrace import Dunetrace  # noqa: E402
from dunetrace import otel as dt_otel  # noqa: E402
from dunetrace.emitters import NoopBatchingEmitter  # noqa: E402
from dunetrace.integrations.otel import (  # noqa: E402
    emit_policy_span,
    emit_signal_span,
    trace_id_hex,
)


def main() -> int:
    # NoopBatchingEmitter: this harness is about OTel export, not Dunetrace's own
    # ingest, so we don't need an ingest endpoint running.
    dt = Dunetrace(emitter=NoopBatchingEmitter())
    if not dt_otel.is_enabled():
        print("OTel export is not enabled. Check DUNETRACE_OTEL_* env and the collector.")
        return 1

    with dt.run("otel-demo-agent", user_input="ship my order", model="gpt-4o") as run:
        run_id = run.run_id

        # LLM call
        run.llm_called("gpt-4o", prompt_tokens=320)
        time.sleep(0.05)
        run.llm_responded(completion_tokens=180, finish_reason="stop", output="On it.")

        # Plain tool
        run.tool_called("order_lookup", {"order_id": "A-1007"})
        time.sleep(0.02)
        run.tool_responded("order_lookup", success=True, output_length=64)

        # HTTP-shaped tool (url in args -> HTTP conventions on the span)
        run.tool_called(
            "api.shipping.com", {"url": "https://api.shipping.com/v1/track", "method": "get"}
        )
        time.sleep(0.02)
        run.tool_responded("api.shipping.com", success=True, output_length=128)

        # Retrieval
        run.retrieval_called("pinecone-kb", query="shipping SLA for priority orders")
        time.sleep(0.02)
        run.retrieval_responded("pinecone-kb", result_count=4, top_score=0.88)

        # Voice
        run.transcription_received(
            "where is my order", confidence=0.93, latency_ms=140, audio_seconds=1.8
        )
        run.voice_activity_detected("silence", duration_ms=600)
        run.tts_generated(
            "Your order ships today.", latency_ms=90, voice_id="rachel", model="eleven_turbo_v2"
        )

        run.final_answer()

    # Server-side findings (what detector_svc would emit for this run).
    tracer = dt_otel.get_tracer()
    emit_signal_span(
        tracer,
        run_id,
        failure_type="SLOW_STEP",
        severity="HIGH",
        confidence=0.87,
        detector_name="SlowStepDetector",
        evidence={"step_index": 2, "duration_ms": 4200},
    )
    emit_policy_span(
        tracer,
        run_id,
        action="switch_model",
        policy_name="cost-guard",
        trigger="cost_usd",
        trigger_value=0.55,
    )

    dt.shutdown()  # flush the export pipeline

    expected = [
        "dunetrace.run",
        "chat gpt-4o",
        "dunetrace.tool.order_lookup",
        "dunetrace.tool.api.shipping.com",
        "dunetrace.retrieval",
        "dunetrace.voice.transcription",
        "dunetrace.voice.tts",
        "dunetrace.signal.SLOW_STEP",
        "dunetrace.policy.switch_model",
    ]
    out = {
        "run_id": run_id,
        "trace_id": trace_id_hex(run_id),
        "expected_spans": expected,
    }
    Path(__file__).with_name("last_run.json").write_text(json.dumps(out, indent=2))
    print("Pushed run. trace_id=%s" % out["trace_id"])
    print("Expected spans:\n  " + "\n  ".join(expected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
