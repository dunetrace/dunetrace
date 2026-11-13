"""
Vapi + Dunetrace integration example.

Vapi is cloud-orchestrated: it runs the STT/LLM/TTS pipeline and posts server
messages to your server URL. The DunetraceVapiAdapter turns those messages into
Dunetrace voice events. You wire it into your webhook handler; Dunetrace buffers
each call and replays it as one run when the call ends.

This script is self-contained: instead of a live phone call, it feeds a
scripted sequence of Vapi server messages through the adapter and prints the
voice events Dunetrace captured. Point the adapter at a running backend
(docker compose up) to see the run and any voice-pack signals in the dashboard.

    PYTHONPATH=packages/sdk-py python examples/vapi_webhook_agent.py

To run it for real, drop the adapter into your webhook route:

    @app.post("/vapi/webhook")
    async def vapi_webhook(payload: dict):
        adapter.handle_message(payload)
        return {}

Coverage note: Vapi server messages do not carry per-transcript STT confidence
or per-event latency, so the confidence and latency detectors get no signal from
Vapi. Turn-taking, speaker confusion, barge-in, and TTS truncation do fire. See
docs/detector-packs/voice.md for the full matrix.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "packages", "sdk-py"),
)

from dunetrace import Dunetrace
from dunetrace.integrations.vapi import DunetraceVapiAdapter

CALL_ID = "call_demo_0001"


def _msg(mtype: str, **fields) -> dict:
    body = {"type": mtype, "call": {"id": CALL_ID}}
    body.update(fields)
    return {"message": body}


# A scripted Vapi call: the caller asks a question, the agent answers, the caller
# barges in, then the call ends.
SCRIPT = [
    _msg("speech-update", status="started", role="customer"),
    _msg("transcript", role="customer", transcriptType="partial", transcript="what's my"),
    _msg(
        "transcript",
        role="customer",
        transcriptType="final",
        transcript="What's my account balance?",
    ),
    _msg("speech-update", status="started", role="assistant"),
    _msg("transcript", role="assistant", transcriptType="final", transcript="Your balance is..."),
    _msg("user-interrupted"),
    # The end-of-call-report carries the recording URL; the adapter captures it
    # automatically (Phase 2.3). Dunetrace links to it, never fetches it.
    _msg(
        "end-of-call-report",
        endedReason="customer-ended-call",
        artifact={"recordingUrl": "https://storage.vapi.ai/demo/call_demo_0001.wav"},
    ),
]


def main() -> None:
    endpoint = os.getenv("DUNETRACE_ENDPOINT", "http://localhost:8001")
    dt = Dunetrace(endpoint=endpoint)
    adapter = DunetraceVapiAdapter(dt, agent_id="vapi-support-line", model="gpt-4o")

    print(f"Replaying a scripted Vapi call ({len(SCRIPT)} server messages)...")
    for message in SCRIPT:
        adapter.handle_message(message)

    dt.shutdown(timeout=3)
    print(
        "Done. One run for call "
        f"{CALL_ID} was sent to {endpoint}. If the voice pack is active for your "
        "org, check the dashboard for turn-taking and barge-in signals."
    )


if __name__ == "__main__":
    main()
