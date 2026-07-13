"""
Voice agent example — instrumenting a real-time voice call with Dunetrace and
the voice detector pack.

What this shows:
  1. Activating the voice pack for your org.
  2. Emitting the four voice event types across a multi-turn call.
  3. A voice policy (inject_recovery_prompt on a slow LLM turn) and reading the
     result off the run.

Run it against a local stack:
    docker compose up -d
    python examples/voice_agent.py

The event emission works against the ingest API; pack activation needs the
Customer API (:8002) and an API key. In local dev (AUTH_MODE=dev) any key
works — this script uses a placeholder and degrades gracefully if the API
isn't reachable, so the instrumentation still runs.

The framework callbacks here are simulated inline. See
docs/integrations/voice-frameworks.md for mapping these hooks onto a real
voice framework's events.
"""

from __future__ import annotations

import urllib.error

from dunetrace import Dunetrace

# api_url points at the Customer API (:8002); the ingest endpoint is separate
# and defaults correctly on its own.
dt = Dunetrace(api_key="dt_dev_test", api_url="http://localhost:8002")


def activate_voice_pack() -> None:
    """Turn the voice pack on for this org. Once per org, not per call."""
    try:
        dt.enable_pack("voice")
        print("voice pack activated:", dt.enabled_packs())
    except (urllib.error.URLError, RuntimeError) as exc:
        print(f"(could not activate pack — is the Customer API running? {exc})")
        print("continuing anyway; the instrumentation below still runs.")


def generate_reply(transcript: str, *, slow: bool = False) -> str:
    """Stand-in for your LLM call."""
    return f"You said: {transcript}. How can I help further?"


def run_one_call() -> None:
    # A voice call is one Dunetrace run — opened when the call connects, kept
    # open for the whole call.
    with dt.run(
        "voice-agent", user_input="book me a table for two", model="gpt-4o-realtime"
    ) as run:
        # ── Turn 1: a clean turn ──────────────────────────────────────────
        run.voice_activity_detected("speech_start")
        run.transcription_received("book me a table for two", confidence=0.94, latency_ms=110)
        run.voice_activity_detected("speech_end", duration_ms=1400)
        run.turn_taking("user_speaking", to_user=True)

        run.llm_called("gpt-4o-realtime", prompt_tokens=180)
        reply = generate_reply("book me a table for two")
        run.llm_responded(completion_tokens=32, latency_ms=780, finish_reason="stop", output=reply)

        run.turn_taking("agent_speaking", from_agent=True)
        run.tts_generated(reply, latency_ms=95, truncated=False)

        # ── Turn 2: a slow LLM turn that trips the recovery policy ─────────
        run.transcription_received("actually make it three", confidence=0.88, latency_ms=130)
        run.turn_taking("user_speaking", to_user=True)

        run.llm_called("gpt-4o-realtime", prompt_tokens=210)
        reply2 = generate_reply("actually make it three", slow=True)
        run.llm_responded(
            completion_tokens=28,
            latency_ms=6200,  # slow!
            finish_reason="stop",
            output=reply2,
        )

        # The policy (configured below) fires on llm_latency_ms > 5000 and sets
        # run.recovery_prompt. Your voice loop reads it and speaks it.
        if run.recovery_prompt:
            print(f"[policy] speaking recovery line: {run.recovery_prompt!r}")
        if run.escalate_to_human:
            print(f"[policy] escalating to human (reason={run.escalation_reason})")
        if run.stop_tts:
            print("[policy] halting current TTS playback")

        run.turn_taking("agent_speaking", from_agent=True)
        run.tts_generated(reply2, latency_ms=90, truncated=False)

        run.final_answer()


def main() -> None:
    activate_voice_pack()

    # A voice policy: if a model turn takes too long, inject a recovery line.
    dt.add_policy(
        name="voice-latency-recovery",
        condition={"trigger": "llm_latency_ms", "operator": "gt", "value": 5000},
        action={
            "type": "inject_recovery_prompt",
            "params": {"prompt": "Sorry for the delay — one moment please."},
        },
    )

    run_one_call()
    dt.shutdown(timeout=5)
    print("done — check the dashboard's shadow view for any voice signals.")


if __name__ == "__main__":
    main()
