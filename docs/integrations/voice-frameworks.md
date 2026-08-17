# Wiring a voice framework to Dunetrace

The [voice pack](../detector-packs/voice.md) reads four event types off a
Dunetrace run. This guide maps those hooks onto the callbacks a typical
real-time voice stack already exposes. Your framework's exact API names will
differ — the mapping is what matters.

> **The code below is illustrative.** It shows the *pattern* for connecting a
> framework's events to Dunetrace's hooks, using plausible callback names. It
> is not copied from any specific vendor SDK — check your framework's own docs
> for the real callback names and payload fields, then adapt.

---

## The mapping

Most voice frameworks emit some form of these five signals. Connect each to
the matching Dunetrace hook:

| Your framework emits | Dunetrace hook |
|---|---|
| final STT transcript for a user turn | `run.transcription_received(text, confidence, latency_ms)` |
| TTS finished synthesizing a response | `run.tts_generated(text, latency_ms, truncated)` |
| VAD state change (speech start/end, silence, barge-in) | `run.voice_activity_detected(type, duration_ms)` |
| conversational floor changed (who's speaking) | `run.turn_taking(action, from_agent, to_user)` |
| the LLM turn itself | the usual `run.llm_called` / `run.llm_responded` |

`voice_activity_detected` `type` ∈ `{speech_start, speech_end, silence,
barge_in}`; `turn_taking` `action` ∈ `{agent_speaking, user_speaking,
both_speaking, neither}`.

### Capturing TTS provider metadata

`tts_generated` takes four optional, keyword-only fields for correlating a
generation back to its TTS provider: `voice_id`, `model`, `provider`, and
`provider_generation_id`. They are unused by the voice pack itself, but if you
run on ElevenLabs, capturing them lets Dunetrace pull ElevenLabs' generation
history and line up cost and voice choices against agent behavior. See the
[ElevenLabs integration](elevenlabs.md). Existing calls that omit them are
unchanged.

---

## One long-lived run per call

A voice call is one Dunetrace run. Open the `dt.run()` context when the call
connects and keep it open for the whole call, emitting events from your
callbacks as they fire. Because callbacks are asynchronous, the cleanest
pattern is to grab the run once and reference it from each handler.

```python
# ── ILLUSTRATIVE — adapt callback names to your framework ──────────────
from dunetrace import Dunetrace

dt = Dunetrace(api_key="dt_live_...", api_url="https://your-dunetrace-host")
dt.enable_pack("voice")   # once per org, not per call

def handle_call(session):
    run_cm = dt.run("voice-agent", model="gpt-4o-realtime")
    run = run_cm.__enter__()

    @session.on("vad")
    def _vad(ev):
        # map your framework's VAD labels to Dunetrace's four
        run.voice_activity_detected(ev.state, duration_ms=ev.duration_ms)

    @session.on("transcript_final")
    def _stt(ev):
        run.turn_taking("user_speaking", to_user=True)
        run.transcription_received(ev.text, confidence=ev.confidence,
                                   latency_ms=ev.latency_ms)

    @session.on("llm_response")
    def _llm(ev):
        run.llm_called(ev.model, prompt_tokens=ev.prompt_tokens)
        run.llm_responded(completion_tokens=ev.completion_tokens,
                          latency_ms=ev.latency_ms, finish_reason=ev.finish_reason,
                          output=ev.text)

    @session.on("tts_complete")
    def _tts(ev):
        run.turn_taking("agent_speaking", from_agent=True)
        run.tts_generated(ev.text, latency_ms=ev.latency_ms, truncated=ev.was_cut_off)

    @session.on("call_ended")
    def _end(ev):
        run.final_answer()
        run_cm.__exit__(None, None, None)
```

If your framework gives you `async`/`await` handlers, the hooks are plain
synchronous method calls — call them directly inside the async handler; they
don't block on I/O.

---

## Barge-in and TTS truncation

Two detectors depend on getting the barge-in / truncation signals right, so
they're worth wiring carefully:

- On a user interruption, emit `run.voice_activity_detected("barge_in", ...)`
  **and**, if the interruption caused you to cut the current TTS short, pass
  `truncated=True` on the `tts_generated` for that response. The pack treats a
  truncation right after a barge-in as correct (the agent yielded), and only
  flags truncations that had *no* barge-in to explain them.
- If a response is cut off for any other reason (length cap, synth error),
  emit `tts_generated(..., truncated=True)` with no preceding `barge_in` —
  that's what `VOICE_TTS_TRUNCATION` is for.

---

## Enforcing voice policy actions

If you configure the [voice policy actions](../detector-packs/voice.md#4-voice-policy-actions),
read the resulting attributes off the run inside your loop — Dunetrace sets
them, your framework acts on them:

```python
# ── ILLUSTRATIVE ──
if run.stop_tts:
    session.stop_playback()
if run.escalate_to_human:
    session.transfer(reason=run.escalation_reason)
if run.recovery_prompt:
    session.say(run.recovery_prompt)
```

---

## Vapi (shipped adapter)

The sections above are illustrative for any in-process framework. Vapi is
different: it is cloud-orchestrated, so Vapi runs the audio pipeline and posts
server messages to your server URL. There is nothing in-process to hook, so
Dunetrace ships a webhook adapter instead of a long-lived run.

```python
from dunetrace import Dunetrace
from dunetrace.integrations.vapi import DunetraceVapiAdapter

dt = Dunetrace()
adapter = DunetraceVapiAdapter(dt, agent_id="support-line", model="gpt-4o")

@app.post("/vapi/webhook")           # your server URL, registered with Vapi
async def vapi_webhook(payload: dict):
    adapter.handle_message(payload)
    return {}
```

The adapter buffers each call's messages (keyed by call id) and replays them
through one `dt.run()` when the call ends (`end-of-call-report`, or a
`status-update` with `status="ended"`). One Vapi call becomes one run, with the
call id threaded as `conversation_id`. Runnable example:
[`examples/vapi_webhook_agent.py`](../../examples/vapi_webhook_agent.py).

**Enforcement caveat.** Vapi owns the audio pipeline, so the voice runtime
policy actions cannot be enforced in-path for Vapi. Detection is post-hoc, per
call. The sub-millisecond runtime firewall claim does not apply to Vapi.

**Detector coverage from Vapi.** Vapi server messages carry turn structure,
transcripts, interruptions, and call lifecycle, but not per-transcript STT
confidence or per-event latency. So these detectors get no signal from Vapi:

| Detector | Fires from Vapi? | Why |
|---|---|---|
| `VOICE_TURN_TAKING_COLLISION` | yes | from `speech-update` |
| `VOICE_SPEAKER_CONFUSION` | yes | transcript while agent speaking |
| `VOICE_BARGE_IN_FAILURE` | yes | from `user-interrupted` + following TTS |
| `VOICE_TTS_TRUNCATION` | yes | truncated TTS without a barge-in |
| `VOICE_SILENCE_TIMEOUT` | partial | needs silence duration, not always present |
| `VOICE_TRANSCRIPTION_CONFIDENCE_DROP` | no | Vapi does not surface STT confidence |
| `VOICE_AUDIO_QUALITY_DEGRADATION` | no | same, confidence-based |
| `VOICE_LATENCY_INDUCED_HANGUP` | no | Vapi does not surface per-event latency |

This is a Vapi surface limit, not a detector defect. An in-process framework
(LiveKit, Deepgram Voice Agent) that exposes confidence and latency lights up
the full pack.

---

## See also

- [Voice detector pack reference](../detector-packs/voice.md)
- [`examples/voice_agent.py`](../../examples/voice_agent.py) — a runnable,
  self-contained version of the pattern above
- [Detector packs overview](../detector-packs/index.md)
