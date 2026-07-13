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

## See also

- [Voice detector pack reference](../detector-packs/voice.md)
- [`examples/voice_agent.py`](../../examples/voice_agent.py) — a runnable,
  self-contained version of the pattern above
- [Detector packs overview](../detector-packs/index.md)
