# Voice agent detector pack

The **voice pack** adds nine failure detectors specific to real-time voice
agents — speech-to-text, text-to-speech, voice-activity detection (VAD), and
turn-taking. It is a [detector pack](./index.md): first-party detection you
activate per org. Built-in detectors keep running regardless; the voice pack
adds to them, it doesn't replace anything.

Pack detectors run **server-side only** (in the detector worker), and — like
every newly added detector — start in **shadow mode**: they evaluate and are
visible in the dashboard's shadow view, but don't fire alerts until an
operator promotes them.

---

## 1. Activate the pack

Activation is **binary per org** — the whole pack is on or off, no
per-detector toggles.

**SDK:**

```python
from dunetrace import Dunetrace

dt = Dunetrace(api_key="dt_live_...", api_url="https://your-dunetrace-host")
dt.enable_pack("voice")
print(dt.enabled_packs())   # ["voice"]
# dt.disable_pack("voice")  # turn it back off
```

**API:**

```bash
curl -X POST https://your-dunetrace-host/v1/orgs/packs/voice \
  -H "Authorization: Bearer dt_live_..."
```

**Dashboard:** **Packs** in the sidebar → **Activate** on the voice card.

The org is always taken from your API key, never from the URL — see
[detector-packs/index.md](./index.md#activation).

---

## 2. Emit voice events

The pack reads four event types off the run. Emit them from your voice
pipeline via the `dt.run()` context manager, alongside the usual
`llm_called`/`tool_called` hooks. All existing hooks are unchanged; these are
purely additive.

| Method | Advances step? | Payload |
|---|---|---|
| `run.transcription_received(text, confidence=1.0, latency_ms=0)` | **yes** | a finalized STT transcript for one user turn |
| `run.tts_generated(text, latency_ms=0, truncated=False)` | no | a synthesized agent response |
| `run.voice_activity_detected(type, duration_ms=0)` | no | a VAD transition |
| `run.turn_taking(action, from_agent=False, to_user=False)` | no | a conversational-floor change |

`voice_activity_detected`'s `type` must be one of `speech_start`,
`speech_end`, `silence`, `barge_in`. `turn_taking`'s `action` must be one of
`agent_speaking`, `user_speaking`, `both_speaking`, `neither`. An invalid
value raises `ValueError` at the call site.

**Why only `transcription_received` advances the step counter:** a normal
voice turn should count as roughly one step, the same as a normal tool/LLM
turn. VAD and turn-taking events fire many times per second of audio; if they
advanced the counter, a real call would blow past the always-on
`RUNAWAY_ITERATION` / `STEP_COUNT_INFLATION` thresholds — which run for every
org whether or not the voice pack is active. So the high-frequency annotation
events don't advance; only the turn-initiating transcription does. This is the
same split the non-voice hooks use (`tool_called` advances, `tool_responded`
doesn't).

```python
with dt.run("voice-agent", user_input=transcript, model="gpt-4o-realtime") as run:
    run.voice_activity_detected("speech_start")
    run.transcription_received(transcript, confidence=0.94, latency_ms=110)
    run.turn_taking("user_speaking", to_user=True)

    run.llm_called("gpt-4o-realtime", prompt_tokens=200)
    reply = generate_reply(transcript)
    run.llm_responded(completion_tokens=40, latency_ms=850, finish_reason="stop", output=reply)

    run.turn_taking("agent_speaking", from_agent=True)
    run.tts_generated(reply, latency_ms=90, truncated=False)
```

See [voice-frameworks.md](../integrations/voice-frameworks.md) for mapping
these hooks onto a real voice framework's callbacks.

---

## 3. The nine detectors

Every detector is silent on a run with no voice events, so activating the pack
on a non-voice agent produces zero signals. Thresholds are class attributes —
tunable the same way as any built-in.

| Detector | Fires when | Default threshold | Severity |
|---|---|---|---|
| `VOICE_TRANSCRIPTION_CONFIDENCE_DROP` | ≥ N transcripts below an absolute confidence floor | `MIN_CONFIDENCE=0.5`, `MIN_COUNT=2` | MEDIUM |
| `VOICE_SILENCE_TIMEOUT` | a `silence` VAD event longer than the ceiling (dead air) | `MAX_SILENCE_MS=8000` | MEDIUM |
| `VOICE_TURN_TAKING_COLLISION` | ≥ N `both_speaking` states (talking over each other) | `MAX_COLLISIONS=2` | MEDIUM |
| `VOICE_LATENCY_INDUCED_HANGUP` | the last turn was a slow TTS with no user reply after it | `MAX_LATENCY_MS=5000` | HIGH |
| `VOICE_AUDIO_QUALITY_DEGRADATION` | transcription confidence trends downward across the call | `MIN_DROP=0.2`, `MIN_SAMPLES=4` | MEDIUM |
| `VOICE_SPEAKER_CONFUSION` | a transcript arrives while the agent itself is speaking (echo / self-hearing) | `MIN_COUNT=1` | HIGH |
| `VOICE_BARGE_IN_FAILURE` | user barged in but the agent produced a full, non-truncated TTS anyway | — | HIGH |
| `VOICE_TTS_TRUNCATION` | ≥ N responses cut off **without** a preceding barge-in to explain it | `MAX_TRUNCATIONS=1` | MEDIUM |
| `VOICE_VAD_FALSE_TRIGGER` | ≥ N speech/barge-in events too short to be real speech | `MAX_FALSE_TRIGGERS=3`, `MIN_DURATION_MS=100` | LOW |

`VOICE_TTS_TRUNCATION` and `VOICE_BARGE_IN_FAILURE` deliberately don't overlap:
a truncation immediately after a `barge_in` is the *correct* behavior (the
agent yielded), so it's excluded from `VOICE_TTS_TRUNCATION`'s count — that
detector only flags truncations with no interruption to justify them.

`VOICE_VAD_FALSE_TRIGGER` only counts events carrying a **positive**
`duration_ms`. A `0` (or absent) duration means "unknown", not "zero-length",
so it isn't counted — if your VAD doesn't report segment length on
`speech_start`, this detector stays silent rather than firing on every start.

Signals are stored with `failure_type` = the detector's name (as TEXT), the
same convention custom Python-class detectors use.

---

## 4. Voice policy actions

Four [policy](../policies.md) actions target voice agents. They follow the
same contract as `switch_model`: **Dunetrace sets intent, your agent's own
loop enforces it** — Dunetrace never touches your audio pipeline. A policy
fires, Dunetrace sets a structured attribute on the run, and your code reads
it between turns.

| Action | Sets on `run` | Params | Your loop does |
|---|---|---|---|
| `stop_current_tts` | `run.stop_tts = True` | — | halt in-progress TTS playback |
| `escalate_to_human` | `run.escalate_to_human = True`, `run.escalation_reason` | `reason` (optional) | transfer the call to a human |
| `inject_recovery_prompt` | `run.recovery_prompt` (last-wins) | `prompt` (required) | speak / prepend the recovery line next turn |
| `slow_response_pace` | `run.response_pace` (last-wins) | `pace` (optional, default `"slow"`) | slow the turn: emit a filler token while thinking, lower the TTS rate |

These are ordinary policy actions — driven by the existing trigger set
(`llm_latency_ms`, `error_count`, `step_count`, `signal`, …), not gated on
pack activation. Example: recover on a slow model response.

```python
dt.add_policy(
    name="voice-latency-recovery",
    condition={"trigger": "llm_latency_ms", "operator": "gt", "value": 5000},
    action={"type": "inject_recovery_prompt",
            "params": {"prompt": "Sorry for the delay — one moment."}},
)

with dt.run("voice-agent", model="gpt-4o-realtime") as run:
    run.llm_called("gpt-4o-realtime")
    reply = generate_reply(transcript)      # slow
    run.llm_responded(latency_ms=6200, finish_reason="stop", output=reply)

    if run.recovery_prompt:                 # policy fired
        speak(run.recovery_prompt)
    if run.response_pace == "slow":         # pace-down policy fired
        emit_filler_token("mm-hmm")
    if run.escalate_to_human:
        transfer_to_agent(reason=run.escalation_reason)
```

`inject_recovery_prompt` content is checked for prompt-injection patterns
server-side, the same as `inject_prompt` — it's spoken text, same risk surface.

---

## Scope

- Pack detectors run only in the **detector worker** (server-side). The SDK's
  in-process runtime path uses the fixed built-in battery and never evaluates
  pack detectors.
- No new `RunState` fields — the detectors read the four event types straight
  out of the run's event list, which both the server and the SDK already
  populate.
- The voice **policy actions** are independent of the voice **pack**: a text
  agent that never reads `run.stop_tts` simply ignores it, and the actions
  work whether or not the pack is activated.
