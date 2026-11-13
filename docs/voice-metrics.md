# Voice metrics

Call-level metrics for voice agents. A **call** is a conversation for a voice
agent: the same conversations/runs model as the [Conversations](dashboard.md)
view, read through a voice lens. There is no separate calls table. A call groups
one or more runs by `conversation_id` (the id you pass to `dt.run(conversation_id=...)`,
or the call id the [Vapi adapter](integrations/voice-frameworks.md#vapi-shipped-adapter)
threads for you).

A conversation shows up as a call once any of its runs carries voice events
(`transcription.received`, `tts.generated`, `voice_activity.detected`,
`turn_taking.changed`). Text-agent conversations stay on the Conversations page
and never appear under Calls.

## The metrics

| Metric | How it is derived |
|---|---|
| Duration | span between the first and last event of the call |
| Completion status | natural, dropped, or escalated (see below) |
| Silence % | sum of `voice_activity.detected` silence durations over call duration |
| Agent vs caller talk ratio | from `turn_taking.changed` transition timestamps: the floor stays with the current speaker until the next transition |
| Voice signal count | voice-pack signals fired on the call's runs |
| Sentiment trend | placeholder until Phase 3 (Feature 6) |

Talk ratio is a timestamp-derived approximation, not an exact playback
measurement. Silence % is only meaningful when your framework emits silence VAD
events; frameworks that do not (for example Vapi) report 0% here.

## Completion status

Three outcomes, in precedence order:

1. **escalated** — an `escalate_to_human` policy fired during the call (observed
   from the `policy.triggered` event it emits).
2. **dropped** — the caller hung up or the call failed. Derived from a
   `VOICE_LATENCY_INDUCED_HANGUP` signal, a run that errored, or a framework
   endedReason that maps to a drop (silence timeout, no answer, provider or
   pipeline error, exceeded max duration).
3. **natural** — none of the above. The call ended normally.

For Vapi, the adapter records the call's `endedReason` so this mapping can run.
The natural-vs-dropped reason mapping lives server-side and is deliberately
small; making it configurable per org is on the backlog.

## API

- `GET /v1/calls` — list calls with metrics. Filters: `agent_id`, `since` (Unix
  timestamp), `completion_status` (natural | dropped | escalated), plus
  `offset` / `limit`.
- `GET /v1/calls/{conversation_id}` — one call's metrics, its runs, its voice
  signals, and an ordered voice event timeline.

A single request aggregates up to the 200 most recent matching calls read-time.
That ceiling is a deliberate bound; if it becomes limiting the metrics move to a
precompute table (backlog).

## Dashboard

The **Calls** page lists calls with duration, outcome, silence %, agent talk
share, and signal count, filterable by agent, outcome, and time range. Click a
call for the detail view: metrics, runs, voice signals, and the voice event
timeline.

## Cost attribution

Each call carries a `cost_usd` total and a `cost_breakdown` by category. Cost is
computed read-time from the call's events and the rates in `voice-pricing.yml`.

| Category | Billing unit | Derived from |
|---|---|---|
| STT | per audio-minute | `audio_seconds` on `transcription.received` |
| TTS | per 1k characters | `len(text)` on `tts.generated` |
| LLM | per token | the run's LLM events + model (shared cost table) |
| Telephony | per minute | call duration |

Honesty rules (constraint: cost math must be truthful):

- STT cost needs the transcribed audio length. Pass `audio_seconds` to
  `run.transcription_received(...)`. When it is absent, STT cost is reported as
  0, "not measured" — never estimated. Vapi does not surface per-transcript audio
  duration, so STT cost is 0 for Vapi calls.
- A category with no configured provider or rate contributes 0, "not priced,"
  rather than a guess. Telephony is off by default.

### voice-pricing.yml

Rates live in `voice-pricing.yml` (mounted into the API; built-in defaults are
used if it is absent). It sets a default provider per category and optional
per-agent overrides:

```yaml
rates_as_of: "2026-07-22"
stt:
  default: deepgram_nova3
  providers:
    deepgram_nova3: { per_minute: 0.0048 }   # list price, verify your contract
tts:
  default: deepgram_aura2
  providers:
    deepgram_aura2: { per_1k_chars: 0.030 }
agent_overrides:
  billing-line: { stt: openai_gpt4o_transcribe }
```

The shipped rates are **list prices as of the `rates_as_of` date**, meant to get
cost tracking working immediately. Verify them against your contract and bump
`rates_as_of` when you do. The API logs a warning at startup if `rates_as_of` is
more than 90 days old.

The `/v1/calls` list supports a `cost_bucket` filter: `low` (<$0.10), `medium`
($0.10 to $1), `high` (>$1). The call list shows a cost column and the detail
view shows the per-category breakdown.

## Call recording

Point Dunetrace at your call audio and it links to it from the run and call
detail, deep-linked to the moment a signal fired. Dunetrace is
storage-agnostic: you pass the URL and metadata, and Dunetrace stores and links
them but **never fetches the audio**.

```python
run.recording_metadata(
    "https://my-bucket.s3.amazonaws.com/calls/abc.wav",
    duration_seconds=142.0,
    format="wav",
    storage_provider="s3",       # display label only: s3 | azure | gcs | https
    start_offset_seconds=0.0,    # when recording began, relative to call start
)
```

The URL is opaque to Dunetrace, so **presigned or otherwise private URLs work as
is** (S3, Azure Blob, GCS, plain HTTPS). The only caveat is expiry: a presigned
URL that has expired will not open when clicked. Because Dunetrace never fetches
the audio, the browser opens the URL directly, subject to your storage's own
auth and CORS.

**Deep-links.** On the call detail, each voice signal gets a "jump to MM:SS"
link (`url#t=<seconds>`) that opens the recording at that moment. The offset is
derived from the signal's structural step time into the call, minus
`start_offset_seconds`. It is approximate: it lands at the right turn, not the
exact millisecond, since Dunetrace aligns to event timing, not the audio itself.

**Vapi.** The Vapi adapter captures the recording automatically: when Vapi's
`end-of-call-report` includes a recording URL, the adapter emits it for you, so
Vapi users get recording correlation with no extra code.
