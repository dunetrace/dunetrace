# ElevenLabs integration

You already see TTS cost and voice metrics in the ElevenLabs dashboard. What you
cannot see there is agent behavior. Did an expensive generation happen on a call
that failed. Does one voice correlate with more user frustration than another.
When TTS got truncated, did the candidate notice.

This integration answers those questions. Dunetrace pulls your ElevenLabs
generation history and correlates each generation to the `tts_generated` event
on the run that produced it. ElevenLabs stays the source of truth for TTS data.
Dunetrace pulls a copy to line it up against the agent behavior it already
tracks.

It is off by default. Orgs that do not configure it see no change.

---

## How it works

A background poller (`services/integrations/integrations_svc/elevenlabs_worker`)
checks each connected org on an interval (default 5 minutes, configurable per
org), fetches new generations from ElevenLabs' `GET /v1/history` since the last
successful poll, and stores them. A correlation pass then matches each stored
generation to a Dunetrace `tts_generated` event for the same org.

The poller and the correlation pass run in their own process, separate from the
evaluation-provider poller. If ElevenLabs is down or rate limited, nothing else
in Dunetrace is affected, and the correlation of one generation never blocks the
fetch of the next.

---

## Setup

### 0. Enable the worker

Call ingestion is served by the `elevenlabs` container, which ships in both
compose files but is **off by default**:

```bash
ELEVENLABS_WORKER_ENABLED=true        # in .env
DUNETRACE_MASTER_KEY=<same value the Customer API uses>
docker compose up -d
```

`DUNETRACE_MASTER_KEY` must match the Customer API's exactly — the API encrypts
your stored ElevenLabs key and this worker is what decrypts it. With the flag
off the container logs one line and exits 0, so `Exited (0)` is the expected
state, not a failure. It runs the same image as the `integrations` worker with a
different entrypoint.

### 1. Get an ElevenLabs API key

In your ElevenLabs account, open the profile menu and copy your API key. Any key
with permission to read history works. The key is validated when you save it, so
a wrong key is rejected immediately.

### 2. Connect it to Dunetrace

From the dashboard, open **Integrations** in the sidebar and connect ElevenLabs
with your API key and a poll interval. Or call the API directly:

```bash
POST /v1/orgs/integrations/elevenlabs
Authorization: Bearer dt_live_...
Content-Type: application/json

{
  "api_key": "sk_...",
  "poll_interval_secs": 300
}
```

`GET /v1/orgs/integrations/elevenlabs` returns configuration and health.
`DELETE` disconnects. The key is encrypted at rest (same Fernet split as the
Langfuse and Slack integrations: `api_svc` encrypts on save, only the poller
decrypts) and never appears in any API response.

The poll interval floor is 60 seconds and the default is 300. ElevenLabs rate
limits by concurrency, not requests per minute, and the poller makes one
sequential request per cycle, so the default sits far under every plan tier.

### 3. Capture correlation metadata (recommended)

Correlation always works on timestamp and character count alone. It works far
better when your `tts_generated` calls carry a few optional fields:

```python
audio = elevenlabs.text_to_speech.convert(text=reply, voice_id=vid, model_id=mid)
run.tts_generated(
    reply,
    latency_ms=elapsed_ms,
    voice_id=vid,                              # narrows the match
    model="eleven_multilingual_v2",
    provider="elevenlabs",
    provider_generation_id=audio.history_item_id,  # deterministic match
)
```

All four fields are optional and keyword-only. Existing calls are unchanged.
`provider_generation_id` is the strongest signal: when your code captures the id
ElevenLabs returns, the match is exact. `voice_id` is the next best and the
easiest to add.

---

## What data flows in

Each generation is stored in `elevenlabs_generations`:

| Field | Source |
|---|---|
| `generation_id` | ElevenLabs `history_item_id` |
| `voice_id`, `voice_name` | the voice used |
| `model` | ElevenLabs `model_id` |
| `character_count` | the per-generation character delta (see note) |
| `cost_credits` | equal to `character_count` on standard TTS plans |
| `text` | the synthesized text, when ElevenLabs returns it |
| `source` | `TTS`, `ConvAI`, and so on |
| `generated_at` | ElevenLabs `date_unix` |

**Character count note.** ElevenLabs returns `character_count_change_from` and
`character_count_change_to`, which are running quota markers. The characters
billed for one generation is the difference between them. Dunetrace stores that
difference. This is verified against the live API, not just the docs.

The first poll after you connect pulls from the connect time forward, not the
beginning of your entire ElevenLabs history. Dunetrace correlates going forward
rather than backfilling an unbounded history.

---

## What analytics you get

Three cross-tool views on the **Voice analytics** page, each joining ElevenLabs
generation data with Dunetrace signals.

**Expensive TTS on failed conversations.** Cost per call, split by outcome, with
the headline number: how much you spent on TTS for calls Dunetrace flagged. A
call is flagged when it carries at least one non-shadow signal.

**Voice choice impact.** Per voice, the share of its runs that fired a signal.
Lower is better. Voices with too few runs are marked, not compared, so a
five-call voice never looks better or worse than it has earned.

**TTS truncation downstream impact.** For runs where the `VOICE_TTS_TRUNCATION`
detector fired, the frustration and abandonment rate compared against runs where
it did not. Answers whether truncation correlates with users noticing.

The raw data is also available:

```
GET /v1/orgs/integrations/elevenlabs/generations?run_id=...      # one run's generations
GET /v1/orgs/integrations/elevenlabs/generations?voice_id=&model=&min_credits=
GET /v1/orgs/integrations/elevenlabs/calls/{conversation_id}/cost
GET /v1/orgs/integrations/elevenlabs/analytics/cost-by-outcome
GET /v1/orgs/integrations/elevenlabs/analytics/voice-impact
GET /v1/orgs/integrations/elevenlabs/analytics/truncation-impact
```

On a run detail page, a correlated generation shows its voice, model, character
count, cost in credits and USD, and a link to the ElevenLabs history page. On a
call detail page, ElevenLabs actual cost shows as its own line under the
estimate.

---

## The correlation model

For each generation, Dunetrace looks at the `tts_generated` events for the same
org within 60 seconds of the generation time, then picks a match strongest
signal first:

| Signal | How it matched | Confidence |
|---|---|---|
| `generation_id` | the event captured the ElevenLabs generation id | 1.0 |
| `exact_text` | the event text equals the synthesized text | 0.97 |
| `voice_char_time` | character count within 10% and voice id agrees | 0.85 |
| `char_time` | character count within 10%, single candidate in the window | 0.70 |

A match below 0.85 is shown with a "verify" indicator in the UI, so a
weak match is never presented as a certain one. The timestamp window is generous
because a `tts_generated` event is emitted after the audio returns, so it lags
ElevenLabs' create time by the synthesis and network latency.

When more than one candidate survives and the signals cannot separate them,
Dunetrace does not guess. It records the generation as unmatched rather than
inventing a correlation.

The tolerances are configurable: `CORRELATION_WINDOW_SECS` (default 60),
`CORRELATION_CHAR_TOLERANCE` (default 0.10), and `CORRELATION_GIVEUP_SECS`
(default 3600).

---

## Troubleshooting correlation failures

A generation that cannot match right away stays pending and is retried. Once it
is older than `CORRELATION_GIVEUP_SECS` (one hour by default) with no match, it
is recorded as unmatched drift with a reason. Match rate and the reason
breakdown are in `get_correlation_metrics`.

| Reason | What it means | What to do |
|---|---|---|
| `no_candidate_events` | no `tts_generated` event for that org near the generation time | Confirm the call was instrumented and `run.tts_generated(...)` fired. The generation may be from traffic Dunetrace never saw. |
| `no_char_match` | events exist in the window, but none within 10% on character count | Check for a clock skew larger than the 60s window, or text differences between what you sent and what ElevenLabs synthesized. |
| `ambiguous_multiple_matches` | several equally plausible events, nothing to separate them | Capture `voice_id`, or better `provider_generation_id`, on `tts_generated`. This is the single biggest lever on match rate. |

If the match rate is lower than you expect, the fastest fix is almost always to
capture more metadata on `tts_generated`. `provider_generation_id` makes the
match exact. `voice_id` resolves most ambiguity on its own.

The `source` field is worth checking too. If your agent runs on ElevenLabs
Conversational AI, its generations carry `source=ConvAI` rather than `TTS`, and
the fields ElevenLabs populates differ. Dunetrace does not over-filter by source,
and logs anything it cannot match as drift rather than dropping it silently.

---

## Failure isolation

ElevenLabs data is augmentation, not a requirement. If the API is unreachable or
rate limited, the poller records the failure, backs off, and retries. Every
other part of Dunetrace keeps working. If ElevenLabs stays unreachable for more
than 30 minutes, an internal `EXTERNAL_INTEGRATION_DOWN` operational signal is
written for visibility, rate limited so a long outage does not repeat it every
cycle.
