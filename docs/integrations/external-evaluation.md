# External Evaluation Integrations (Phase 2)

If you already run Langfuse, LangSmith, or Braintrust for evaluation, Dunetrace
can pull those results in and surface them alongside its own structural and
semantic signals — one dashboard, one alert channel, instead of switching
between tools. This is separate from Dunetrace's own [native semantic
evaluation](../semantic-evaluation.md); use either, or both.

All three pull integrations, plus the generic push endpoint below, write
signals into the same `failure_signals` table structural detectors and native
semantic evaluators use, tagged with `source` set to the provider's own name
(`"langfuse"`, `"langsmith"`, `"braintrust"`, or whatever you supply to the
push endpoint) — they show up in the same run detail view, the same
`GET /v1/agents/{id}/signals` list, and the same performance trends as
everything else.

---

## Pull integrations: Langfuse, LangSmith, Braintrust

A background poller (`services/integrations/integrations_svc`) checks each
connected provider on an interval (default 60s, configurable per
integration), fetches new evaluation results since the last successful poll,
and correlates each one to a Dunetrace run via `trace_id` — the same
`trace_id` your OTel-instrumented agent already sends (see
`POST /v1/otlp/traces` in `docs/architecture.md`), so no extra wiring is
needed if you're already sending traces to both systems.

### Connect a provider

```bash
POST /v1/orgs/integrations/langfuse
Authorization: Bearer dt_live_...
Content-Type: application/json

{
  "endpoint_url": "https://cloud.langfuse.com",
  "public_key": "pk-lf-...",
  "secret_key": "sk-lf-...",
  "poll_interval_secs": 60
}
```

```bash
POST /v1/orgs/integrations/langsmith
{
  "endpoint_url": "https://api.smith.langchain.com",
  "api_key": "ls-...",
  "project_name": "my-project",
  "poll_interval_secs": 60
}
```

```bash
POST /v1/orgs/integrations/braintrust
{
  "endpoint_url": "https://api.braintrust.dev",
  "api_key": "bt-...",
  "project_id": "proj-...",
  "poll_interval_secs": 60
}
```

`GET`/`DELETE /v1/orgs/integrations/{provider}` check status or disconnect.
Credentials are encrypted at rest (same `encrypt_credentials`/
`decrypt_credentials` split as the Slack/Linear integrations — `api_svc`
encrypts-only, the poller decrypts-only) and never appear in any API
response.

### Failure handling

A provider being unreachable doesn't crash the poller or affect any other
integration — each poll is independent, wrapped in its own try/except, with
`consecutive_failures` tracked per integration. If a single provider stays
down past **30 minutes**, an internal `EXTERNAL_INTEGRATION_DOWN` operational
signal is written (shadow-only — this is an ops signal about the integration
itself, not a customer-facing agent failure) so it's visible for debugging
without alerting on every transient network blip.

### What gets ingested

Each provider's raw score is normalized to Dunetrace's own signal shape:
`failure_type` becomes `{PROVIDER}_{METRIC}` (e.g. `LANGFUSE_HALLUCINATION`),
confidence is derived from the provider's numeric score when it's in 0–1
range (falls back to a neutral 0.5 for categorical/out-of-range values), and
the original evaluation URL is preserved for click-through from the
dashboard back to the provider's own UI.

Signals from these integrations are written with `shadow=FALSE` — alert-
eligible immediately. Unlike Dunetrace's own semantic evaluators, there's no
local false-positive management (confidence floor, grouping, feedback loop)
for third-party scores: the customer explicitly configured the integration
and is trusted to have already judged their provider's evaluation quality.
This is a documented gap, not a silent one.

---

## Generic push endpoint

For evaluation tools without a dedicated poller — a custom eval pipeline, a
framework with no pull-friendly API, or a one-off backfill — push results
directly:

```bash
POST /v1/semantic-signals/external
Authorization: Bearer dt_live_...
Content-Type: application/json

{
  "trace_id": "...",
  "provider": "custom_eval_pipeline",
  "name": "hallucination",
  "external_id": "eval-run-4471",
  "value": 0.87,
  "comment": "flagged by internal QA pipeline"
}
```

This is a synchronous, customer-facing call — unlike the pull integrations'
background loop, a bad request or an unmatched `trace_id` is a **404**
directly in the response, not a silently-dropped background failure.
`external_id` is a caller-supplied idempotency key: retrying the same
`(provider, external_id)` pair after a timeout is a no-op, never a duplicate
signal. Either `value` (numeric) or `string_value` (free text) is required —
`comment` is optional context shown in the dashboard.

Rate limited per API key, same infrastructure as every other ingest-side
endpoint.

---

## Which integration should I use?

| | Pull (Langfuse/LangSmith/Braintrust) | Generic push | Dunetrace native semantic |
|---|---|---|---|
| Setup | Connect once, background polling | One HTTP call per result, you control timing | Zero setup — sampling-based, automatic |
| Best for | Already running one of these three | Custom pipelines, backfills, unsupported tools | Don't want to run/pay for a separate eval system |
| False-positive management | None (trust the provider's own judgment) | None | Confidence floor, grouping, feedback loop, second opinion |
