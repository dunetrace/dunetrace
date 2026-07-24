# OpenTelemetry ingestion

If your agents already emit OpenTelemetry traces, Dunetrace can run its
structural detection on them without you adding the Dunetrace SDK. Point your
existing OTel pipeline at Dunetrace and your `gen_ai.*` spans become Dunetrace
runs, with the same detectors (tool loops, cost spikes, context bloat, and the
rest) that native SDK instrumentation gets.

This is the opposite direction from [OpenTelemetry export](opentelemetry.md),
where Dunetrace emits its runs as OTel spans to your backend. Ingestion brings
your OTel spans into Dunetrace. The two are independent and can both be on.

Tested emitters (setup guides):
- [OpenLIT](otel-with-openlit.md)
- [OpenTelemetry contrib GenAI instrumentation](otel-with-otel-contrib.md)
- [Traceloop / OpenLLMetry](otel-with-traceloop.md)

## Two ways in

**1. Send to the Dunetrace OTLP endpoint (most common).** Add Dunetrace as an
OTLP destination in your collector, or export OTLP directly from your app.

```
POST https://<your-dunetrace-ingest>/v1/otlp/traces
```

It accepts OTLP/HTTP as JSON or protobuf, gzip-compressed or not, the same
formats a standard OTel Collector and the Python `OTLPSpanExporter` send.

**2. Attach the SDK receiver in-process.** If you already build your own
`TracerProvider`, add `DunetraceOTelReceiver` as a span processor. It translates
spans into Dunetrace calls locally, no separate endpoint.

```python
from dunetrace import Dunetrace
from dunetrace.integrations.otel_receiver import DunetraceOTelReceiver

dt = Dunetrace(api_key="dt_live_...")
DunetraceOTelReceiver.attach(provider, dt, agent_id="my-agent")
```

## Configuring your collector

Point an OTLP exporter at the Dunetrace endpoint and authenticate (see Auth).

```yaml
exporters:
  otlphttp/dunetrace:
    endpoint: https://<your-dunetrace-ingest>
    headers:
      Authorization: "Bearer dt_live_..."

service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [otlphttp/dunetrace]
```

The `otlphttp` exporter appends `/v1/traces`, so set the endpoint to your ingest
host and Dunetrace serves OTLP at `/v1/otlp/traces` behind your gateway, or send
directly to the full path from an app-level exporter.

## Auth and org attribution

Every incoming span is attributed to a Dunetrace org. Provide the API key any one
of these ways:

- `Authorization: Bearer <key>`
- `X-Dunetrace-API-Key: <key>`
- a `dunetrace.api_key` resource attribute (for emitters that set resource
  attributes but not request headers)

Missing or invalid auth is rejected with 401. OTel ingestion can be turned off
per org (a kill switch on top of rate limiting); it is on by default.

## Semantic conventions supported

Dunetrace reads the OpenTelemetry GenAI conventions first, then the
OpenLLMetry and vector-store attribute keys real emitters also use.

**LLM spans** (classified by `gen_ai.system` / `gen_ai.provider.name` /
`gen_ai.request.model` / `llm.request.model`):
- model: `gen_ai.request.model`, then `gen_ai.response.model`, then
  `llm.request.model`
- tokens: `gen_ai.usage.input_tokens` / `output_tokens` (or the older
  `prompt_tokens` / `completion_tokens`)
- output text: `gen_ai.completion`, or the structured `gen_ai.output.messages`
- finish reason: `gen_ai.response.finish_reasons` and variants

**Tool spans** (`gen_ai.tool.name` / `tool.name`): tool name, args
(`gen_ai.tool.call.arguments` / `traceloop.entity.input`), result status.

**Retrieval spans** (`retrieval.*` / `vector_db.*`): index / vector store,
document count, top score, retrieved content.

A trace whose only span is the LLM call (a bare `chat.completions.create` with
no surrounding agent span) is handled: that root span is translated as the LLM
call, not just used as the run boundary.

Voice events have no OTel convention and are better sent via the Dunetrace SDK.

## Limits

The receiver protects itself and every other org:
- Request bodies over 10MB are rejected (413); gzip bodies that expand past the
  decompression limit are rejected as gzip bombs.
- Per-org span rate limiting (default 1000 spans/sec) returns 429 with
  Retry-After. One org's burst cannot starve another's ingestion.
- Long attribute values are truncated to keep one oversized span from bloating
  storage.

## Seeing that it worked

The dashboard's **OTel Receiver** page shows, per org: spans received, events
translated, spans rejected with a per-reason breakdown, auth failures, and
rate-limit hits, plus anomaly flags (a sudden drop usually means an integration
broke). Start there when spans are not appearing.

## Troubleshooting

**Spans do not appear.** Check the OTel Receiver dashboard. If `auth_failures`
is climbing, the key or header is wrong. If `rejected` is climbing, the reason
breakdown says why (malformed, oversized, rate_limited, disabled_org). If
`received` is climbing but `translated` is not, the spans are arriving but not
classifying as LLM/tool/retrieval (see conventions above).

**403 disabled.** OTel ingestion is turned off for the org. An admin re-enables
it.

**415 unsupported content-type.** Send `application/json` or
`application/x-protobuf`.

**429.** You are over the per-org span rate. Honor the `Retry-After` header, or
ask to raise the limit.
