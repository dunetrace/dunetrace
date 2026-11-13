# OpenTelemetry export

Dunetrace can emit its runs as OpenTelemetry spans to any OTLP endpoint, so the
same run/LLM/tool/retrieval/voice data that populates the Dunetrace dashboard
also flows into whatever observability stack you already run: Datadog, Grafana
Tempo, Honeycomb, Signoz, Jaeger, or a self-hosted OTel collector.

This is dual emission, not a replacement. Dunetrace's own ingest keeps working
exactly as before. OTel export is additive and opt-in: set two env vars and
your agent spans start showing up in your backend, with no code changes.

- [Datadog setup](otel-datadog.md)
- [Grafana Tempo setup](otel-grafana.md)
- [Signoz setup](otel-signoz.md)
- [Honeycomb setup](otel-honeycomb.md)

## Enable it

Install the SDK with the `otel` extra, then set the endpoint:

```bash
pip install 'dunetrace[otel]'

export DUNETRACE_OTEL_ENABLED=1
export DUNETRACE_OTEL_ENDPOINT=http://localhost:4317   # your OTLP collector
```

That is the whole change. The SDK reads these at startup and, when enabled,
builds an OTLP exporter on a background pipeline. Your existing `dt.run(...)`,
`@dt.agent`, and `auto_instrument()` code is untouched.

### Configuration

All config is env-driven. Nothing here is required except enabling export and
setting an endpoint.

| Variable | Default | Meaning |
|---|---|---|
| `DUNETRACE_OTEL_ENABLED` | `false` | Turn export on. Must be set explicitly. |
| `DUNETRACE_OTEL_ENDPOINT` | (none) | OTLP endpoint URL. Required when enabled. |
| `DUNETRACE_OTEL_PROTOCOL` | `grpc` | `grpc` or `http/protobuf`. |
| `DUNETRACE_OTEL_HEADERS` | (none) | Auth headers, `k1=v1,k2=v2` (e.g. `DD-API-KEY=xxx`). |
| `DUNETRACE_OTEL_SERVICE_NAME` | `dunetrace` | `service.name` on every span. |
| `DUNETRACE_OTEL_SAMPLING_RATIO` | `1.0` | Head sampling ratio, 0.0 to 1.0. |
| `DUNETRACE_OTEL_CAPTURE_CONTENT` | `true` | Include content attributes (see PII below). |
| `DUNETRACE_ORG_ID` | (none) | Optional `dunetrace.org_id` resource label. |

The standard `OTEL_*` env vars the OTLP exporter reads natively still apply on
top of these, e.g. `OTEL_EXPORTER_OTLP_INSECURE=true` to talk plaintext gRPC to
a local collector.

## What gets exported

One agent run is one trace. Its spans:

```
dunetrace.run                          the run, root span
├── chat gpt-4o                        each LLM call (GenAI conventions)
├── dunetrace.tool.{name}              each tool call
├── dunetrace.retrieval                each retrieval
├── dunetrace.voice.transcription      STT (voice agents)
├── dunetrace.voice.tts                TTS (voice agents)
├── event dunetrace.voice.{vad}        VAD transitions, as span events on the run
├── dunetrace.signal.{failure_type}    detector signals (emitted server-side)
└── dunetrace.policy.{action}          policy interventions (emitted server-side)
```

## Semantic conventions

Dunetrace follows the standard OTel conventions where they exist, and namespaces
its own concepts under `dunetrace.*` where they do not.

**LLM calls** use the [OpenTelemetry GenAI conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/):
`gen_ai.operation.name`, `gen_ai.provider.name` (the current attribute, not the
deprecated `gen_ai.system`), `gen_ai.request.model`, `gen_ai.usage.input_tokens`,
`gen_ai.usage.output_tokens`, `gen_ai.response.finish_reasons`. Cost and
truncation are Dunetrace-specific: `dunetrace.llm.cost_usd`,
`dunetrace.llm.output_truncated`.

**HTTP-shaped tool calls** (tools that call an external API) use the current
stable [HTTP conventions](https://opentelemetry.io/docs/specs/semconv/http/):
`http.request.method`, `server.address`, `url.full`, `http.response.status_code`,
alongside the `dunetrace.tool.*` attributes.

**Runs, tools, retrievals, and voice events** have no standard OTel operation
type, so they use `dunetrace.*` attributes: `dunetrace.run.id`,
`dunetrace.run.agent_id`, `dunetrace.run.status`, `dunetrace.run.duration_ms`,
`dunetrace.tool.name`, `dunetrace.tool.result_status`,
`dunetrace.retrieval.vector_store`, `dunetrace.retrieval.document_count`,
`dunetrace.voice.confidence`, and so on.

The GenAI conventions are still in Development status upstream, so specific
attribute names may shift. Dunetrace tracks the stable set.

## Trace correlation

A run's trace ID is derived deterministically from its Dunetrace run ID, so a
run maps to exactly one trace and you can jump between the two systems:

- In your OTel backend, every span carries `dunetrace.run.id`. Search for it to
  find the run.
- In your own code, the active run exposes `run.otel_trace_id` and
  `run.otel_span_id` (W3C hex) when export is on, so you can log them next to
  your application logs or link straight to the run in your backend.

Because the trace ID is deterministic, the detector service can attach signal
and policy spans to a run's trace after the fact (see below) even though the SDK
already closed the run's root span.

## Signals and policies

Signals (tool loops, cost spikes, and the rest) and policy interventions are
decided server-side by the detector service, after your agent's run has already
finished and its spans have been exported. An OTel span event cannot be added to
a span that is already closed, so Dunetrace emits these as child spans in the
run's trace instead: `dunetrace.signal.{failure_type}` and
`dunetrace.policy.{action}`, parented onto the run by its deterministic trace ID.
A HIGH or CRITICAL signal also marks its span as an error.

To turn this on, the detector service needs the same `DUNETRACE_OTEL_*` env set
in its own environment. See [operations](../operations.md).

## PII and content

Dunetrace is raw-by-default, so content attributes (tool args, request URLs,
retrieval queries, tool error messages, and string signal evidence) are included
by default. Set `DUNETRACE_OTEL_CAPTURE_CONTENT=false` to drop them and keep only
metadata (names, counts, statuses, latencies, hostnames, token counts). Voice
spans never carry raw transcript or TTS text either way, only character counts
and provider metadata.

## Failure isolation

OTel export can never affect your agent. Export runs on a background thread with
a bounded queue that drops spans on overflow rather than blocking. A circuit
breaker stops export attempts for 60 seconds after 5 failures in 60 seconds, so
a dead or slow collector is not retried on every batch. A missing
`opentelemetry` install, a bad endpoint, or a broken pipeline all degrade to
"OTel disabled" with a logged warning, never an exception in your code.

## Troubleshooting

**Nothing shows up.** Confirm `DUNETRACE_OTEL_ENABLED` and
`DUNETRACE_OTEL_ENDPOINT` are set in the process actually running the agent. Turn
on `logging.getLogger("dunetrace.otel").setLevel(logging.DEBUG)` to see whether
the provider built. A quick check: point at a local collector with a debug
exporter (see `otel/` in the repo) and watch the spans print.

**Connection refused on gRPC.** For a plaintext local collector, set
`OTEL_EXPORTER_OTLP_INSECURE=true`, or use `DUNETRACE_OTEL_PROTOCOL=http/protobuf`
with the `:4318` endpoint.

**Spans arrive but signals do not.** Signal and policy spans come from the
detector service, not the SDK. Make sure the detector service has the
`DUNETRACE_OTEL_*` env set too.

**Content missing.** `DUNETRACE_OTEL_CAPTURE_CONTENT` is probably `false`.

## Try it locally

The repo ships a ready-to-run harness at `otel/`: a collector plus Jaeger and
Grafana Tempo, an exerciser that pushes a full run, and a verifier.

```bash
docker compose -f otel/docker-compose.yml up -d
python otel/exercise_agent.py
python otel/verify.py
```
