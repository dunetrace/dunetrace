# Send Dunetrace to Signoz

Signoz is an open-source, OTel-native observability stack. It ships its own OTel
Collector, so you point the Dunetrace SDK at the Signoz collector's OTLP
endpoint.

See [OpenTelemetry export](opentelemetry.md) for the full attribute set and the
`DUNETRACE_OTEL_*` config.

## Self-hosted Signoz

A local Signoz install exposes its collector's OTLP receiver on 4317 (gRPC) and
4318 (HTTP). Point the SDK at it:

```bash
export DUNETRACE_OTEL_ENABLED=1
export DUNETRACE_OTEL_ENDPOINT=http://localhost:4317
export OTEL_EXPORTER_OTLP_INSECURE=true
```

If your agent runs in a container on the same Docker network as Signoz, use the
collector's service name instead of localhost, e.g.
`http://signoz-otel-collector:4317`.

## Signoz Cloud

Signoz Cloud gives you a region endpoint and an ingestion key. Send the key as
an OTLP header:

```bash
export DUNETRACE_OTEL_ENABLED=1
export DUNETRACE_OTEL_PROTOCOL=http/protobuf
export DUNETRACE_OTEL_ENDPOINT=https://ingest.<region>.signoz.cloud:443
export DUNETRACE_OTEL_HEADERS="signoz-ingestion-key=<your-key>"
```

## Where to look in Signoz

- **Traces**: filter by `service.name = dunetrace`, or search the
  `dunetrace.run.id` attribute. The trace opens as `dunetrace.run` with the
  LLM/tool/retrieval/voice spans nested under it.
- The GenAI attributes (`gen_ai.provider.name`, `gen_ai.request.model`,
  `gen_ai.usage.*`) show as span tags on the `chat {model}` spans.
- `dunetrace.signal.*` spans with ERROR status surface in the errors view.
