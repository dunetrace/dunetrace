# Send Dunetrace to Honeycomb

Honeycomb has a native OTLP endpoint, so the Dunetrace SDK sends to it directly
with no collector. You authenticate with an API key header.

See [OpenTelemetry export](opentelemetry.md) for the full attribute set and the
`DUNETRACE_OTEL_*` config.

## Direct OTLP

```bash
export DUNETRACE_OTEL_ENABLED=1
export DUNETRACE_OTEL_PROTOCOL=http/protobuf
export DUNETRACE_OTEL_ENDPOINT=https://api.honeycomb.io
export DUNETRACE_OTEL_HEADERS="x-honeycomb-team=<your-api-key>"
```

For the EU instance, use `https://api.eu1.honeycomb.io`.

gRPC works too (`DUNETRACE_OTEL_PROTOCOL=grpc`, endpoint
`https://api.honeycomb.io:443`); HTTP is simplest to start with.

Honeycomb routes spans into a dataset by `service.name`, so your runs land in a
dataset named `dunetrace` (or whatever you set `DUNETRACE_OTEL_SERVICE_NAME` to).

## Where to look in Honeycomb

- Open the `dunetrace` dataset. A run is a trace rooted at `dunetrace.run`.
- Query on any attribute: `dunetrace.run.agent_id`, `gen_ai.request.model`,
  `dunetrace.signal.type`, and so on. For example, group by
  `dunetrace.signal.type` where `dunetrace.signal.severity = HIGH` to see which
  failures fire most.
- The `chat {model}` spans carry the GenAI attributes; the
  `dunetrace.tool.*` and `dunetrace.retrieval` spans carry their own.

This backend is documented from the OTLP standard. The attributes are the same
ones verified against Jaeger and Grafana Tempo, so they render the same way in
Honeycomb.
