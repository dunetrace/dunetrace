# Send Dunetrace to Grafana Tempo

Grafana Tempo stores traces and Grafana explores them. Tempo has a native OTLP
receiver, so you can point the SDK at Tempo directly, or at a collector that
forwards to it.

See [OpenTelemetry export](opentelemetry.md) for the full attribute set and the
`DUNETRACE_OTEL_*` config.

## Grafana Cloud

Grafana Cloud gives you an OTLP endpoint and a token. Send OTLP over HTTP with a
basic-auth header:

```bash
export DUNETRACE_OTEL_ENABLED=1
export DUNETRACE_OTEL_PROTOCOL=http/protobuf
export DUNETRACE_OTEL_ENDPOINT=https://otlp-gateway-<region>.grafana.net/otlp
export DUNETRACE_OTEL_HEADERS="Authorization=Basic <base64 instanceID:token>"
```

Build the header value from `<instanceID>:<token>` base64-encoded, as shown in
your Grafana Cloud OTLP connection page.

## Self-hosted Tempo

Point the SDK straight at Tempo's OTLP receiver:

```bash
export DUNETRACE_OTEL_ENABLED=1
export DUNETRACE_OTEL_ENDPOINT=http://tempo:4317
export OTEL_EXPORTER_OTLP_INSECURE=true
```

A minimal Tempo config with the OTLP receiver on 4317 (see `otel/tempo.yaml` in
the repo for a full one):

```yaml
distributor:
  receivers:
    otlp:
      protocols:
        grpc:
          endpoint: 0.0.0.0:4317
```

Add Tempo as a Grafana datasource (URL `http://tempo:3200`). The repo's
`otel/grafana-datasources.yaml` provisions this automatically.

## Where to look in Grafana

- **Explore > Tempo**: search by trace ID (the value of `run.otel_trace_id`, or
  `dunetrace.run.id` mapped through the deterministic derivation) or use TraceQL,
  e.g. `{ .dunetrace.run.agent_id = "checkout" }`.
- The run renders as `dunetrace.run` with the LLM/tool/retrieval/voice spans
  nested under it, and any `dunetrace.signal.*` / `dunetrace.policy.*` spans in
  the same trace.

## Logs (optional)

Tempo is traces only. If you also want Dunetrace events as logs in Loki, use the
SDK's NDJSON stdout mode (`emit_as_json=True`) and scrape stdout with Promtail or
the Grafana Agent. That path is independent of OTLP trace export.
