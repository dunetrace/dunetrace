# Send Dunetrace to Datadog

Dunetrace exports OTLP, and Datadog ingests OTLP two ways: through an OTel
Collector with the Datadog exporter, or straight into the Datadog Agent's OTLP
receiver. Datadog reads the GenAI conventions natively, so Dunetrace's LLM spans
land in APM and LLM Observability without extra mapping.

See [OpenTelemetry export](opentelemetry.md) for the full attribute set and the
`DUNETRACE_OTEL_*` config.

## Option A: OTel Collector with the Datadog exporter

Point the SDK at a collector, and give the collector a Datadog exporter.

```bash
export DUNETRACE_OTEL_ENABLED=1
export DUNETRACE_OTEL_ENDPOINT=http://localhost:4317
```

Collector config (`collector-config.datadog.yaml`, also in `otel/` in the repo):

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 1s

exporters:
  datadog:
    api:
      key: ${env:DD_API_KEY}
      site: ${env:DD_SITE}      # datadoghq.com, or datadoghq.eu for EU

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [datadog]
```

Run it (contrib image, which includes the Datadog exporter):

```bash
export DD_API_KEY=xxxx
export DD_SITE=datadoghq.com
docker run --rm -p 4317:4317 -p 4318:4318 \
  -e DD_API_KEY -e DD_SITE \
  -v "$PWD/collector-config.datadog.yaml:/etc/otel/config.yaml:ro" \
  otel/opentelemetry-collector-contrib:0.111.0 --config=/etc/otel/config.yaml
```

## Option B: straight to the Datadog Agent

The Datadog Agent (7.x) has its own OTLP receiver. Enable it in `datadog.yaml`:

```yaml
otlp_config:
  receiver:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
```

Then point the SDK at the Agent and skip the collector:

```bash
export DUNETRACE_OTEL_ENABLED=1
export DUNETRACE_OTEL_ENDPOINT=http://localhost:4317
```

## Sending the API key from the SDK directly

If you export OTLP to a Datadog intake that authenticates with a header rather
than running a collector, pass it as an OTLP header:

```bash
export DUNETRACE_OTEL_HEADERS="DD-API-KEY=xxxx"
```

## Where to look in Datadog

- **APM > Traces**: search `dunetrace.run.id:<id>` or filter by
  `service:dunetrace`.
- **LLM Observability**: the `chat {model}` spans appear with their
  `gen_ai.*` attributes (provider, model, input/output tokens).
- A HIGH or CRITICAL `dunetrace.signal.*` span shows as an error on the trace.
