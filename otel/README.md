# OTel export test harness

Self-contained harness for verifying that the Dunetrace SDK exports spans to real
OTel backends with the right shape. Runs independently of the main Dunetrace
stack: the SDK exports OTLP to a collector, and the collector fans out to Jaeger
and Grafana Tempo (and, with a key, Datadog).

## Quick start

```bash
docker compose -f otel/docker-compose.yml up -d
python otel/exercise_agent.py     # push a full run through the SDK
python otel/verify.py             # assert the trace landed with the right shape
```

`verify.py` prints the spans and exits 0 on PASS. To see the raw spans the
collector received:

```bash
docker compose -f otel/docker-compose.yml logs collector
```

UIs: Jaeger at http://localhost:16686, Grafana (Tempo) at http://localhost:3001.

Tear down with `docker compose -f otel/docker-compose.yml down`.

## What the exerciser produces

One run exercising every span type, plus the server-side findings the detector
service would emit:

```
dunetrace.run
├── chat gpt-4o                       (gen_ai.* conventions)
├── dunetrace.tool.order_lookup
├── dunetrace.tool.api.shipping.com   (http.request.method, server.address, url.full)
├── dunetrace.retrieval
├── dunetrace.voice.transcription
├── dunetrace.voice.tts
├── event dunetrace.voice.silence     (VAD, on the run span)
├── dunetrace.signal.SLOW_STEP        (emitted server-side, ERROR status)
└── dunetrace.policy.switch_model     (emitted server-side)
```

## Pointing the SDK at each backend

The SDK only needs the OTLP endpoint. The collector does the fan-out, so you
usually point the SDK at the collector and change the collector's exporters.

```bash
export DUNETRACE_OTEL_ENABLED=1
export DUNETRACE_OTEL_ENDPOINT=http://localhost:4317   # collector, gRPC
export DUNETRACE_OTEL_PROTOCOL=grpc
```

### Jaeger (open-source, OTel-native)

Runs in this harness. Query API at http://localhost:16686. `verify.py` reads it.
This is the "Signoz or Jaeger" open-source slot; Signoz notes are below.

### Grafana Tempo + Grafana

Runs in this harness. Tempo query API at http://localhost:3200; Grafana is
pre-provisioned with Tempo as a datasource at http://localhost:3001 (Explore ->
Tempo -> search by trace ID). Loki is not needed for traces; add it only if you
also export logs.

### Datadog

Needs a `DD_API_KEY` (test tier is fine). Datadog reads the GenAI conventions
natively, so the LLM spans land in LLM Observability without extra mapping.

```bash
export DD_API_KEY=xxxx;  export DD_SITE=datadoghq.com   # datadoghq.eu for EU
docker compose -f otel/docker-compose.yml stop collector
docker run --rm --network otel_default \
  -e DD_API_KEY -e DD_SITE -p 4317:4317 -p 4318:4318 \
  -v "$PWD/otel/collector-config.datadog.yaml:/etc/otel/config.yaml:ro" \
  otel/opentelemetry-collector-contrib:0.111.0 --config=/etc/otel/config.yaml
```

Or send OTLP straight to the Datadog Agent (DD Agent 7.x has an OTLP receiver on
4317) and skip the collector entirely.

### Signoz (open-source alternative to Jaeger)

Signoz ships its own OTel collector. Point the SDK (or this harness's collector)
at the Signoz collector's OTLP endpoint (default `localhost:4317` when Signoz
runs locally). Its Traces explorer groups by `service.name` (= `dunetrace`) and
shows the `dunetrace.*` and `gen_ai.*` attributes as span tags.

## Env the SDK reads

See `dunetrace/otel.py`. `DUNETRACE_OTEL_ENDPOINT`, `_PROTOCOL` (grpc |
http/protobuf), `_HEADERS`, `_SERVICE_NAME`, `_SAMPLING_RATIO`,
`_CAPTURE_CONTENT`, and `DUNETRACE_ORG_ID`. `OTEL_EXPORTER_OTLP_INSECURE=true`
lets gRPC talk to a plaintext local collector.
