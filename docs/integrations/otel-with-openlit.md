# Send OpenLIT traces to Dunetrace

[OpenLIT](https://github.com/openlit/openlit) is open-source OpenTelemetry
instrumentation for LLM apps. It emits `gen_ai.*` spans that Dunetrace ingests
and runs detectors on. See [OpenTelemetry ingestion](otel-ingestion.md) for the
full picture; this is the OpenLIT-specific setup.

## Direct OTLP from your app

OpenLIT sends OTLP, so point it at the Dunetrace ingest endpoint with your API
key as a header.

```python
import openlit

openlit.init(
    otlp_endpoint="https://<your-dunetrace-ingest>/v1/otlp",
    otlp_headers="Authorization=Bearer dt_live_...",
)
```

`openlit.init()` instruments your OpenAI, Anthropic, and other supported clients.
Every LLM call then ships as an OTLP span to Dunetrace.

## Through a collector

If you already run a collector, let OpenLIT send to it (its default
`localhost:4318`) and add Dunetrace as an `otlphttp` exporter on the collector
(see the collector snippet in [OpenTelemetry ingestion](otel-ingestion.md)).

## What Dunetrace reads

OpenLIT sets `gen_ai.operation.name`, `gen_ai.provider.name`, and
`gen_ai.request.model`, which Dunetrace classifies as an LLM call and maps to
model and, when present, token counts. A bare LLM call with no surrounding agent
span still produces a Dunetrace run (the single span is translated as the call).

## Verify

Open the dashboard's **OTel Receiver** page. `spans_received` and
`events_translated` should climb after a few calls. If `received` climbs but
`translated` does not, the spans are arriving but not classifying; check that
`gen_ai.request.model` is set.
