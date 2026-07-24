# Send Traceloop / OpenLLMetry traces to Dunetrace

[Traceloop](https://www.traceloop.com/) (OpenLLMetry) is widely used OpenTelemetry
instrumentation for LLM apps. Its `gen_ai.*` spans ingest into Dunetrace. See
[OpenTelemetry ingestion](otel-ingestion.md) for the full picture; this is the
Traceloop-specific setup.

## With the Traceloop SDK

Traceloop sends OTLP. Point it at the Dunetrace ingest endpoint with your key.

```python
from traceloop.sdk import Traceloop

Traceloop.init(
    api_endpoint="https://<your-dunetrace-ingest>/v1/otlp",
    headers={"Authorization": "Bearer dt_live_..."},
)
```

`Traceloop.init()` instruments your LLM and vector-store clients. Each call ships
to Dunetrace as an OTLP span.

## With the bare instrumentation + your own provider

If you use `opentelemetry-instrumentation-openai` (the OpenLLMetry instrumentor)
with your own `TracerProvider`, add a Dunetrace OTLP exporter to it, the same way
as the [OTel contrib guide](otel-with-otel-contrib.md), and call
`OpenAIInstrumentor().instrument()`.

## What Dunetrace reads

Traceloop uses `gen_ai.provider.name`, `gen_ai.request.model`,
`gen_ai.usage.input_tokens` / `output_tokens`, and puts the assistant reply under
the structured `gen_ai.output.messages`. Dunetrace maps all of these, including
parsing `gen_ai.output.messages` for the output text, so runs carry model,
tokens, and the response content.

## Verify

The dashboard's **OTel Receiver** page should show `spans_received` and
`events_translated` climbing after a few calls.
