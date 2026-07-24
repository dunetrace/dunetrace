# Send OpenTelemetry contrib GenAI traces to Dunetrace

The official OpenTelemetry Python GenAI instrumentation
(`opentelemetry-instrumentation-openai-v2`) emits the current GenAI convention
spans. Dunetrace ingests them directly. See
[OpenTelemetry ingestion](otel-ingestion.md) for the full picture; this is the
contrib-specific setup.

## Install and instrument

```bash
pip install opentelemetry-instrumentation-openai-v2 \
            opentelemetry-exporter-otlp-proto-http
```

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(
        OTLPSpanExporter(
            endpoint="https://<your-dunetrace-ingest>/v1/otlp/traces",
            headers={"Authorization": "Bearer dt_live_..."},
        )
    )
)
trace.set_tracer_provider(provider)

OpenAIInstrumentor().instrument()
# Your openai calls now ship to Dunetrace as OTLP spans.
```

To also capture prompt and completion text, set
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT` (it is off by
default). Dunetrace reads the content when present but never requires it.

## What Dunetrace reads

This instrumentation uses the current convention: `gen_ai.provider.name`,
`gen_ai.request.model`, `gen_ai.response.model`,
`gen_ai.usage.input_tokens` / `output_tokens`, and
`gen_ai.response.finish_reasons`. Dunetrace maps all of these, so runs carry
model, both token counts, and the finish reason.

## Verify

The dashboard's **OTel Receiver** page should show `spans_received` and
`events_translated` climbing. A single `chat.completions.create()` with no agent
span still produces a run.
