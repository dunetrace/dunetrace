# Real emitter capture fixtures (Phase 4)

OTLP spans captured from the actual instrumentation of three LLM-observability
emitters. `test_otlp_emitter_integration.py` replays them through the real
translator (`otlp_to_events`) and, where the detector service is importable,
through the real run reconstruction and detector suite.

## How they were captured

A real `openai` SDK `chat.completions.create()` call was driven through an
`httpx.MockTransport` while each emitter's instrumentation was active. Only the
model's HTTP response is stubbed. The span attributes are each emitter's genuine
production output, so these are real spans, not hand-written fixtures.

Versions: `opentelemetry-instrumentation-openai-v2` (OTel-contrib GenAI),
`opentelemetry-instrumentation-openai==0.62.1` (Traceloop / OpenLLMetry),
`openlit==1.44.0`, `openai` 1.x/2.x.

## What each emitter emits

| | Span name | model | tokens | output text |
|---|---|---|---|---|
| OTel-contrib | `chat gpt-4o` | `gen_ai.request.model` + `gen_ai.response.model` | `gen_ai.usage.input_tokens` / `output_tokens` | (content capture off by default) |
| Traceloop | `openai.chat` | `gen_ai.request.model` + `gen_ai.response.model` | `gen_ai.usage.input_tokens` / `output_tokens` | **`gen_ai.output.messages`** (structured) |
| OpenLIT | `chat gpt-4o` | `gen_ai.request.model` | absent (see below) | absent |

All three use the current GenAI namespace (`gen_ai.operation.name`,
`gen_ai.provider.name`, `gen_ai.request.model`), which the translator already
classified as LLM and mapped correctly.

## Gaps found, and the fixes

Two real gaps surfaced only because these are real emitter spans:

1. **Single-span traces dropped the LLM call.** All three emitters produce one
   span per bare LLM call, and that span is the trace root (no parent). The
   mapper treated the root purely as the run boundary and never emitted
   `llm.called` / `llm.responded`, so the LLM data was lost. Fixed in
   `otel.py`: a root span that classifies as a real operation (llm/tool/
   retrieval) is now translated too, not just used as the run boundary.

2. **Traceloop output text was not captured.** Traceloop emits the assistant
   reply under `gen_ai.output.messages` (a structured list of message parts),
   not `gen_ai.completion`. Fixed in both `otel.py` and the SDK receiver
   `otel_receiver.py`: `gen_ai.output.messages` / `gen_ai.input.messages` are
   now parsed for their text.

## Notes

- **OpenLIT** hit an internal `AttributeError` extracting tokens/content against
  the stubbed response (an openlit-side issue with the mock, not a Dunetrace
  translation gap), so its capture carries the core `gen_ai.*` attributes but no
  tokens and an ERROR span status. Dunetrace still translates the model cleanly
  and the run resolves to `run.errored`, which is correct for an errored span.

## Reproduce

The capture driver (`capture_emitter.py`) is not committed since it needs the
emitter packages installed. To regenerate: install an emitter, run a stubbed
`chat.completions.create()` under its instrumentation with an
`InMemorySpanExporter`, and serialize the finished spans to the OTLP
`resourceSpans` shape these files use.
