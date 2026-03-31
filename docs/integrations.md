# Integrations

---

## LangChain / LangGraph

```bash
pip install 'dunetrace[langchain]' langchain-openai langgraph python-dotenv       # OpenAI
pip install 'dunetrace[langchain]' langchain-anthropic langgraph python-dotenv    # Anthropic
pip install 'dunetrace[langchain]' langchain-google-genai langgraph python-dotenv # Gemini
```

```python
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dunetrace import Dunetrace
from dunetrace.integrations.langchain import DunetraceCallbackHandler

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Results for {query}"

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [web_search]
system_prompt = "You are a helpful assistant."
user_input = "What is the capital of France?"

dt = Dunetrace()
callback = DunetraceCallbackHandler(dt, agent_id="my-agent")

agent = create_react_agent(llm, tools, prompt=system_prompt)
result = agent.invoke(
    {"messages": [("human", user_input)]},
    config={"callbacks": [callback]},
)
dt.shutdown()
```

---

## Manual instrumentation

Use this when no native integration exists for your framework.

```python
from dunetrace import Dunetrace

dt = Dunetrace()
user_input = "What is the capital of France?"

with dt.run("my-agent", user_input=user_input, model="gpt-4o", tools=["search"]) as run:
    run.llm_called("gpt-4o", prompt_tokens=150)
    run.llm_responded(finish_reason="tool_calls", latency_ms=320)

    run.tool_called("search", {"query": user_input})
    run.tool_responded("search", success=True, output_length=512)

    run.llm_called("gpt-4o", prompt_tokens=480)
    run.llm_responded(finish_reason="stop", output_length=120)
    run.final_answer()

dt.shutdown()
```

---

## Grafana / Loki

```python
dt = Dunetrace(emit_as_json=True)
```

Writes every event to stdout as a Loki-compatible NDJSON line. Each line includes `ts`, `level`, `logger`, `event_type`, `agent_id`, `run_id`, `step_index`, and `payload`. Works alongside HTTP ingest, both can be active at the same time.

Minimal Promtail pipeline stage:

```yaml
pipeline_stages:
  - json:
      expressions: {ts: ts, event_type: event_type, agent_id: agent_id}
  - timestamp:
      source: ts
      format: RFC3339Nano
  - labels:
      agent_id:
      event_type:
```

---

## OpenTelemetry

```bash
pip install 'dunetrace[otel]' opentelemetry-exporter-otlp-proto-grpc
```

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from dunetrace.integrations.otel import DunetraceOTelExporter

resource = Resource.create({
    "service.name": "my-agent-service",
    "deployment.environment": "production",
})
provider = TracerProvider(resource=resource)
provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

dt = Dunetrace(otel_exporter=DunetraceOTelExporter(provider))
```

Each agent run produces a trace with a deterministic `trace_id` derived from `run_id`, so you can correlate Dunetrace signals with infra metrics in Tempo, Honeycomb, Datadog, or Jaeger:

```
Trace
└── Span: "agent_run"     [dunetrace.agent_id, dunetrace.model, dunetrace.tools, …]
    ├── Span: "llm_call"  [gen_ai.request.model, gen_ai.usage.input_tokens, …]
    ├── Span: "tool_call" [dunetrace.tool_name, dunetrace.success, dunetrace.latency_ms]
    └── Span: "retrieval" [dunetrace.index_name, dunetrace.result_count, dunetrace.top_score]
```

Failure signals detected at run end are written as indexed attributes on the root span (`dunetrace.signal.0.failure_type`, `.severity`, `.confidence`). HIGH/CRITICAL signals set `span.status = ERROR`.

Use `endpoint=None` to run OTel-only with no HTTP ingest:

```python
dt = Dunetrace(endpoint=None, otel_exporter=DunetraceOTelExporter(provider))
```

**With LangChain:** `DunetraceCallbackHandler` and `DunetraceOTelExporter` are independent and both active simultaneously without any extra configuration.

For deeper OTel internals (span structure, known gaps, parallel tool calls) see [architecture.md](architecture.md#otel-span-exporter).

---

## Examples

**Basic agent** (no framework, simulates tool loops, prompt injection, RAG failures):

```bash
cd packages/sdk-py
pip install dunetrace
python examples/basic_agent.py
```

**LangChain agent** (real OpenAI calls, auto-instrumented via callback):

```bash
cd packages/sdk-py
pip install 'dunetrace[langchain]' langchain-openai langgraph python-dotenv
```

Add your key to the root `.env`:

```
OPENAI_API_KEY=sk-...
```

```bash
python examples/langchain_agent.py

# Force a tool-loop scenario:
SCENARIO=tool_loop python examples/langchain_agent.py
```

Both examples send events to `http://localhost:8001` by default. Override with `DUNETRACE_ENDPOINT=http://your-host:8001`.
