# Integrating a Haystack Pipeline with Dunetrace

This guide covers adding Dunetrace monitoring to a Haystack 2.x pipeline. The integration is a single tracer registration — no changes to your pipeline or component code.

---

## How It Works

`DunetraceHaystackTracer` implements Haystack's `Tracer` protocol. Register it once with `haystack.tracing.enable_tracing()` and every subsequent pipeline run is automatically tracked.

| Haystack event | Dunetrace event |
|---|---|
| `haystack.pipeline.run` starts | `RUN_STARTED` |
| `haystack.component.run` for a `*Generator` / `*LLM*` | `LLM_CALLED` → `LLM_RESPONDED` (with token counts + latency) |
| `haystack.component.run` for a `*Retriever*` | `RETRIEVAL_CALLED` → `RETRIEVAL_RESPONDED` (with result count + top score) |
| `haystack.component.run` for a `ToolInvoker` / `ComponentTool` | `TOOL_CALLED` → `TOOL_RESPONDED` |
| `haystack.pipeline.run` completes | `RUN_COMPLETED` |
| Any unhandled exception | `RUN_ERRORED` |

Token counts (prompt + completion) are extracted from component output metadata — no extra configuration needed for OpenAI, Anthropic, Amazon Bedrock, Cohere, Gemini, Mistral, and Ollama generators.

Content tracing is enabled so that Haystack passes component I/O to the tracer, and that text — prompts, documents, outputs — is sent to the backend as-is.

---

## Prerequisites

- Dunetrace backend running (`docker compose up -d`)
- Python 3.11+

> **Local dev — no API key needed.** The backend accepts requests without any API key when running locally. API keys are only required for production deployments.

---

## Step 1: Install Dependencies

```bash
pip install 'dunetrace[haystack]'
```

This installs the SDK along with `haystack-ai`. Also install your LLM provider package:

```bash
pip install haystack-ai openai             # OpenAI generators
pip install haystack-ai anthropic          # Anthropic generators
pip install haystack-ai google-generativeai # Gemini
```

---

## Step 2: Register the Tracer

Call `haystack.tracing.enable_tracing()` once at startup, before running any pipeline.

```python
import haystack.tracing
from dunetrace import Dunetrace
from dunetrace.integrations.haystack import DunetraceHaystackTracer

# Local dev — no api_key needed
dt = Dunetrace(endpoint="http://localhost:8001")

# Production
# dt = Dunetrace(endpoint="https://your-dunetrace-ingest", api_key="dt_live_...")

haystack.tracing.enable_tracing(
    DunetraceHaystackTracer(
        dt,
        agent_id="my-haystack-pipeline",   # identifies this agent in the dashboard
        system_prompt=SYSTEM_PROMPT,        # optional — helps pattern analysis
        model="gpt-4o-mini",               # declared model for this agent
        tools=["web_search", "calculator"], # tool names for detector context
    )
)
```

**Constructor parameters:**

| Parameter | Required | Description |
|---|---|---|
| `client` | Yes | Your `Dunetrace` instance |
| `agent_id` | Yes | Identifier for this agent — must match your `api_keys` row |
| `system_prompt` | No | System prompt string — used to compute a version fingerprint |
| `model` | No | Default model name for display and detector context |
| `tools` | No | List of tool name strings — used by detectors like `TOOL_AVOIDANCE` |

---

## Step 3: Run Your Pipeline Normally

No changes to pipeline or component code are needed.

### Simple LLM pipeline

```python
from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

pipeline = Pipeline()
pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini"))

result = pipeline.run({
    "llm": {"messages": [ChatMessage.from_user("What is the capital of France?")]}
})
print(result["llm"]["replies"][0].text)
```

### RAG pipeline (retriever + generator)

```python
from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.dataclasses import ChatMessage, Document

store = InMemoryDocumentStore()
store.write_documents([Document(content="Paris is the capital of France.")])

rag = Pipeline()
rag.add_component("retriever", InMemoryBM25Retriever(document_store=store))
rag.add_component("prompt",    ChatPromptBuilder(template=[
    ChatMessage.from_system("Answer using only these documents:\n{% for d in documents %}{{ d.content }}\n{% endfor %}"),
    ChatMessage.from_user("{{ question }}"),
]))
rag.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini"))
rag.connect("retriever.documents", "prompt.documents")
rag.connect("prompt.prompt", "llm.messages")

result = rag.run({
    "retriever": {"query": "What is the capital of France?"},
    "prompt":    {"question": "What is the capital of France?"},
})
# Both RETRIEVAL_CALLED/RESPONDED and LLM_CALLED/RESPONDED are emitted automatically.
```

### Haystack Agent component (2.3+)

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.tools import ComponentTool
from haystack import Pipeline

@component
class WebSearch:
    @component.output_types(results=list)
    def run(self, query: str) -> dict:
        return {"results": [f"Search results for: {query}"]}

llm   = OpenAIChatGenerator(model="gpt-4o-mini")
agent = Agent(chat_generator=llm, tools=[ComponentTool(component=WebSearch())])

pipeline = Pipeline()
pipeline.add_component("agent", agent)

result = pipeline.run({"agent": {"messages": [ChatMessage.from_user("Search for AI trends")]}})
# The agent's internal LLM calls and tool invocations are all tracked.
```

---

## Step 4: Shutdown on Process Exit

```python
import atexit
atexit.register(dt.shutdown)
```

Or explicitly:

```python
dt.shutdown(timeout=5)  # waits up to 5 seconds to flush pending events
```

---

## Complete Example

```python
import os
import atexit
import haystack.tracing
from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.dataclasses import ChatMessage, Document
from dunetrace import Dunetrace
from dunetrace.integrations.haystack import DunetraceHaystackTracer

SYSTEM_PROMPT = "You are a research assistant. Use the provided documents to answer questions."

dt = Dunetrace(
    endpoint=os.environ.get("DUNETRACE_ENDPOINT", "http://localhost:8001"),
    api_key=os.environ.get("DUNETRACE_API_KEY", ""),
)
atexit.register(dt.shutdown)

haystack.tracing.enable_tracing(
    DunetraceHaystackTracer(
        dt,
        agent_id="rag-pipeline",
        system_prompt=SYSTEM_PROMPT,
        model="gpt-4o-mini",
    )
)

store = InMemoryDocumentStore()
store.write_documents([
    Document(content="Paris is the capital of France."),
    Document(content="Berlin is the capital of Germany."),
])

rag = Pipeline()
rag.add_component("retriever", InMemoryBM25Retriever(document_store=store))
rag.add_component("prompt",    ChatPromptBuilder(template=[
    ChatMessage.from_system(SYSTEM_PROMPT + "\n\nDocuments:\n{% for d in documents %}{{ d.content }}\n{% endfor %}"),
    ChatMessage.from_user("{{ question }}"),
]))
rag.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini"))
rag.connect("retriever.documents", "prompt.documents")
rag.connect("prompt.prompt",       "llm.messages")

result = rag.run({
    "retriever": {"query": "What is the capital of France?"},
    "prompt":    {"question": "What is the capital of France?"},
})
print(result["llm"]["replies"][0].text)
```

---

## Async Pipelines

`DunetraceHaystackTracer` works with `AsyncPipeline` without any changes. The `current_span()` implementation uses a `ContextVar` so spans are correctly isolated across concurrent coroutines.

```python
from haystack.core.pipeline.async_pipeline import AsyncPipeline

async_pipeline = AsyncPipeline()
async_pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini"))
# haystack.tracing.enable_tracing(...) is already set from startup

result = await async_pipeline.run({
    "llm": {"messages": [ChatMessage.from_user("Hello")]}
})
```

---

## What Is and Isn't Captured

**Captured automatically:**
- Every LLM call: model name, token counts (prompt + completion), latency, finish reason, raw prompt/completion text
- Every retrieval: index/component name, result count, top similarity score
- Every tool call via `ToolInvoker` or `ComponentTool`: name, success/failure, output length, raw arguments and output
- Run-level: total steps, latency, exit reason, raw user input text
- Error messages, raw text

**Not captured:**
- Intermediate `PromptBuilder` / `OutputAdapter` outputs (classified as "other" — no events)
- Streaming token counts (Haystack does not expose these in span metadata)
- Custom components with no standard type keywords — classify by adding `"Generator"` or `"Retriever"` to the class name, or track manually via `dt.run()` context manager

---

## Verify the Integration

Run your pipeline once, then check:

1. **Dashboard** (`http://your-dashboard:3000`) — the run should appear within 15 seconds
2. **Runs API** — `GET http://your-ingest:8002/v1/runs?agent_id=rag-pipeline`

To confirm detector signals fire end-to-end, run a pipeline that calls a retriever but returns empty results — this triggers `RAG_EMPTY_RETRIEVAL`. Or call the same retriever component 3+ times in one pipeline run to trigger `TOOL_LOOP`.

---

## Troubleshooting

**No runs appear in the dashboard**
- Ensure `haystack.tracing.enable_tracing(...)` is called before `pipeline.run()`
- Call `dt.shutdown()` before process exit to flush the event buffer
- Use `debug=True` in the `Dunetrace()` constructor for verbose logging

**Token counts are missing**
- Token counts come from `replies[0].meta["usage"]`. If the provider does not populate this (some streaming or proxy configurations), token fields are omitted from the event — detectors still work but cost-based signals (`COST_SPIKE`) may not fire

**Custom component not tracked**
- Only components whose class name contains a recognized keyword are classified. For custom generators, extend `OpenAIChatGenerator` or include `"Generator"` in the class name. For custom retrievers, include `"Retriever"` in the class name.

**Detectors fire too aggressively**
- Tune thresholds in `detectors.yml` on the server and restart the detector service. Per-agent overrides are supported — use your `agent_id` as the category name.
