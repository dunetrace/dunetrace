# Integrating a Haystack Pipeline with Dunetrace

<!--dunetrace:instrument
framework: haystack
language: python
install: pip install 'dunetrace[haystack]'
primary_symbol: dunetrace.integrations.haystack.DunetraceHaystackTracer
mechanism: tracer
opens_own_run: true
requires_run_context: false
target_file: the module that builds the Pipeline, before the first run()
target_example: `app/pipeline.py`, `main.py`
target_hints: ["Pipeline()", "pipeline.run(", "add_component("]
emits: [run.started, run.completed, run.errored, llm.called, llm.responded, tool.called, tool.responded, retrieval.called, retrieval.responded]
verify_cmd: python packages/sdk-py/examples/haystack_agent.py
-->

## Quick Start

```bash
pip install 'dunetrace[haystack]'
```

```python
import haystack.tracing
from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from dunetrace import Dunetrace
from dunetrace.integrations.haystack import DunetraceHaystackTracer

dt = Dunetrace()   # local dev, no API key needed
haystack.tracing.enable_tracing(
    DunetraceHaystackTracer(dt, agent_id="my-pipeline", model="gpt-4o-mini")
)

pipeline = Pipeline()
pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini"))
result = pipeline.run({"llm": {"messages": [ChatMessage.from_user("What is the capital of France?")]}})
print(result["llm"]["replies"][0].text)

dt.shutdown()
```

Start the backend once, locally, before running this: `docker compose up -d`.

## Where this goes

`enable_tracing()` is process-global and goes **once at startup**,
before any pipeline runs. Nothing changes at the pipeline or component level.

Find your pipeline construction with:

```bash
grep -rn "Pipeline()\|pipeline.run(" --include=*.py .
```

Typically `app/pipeline.py` or `main.py`. Put `enable_tracing(...)` at module
import time or in your app startup hook, not inside a request handler, since
calling it per request re-registers the tracer on every call.

## What this does

`DunetraceHaystackTracer` implements Haystack's `Tracer` protocol. Register it once with `enable_tracing()` and every pipeline run afterward is tracked automatically — no changes to pipeline or component code. Generators become LLM events, retrievers become retrieval events, `ToolInvoker`/`ComponentTool` become tool events; components are classified by keywords in their class name (`Generator`, `Retriever`).

## Verification

Run your pipeline once, then check the dashboard at `http://localhost:3000` — the run should appear within ~15 seconds. To confirm a signal fires end-to-end, run a pipeline whose retriever returns 0 results (`RAG_EMPTY_RETRIEVAL`), or call the same component 3+ times in one run (`TOOL_LOOP`).


### If nothing arrives

Work down this list. The first two cover almost every case.

1. **Is a run open?** This integration opens its own run, so there is nothing to wrap. Confirm the registration call (`DunetraceHaystackTracer`) actually ran, and ran **before** the first agent invocation.

2. **Did the process flush?** Events ship from a background thread. Call
   `dt.shutdown()` before the process exits, or the buffer dies with it.

3. **Is the backend reachable?** Both should return `{"status":"ok",...}`:

   ```bash
   curl -s localhost:8001/ready   # ingest
   curl -s localhost:8002/ready   # customer API
   ```

4. **Did anything land?** If your agent id is listed here, instrumentation is
   working and the problem is downstream:

   ```bash
   curl -s localhost:8002/v1/agents
   curl -s "localhost:8002/v1/agents/<your-agent-id>/runs?limit=5"
   ```

5. **Turn on debug logging.** `Dunetrace(debug=True)` logs every event as it is buffered and
   every batch as it ships.

**Runs appear but no signals?** That is usually correct, not a fault. The
detector polls every 5 seconds, and a healthy run produces no signals. Several
detectors also need cross-run baselines and stay dormant until the agent has
run history. Check `docker compose logs detector` if you expected one.

---

## Advanced (optional)

### RAG pipelines

Retriever → generator pipelines need no extra code — both `RETRIEVAL_CALLED`/`RETRIEVAL_RESPONDED` and `LLM_CALLED`/`LLM_RESPONDED` are emitted automatically as the pipeline runs.

### Async pipelines

`DunetraceHaystackTracer` works with `AsyncPipeline` unchanged — spans are isolated across concurrent coroutines via `ContextVar`.

### What's captured

Every LLM call (model, tokens, latency, raw prompt/completion — no config needed for OpenAI, Anthropic, Bedrock, Cohere, Gemini, Mistral, Ollama), every retrieval (component name, count, top score), every tool call via `ToolInvoker`/`ComponentTool`. Not captured: intermediate `PromptBuilder`/`OutputAdapter` outputs, streaming token counts, or custom components whose class name lacks a recognized keyword (add `"Generator"`/`"Retriever"` to the name, or track manually via `dt.run()`).

### Troubleshooting

- **No runs appear** — confirm `enable_tracing(...)` runs before `pipeline.run()`, and `dt.shutdown()` is called
- **Token counts missing** — come from `replies[0].meta["usage"]`; if the provider/proxy doesn't populate it, those fields are omitted (`COST_SPIKE` may not fire, other detectors still work)
- **Custom component not tracked** — extend a recognized base class or include `"Generator"`/`"Retriever"` in the class name
- **Detectors fire too aggressively** — tune `detectors.yml` per `agent_id` and restart the detector service
