# Integrating a LangChain Agent with Dunetrace

> **Looking for `dt.auto_instrument()` instead?** It patches LangChain for you, no callback object needed — but requires wrapping the top-level call in `dt.run(...)`. See [auto-instrumentation.md](./integrations/auto-instrumentation.md).

## Quick Start

```bash
pip install 'dunetrace[langchain]' langchain-openai
```

```python
from dunetrace import Dunetrace
from dunetrace.integrations.langchain import DunetraceCallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

dt = Dunetrace()   # local dev, no API key needed
callback = DunetraceCallbackHandler(dt, agent_id="my-agent", model="gpt-4o")

agent = create_react_agent(ChatOpenAI(model="gpt-4o"), tools=[])
result = agent.invoke(
    {"messages": [("human", "What is the capital of France?")]},
    config={"callbacks": [callback]},   # <-- this is the whole integration
)

dt.shutdown()
```

Start the backend once, locally, before running this: `docker compose up -d`.

## What this does

`DunetraceCallbackHandler` plugs into LangChain's callback system and translates every LLM call, tool call, and retriever call into Dunetrace events automatically — no changes to your agent logic. Works with LangGraph (`create_react_agent`, custom graphs) and the older `AgentExecutor` the same way; just pass the same `callback` in `config={"callbacks": [...]}`. Async (`ainvoke`) works identically to sync (`invoke`).

## Constructor options

| Parameter | Required | Description |
|---|---|---|
| `agent_id` | Yes | Identifier shown in the dashboard |
| `system_prompt` | No | Used to compute a version fingerprint when your prompt changes |
| `model` | No | Model name for display and detector context |
| `tools` | No | Tool name list — used by `TOOL_AVOIDANCE` |

## Verification

```bash
SCENARIO=tool_loop python examples/langchain_agent.py
```

Calls `web_search` six times in one run, triggering `TOOL_LOOP`. Check the dashboard at `http://localhost:3000` — the signal should appear within ~15 seconds.

---

## Advanced (optional)

### RAG agents

Retriever calls are captured automatically — no extra code. `result_count` and `top_score` are extracted from document metadata (`score`, `relevance_score`, or `similarity`), feeding `RAG_EMPTY_RETRIEVAL`.

### Accessing the run inside a tool

```python
from dunetrace import get_current_run

@tool
def web_search(query: str) -> str:
    run = get_current_run()
    if rate_limited():
        run.external_signal("rate_limit", source="serpapi")
    return do_search(query)
```

### Concurrent invocations

The handler is thread-safe — one instance can be shared across concurrent `invoke()` calls, each tracked independently by LangChain's own root `run_id`. Stale runs (never completed after 30 minutes) are pruned automatically.

### What's captured

Every LLM call (model, tokens, latency, raw prompt/completion), every tool call (name, success/failure, raw args/output — including framework-handled errors via `handle_tool_error=True`), every retriever call (index, count, score, raw query), and run-level totals. Not captured: intermediate sub-chain inputs/outputs (only the root chain boundary is a run), and streaming token counts in some provider/version combinations.

### Troubleshooting

- **No runs appear** — confirm `dt.shutdown()` was called; try `Dunetrace(debug=True)` for verbose logs
- **Token counts missing** — some providers/LangChain versions omit `token_usage`; the handler falls back to `usage_metadata`, and omits the fields entirely if both are absent (doesn't break detectors)
- **Detectors fire too aggressively** — tune thresholds in `detectors.yml` and restart the detector service
