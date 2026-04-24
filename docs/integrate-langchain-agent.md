# Integrating a LangChain Agent with Dunetrace

This guide covers adding Dunetrace monitoring to a LangChain or LangGraph agent already running in production. The integration is a single callback — no changes to your agent logic.

---

## How It Works

`DunetraceCallbackHandler` plugs into LangChain's callback system and translates LangChain events into Dunetrace events automatically:

| LangChain event | Dunetrace event |
|---|---|
| `on_chain_start` (outermost) | `RUN_STARTED` |
| `on_chat_model_start` / `on_llm_start` | `LLM_CALLED` |
| `on_llm_end` | `LLM_RESPONDED` (with token counts + latency) |
| `on_tool_start` / `on_agent_action` | `TOOL_CALLED` |
| `on_tool_end` | `TOOL_RESPONDED` |
| `on_retriever_start` | `RETRIEVAL_CALLED` |
| `on_retriever_end` | `RETRIEVAL_RESPONDED` (with result count + top score) |
| `on_chain_end` (outermost) | `RUN_COMPLETED` |
| `on_chain_error` / `on_llm_error` / `on_tool_error` | `RUN_ERRORED` |

Sub-chains (tool nodes, LLM nodes, retriever nodes) are automatically attributed to the correct root invocation. Running multiple concurrent `invoke()` calls with the same handler is safe — each invocation is tracked independently by LangChain's root `run_id`.

---

## Prerequisites

- Dunetrace backend running (`docker compose up -d`)
- Python 3.11+

> **Local dev — no API key needed.** The backend accepts requests without any API key when running locally. API keys are only required for production deployments — see [the main integration guide](./integrate-custom-python-agent.md#step-1-generate-an-api-key-production-only).

---

## Step 1: Install Dependencies

```bash
pip install 'dunetrace[langchain]'
```

This installs the SDK with the `langchain-core` dependency. Also install your LangChain stack:

```bash
# LangGraph (recommended)
pip install langchain-openai langgraph

# or LangChain with AgentExecutor (older pattern)
pip install langchain langchain-openai
```

---

## Step 2: Create the Callback Handler

Instantiate `DunetraceCallbackHandler` once at startup and reuse it across all invocations.

```python
from dunetrace import Dunetrace
from dunetrace.integrations.langchain import DunetraceCallbackHandler

# Local dev — no api_key needed
dt = Dunetrace(endpoint="http://localhost:8001")

# Production
# dt = Dunetrace(endpoint="https://your-dunetrace-ingest", api_key="dt_live_...")

callback = DunetraceCallbackHandler(
    dt,
    agent_id="my-langchain-agent",
    system_prompt=SYSTEM_PROMPT,     # used to compute agent version hash
    model="gpt-4o",                  # declared model for this agent
    tools=["web_search", "calculator"],  # tool names for detector context
)
```

**Constructor parameters:**

| Parameter | Required | Description |
|---|---|---|
| `client` | Yes | Your `Dunetrace` instance |
| `agent_id` | Yes | Identifier for this agent — must match your `api_keys` row |
| `system_prompt` | No | System prompt string — used to compute a version fingerprint when your prompt changes |
| `model` | No | Model name for display and detector context. Defaults to `"unknown"` |
| `tools` | No | List of tool name strings — used by detectors like `TOOL_AVOIDANCE` |

---

## Step 3: Pass the Callback on Each Invocation

### LangGraph (create_react_agent)

> **Note:** In LangGraph V1.0+, `create_react_agent` emits a deprecation warning when imported from `langgraph.prebuilt`. This does not affect functionality — it will be removed in V2.0.

```python
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="gpt-4o", temperature=0)

@tool
def web_search(query: str) -> str:
    """Search the web for information on a topic."""
    return f"Results for: {query}"

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))  # noqa: S307

agent = create_react_agent(llm, [web_search, calculator], prompt=SYSTEM_PROMPT)

result = agent.invoke(
    {"messages": [("human", "What is 42 * 17?")]},
    config={"callbacks": [callback]},   # <-- add this
)
```

### LangGraph async

```python
result = await agent.ainvoke(
    {"messages": [("human", "What is 42 * 17?")]},
    config={"callbacks": [callback]},
)
```

The same handler works for both `invoke()` and `ainvoke()`. LangChain runs sync callbacks in a thread-pool executor for async invocations, so no async variant is needed.

### LangChain AgentExecutor (older pattern)

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

result = executor.invoke(
    {"input": "What is the capital of France?"},
    config={"callbacks": [callback]},   # <-- add this
)
```

---

## Step 4: Shutdown on Process Exit

```python
import atexit
atexit.register(dt.shutdown)
```

Or explicitly with a timeout:

```python
dt.shutdown(timeout=5)  # waits up to 5 seconds to flush pending events
```

---

## Complete Example

```python
import os
import atexit
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from dunetrace import Dunetrace
from dunetrace.integrations.langchain import DunetraceCallbackHandler

SYSTEM_PROMPT = (
    "You are a research assistant. "
    "Use the search tool to find information before answering."
)

dt = Dunetrace(
    endpoint=os.environ.get("DUNETRACE_ENDPOINT", "http://localhost:8001"),
    api_key=os.environ.get("DUNETRACE_API_KEY", ""),  # omit for local dev
)
atexit.register(dt.shutdown)

@tool
def web_search(query: str) -> str:
    """Search the web for information on a topic."""
    # your actual search logic here
    return f"Results for: {query}"

tools = [web_search]

callback = DunetraceCallbackHandler(
    dt,
    agent_id="my-langchain-agent",
    system_prompt=SYSTEM_PROMPT,
    model="gpt-4o",
    tools=[t.name for t in tools],
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

result = agent.invoke(
    {"messages": [("human", "What is the capital of France?")]},
    config={"callbacks": [callback]},
)
print(result["messages"][-1].content)
```

---

## Accessing the Run Inside Tool Functions

`get_current_run()` returns the active `RunContext` inside any LangChain tool callback — useful for emitting infrastructure signals (rate limits, cache misses) without plumbing the run object through your call stack.

```python
from dunetrace import get_current_run

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    run = get_current_run()

    # Emit an infrastructure signal without advancing the step counter
    if rate_limited():
        run.external_signal("rate_limit", source="serpapi")

    result = do_search(query)
    return result
```

`get_current_run()` returns `None` when called outside an active run — always guard with `if run:`.

---

## RAG Agents (Retriever Auto-Instrumentation)

If your agent uses a LangChain retriever, retrieval events are captured automatically — no extra code needed.

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool

vectorstore = FAISS.from_texts(["..."], OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(retriever, "search_docs", "Search product documentation.")
tools = [retriever_tool]

# callback captures on_retriever_start / on_retriever_end automatically
agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
result = agent.invoke(
    {"messages": [("human", "How do I configure feature X?")]},
    config={"callbacks": [callback]},
)
```

The handler extracts `result_count` and `top_score` from document metadata fields (`score`, `relevance_score`, or `similarity`) if present. These feed the `RAG_EMPTY_RETRIEVAL` detector.

---

## Sharing One Handler Across Concurrent Calls

The handler is thread-safe. A single instance can be shared across concurrent `invoke()` calls — each invocation is tracked by LangChain's root `run_id` and does not collide with others.

```python
# This is safe — one handler, many concurrent calls
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(agent.invoke, {"messages": [("human", q)]},
                           {"callbacks": [callback]}) for q in queries]
    results = [f.result() for f in futures]
```

Stale runs (invocations that never completed after 30 minutes) are pruned automatically on each new `on_chain_start`.

---

## What Is and Isn't Captured

**Captured automatically:**
- Every LLM call: model name, token counts (prompt + completion), latency, finish reason
- Every tool call: tool name, success/failure, output length
- Every retriever call: index name, result count, top similarity score
- Run-level: total steps, tool call count, exit reason

**Not captured (privacy — hashed in-process):**
- User input text
- LLM prompts and completions
- Tool arguments and outputs
- Error messages
- Retrieval queries

**Not captured (LangChain limitation):**
- Intermediate chain inputs/outputs for sub-chains (only root chain boundary is tracked as run start/end)
- Streaming token counts (may be absent depending on provider and LangChain version)

---

## Verify the Integration

Run your agent once, then check:

1. **Dashboard** (`http://your-dashboard:3000`) — the run should appear within 15 seconds
2. **Runs API** — `GET http://your-ingest:8002/v1/runs?agent_id=my-langchain-agent`
3. **Simulate a tool loop** — use the built-in scenario to confirm signals fire end-to-end:

```bash
SCENARIO=tool_loop python examples/langchain_agent.py
```

This calls `web_search` six times in one run, which triggers `TOOL_LOOP` (threshold: 3 calls in a 5-call window). The signal should appear in the dashboard within ~15 seconds.

For local testing before pointing at production, omit the `api_key` parameter — the backend accepts unauthenticated requests in dev mode:

```python
dt = Dunetrace(endpoint="http://localhost:8001")  # dev mode, no key required
```

---

## Connect Langfuse for Deep Analysis

If you run Langfuse alongside Dunetrace, you can wire them together so that when a signal fires the dashboard offers an **"Explain with Langfuse ↗"** button that pulls the actual trace and produces a specific root-cause explanation.

### How the IDs align

`DunetraceCallbackHandler` sets its `run_id` from the LangChain root `run_id` (the UUID that LangGraph assigns to the top-level chain). Langfuse v4 independently uses the same LangChain root `run_id` as its `trace_id`, but in 32-character hex format (no dashes). They represent the same run:

| System | ID format | Example |
|--------|-----------|---------|
| Dunetrace `run_id` | UUID with dashes | `b5ed23be-e4f0-43bc-8625-...` |
| Langfuse `trace_id` | 32-char hex, no dashes | `b5ed23bee4f043bc8625...` |

The Dunetrace API normalises the format automatically when querying Langfuse.

### Access both IDs after a run

```python
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler  # v4+

lf_cb = LangfuseCallbackHandler()   # reads LANGFUSE_* from env
result = agent.invoke(
    {"messages": [("human", query)]},
    config={"callbacks": [dt_callback, lf_cb]},
)

dt.shutdown(timeout=5)
import langfuse as lf_module
lf_module.get_client().flush()      # ensure trace is uploaded

dt_run_id   = dt_callback.last_run_id   # Dunetrace run ID
lf_trace_id = lf_cb.last_trace_id       # Langfuse trace ID
```

### Call the explain endpoint

```bash
POST /v1/signals/{signal_id}/explain
Authorization: Bearer <key>
Content-Type: application/json

{"langfuse_trace_id": "<lf_trace_id>"}
```

See [docs/integrations.md#langfuse](integrations.md#langfuse) for full setup — credentials, `.env` vars, and the complete runnable example.

---

## Troubleshooting

**No runs appear in the dashboard**
- Check that `dt.shutdown()` was called — events are flushed by the background drain thread, and without shutdown some may be lost on process exit
- Confirm the `api_key` in the handler matches an `active = TRUE` row in `api_keys`
- Try `debug=True` in the `Dunetrace()` constructor to enable verbose logging

**Token counts are missing**
- Some providers and LangChain versions do not populate `token_usage` in `llm_output`. The handler falls back to `usage_metadata` on the message object. If both are absent, token fields are omitted from the event — this is expected behavior and does not break detectors

**Detectors fire too aggressively for my search agent**
- Tune thresholds in `detectors.yml` on the server (see the main integration guide) and restart the detector service
