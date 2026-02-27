# SDK Reference

The Dunetrace Python SDK. Zero external dependencies. Works with any Python agent framework.

---

## Installation

```bash
pip install dunetrace-sdk

# With LangChain adapter
pip install dunetrace-sdk[langchain]
```

---

## Quick Start

### Option A — Context Manager (recommended)

```python
from dunetrace import Dunetrace

dt = Dunetrace(
    api_key="dt_live_your_key_here",
    agent_id="research-agent",
)

with dt.run(user_input, system_prompt=PROMPT, model="gpt-4o", tools=TOOLS) as run:
    # LLM call
    run.llm_called("gpt-4o", prompt_tokens=512)
    response = llm.invoke(messages)
    run.llm_responded("gpt-4o", finish_reason="tool_calls",
                      output_length=len(response.content))

    # Tool call
    run.tool_called("web_search", {"query": query})
    result = search_tool.run(query)
    run.tool_responded("web_search", success=True, output_length=len(result))

    # RAG retrieval
    docs = vectorstore.similarity_search(query, k=4)
    run.retrieval_completed("knowledge-base",
                            result_count=len(docs),
                            top_score=docs[0].metadata.get("score"))

    run.set_exit_reason("final_answer")
# run.completed emitted automatically on context exit
# run.errored emitted automatically if an exception is raised
```

### Option B — LangChain Callback (one-liner)

```python
from dunetrace import Dunetrace
from dunetrace.adapters.langchain import DunetraceCallback

dt = Dunetrace(api_key="dt_live_...", agent_id="my-agent")

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=[DunetraceCallback(
        client=dt,
        system_prompt=SYSTEM_PROMPT,
        model="gpt-4o-mini",
        tools=[t.name for t in tools],
    )],
)
```

No other changes needed. All events are emitted automatically via LangChain's callback hooks.

---

## `Dunetrace` Client

### Constructor

```python
Dunetrace(
    api_key:           str,
    agent_id:          str,
    ingest_url:        str   = "https://ingest.dunetrace.io/v1/ingest",
    buffer_size:       int   = 10_000,
    flush_interval_ms: int   = 200,
    debug:             bool  = False,
)
```

| Parameter | Description |
|---|---|
| `api_key` | Your Dunetrace API key. Use `"dt_dev_test"` in local dev. |
| `agent_id` | A stable identifier for this agent. Used to group runs. |
| `ingest_url` | Override to point at a local instance during development. |
| `buffer_size` | Max events to buffer in memory. Oldest events dropped when full. |
| `flush_interval_ms` | How often the drain thread ships batches when idle. |
| `debug` | Enables `DEBUG` log level for verbose output. |

### `dt.run()`

Context manager that wraps a single agent run.

```python
with dt.run(
    user_input:    str,
    system_prompt: str  = "",
    model:         str  = "unknown",
    tools:         list = [],
    parent_run_id: str  = None,     # for sub-agent / multi-agent runs
) as run:
    ...
```

Automatically emits:
- `run.started` on enter with `input_hash`, `model`, and `tools`
- `run.completed` on clean exit with `total_steps` and `exit_reason`
- `run.errored` if an exception is raised (then re-raises it)

### `dt.shutdown(timeout=5.0)`

Flushes the buffer and stops the drain thread. Call this at the end of scripts or when terminating the process gracefully.

```python
dt.shutdown(timeout=5.0)
```

---

## `RunContext` Methods

The `run` object yielded by `dt.run()`.

### LLM calls

```python
run.llm_called(model: str, prompt_tokens: int = 0)
run.llm_responded(
    model:         str,
    finish_reason: str  = "stop",    # "stop" | "tool_calls" | "length"
    output_length: int  = 0,
    latency_ms:    int  = 0,
)
```

### Tool calls

```python
run.tool_called(tool_name: str, args: dict = {})
run.tool_responded(
    tool_name:     str,
    success:       bool = True,
    output_length: int  = 0,
    error:         str  = None,
)
```

### RAG retrieval

```python
run.retrieval_completed(
    index_name:   str,
    result_count: int,
    top_score:    float = None,   # relevance score 0.0–1.0
)
```

### Other

```python
run.set_exit_reason(reason: str)
# reason: "final_answer" | "error" | "max_iterations" | "stalled" | any string

run.run_id          # str — the UUID for this run
run.agent_version   # str — deterministic hash of config
run.step            # int — current step index
```

---

## Privacy Model

**No raw content is ever sent to Dunetrace.**

Every content field — user inputs, prompts, tool arguments, tool outputs, LLM responses — is SHA-256 hashed before being included in an event payload. The hash is truncated to 16 characters.

```python
def hash_content(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]
```

What this means in practice:

- Dunetrace can detect that the same tool was called with the same arguments (same hash) — enough for loop detection
- Dunetrace cannot reconstruct what the arguments were
- You can share Dunetrace signal data with your security team without exposing user data

The only metadata sent in plain text: event types, tool names, model names, step indices, timestamps, and token counts.

---

## Agent Versioning

The `agent_version` is computed automatically from your agent's configuration:

```python
def agent_version(system_prompt: str, model: str, tools: List[str]) -> str:
    content = system_prompt + model + "".join(sorted(tools))
    return hashlib.sha256(content.encode()).hexdigest()[:8]
```

Any change to your system prompt, model, or tool list produces a new version hash. This means:

- You can track whether a prompt change improved or worsened signal rates
- The detector never compares runs across different agent configurations
- Deploys are automatically detected without any manual versioning

---

## Multi-Agent / Sub-Agent Runs

For agents that spawn sub-agents, pass the parent's `run_id` to link them:

```python
with dt.run(user_input, ...) as parent:
    # ...
    with dt.run(sub_task, parent_run_id=parent.run_id) as child:
        # child run is linked to parent in the DB
        ...
```

Sub-runs appear as linked runs in the customer API and dashboard.

---

## LangChain Adapter Reference

The `DunetraceCallback` handles all LangChain hook methods:

| LangChain Hook | Dunetrace Event |
|---|---|
| `on_chain_start` | `run.started` |
| `on_chain_end` | `run.completed` |
| `on_llm_start` | `llm.called` |
| `on_llm_end` | `llm.responded` |
| `on_agent_action` | `tool.called` |
| `on_tool_end` | `tool.responded` |
| `on_tool_error` | `tool.responded` (success=False) |

```python
DunetraceCallback(
    client:        Dunetrace,
    system_prompt: str        = "",
    model:         str        = "unknown",
    tools:         List[str]  = [],
)
```

---

## Local Development

Point the SDK at your local ingest instance:

```python
dt = Dunetrace(
    api_key="dt_dev_test",
    agent_id="my-agent",
    ingest_url="http://localhost:8001/v1/ingest",
    debug=True,   # verbose logging
)
```

In `AUTH_MODE=dev` (the default), any non-empty API key is accepted by both the ingest and customer APIs.
