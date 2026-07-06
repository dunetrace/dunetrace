# Integrating an OpenAI Agents SDK Agent with Dunetrace

The [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) ships a built-in tracing interface. Dunetrace plugs into it as a **trace processor** — one `add_dunetrace_processor(...)` call at startup instruments every run, LLM generation, and function-tool call. No monkey-patching, no changes to your agent definition.

---

## Prerequisites

- Dunetrace backend running (`docker compose up -d`)
- Python 3.11+
- openai-agents (the Agents SDK)

> **Local dev — no API key needed.** The backend accepts requests without any API key when running locally.

---

## Install

```bash
pip install 'dunetrace[openai-agents]' python-dotenv
```

---

## How it works

The Agents SDK emits a *trace* per run and *spans* for each step inside it. Dunetrace maps them to its canonical events:

| SDK trace / span | Dunetrace event |
| --- | --- |
| Trace start / end | `run.started` / `run.completed` |
| `generation` / `response` span | `llm.called` / `llm.responded` (model, token counts, latency) |
| `function` span | `tool.called` / `tool.responded` (tool name, latency, success) |
| `handoff` span | `tool.called` / `tool.responded` with `tool_name` `handoff:<to_agent>` |

Each SDK `trace_id` becomes the Dunetrace `run_id`, so a run lines up across tools that share the same trace. Content fields (prompts, tool args, outputs) are sent to the backend as-is.

The processor is registered *alongside* the SDK's default exporter, so existing OpenAI tracing (and any other processors like Langfuse) keep working.

> **One processor per process.** The Agents SDK trace provider is process-global — every registered processor sees every trace. Registering a second Dunetrace processor (e.g. for a different `agent_id`) would re-emit each run under both ids, so `add_dunetrace_processor` refuses the second registration and reuses the first (logging a warning). Use a single processor per process.

---

## Integration

```python
from agents import Agent, Runner, function_tool
from dunetrace import Dunetrace
from dunetrace.integrations.openai_agents import add_dunetrace_processor

dt = Dunetrace(endpoint="http://localhost:8001")

@function_tool
def web_search(query: str) -> str:
    """Search the web."""
    return f"Results for {query}"

agent = Agent(
    name="my-agent",
    instructions="You are helpful. Search before answering.",
    model="gpt-4o-mini",
    tools=[web_search],
)

# Register once at startup.
add_dunetrace_processor(
    dt,
    agent_id="my-agent",
    system_prompt="You are helpful. Search before answering.",
    model="gpt-4o-mini",
    tools=["web_search"],
)

result = Runner.run_sync(agent, "What is the capital of France?")
print(result.final_output)

dt.shutdown()
```

The run appears in the dashboard under `my-agent`. Every LLM generation is tracked — `llm.called` / `llm.responded` capture model name, prompt + completion token counts, and latency.

> `agent_id`, `system_prompt`, `model`, and `tools` only set the run's metadata and version fingerprint — they do not have to match the live `Agent` exactly, but keeping them in sync gives the cleanest dashboard grouping.

---

## Replacing the default processor

`add_dunetrace_processor` registers *in addition to* existing processors. To run Dunetrace as the **only** processor (e.g. to suppress the default OpenAI exporter), build the processor yourself and call `set_trace_processors`:

```python
from agents import set_trace_processors
from dunetrace.integrations.openai_agents import DunetraceTracingProcessor

processor = DunetraceTracingProcessor(dt, agent_id="my-agent", model="gpt-4o-mini")
set_trace_processors([processor])
```

---

## Concurrency

A single processor instance is safe to share across concurrent runs. Each SDK `trace_id` maps to its own run context, so parallel `Runner.run(...)` calls and multi-agent handoffs (which share one trace) don't collide.

---

## Verify

```bash
docker compose up -d
OPENAI_API_KEY=sk-… python packages/sdk-py/examples/openai_agents_agent.py
```

Open the dashboard at `http://localhost:3000`. The run should appear within 15 seconds.

**Trigger a tool-loop scenario:**

```bash
SCENARIO=tool_loop OPENAI_API_KEY=sk-… python packages/sdk-py/examples/openai_agents_agent.py
```

---

## Troubleshooting

**No runs appear in the dashboard**

- Confirm `add_dunetrace_processor(...)` runs *before* `Runner.run(...)`.
- Confirm `dt.shutdown()` (or `dt.flush()`) is called so buffered events are sent.
- Try `Dunetrace(debug=True)` for verbose logging.

**Token counts missing**

- Token counts come from the span's `usage` (chat completions) or the nested `response.usage` (Responses API). If the provider omits usage — common for streaming runs — the token fields are absent. Detectors still run on step counts and tool patterns.
