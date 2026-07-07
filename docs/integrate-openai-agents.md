# Integrating an OpenAI Agents SDK Agent with Dunetrace

## Quick Start

```bash
pip install 'dunetrace[openai-agents]'
```

```python
from agents import Agent, Runner
from dunetrace import Dunetrace
from dunetrace.integrations.openai_agents import add_dunetrace_processor

dt = Dunetrace()   # local dev, no API key needed

agent = Agent(name="my-agent", instructions="You are helpful.", model="gpt-4o-mini")

add_dunetrace_processor(dt, agent_id="my-agent", model="gpt-4o-mini")   # register once, before running

result = Runner.run_sync(agent, "What is the capital of France?")
print(result.final_output)

dt.shutdown()
```

Start the backend once, locally, before running this: `docker compose up -d`.

## What this does

The Agents SDK has a built-in tracing interface — Dunetrace plugs into it as a trace processor, registered *alongside* any existing processors (e.g. the SDK's own default exporter). Every run, LLM generation, function-tool call, and handoff between agents is captured with no monkey-patching and no changes to your agent definition. Each SDK `trace_id` becomes the Dunetrace `run_id`.

**One processor per process** — the trace provider is process-global, so a second `add_dunetrace_processor` call (e.g. for a different `agent_id`) is refused (with a logged warning) rather than double-emitting every run. Use one processor per process.

## Verification

```bash
OPENAI_API_KEY=sk-... SCENARIO=tool_loop python packages/sdk-py/examples/openai_agents_agent.py
```

Check the dashboard at `http://localhost:3000` — the run and the `TOOL_LOOP` signal should appear within ~15 seconds.

---

## Advanced (optional)

### Replacing the default processor

`add_dunetrace_processor` adds to existing processors. To make Dunetrace the *only* one:

```python
from agents import set_trace_processors
from dunetrace.integrations.openai_agents import DunetraceTracingProcessor

set_trace_processors([DunetraceTracingProcessor(dt, agent_id="my-agent", model="gpt-4o-mini")])
```

### Concurrency

A single processor is safe to share across concurrent runs — each SDK `trace_id` maps to its own run context, so parallel `Runner.run()` calls and multi-agent handoffs (which share one trace) don't collide.

### Troubleshooting

- **No runs appear** — confirm `add_dunetrace_processor` runs before `Runner.run(...)`, and `dt.shutdown()` (or `dt.flush()`) is called
- **Token counts missing** — come from the span's `usage`; streaming runs commonly omit this — detectors still run on step counts and tool patterns
