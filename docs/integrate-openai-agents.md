# Integrating an OpenAI Agents SDK Agent with Dunetrace

<!--dunetrace:instrument
framework: openai-agents
language: python
install: pip install 'dunetrace[openai-agents]'
primary_symbol: dunetrace.integrations.openai_agents.add_dunetrace_processor
mechanism: trace-processor
opens_own_run: true
requires_run_context: false
target_file: startup, before the first Runner.run()
target_example: `main.py`, `src/app.py`
target_hints: ["Runner.run", "Runner.run_sync", "from agents import"]
emits: [run.started, run.completed, run.errored, llm.called, llm.responded, tool.called, tool.responded]
verify_cmd: OPENAI_API_KEY=sk-... SCENARIO=tool_loop python packages/sdk-py/examples/openai_agents_agent.py
-->

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

## Where this goes

`add_dunetrace_processor()` is process-global and goes **once at
startup**, before the first `Runner.run()`. Agent definitions are untouched.

Find the entry point with:

```bash
grep -rn "Runner.run" --include=*.py .
```

Register the processor in `main.py` or your app startup, above that call. A
second `add_dunetrace_processor()` in the same process is refused with a logged
warning rather than double-emitting, so one per process.

## What this does

The Agents SDK has a built-in tracing interface — Dunetrace plugs into it as a trace processor, registered *alongside* any existing processors (e.g. the SDK's own default exporter). Every run, LLM generation, function-tool call, and handoff between agents is captured with no monkey-patching and no changes to your agent definition. Each SDK `trace_id` becomes the Dunetrace `run_id`.

**One processor per process** — the trace provider is process-global, so a second `add_dunetrace_processor` call (e.g. for a different `agent_id`) is refused (with a logged warning) rather than double-emitting every run. Use one processor per process.

## Verification

```bash
OPENAI_API_KEY=sk-... SCENARIO=tool_loop python packages/sdk-py/examples/openai_agents_agent.py
```

Check the dashboard at `http://localhost:3000` — the run and the `TOOL_LOOP` signal should appear within ~15 seconds.


### If nothing arrives

Work down this list. The first two cover almost every case.

1. **Is a run open?** This integration opens its own run, so there is nothing to wrap. Confirm the registration call (`add_dunetrace_processor`) actually ran, and ran **before** the first agent invocation.

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
