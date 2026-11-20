# Integrating an AutoGen Agent with Dunetrace

<!--dunetrace:instrument
framework: autogen
language: python
install: pip install dunetrace autogen-agentchat autogen-ext
primary_symbol: dunetrace.integrations.autogen.DunetraceAutoGenObserver
mechanism: client-wrap
opens_own_run: true
requires_run_context: false
target_file: the module that constructs the model client and runs the team
target_example: `main.py`, `src/team.py`
target_hints: ["OpenAIChatCompletionClient", "AssistantAgent(", "RoundRobinGroupChat", ".run(task="]
emits: [run.started, run.completed, run.errored, llm.called, llm.responded, tool.called, tool.responded]
verify_cmd: OPENAI_API_KEY=sk-... SCENARIO=tool_loop python packages/sdk-py/examples/autogen_agent.py
-->

## Quick Start

```bash
pip install dunetrace autogen-agentchat autogen-ext
```

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dunetrace import Dunetrace
from dunetrace.integrations.autogen import DunetraceAutoGenObserver

dt       = Dunetrace()   # local dev, no API key needed
observer = DunetraceAutoGenObserver(dt, agent_id="my-agent", model="gpt-4o-mini")

async def main():
    base_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    dt_client   = observer.wrap_client(base_client)   # instruments every LLM call

    assistant = AssistantAgent("assistant", model_client=dt_client)

    async with observer.run(user_input="What is the capital of France?"):
        result = await assistant.run(task="What is the capital of France?")

    await base_client.close()
    dt.shutdown()

asyncio.run(main())
```

Start the backend once, locally, before running this: `docker compose up -d`.

## Where this goes

`observer.wrap_client()` goes where the model client is constructed.
`observer.run()` goes around the team or agent execution.

Find both with:

```bash
grep -rn "OpenAIChatCompletionClient\|AssistantAgent(\|\.run(task=" --include=*.py .
```

Usually the same module, typically `main.py`. Wrap **each** agent's model client
separately if agents use different clients; they all report into the same
`observer.run()`.

## What this does

`observer.wrap_client()` instruments a model client so every `create()` call — model name, token counts, latency — is captured automatically. `observer.run()` opens a Dunetrace run around the whole conversation, so a multi-agent team's calls all land under one run.

For a team with multiple agents, wrap each agent's model client separately — each still reports into the same run:

```python
agent_a = AssistantAgent("researcher", model_client=observer.wrap_client(base_a))
agent_b = AssistantAgent("writer",     model_client=observer.wrap_client(base_b))
```

## Verification

```bash
OPENAI_API_KEY=sk-... SCENARIO=tool_loop python packages/sdk-py/examples/autogen_agent.py
```

Check the dashboard at `http://localhost:3000` — the run (and, for the tool-loop scenario, a `TOOL_LOOP` signal) should appear within ~15 seconds.


### If nothing arrives

Work down this list. The first two cover almost every case.

1. **Is a run open?** This integration opens its own run, so there is nothing to wrap. Confirm the registration call (`DunetraceAutoGenObserver`) actually ran, and ran **before** the first agent invocation.

2. **Did the process flush?** A normal exit flushes what is still buffered, but
   a process that is killed, exits through `os._exit()`, or is frozen (a
   serverless function after it returns) loses it. Call `dt.shutdown()` before
   the process ends, or `dt.flush(block=True)` before a handler returns. See
   [Troubleshooting](troubleshooting.md#14-did-the-process-flush-before-exit).

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

5. **Watch the events.** `Dunetrace(emit_as_json=True)` also prints every event
   as a JSON line, and `Dunetrace(debug=True)` logs each batch as it ships, with
   its event count and HTTP status.

**Runs appear but no signals?** That is usually correct, not a fault. The
detector polls every 5 seconds, and a healthy run produces no signals. Only
`STEP_COUNT_INFLATION` waits for run history (20 clean runs); the other
detectors use static thresholds until a baseline exists. Check
`docker compose logs detector` if you expected one, and see
[Troubleshooting](troubleshooting.md#2-my-run-completed-but-i-see-no-signals).

---

## Advanced (optional)

### Troubleshooting

- **No runs appear** — confirm the team execution runs inside `observer.run()`, and `dt.shutdown()` was called; try `Dunetrace(debug=True)` for verbose logs
- **Token counts missing** — extracted from the model client's response metadata; if the provider omits usage, detectors still run on step counts and tool patterns
