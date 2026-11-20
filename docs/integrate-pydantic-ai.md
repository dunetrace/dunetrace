# Integrating a Pydantic AI Agent with Dunetrace

<!--dunetrace:instrument
framework: pydantic-ai
language: python
install: pip install dunetrace pydantic-ai
primary_symbol: dunetrace.Dunetrace.run
mechanism: manual
opens_own_run: false
requires_run_context: true
target_file: the function that calls Agent.run() or Agent.iter()
target_example: `main.py`, `src/agent.py`
target_hints: ["Agent(", "agent.run(", "agent.iter("]
emits: [run.started, run.completed, run.errored, llm.called, llm.responded]
verify_cmd: python main.py  # then check the dashboard
-->

## Quick Start

```bash
pip install dunetrace pydantic-ai
```

`Agent.iter()` is async, so the whole thing lives in an `async def`. Save as
`agent.py`:

```python
# agent.py
import asyncio

from pydantic_ai import Agent

from dunetrace import Dunetrace

dt = Dunetrace()   # local dev, no API key needed
agent = Agent("openai:gpt-4o-mini", instructions="You are a helpful AI assistant.")


async def main() -> None:
    with dt.run("my-agent", user_input="Explain RAG.", model="gpt-4o-mini") as run:
        run.llm_called("gpt-4o-mini")

        async with agent.iter("Explain RAG.") as agent_run:
            async for _ in agent_run:
                pass
            usage = agent_run.result.usage()

        run.llm_responded(
            prompt_tokens=usage.request_tokens,
            completion_tokens=usage.response_tokens,
            finish_reason="stop",
        )
        run.final_answer()


if __name__ == "__main__":
    asyncio.run(main())
    dt.shutdown()
```

```bash
OPENAI_API_KEY=sk-... python agent.py
```

Start the backend once, locally, before running this: `docker compose up -d`.

## Where this goes

`dt.run()` goes around the `Agent.iter()` block, in the function that
owns one agent turn.

```bash
grep -rn "agent.run(\|agent.iter(" --include=*.py .
```

Pydantic AI has no callback integration in this SDK, so the LLM events are
emitted by hand around the iteration. Usage is only available after the agent
finishes, which is why `llm_responded()` comes after the loop.

## What this does

Pydantic AI exposes agent execution through `Agent.iter()`, which lets you observe the run as it happens rather than just getting a final result. Wrap the whole thing in `dt.run()` so `llm_called()`/`llm_responded()` around the iteration groups it as one Dunetrace run, with token usage read from `agent_run.result.usage()` once the agent finishes.

## Verification

```bash
docker compose up -d
```

Run your instrumented Pydantic AI application, then open the dashboard at `http://localhost:3000` — the run should appear with its LLM events and usage information.


### If nothing arrives

Work down this list. The first two cover almost every case.

1. **Is a run open?** This integration attaches to a run that is already open. Outside one it emits nothing, silently. Confirm your top-level call is inside `dt.run(...)` (or a decorator that opens one).

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
