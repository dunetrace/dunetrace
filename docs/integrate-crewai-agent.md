# Integrating a CrewAI Agent with Dunetrace

<!--dunetrace:instrument
framework: crewai
language: python
install: pip install dunetrace crewai
primary_symbol: dunetrace.integrations.crewai.DunetraceCrewCallback
mechanism: global-hooks
opens_own_run: false
requires_run_context: true
target_file: the module that constructs the Crew and calls kickoff()
target_example: `main.py`, `src/crew.py`, or a `@CrewBase` class module
target_hints: ["Crew(", "crew.kickoff(", "@CrewBase", "Process.sequential"]
emits: [run.started, run.completed, run.errored, llm.called, llm.responded, tool.called, tool.responded, memory.written, memory.read, memory.cleared]
verify_cmd: OPENAI_API_KEY=sk-... SCENARIO=tool_loop python packages/sdk-py/examples/crewai_agent.py
-->

> **Looking for `dt.auto_instrument()` instead?** It installs these hooks for you and makes `dt.run()` optional. See [auto-instrumentation.md](./integrations/auto-instrumentation.md).

## Quick Start

```bash
pip install dunetrace crewai
```

```python
from crewai import Agent, Crew, Task, Process
from dunetrace import Dunetrace
from dunetrace.integrations.crewai import DunetraceCrewCallback

dt = Dunetrace()   # local dev, no API key needed
cb = DunetraceCrewCallback(dt, agent_id="my-crew", model="gpt-4o-mini")
cb.install()        # registers global LLM + tool hooks

researcher = Agent(role="Researcher", goal="...", backstory="...", llm="gpt-4o-mini")
task = Task(description="Research AI trends", agent=researcher, expected_output="A summary")
crew = Crew(agents=[researcher], tasks=[task], process=Process.sequential)

with dt.run("my-crew", user_input="AI trends", model="gpt-4o-mini") as run:
    result = crew.kickoff()
    run.final_answer()

cb.uninstall()
dt.shutdown()
```

Start the backend once, locally, before running this: `docker compose up -d`.

## Where this goes

`cb.install()` goes once at startup, before any crew runs. The
`dt.run()` goes around `crew.kickoff()`.

Find the kickoff with:

```bash
grep -rn "kickoff(\|Crew(" --include=*.py .
```

In a `crewai create`-scaffolded project that is `src/<project>/main.py`, with
the crew itself assembled in `src/<project>/crew.py`. Install the hooks in
`main.py` and wrap the `kickoff()` call there.

`install()` registers **global** hooks, so it affects every CrewAI agent in the
process, not just this crew. Call `uninstall()` if you need to scope it.

If you would rather not wrap `kickoff()`, use `dt.auto_instrument(["crewai"])`
instead: CrewAI is the one integration that patches the true top-level entry
point and can open its own run. See
[auto-instrumentation.md](./integrations/auto-instrumentation.md).

## What this does

`DunetraceCrewCallback` registers global hooks on CrewAI's LLM and tool call lifecycle — every LLM call and tool call across every agent in the crew is captured automatically. Wrapping `crew.kickoff()` in `dt.run()` groups them all under one Dunetrace run.

`install()` is idempotent and affects every CrewAI agent in the process, not just the current crew — call `uninstall()` when done if you need to scope it.

## Verification

```bash
OPENAI_API_KEY=sk-... SCENARIO=tool_loop python packages/sdk-py/examples/crewai_agent.py
```

Forces `web_search` to be called repeatedly with the same arguments, triggering `TOOL_LOOP`. Check the dashboard at `http://localhost:3000` — the signal should appear within ~15 seconds.


### If nothing arrives

Work down this list. The first two cover almost every case.

1. **Is a run open?** This integration attaches to a run that is already open. Outside one it emits nothing, silently. Confirm your top-level call is inside `dt.run(...)` (or a decorator that opens one).

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

### Troubleshooting

- **No runs appear** — confirm `cb.install()` ran before `crew.kickoff()`, and `dt.shutdown()` was called after; try `Dunetrace(debug=True)` for verbose logs
- **Token counts missing** — CrewAI routes calls through LiteLLM; if the provider doesn't return usage, token fields are simply omitted (detectors still work)
