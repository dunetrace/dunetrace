# Integrating a CrewAI Agent with Dunetrace

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

## What this does

`DunetraceCrewCallback` registers global hooks on CrewAI's LLM and tool call lifecycle — every LLM call and tool call across every agent in the crew is captured automatically. Wrapping `crew.kickoff()` in `dt.run()` groups them all under one Dunetrace run.

`install()` is idempotent and affects every CrewAI agent in the process, not just the current crew — call `uninstall()` when done if you need to scope it.

## Verification

```bash
OPENAI_API_KEY=sk-... SCENARIO=tool_loop python packages/sdk-py/examples/crewai_agent.py
```

Forces `web_search` to be called repeatedly with the same arguments, triggering `TOOL_LOOP`. Check the dashboard at `http://localhost:3000` — the signal should appear within ~15 seconds.

---

## Advanced (optional)

### Troubleshooting

- **No runs appear** — confirm `cb.install()` ran before `crew.kickoff()`, and `dt.shutdown()` was called after; try `Dunetrace(debug=True)` for verbose logs
- **Token counts missing** — CrewAI routes calls through LiteLLM; if the provider doesn't return usage, token fields are simply omitted (detectors still work)
