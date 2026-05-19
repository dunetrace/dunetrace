# Integrating a CrewAI Agent with Dunetrace

`DunetraceCrewCallback` hooks into CrewAI 1.x's global hook system to track every LLM and tool call. Wrap the crew kickoff with `dt.run()` to group all events under one run.

---

## Prerequisites

- Dunetrace backend running (`docker compose up -d`)
- Python 3.11+
- CrewAI 1.x

> **Local dev — no API key needed.** The backend accepts requests without any API key when running locally.

---

## Install

```bash
pip install dunetrace crewai python-dotenv
```

---

## How it works

`DunetraceCrewCallback` registers four global hooks on `install()`:

| Hook | Dunetrace event emitted |
|---|---|
| `register_before_llm_call_hook` | `llm.called` |
| `register_after_llm_call_hook` | `llm.responded` (token counts + latency) |
| `register_before_tool_call_hook` | `tool.called` |
| `register_after_tool_call_hook` | `tool.responded` (success/failure) |

Wrap the crew `kickoff()` with `dt.run()` to group all agent events under one Dunetrace run.

---

## Integration

```python
from crewai import Agent, Crew, Task, Process
from crewai.tools import tool
from dunetrace import Dunetrace
from dunetrace.integrations.crewai import DunetraceCrewCallback

dt = Dunetrace(endpoint="http://localhost:8001")
cb = DunetraceCrewCallback(
    dt,
    agent_id="research-crew",
    model="gpt-4o-mini",
    tools=["web_search"],
)
cb.install()   # register global LLM + tool hooks

@tool("web_search")
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Results for {query}"

# CrewAI 1.x: pass model as a plain string — routed via LiteLLM
researcher = Agent(
    role="Senior Researcher",
    goal="Find accurate information on any topic",
    backstory="Expert researcher who always uses web_search",
    llm="gpt-4o-mini",
    tools=[web_search],
)
writer = Agent(
    role="Content Writer",
    goal="Write clear summaries from research",
    backstory="Skilled at turning research into readable content",
    llm="gpt-4o-mini",
)

research_task = Task(
    description="Research AI trends for 2025",
    agent=researcher,
    expected_output="Key findings report",
)
write_task = Task(
    description="Summarize the research",
    agent=writer,
    expected_output="One-paragraph summary",
    context=[research_task],
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
)

with dt.run(
    "research-crew",
    user_input="AI trends 2025",
    model="gpt-4o-mini",
    tools=["web_search"],
) as run:
    result = crew.kickoff()
    run.final_answer()

cb.uninstall()   # remove hooks when done
dt.shutdown()
```

---

## API notes

- `install()` is idempotent — safe to call multiple times.
- `uninstall()` is a no-op if not installed.
- The callback is global — it affects all CrewAI agents in the process, not just the ones in the current crew.

---

## Verify

```bash
docker compose up -d
OPENAI_API_KEY=sk-… python examples/crewai_agent.py
```

Open the dashboard at `http://localhost:3000`. The run should appear within 15 seconds.

**Trigger a tool-loop scenario:**

```bash
SCENARIO=tool_loop OPENAI_API_KEY=sk-… python packages/sdk-py/examples/crewai_agent.py
```

This forces `web_search` to be called repeatedly with the same arguments, triggering `TOOL_LOOP`. The signal should appear in the dashboard within ~15 seconds.

---

## Troubleshooting

**No runs appear in the dashboard**
- Confirm `cb.install()` was called before `crew.kickoff()`
- Confirm `dt.shutdown()` is called after the run
- Try `debug=True` in `Dunetrace(debug=True)` for verbose logging

**Token counts missing**
- CrewAI routes LLM calls through LiteLLM. Token counts are extracted from LiteLLM's response metadata. If the provider does not return usage, token fields are omitted — this is expected and does not break detectors.
