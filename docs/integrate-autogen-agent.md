# Integrating an AutoGen Agent with Dunetrace

`DunetraceAutoGenObserver` wraps a multi-agent AutoGen conversation with a single Dunetrace run. Wrap the model client with `observer.wrap_client()` to instrument every LLM call, then use `observer.run()` as a context manager around the team execution.

---

## Prerequisites

- Dunetrace backend running (`docker compose up -d`)
- Python 3.11+
- autogen-agentchat >= 0.4 (tested with 0.7.x)

> **Local dev — no API key needed.** The backend accepts requests without any API key when running locally.

---

## Install

```bash
pip install dunetrace autogen-agentchat autogen-ext python-dotenv
```

---

## How it works

| What you call | What Dunetrace captures |
|---|---|
| `observer.wrap_client(base_client)` | Wraps every `model_client.create()` call — emits `llm.called` / `llm.responded` with model, token counts, and latency |
| `async with observer.run(user_input=...)` | Emits `run.started` / `run.completed` |
| Tool functions registered in `FunctionTool` | Emits `tool.called` / `tool.responded` automatically |

---

## Integration

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dunetrace import Dunetrace
from dunetrace.integrations.autogen import DunetraceAutoGenObserver

dt       = Dunetrace(endpoint="http://localhost:8001")
observer = DunetraceAutoGenObserver(dt, agent_id="my-autogen", model="gpt-4o-mini")

def web_search(query: str) -> str:
    """Search the web."""
    return f"Results for {query}"

async def main():
    base_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    dt_client   = observer.wrap_client(base_client)   # instruments every LLM call

    assistant = AssistantAgent(
        "assistant",
        model_client=dt_client,
        tools=[FunctionTool(web_search, description="Search the web.")],
        system_message="You are helpful. Say TERMINATE when done.",
    )
    team = RoundRobinGroupChat(
        [assistant],
        termination_condition=MaxMessageTermination(5),
    )

    query = "What is the capital of France?"
    async with observer.run(user_input=query):
        result = await team.run(task=query)

    await base_client.close()
    dt.shutdown()

asyncio.run(main())
```

The run appears in the dashboard under `my-autogen`. Every `model_client.create()` call is tracked — `llm.called` / `llm.responded` events capture model name, prompt + completion token counts, and latency.

---

## Multi-agent setup

When your team has multiple agents, wrap each agent's model client separately. Each wrapped client reports into the same Dunetrace run:

```python
base_a = OpenAIChatCompletionClient(model="gpt-4o-mini")
base_b = OpenAIChatCompletionClient(model="gpt-4o")

agent_a = AssistantAgent("researcher", model_client=observer.wrap_client(base_a), ...)
agent_b = AssistantAgent("writer",     model_client=observer.wrap_client(base_b), ...)
```

---

## Verify

```bash
docker compose up -d
OPENAI_API_KEY=sk-… python packages/sdk-py/examples/autogen_agent.py
```

Open the dashboard at `http://localhost:3000`. The run should appear within 15 seconds.

**Trigger a tool-loop scenario:**

```bash
SCENARIO=tool_loop OPENAI_API_KEY=sk-… python packages/sdk-py/examples/autogen_agent.py
```

---

## Troubleshooting

**No runs appear in the dashboard**
- Confirm `observer.run()` context manager wraps the team execution
- Confirm `dt.shutdown()` is called after the run
- Try `debug=True` in `Dunetrace(debug=True)` for verbose logging

**Token counts missing**
- Token counts are extracted from the AutoGen model client response metadata. If the provider omits usage, token fields are absent — detectors still run on step counts and tool patterns.
