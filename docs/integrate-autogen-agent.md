# Integrating an AutoGen Agent with Dunetrace

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

---

## Advanced (optional)

### Troubleshooting

- **No runs appear** — confirm the team execution runs inside `observer.run()`, and `dt.shutdown()` was called; try `Dunetrace(debug=True)` for verbose logs
- **Token counts missing** — extracted from the model client's response metadata; if the provider omits usage, detectors still run on step counts and tool patterns
