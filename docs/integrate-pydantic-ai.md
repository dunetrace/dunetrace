# Integrating a Pydantic AI Agent with Dunetrace

## Quick Start

```bash
pip install dunetrace pydantic-ai
```

```python
from pydantic_ai import Agent
from dunetrace import Dunetrace

dt = Dunetrace()   # local dev, no API key needed
agent = Agent("openai:gpt-4o-mini", instructions="You are a helpful AI assistant.")

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

dt.shutdown()
```

Start the backend once, locally, before running this: `docker compose up -d`.

## What this does

Pydantic AI exposes agent execution through `Agent.iter()`, which lets you observe the run as it happens rather than just getting a final result. Wrap the whole thing in `dt.run()` so `llm_called()`/`llm_responded()` around the iteration groups it as one Dunetrace run, with token usage read from `agent_run.result.usage()` once the agent finishes.

## Verification

```bash
docker compose up -d
```

Run your instrumented Pydantic AI application, then open the dashboard at `http://localhost:3000` — the run should appear with its LLM events and usage information.
