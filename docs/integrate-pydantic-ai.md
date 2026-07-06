# Integrating a Pydantic AI Agent with Dunetrace

`Pydantic AI` provides the `Agent.iter()` API, which exposes the execution lifecycle of an agent. This allows Dunetrace to observe agent execution, capture token usage from the agent's final result, and emit LLM events for tracing.

Wrap the agent execution with `dt.run()` to group all events under a single Dunetrace run.

---

## Prerequisites

- Dunetrace backend running (`docker compose up -d`)
- Python 3.11+
- Pydantic AI

> **Local dev — no API key needed.** The backend accepts requests without any API key when running locally.

---

## Install

```bash
pip install dunetrace pydantic-ai python-dotenv
```

---

## How it works

Pydantic AI exposes agent execution through `Agent.iter()`, allowing applications to observe the lifecycle of an agent run.

Within a Dunetrace run:

- Start the agent using `Agent.iter()`.
- Observe the execution as the agent progresses through its graph.
- Read token usage from the final result using `final_result.usage()`.
- Emit `run.llm_responded()` using the collected usage information.

This groups the complete agent execution under a single Dunetrace run, making it easier to trace LLM activity and monitor usage.

## Integration

```python
from pydantic_ai import Agent
from dunetrace import Dunetrace

dt = Dunetrace(endpoint="http://localhost:8001")

agent = Agent(
    "openai:gpt-4o-mini",
    instructions="You are a helpful AI assistant.",
)

with dt.run(
    "pydantic-ai-agent",
    user_input="Explain what Retrieval-Augmented Generation is.",
    model="gpt-4o-mini",
) as run:
    run.llm_called("gpt-4o-mini")

    async with agent.iter(
        "Explain what Retrieval-Augmented Generation is."
    ) as agent_run:
        async for _ in agent_run:
            pass

        final_result = agent_run.result
        usage = final_result.usage()

    run.llm_responded(
        prompt_tokens=usage.request_tokens,
        completion_tokens=usage.response_tokens,
        finish_reason="stop",
    )

    run.final_answer()

dt.shutdown()
```

---

## API notes

- `Agent.iter()` exposes the execution lifecycle of an agent, allowing applications to observe agent execution.
- Token usage is available from the final result via `final_result.usage()`.
- Wrap agent execution with `dt.run()` to group all LLM activity under a single Dunetrace run.

---

## Verify

1. Start the Dunetrace backend:

```bash
docker compose up -d
```

2. Run your Pydantic AI application that has been instrumented with Dunetrace.

3. Open the dashboard at `http://localhost:3000`.

The agent run should appear in the dashboard with its associated LLM events and usage information.

---