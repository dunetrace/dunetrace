# Integrating a smolagents Agent with Dunetrace

This guide covers adding Dunetrace monitoring to a Hugging Face `smolagents` agent. Because `smolagents` operates without a centralized tracing interface like some other frameworks, we integrate Dunetrace using the built-in **Step Callback** pattern.

---

## How It Works

`smolagents` allows you to attach callback functions to an agent via its `step_callbacks` list. These functions are executed at the end of every agent step. 

We can create a lightweight callback that inspects the step for any tool usage and emits `TOOL_CALLED` and `TOOL_RESPONDED` events to Dunetrace. By wrapping the entire `agent.run()` call within a Dunetrace run context (`dt.run()`), we capture the complete lifecycle of the agent.

| smolagents event | Dunetrace event |
|---|---|
| `with dt.run(...):` block starts | `RUN_STARTED` |
| Callback finds tool calls in `ActionStep` | `TOOL_CALLED` → `TOOL_RESPONDED` |
| `with dt.run(...):` block exits | `RUN_COMPLETED` |
| Any unhandled exception | `RUN_ERRORED` |

---

## Prerequisites

- Dunetrace backend running (`docker compose up -d`)
- Python 3.11+

> **Local dev — no API key needed.** The backend accepts requests without any API key when running locally. API keys are only required for production deployments.

---

## Step 1: Install Dependencies

```bash
pip install dunetrace smolagents
```

You may also need to install your preferred LLM provider package if you are not using the default Hugging Face models.

---

## Step 2: Register the Callback & Run

You can define the callback inside your execution scope so it has access to the active `run` object, and attach it to the agent before it starts running.

```python
import os
import atexit
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel
from dunetrace import Dunetrace

# 1. Initialize Dunetrace
dt = Dunetrace(
    endpoint=os.environ.get("DUNETRACE_ENDPOINT", "http://localhost:8001"),
    api_key=os.environ.get("DUNETRACE_API_KEY", ""),
)
atexit.register(dt.shutdown)

# 2. Define a callback that relies on an active run reference
active_run = None

def dunetrace_callback(step_log, agent=None, **kwargs):
    # If we are not inside a dt.run() context, do nothing
    if not active_run:
        return
        
    # smolagents ActionStep holds tool calls and their observations
    if hasattr(step_log, 'tool_calls') and step_log.tool_calls:
        for tool_call in step_log.tool_calls:
            # Emit tool called event
            active_run.tool_called(tool_call.name, tool_call.arguments)
            
            # Emit tool responded event if observation is available
            if hasattr(step_log, 'observations'):
                success = "Error" not in str(step_log.observations)
                active_run.tool_responded(
                    tool_call.name, 
                    success=success, 
                    output_length=len(str(step_log.observations))
                )

# 3. Create your agent normally, passing the callback
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=InferenceClientModel("Qwen/Qwen2.5-Coder-32B-Instruct"),
    step_callbacks=[dunetrace_callback]
)

# 4. Wrap your execution in the Dunetrace run context
with dt.run(
    agent_id="my-smolagent",
    system_prompt=agent.system_prompt,
    model="huggingface-model",
    tools=list(agent.tools.keys())
) as run:
    
    # Set the active run for the callback
    active_run = run
    try:
        result = agent.run("What is the capital of France and what is its population?")
        run.final_answer()
        print("Agent Result:", result)
    finally:
        # Clean up
        active_run = None

```

---

## What Is and Isn't Captured

**Captured automatically via the callback:**
- Every tool call: name, success/failure, output length
- Run-level: total steps, latency, exit reason

**Not captured (privacy — hashed in-process):**
- Exact tool arguments and textual observations (Dunetrace hashes these automatically when tracking)
- Inner reasoning text / code snippets of the agent step

**Not captured by default:**
- Individual `LLM_CALLED` / `LLM_RESPONDED` token usage metrics. If you need precise token metrics, you can manually emit those in the callback by inspecting `step_log.llm_calls` (if populated by your chosen model engine) and calling `run.llm_called(...)` and `run.llm_responded(...)`.

---

## Verify the Integration

Run your python script once, then check:

1. **Dashboard** (`http://localhost:3000`) — the run should appear within 15 seconds.
2. **Runs API** — `GET http://localhost:8002/v1/runs?agent_id=my-smolagent`

To confirm Dunetrace detectors fire correctly, try giving the agent a task that causes it to fail a tool call multiple times, which should trigger a `TOOL_LOOP` signal in the dashboard.
