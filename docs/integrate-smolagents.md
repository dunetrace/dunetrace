# Integrating a smolagents Agent with Dunetrace

## Quick Start

```bash
pip install dunetrace smolagents
```

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel
from dunetrace import Dunetrace

dt = Dunetrace()   # local dev, no API key needed
active_run = None

def dunetrace_callback(step_log, agent=None, **kwargs):
    if not active_run:
        return
    for tool_call in getattr(step_log, "tool_calls", None) or []:
        active_run.tool_called(tool_call.name, tool_call.arguments)
        obs = getattr(step_log, "observations", None)
        active_run.tool_responded(tool_call.name, success=obs is not None, output_length=len(str(obs or "")))

agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=InferenceClientModel("Qwen/Qwen2.5-Coder-32B-Instruct"),
    step_callbacks=[dunetrace_callback],
)

with dt.run(agent_id="my-agent", model="huggingface-model") as run:
    active_run = run
    try:
        result = agent.run("What is the capital of France?")
    finally:
        active_run = None

dt.shutdown()
```

Start the backend once, locally, before running this: `docker compose up -d`.

## What this does

`smolagents` has no built-in tracing interface, so this uses its `step_callbacks` hook instead: a lightweight function runs at the end of every agent step, inspects it for tool calls, and emits `tool_called`/`tool_responded` to Dunetrace. Wrapping `agent.run()` in `dt.run()` captures the run boundary.

## Verification

Run your script once, then check the dashboard at `http://localhost:3000` — the run should appear within ~15 seconds. Give the agent a task that fails the same tool call repeatedly to confirm `TOOL_LOOP` fires.

---

## Advanced (optional)

### What isn't captured by this callback

- Raw tool output text — only `output_length` is passed; add your own event if you need the text itself
- The agent's reasoning/code (`step_log.model_output` / `step_log.code_action`) — read these yourself if you want them captured
- LLM token usage (`llm_called`/`llm_responded`) — not emitted by default; call them yourself inside the callback if `step_log.llm_calls` is populated by your model engine
