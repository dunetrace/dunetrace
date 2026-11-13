# Integrating a smolagents Agent with Dunetrace

<!--dunetrace:instrument
framework: smolagents
language: python
install: pip install dunetrace smolagents
primary_symbol: dunetrace.Dunetrace.run
mechanism: step-callback
opens_own_run: false
requires_run_context: true
target_file: the module that constructs the CodeAgent and calls agent.run()
target_example: `main.py`, `src/agent.py`
target_hints: ["CodeAgent(", "ToolCallingAgent(", "agent.run("]
emits: [run.started, run.completed, run.errored, tool.called, tool.responded]
verify_cmd: python main.py  # then check the dashboard
-->

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

## Where this goes

Two edits in the same module: `step_callbacks=[...]` on the agent
constructor, and `dt.run()` around `agent.run()`.

```bash
grep -rn "CodeAgent(\|ToolCallingAgent(\|agent.run(" --include=*.py .
```

smolagents exposes no tracing interface, so the callback is the only hook. It
fires per step and needs the run already open, which is what the `dt.run()`
wrapper provides.

## What this does

`smolagents` has no built-in tracing interface, so this uses its `step_callbacks` hook instead: a lightweight function runs at the end of every agent step, inspects it for tool calls, and emits `tool_called`/`tool_responded` to Dunetrace. Wrapping `agent.run()` in `dt.run()` captures the run boundary.

## Verification

Run your script once, then check the dashboard at `http://localhost:3000` — the run should appear within ~15 seconds. Give the agent a task that fails the same tool call repeatedly to confirm `TOOL_LOOP` fires.


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

### What isn't captured by this callback

- Raw tool output text — only `output_length` is passed; add your own event if you need the text itself
- The agent's reasoning/code (`step_log.model_output` / `step_log.code_action`) — read these yourself if you want them captured
- LLM token usage (`llm_called`/`llm_responded`) — not emitted by default; call them yourself inside the callback if `step_log.llm_calls` is populated by your model engine
