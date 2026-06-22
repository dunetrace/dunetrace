# Hermes Agent Integration

Monitor [Hermes Agent](https://github.com/nousresearch/hermes-agent) runs with Dunetrace. Every `run_conversation()` call is tracked as a Dunetrace run — tool calls, LLM calls, token usage, and latency are captured automatically through Hermes's plugin hook system.

---

## Install

```bash
pip install dunetrace hermes-agent
```

---

## Usage

```python
from run_agent import AIAgent
from dunetrace import Dunetrace
from dunetrace.integrations.hermes import DunetraceHermesPlugin

dt = Dunetrace(endpoint="http://localhost:8001")

plugin = DunetraceHermesPlugin(
    dt,
    agent_id="my-hermes-agent",
    model="hermes-3-llama-3.1-70b",
    system_prompt="You are a research assistant.",
    tools=["web_search", "calculator"],
)
plugin.attach()   # registers hooks with the Hermes global PluginManager

agent = AIAgent(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    quiet_mode=True,
)

result = agent.run_conversation("What is the capital of France?")
dt.flush()
```

Call `plugin.attach()` once before any `run_conversation()` calls. Multiple calls to `attach()` append duplicate handlers — create one plugin instance per process.

---

## How it works

Dunetrace hooks into Hermes's plugin system without modifying agent code. Each `run_conversation()` call maps to one Dunetrace run:

| Hermes hook | Dunetrace event | What it captures |
|---|---|---|
| `pre_llm_call` | `run.started` | User message hash, model |
| `pre_api_request` | `llm.called` | Model name, approximate input tokens |
| `post_api_request` | `llm.responded` | Actual token counts, latency, finish reason |
| `pre_tool_call` | `tool.called` | Tool name, args hash (never raw args) |
| `post_tool_call` | `tool.responded` | Success/failure, output length, latency, error hash |
| `on_session_end` | `run.completed` / `run.errored` | Total duration, interrupted flag |

The Hermes `turn_id` is used as the Dunetrace `run_id`. One user message = one run, regardless of how many LLM calls or tool calls happen within it.

---

## Persistent plugin (CLI users)

If you use the `hermes` CLI, install the plugin once so it activates automatically:

```bash
mkdir -p ~/.hermes/plugins/dunetrace
```

Create `~/.hermes/plugins/dunetrace/plugin.yaml`:
```yaml
name: dunetrace
version: "1.0"
description: "Dunetrace observability"
```

Create `~/.hermes/plugins/dunetrace/__init__.py`:
```python
import os
from dunetrace import Dunetrace
from dunetrace.integrations.hermes import DunetraceHermesPlugin

def register(ctx):
    dt = Dunetrace(
        endpoint=os.getenv("DUNETRACE_API_URL", "http://localhost:8001"),
        api_key=os.getenv("DUNETRACE_API_KEY", ""),
    )
    plugin = DunetraceHermesPlugin(
        dt,
        agent_id=os.getenv("DUNETRACE_AGENT_ID", "hermes"),
    )
    ctx.register_hook("pre_llm_call",    plugin._pre_llm_call)
    ctx.register_hook("pre_api_request", plugin._pre_api_request)
    ctx.register_hook("post_api_request",plugin._post_api_request)
    ctx.register_hook("pre_tool_call",   plugin._pre_tool_call)
    ctx.register_hook("post_tool_call",  plugin._post_tool_call)
    ctx.register_hook("on_session_end",  plugin._on_session_end)
```

Enable it:
```bash
hermes plugins enable dunetrace
```

Configure via environment:

| Variable | Default | Description |
|---|---|---|
| `DUNETRACE_API_URL` | `http://localhost:8001` | Ingest endpoint |
| `DUNETRACE_API_KEY` | (empty) | API key for production |
| `DUNETRACE_AGENT_ID` | `hermes` | Agent identifier in the dashboard |

---

## What gets detected

All 17 built-in detectors run automatically. Scenarios confirmed end-to-end:

| Scenario | Signal |
|---|---|
| Same tool called 5× with identical args | `TOOL_LOOP` (HIGH) |
| Tool fails 3× in a row | `RETRY_STORM` (HIGH) |
| 5 consecutive LLM calls after last tool use | `REASONING_STALL` (HIGH) |
| Tool fails on first step | `FIRST_STEP_FAILURE` (MEDIUM) |
| Normal run: tool → answer | (no signal) ✓ |
| Interrupted run | `run.errored` recorded, no false signal |

---

## Running the example

```bash
# All synthetic scenarios (no API key needed):
PYTHONPATH=packages/sdk-py python packages/sdk-py/examples/hermes_agent.py

# Specific scenario:
SCENARIO=tool_loop PYTHONPATH=packages/sdk-py python packages/sdk-py/examples/hermes_agent.py

# Real Hermes agent:
SCENARIO=real OPENAI_API_KEY=sk-... python packages/sdk-py/examples/hermes_agent.py

# Real Hermes agent provoked into a tool loop:
SCENARIO=real_loop OPENAI_API_KEY=sk-... python packages/sdk-py/examples/hermes_agent.py
```

Available scenarios: `happy`, `tool_loop`, `retry`, `abandon`, `edge`, `real`, `real_loop`, `all` (default).

---

## Privacy

All content fields are hashed before transmission — raw tool arguments, user messages, and LLM outputs never leave your process.

| Field | Transmitted as |
|---|---|
| User message | `input_hash` (SHA-256) |
| Tool arguments | `args_hash` (SHA-256 of `str(args)`) |
| Error messages | `error_hash` (SHA-256) |
| Token counts, latency | plain numbers |
