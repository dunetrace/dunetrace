# Hermes Agent Integration

## Quick Start

```bash
pip install dunetrace hermes-agent
```

```python
from run_agent import AIAgent
from dunetrace import Dunetrace
from dunetrace.integrations.hermes import DunetraceHermesPlugin

dt = Dunetrace()   # local dev, no API key needed
plugin = DunetraceHermesPlugin(dt, agent_id="my-agent", model="hermes-3-llama-3.1-70b")
plugin.attach()   # registers hooks with the Hermes global PluginManager, once per process

agent = AIAgent(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini", quiet_mode=True)
result = agent.run_conversation("What is the capital of France?")

dt.flush()
```

Start the backend once, locally, before running this: `docker compose up -d`.

## What this does

Dunetrace hooks into [Hermes Agent](https://github.com/nousresearch/hermes-agent)'s plugin system — every LLM call, tool call, and session boundary is captured with no changes to your agent code. Each `run_conversation()` call becomes one Dunetrace run (Hermes's `turn_id` is used as the `run_id`).

## Verification

```bash
SCENARIO=tool_loop PYTHONPATH=packages/sdk-py python packages/sdk-py/examples/hermes_agent.py
```

Check the dashboard at `http://localhost:3000` — the run and the `TOOL_LOOP` signal should appear within ~15 seconds. Other scenarios: `happy`, `retry`, `abandon`, `edge`, `real`, `real_loop`, `all` (default).

---

## Advanced (optional)

### Persistent plugin (CLI users)

Install the plugin once so it activates for every `hermes` CLI session — see `~/.hermes/plugins/` in the Hermes docs for the plugin registration format. Configure via env vars: `DUNETRACE_API_URL`, `DUNETRACE_API_KEY`, `DUNETRACE_AGENT_ID`.

### Data handling

User message, tool arguments, and error messages are sent to the backend as-is (`input_text`, `args`, `error`); token counts and latency as plain numbers.
