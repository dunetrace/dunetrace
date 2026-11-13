# Hermes Agent Integration

<!--dunetrace:instrument
framework: hermes
language: python
install: pip install dunetrace hermes-agent
primary_symbol: dunetrace.integrations.hermes.DunetraceHermesPlugin
mechanism: plugin
opens_own_run: true
requires_run_context: false
target_file: startup, before the first run_conversation()
target_example: `main.py`
target_hints: ["AIAgent(", "run_conversation("]
emits: [run.started, run.completed, run.errored, llm.called, llm.responded, tool.called, tool.responded]
verify_cmd: SCENARIO=tool_loop PYTHONPATH=packages/sdk-py python packages/sdk-py/examples/hermes_agent.py
-->

## Quick Start

```bash
pip install dunetrace hermes-agent
```

```python
# agent.py
import os

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

## Where this goes

`plugin.attach()` registers with the Hermes global `PluginManager` and
goes **once per process**, before the first `run_conversation()`.

```bash
grep -rn "AIAgent(\|run_conversation(" --include=*.py .
```

Each `run_conversation()` call becomes one Dunetrace run, keyed on Hermes's own
`turn_id`.

## What this does

Dunetrace hooks into [Hermes Agent](https://github.com/nousresearch/hermes-agent)'s plugin system — every LLM call, tool call, and session boundary is captured with no changes to your agent code. Each `run_conversation()` call becomes one Dunetrace run (Hermes's `turn_id` is used as the `run_id`).

## Verification

```bash
SCENARIO=tool_loop PYTHONPATH=packages/sdk-py python packages/sdk-py/examples/hermes_agent.py
```

Check the dashboard at `http://localhost:3000` — the run and the `TOOL_LOOP` signal should appear within ~15 seconds. Other scenarios: `happy`, `retry`, `abandon`, `edge`, `real`, `real_loop`, `all` (default).


### If nothing arrives

Work down this list. The first two cover almost every case.

1. **Is a run open?** This integration opens its own run, so there is nothing to wrap. Confirm the registration call (`DunetraceHermesPlugin`) actually ran, and ran **before** the first agent invocation.

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

### Persistent plugin (CLI users)

Install the plugin once so it activates for every `hermes` CLI session — see `~/.hermes/plugins/` in the Hermes docs for the plugin registration format. Configure via env vars: `DUNETRACE_API_URL`, `DUNETRACE_API_KEY`, `DUNETRACE_AGENT_ID`.

### Data handling

User message, tool arguments, and error messages are sent to the backend as-is (`input_text`, `args`, `error`); token counts and latency as plain numbers.
