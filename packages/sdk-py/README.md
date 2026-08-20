# Dunetrace SDK

Runtime observability for AI agents. Detects tool loops, context bloat, prompt injection, and 20 other failure patterns in real-time — with a Slack alert while the run is still live.

Zero external dependencies.

## Install

```bash
pip install dunetrace                    # core SDK
pip install 'dunetrace[langchain]'       # + LangChain / LangGraph
pip install 'dunetrace[otel]'            # + OpenTelemetry exporter
```

## Quickstart

**LangChain / LangGraph**

```python
from dunetrace import Dunetrace
from dunetrace.integrations.langchain import DunetraceCallbackHandler

dt = Dunetrace()
callback = DunetraceCallbackHandler(dt, agent_id="my-agent")

result = agent.invoke(input, config={"callbacks": [callback]})
dt.shutdown()
```

**Pure Python / custom agent — decorator style**

```python
from dunetrace import Dunetrace

dt = Dunetrace()

@dt.tool                                  # auto-emits tool.called / tool.responded
def web_search(query: str) -> list: ...   # args are transmitted as-is

@dt.trace                                 # agent_id defaults to "my_agent"
def my_agent(question: str) -> str:
    return web_search(question)[0]        # zero SDK calls needed inside function bodies
```

`@dt.trace` supports bare usage (`@dt.trace` with no parens), explicit agent ID (`@dt.trace("research-agent")`), and keyword args (`@dt.trace(model="gpt-4o")`). `@dt.tool` works on both sync and async functions and is a no-op when called outside a run context.

**Or with `@dt.agent` + auto-instrumentation:**

```python
dt.init(agent_id="my-agent")   # patches openai, anthropic, mistral, httpx, requests, langchain, crewai

@dt.agent(model="gpt-4o")      # agent_id inherited from init()
def run_agent(query: str) -> str:
    return openai_client.chat.completions.create(...).choices[0].message.content
```

LangChain/LangGraph and CrewAI agents need zero manual callback wiring — see [docs/integrations/auto-instrumentation.md](../../docs/integrations/auto-instrumentation.md) for how agent attribution is resolved.

**FastAPI / Flask** — one line each, see [docs/integrate-custom-python-agent.md](../../docs/integrate-custom-python-agent.md).

## What it detects

31 detectors run on every completed run — no configuration, no LLM. A few of the main ones:

| Detector                  | What it catches                                            | Severity    |
| ------------------------- | ---------------------------------------------------------- | ----------- |
| `TOOL_LOOP`               | Same tool called 3+ times in a 5-call window               | HIGH        |
| `RETRY_STORM`             | Same tool fails 3+ times in a row                          | HIGH        |
| `PROMPT_INJECTION_SIGNAL` | Input matches known injection / jailbreak patterns         | CRITICAL    |
| `COST_SPIKE`              | Total tokens 3× above per-agent P75 baseline               | MEDIUM      |
| `PREMATURE_TERMINATION`   | Agent claims success right after a tool call it made actually failed | HIGH/CRITICAL |
| `RUNAWAY_ITERATION`       | Step or cost ceiling crossed with no completion signal      | HIGH/CRITICAL |

→ [docs/detectors.md](../../docs/detectors.md) for the full list of 31 detectors


## Output modes


| Mode                  | How to enable                                   | Destination                                      |
| --------------------- | ----------------------------------------------- | ------------------------------------------------ |
| HTTP ingest (default) | `endpoint="http://…"`                           | Dunetrace backend → detection, alerts, dashboard |
| Loki NDJSON           | `emit_as_json=True`                             | stdout → Promtail / Grafana Alloy                |
| OpenTelemetry         | `otel_exporter=DunetraceOTelExporter(provider)` | Tempo, Honeycomb, Datadog, Jaeger                |


## Backend

```bash
git clone https://github.com/dunetrace/dunetrace
cd dunetrace && cp .env.example .env && docker compose up -d
```

Dashboard → `http://localhost:3000` · Ingest → `http://localhost:8001`

## Deploy markers

Annotate the detector timeline with release boundaries so you can correlate failure spikes with deploys:

```python
# Call from your deploy script, CI/CD pipeline, or app startup
dt.mark_deploy("my-agent", version="v1.4.2", commit="abc1234", env="production")
```

The dashboard renders blue dashed vertical lines at each deploy timestamp on the 30-day detector rate chart. Fire-and-forget — runs on a background thread, never blocks the caller.

Additional keyword arguments are stored as `meta` and shown on hover.

## Policies

Runtime guardrails that fire mid-run — before a failure propagates. Define conditions with any supported trigger and attach a `stop`, `switch_model`, `inject_prompt`, or `log` action.

```python
from dunetrace import Dunetrace

dt = Dunetrace()

# Stop the run if tool call count exceeds 5
dt.add_policy(
    name="cap tool calls",
    condition={"trigger": "tool_call_count", "operator": "gt", "value": 5},
    action={"type": "stop"},
)

# Downgrade model when cost exceeds $0.50
dt.add_policy(
    name="cost cap",
    condition={"trigger": "cost_usd", "operator": "gt", "value": 0.50},
    action={"type": "switch_model", "params": {"model": "gpt-4o-mini"}},
)

# Inject a corrective prompt when a loop is detected
dt.add_policy(
    name="loop fix",
    condition={"trigger": "signal", "operator": "eq", "value": "TOOL_LOOP"},
    action={"type": "inject_prompt", "params": {"prompt": "Stop repeating tool calls. Summarise what you know and answer."}},
)

with dt.run("my-agent", user_input=query, tools=["search"]) as run:
    ...
    # After a stop policy fires, PolicyViolation is raised
    # After switch_model fires, check run.model_override
    # After inject_prompt fires, check run.pop_prompt_addition()
```

Policies can also be defined in the dashboard and fetched automatically at run start (60-second TTL cache per agent). See [docs/policies.md](../../docs/policies.md) for the full reference.

## MCP server

Query agent signals directly from Claude Code, Cursor, or any MCP-compatible editor — no context switch to the dashboard required.

```bash
pip install dunetrace-mcp
```

Ten tools: `list_agents`, `get_agent_signals`, `get_agent_health`, `get_run_detail`, `get_agent_runs`, `search_signals`, `get_signal_detail`, `get_agent_patterns`, `summarize_agent`, `get_instrumentation_guide`.

**Claude Code** — add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "dunetrace": {
      "command": "dunetrace-mcp",
      "env": {
        "DUNETRACE_API_URL": "http://localhost:8002",
        "DUNETRACE_API_KEY": "dt_dev_test"
      }
    }
  }
}
```

**Cursor** — add to `.cursor/mcp.json` in your project root (same shape as above).

Once connected, ask your editor things like:
- *"Is my agent healthy?"*
- *"What failed in the last 24 hours?"*
- *"Show me signal #42 with its fix."*
- *"Is this failure systemic or a one-off?"*

→ [docs/mcp-server.md](../../docs/mcp-server.md)

## Tests

```bash
python -m unittest discover -s tests -v          # SDK tests (no network required)
cd ../mcp-server && python -m pytest tests/ -v   # MCP server tests (no network required)
```

SDK: 620 tests · MCP server: 154 tests — both run fully offline.

## Links

- [LangChain / LangGraph](../../docs/integrate-langchain-agent.md)
- [Hermes Agent (Nous Research)](../../docs/integrate-hermes-agent.md)
- [CrewAI](../../docs/integrate-crewai-agent.md)
- [Pydantic AI](../../docs/integrate-pydantic-ai.md)
- [AutoGen](../../docs/integrate-autogen-agent.md)
- [OpenAI Agents SDK](../../docs/integrate-openai-agents.md)
- [Haystack 2.x](../../docs/integrate-haystack-agent.md)
- [LlamaIndex](../../docs/integrate-llamaindex.md)
- [LiteLLM](../../docs/integrate-litellm.md)
- [All integration guides](../../docs/)
- [GitHub](https://github.com/dunetrace/dunetrace)
- [Issues](https://github.com/dunetrace/dunetrace/issues)

