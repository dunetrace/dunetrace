# Dunetrace

![Dunetrace](dunetrace.png)

**Real-time failure detection for production AI agents - Slack alert within 15 seconds.**

[![PyPI version](https://img.shields.io/pypi/v/dunetrace.svg)](https://pypi.org/project/dunetrace/)
[![Python versions](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/dunetrace/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/dunetrace?period=total&units=INTERNATIONAL_SYSTEM&left_color=grey&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/dunetrace)
[![npm version](https://img.shields.io/npm/v/dunetrace.svg)](https://www.npmjs.com/package/dunetrace)
[![CI](https://img.shields.io/github/actions/workflow/status/dunetrace/dunetrace/ci.yml?branch=main&label=CI&logo=github)](https://github.com/dunetrace/dunetrace/actions/workflows/ci.yml)
[![CodeQL](https://img.shields.io/github/actions/workflow/status/dunetrace/dunetrace/codeql.yml?branch=main&label=CodeQL&logo=github)](https://github.com/dunetrace/dunetrace/actions/workflows/codeql.yml)
[![GitHub Stars](https://img.shields.io/github/stars/dunetrace/dunetrace?style=flat&logo=github)](https://github.com/dunetrace/dunetrace)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/badge/Discord-Join%20Server-5865F2?logo=discord&logoColor=white)](https://discord.gg/yxFjATwHW4)

**If Dunetrace helps you, consider giving it a ⭐ on top right, it helps others find the project.**

![Slack alert](slack-alert.png)

---

## The problem

AI agents fail silently:

- ✓ API returns 200 &nbsp; ✓ Logs are clean
- ✗ Agent called the same tool 12 times, burned $10, and gave the user a wrong answer

Langfuse and LangSmith answer "what happened?" — after you already know it broke.
Dunetrace answers **"is something breaking right now?"** and fires an alert in 15 seconds.

---

## Why it's different

| | Dunetrace | Langfuse / LangSmith |
|---|---|---|
| **When it fires** | Within 15s of run completion | You query it after you notice a problem |
| **What it watches** | Structural failure patterns | Raw trace data |
| **Alert channel** | Slack / webhook / Dashboard | Dashboard only |
| **Fix path** | One-click prompt apply or GitHub PR | Manual |

---

## Quick Start

**1. Start the backend**
```bash
git clone https://github.com/dunetrace/dunetrace
cd dunetrace && cp .env.example .env
docker compose -f docker-compose.ghcr.yml up -d
pip install -r requirements.txt
```

**2. Install the SDK**
```bash
pip install dunetrace                       # Python
npm install dunetrace                       # Node.js / TypeScript
```

**3. Instrument your agent**

**Python**
```python
from dunetrace import Dunetrace

dt = Dunetrace()

@dt.tool
def web_search(query: str) -> list: ...

@dt.trace
def my_agent(question: str) -> str:
    return web_search(question)[0]
```

**TypeScript / Node.js**
```typescript
import { Dunetrace } from "dunetrace";
import OpenAI from "openai";

const dt     = new Dunetrace();
const openai = dt.wrapOpenAI(new OpenAI());

await dt.run("my-agent", { model: "gpt-4o" }, async (run) => {
  await openai.chat.completions.create({ model: "gpt-4o", messages });
  run.finalAnswer();
});
```

**Try the built-in failure scenarios**
```bash
cd packages/sdk-py

python examples/basic_agent.py                          # No LLM calls
SCENARIO=tool_loop python examples/langchain_agent.py   # TOOL_LOOP via LangChain
SCENARIO=failures python examples/decorator_agent.py    # TOOL_LOOP, RETRY_STORM, RAG_EMPTY_RETRIEVAL
SCENARIO=tool_loop python examples/langfuse_agent.py    # TOOL_LOOP + Langfuse explain
```

Open the dashboard: **[http://localhost:3000](http://localhost:3000)**

---

## Detectors

17 detectors run on every completed run — no configuration, no LLM.

| Signal | What it catches |
|---|---|
| `TOOL_LOOP` | Same tool called repeatedly with identical args |
| `TOOL_THRASHING` | Oscillating between two tools, unable to commit |
| `RETRY_STORM` | Tool failing, agent retrying it repeatedly |
| `CASCADING_TOOL_FAILURE` | Multiple different tools failing in sequence |
| `CONTEXT_BLOAT` | Prompt tokens growing unsustainably across LLM calls |
| `LLM_TRUNCATION_LOOP` | Model output truncated repeatedly |
| `GOAL_ABANDONMENT` | Agent stopped using tools before finishing |
| `REASONING_STALL` | Too many LLM calls per tool call — agent deliberating in circles |
| `TOOL_AVOIDANCE` | Agent answered without using any tools |
| `RAG_EMPTY_RETRIEVAL` | Retrieval returned nothing, agent answered anyway |
| `EMPTY_LLM_RESPONSE` | Model returned an empty response |
| `FIRST_STEP_FAILURE` | Failed on the first step — config or setup issue |
| `SLOW_STEP` | Single step latency well above threshold |
| `STEP_COUNT_INFLATION` | Far more steps than the agent's baseline |
| `SESSION_LATENCY` | Wall-clock run time anomalously long vs per-agent baseline |
| `COST_SPIKE` | Total token consumption unusually high vs per-agent baseline |
| `PROMPT_INJECTION_SIGNAL` | Input matched adversarial injection patterns |

Each alert includes: what fired, why it matters, a concrete fix, and a rate context line (first occurrence / recurring / systemic).

→ [docs/detectors.md](docs/detectors.md)

**Custom detectors** — write a detector in plain English. Dunetrace translates it to a structured condition set, runs it in shadow mode against real traffic, and lets you review the fire rate before any alert fires. In the dashboard: **Config → Custom detectors → Add detector**.

---

## Dashboard

![Overview](dashboard.png)

Live at **[http://localhost:3000](http://localhost:3000)**. Auto-refreshes every 15s.

→ [docs/dashboard.md](docs/dashboard.md)

---

## Alerts

Slack and generic webhook (PagerDuty, Linear, custom).

```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
SLACK_MIN_SEVERITY=LOW   # LOW | MEDIUM | HIGH | CRITICAL
```

A weekly digest (Monday 9am UTC) summarises top failure types and systemic patterns. Enable with `DIGEST_ENABLED=true`.

→ [docs/alerts.md](docs/alerts.md)

---

## Diagnose with Langfuse

Connect Langfuse to get LLM-powered root-cause analysis on any signal. Click **Explain +** on any alert — Dunetrace fetches the full trace, extracts the system prompt, and returns a specific root cause and fix.

- **Prompt fixes** → **Apply via Langfuse** creates a new prompt version in one click
- **Code/infra fixes** → **Open PR on GitHub** creates a draft PR with a unified diff

Fix effectiveness is tracked automatically.

→ [docs/integrate-langfuse.md](docs/integrate-langfuse.md)

---

## Policies

Runtime guardrails that fire mid-run — before a failure propagates.

```python
dt.add_policy(
    name="cap tool calls",
    condition={"trigger": "tool_call_count", "operator": "gt", "value": 5},
    action={"type": "stop"},
)
dt.add_policy(
    name="cost cap",
    condition={"trigger": "cost_usd", "operator": "gt", "value": 0.50},
    action={"type": "switch_model", "params": {"model": "gpt-4o-mini"}},
)
```

Policies can also be created in the dashboard and fetched automatically by the SDK (60s TTL).

→ [docs/policies.md](docs/policies.md)

---

## MCP server

Query agent signals directly from Claude Code, Cursor, or Codex — without leaving your editor.

```bash
pip install dunetrace-mcp
```

<details>
<summary>10 tools — ask your editor things like "what failed in the last 24 hours?"</summary>

| Tool | What you can ask |
|---|---|
| `list_agents` | "Which agents are monitored and how healthy are they?" |
| `get_agent_signals` | "What failures did my agent have today?" |
| `get_agent_health` | "Show me the health score breakdown for my agent." |
| `get_signal_detail` | "Show me signal #42 with full evidence and fix code." |
| `get_agent_patterns` | "Is this failure systemic or a one-off?" |
| `get_run_detail` | "Walk me through run abc123 step by step." |
| `get_agent_runs` | "List recent runs for my agent with their status." |
| `search_signals` | "Show me all CRITICAL signals in the last 24 hours." |
| `summarize_agent` | "Give me a one-shot diagnosis of my agent." |
| `get_agent_token_stats` | "How much is my agent wasting on failed runs?" |

</details>

**Claude Code**: registered automatically in `~/.claude.json` after `pip install dunetrace-mcp`. Restart Claude Code to load.

**Cursor**: add `.cursor/mcp.json` to your project root:

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

→ [docs/mcp-server.md](docs/mcp-server.md)

---

## Data handling

Prompts, tool arguments, and model outputs are sent to the backend over TLS and
stored there — content-aware detectors (prompt injection in retrieved documents,
tool argument fabrication, silent failures) need to see what the agent actually
said and did. Trust is handled the way any SaaS handles it: encryption in
transit and at rest, SOC2, no training on customer data. If you need an
air-gapped deployment, self-host — same code, same detectors, your
infrastructure.

→ [docs/architecture.md](docs/architecture.md) · upgrading a self-hosted install? see [docs/migrations/multi-tenancy-v0.5.0.md](docs/migrations/multi-tenancy-v0.5.0.md)

---

## Architecture

```
Agent Code
  └─► Dunetrace SDK        (raw content → ingest events)
        └─► Ingest API      (POST /v1/ingest → Postgres)
                ├─► Detector       (poll → 17 detectors → signals)
                ├─► Alerts         (poll → explain → Slack / webhook)
                └─► Customer API   (runs, signals, explanations → dashboard)
```

---

## Integrations

<details>
<summary>LangChain, CrewAI, AutoGen, Haystack, LlamaIndex, TypeScript, and more</summary>

- [Custom Python agent](docs/integrate-custom-python-agent.md)
- [LangChain / LangGraph](docs/integrate-langchain-agent.md)
- [CrewAI](docs/integrate-crewai-agent.md)
- [Pydantic AI](docs/integrate-pydantic-ai.md)
- [AutoGen (Microsoft)](docs/integrate-autogen-agent.md)
- [OpenAI Agents SDK](docs/integrate-openai-agents.md)
- [Haystack 2.x](docs/integrate-haystack-agent.md)
- [Hermes Agent (Nous Research)](docs/integrate-hermes-agent.md)
- [LlamaIndex](docs/integrate-llamaindex.md)
- [smolagents (Hugging Face)](docs/integrate-smolagents.md)
- [TypeScript / JavaScript](docs/integrate-typescript-agent.md)
- [Langdock](docs/integrate-langdock.md)
- [Langfuse](docs/integrate-langfuse.md)
- [Policies](docs/policies.md)
- [MCP server (Claude Code, Cursor, Codex)](docs/mcp-server.md)

</details>

---

## Contributing

Fork, branch, change, `make test`, PR. For larger changes (new detectors, architecture changes), open an issue first.

Requires Python 3.11+, Node.js 18+, Docker + Docker Compose.

## ⭐ Star this if it saves you debugging time

[![Star History Chart](https://api.star-history.com/svg?repos=dunetrace/dunetrace&type=Date)](https://star-history.com/#dunetrace/dunetrace&Date)

## Contact

[dunetrace@gmail.com](mailto:dunetrace@gmail.com)

## License

[Apache 2.0](LICENSE)
