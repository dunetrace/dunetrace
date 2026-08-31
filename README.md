# Dunetrace

![Dunetrace](dunetrace.png)

**Runtime reliability for AI agents. Structural and semantic detection, runtime prevention, native root cause, and one-click fixes.**

[![PyPI version](https://img.shields.io/pypi/v/dunetrace.svg)](https://pypi.org/project/dunetrace/)
[![Python versions](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/dunetrace/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/dunetrace?period=total&units=INTERNATIONAL_SYSTEM&left_color=grey&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/dunetrace)
[![npm version](https://img.shields.io/npm/v/dunetrace.svg)](https://www.npmjs.com/package/dunetrace)
[![CI](https://img.shields.io/github/actions/workflow/status/dunetrace/dunetrace/ci.yml?branch=main&label=CI&logo=github)](https://github.com/dunetrace/dunetrace/actions/workflows/ci.yml)
[![CodeQL](https://img.shields.io/github/actions/workflow/status/dunetrace/dunetrace/codeql.yml?branch=main&label=CodeQL&logo=github)](https://github.com/dunetrace/dunetrace/actions/workflows/codeql.yml)
[![GitHub Stars](https://img.shields.io/github/stars/dunetrace/dunetrace?style=flat&logo=github)](https://github.com/dunetrace/dunetrace)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/badge/Discord-Join%20Server-5865F2?logo=discord&logoColor=white)](https://discord.gg/yxFjATwHW4)

![Slack alert](slack-alert.png)

---

## Star us ⭐

**If Dunetrace helps you, consider giving it a ⭐ on top right, it helps others find the project.**

---

## The problem

AI agents fail silently:

- ✓ API returns 200 &nbsp; ✓ Latency is normal &nbsp; ✓ Cost looks normal
- ✗ The upstream API returned an error body. The agent invented the numbers and reported success.
- ✗ Two agents delegated in a circle. Eight runs, all green, no progress.
- ✗ A document your agent read last week wrote an instruction into its memory. It fired today.

Tracers answer "what happened?" — after you already know it broke. Dunetrace answers
**"is something breaking right now?"** with deterministic, zero-LLM checks on every run,
and in the request path, where a policy can block the action before it executes.

---

## Five pillars, one platform

Dunetrace covers the full agent reliability lifecycle, not just one slice of it:

| | Pillar | What it does |
|---|---|---|
| 1 | **Sessions & Events** | Every run, every tool call, every LLM exchange — the raw data everything else is built on |
| 2 | **Structural Detection** | 34 zero-LLM detectors (31 of them in-path, sub-500μs per hook) — the always-on first line → [docs/detectors.md](docs/detectors.md) |
| 3 | **Semantic Evaluation** | LLM-based judgment (hallucination, task completion, cross-turn frustration) — post-hoc, sampling-based, opt-in → [docs/semantic-evaluation.md](docs/semantic-evaluation.md) |
| 4 | **Runtime Prevention** | Policies that stop, redirect, or downgrade a run *while it's happening* — the differentiator no tracer offers → [docs/policies.md](docs/policies.md) |
| 5 | **Root Cause & Fix** | Native root-cause analysis, auto-applied policy fixes, or a one-click draft PR → [Diagnose & fix](#diagnose--fix) |

---

## Quick Start

See the [examples index](examples/README.md) for ready‑to‑run examples.

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
import openai

dt = Dunetrace()
dt.init(agent_id="support-agent")   # auto-instruments installed clients (OpenAI, Anthropic, Mistral, Bedrock, LangChain, CrewAI, httpx, requests)

@dt.agent("support-agent", model="gpt-4o")
def my_agent(question: str) -> str:
    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}],
    )
    return resp.choices[0].message.content   # LLM + tool calls tracked automatically, no manual hooks
```

**TypeScript / Node.js**
```typescript
import { Dunetrace, autoInstrument } from "dunetrace";
import OpenAI from "openai";

const dt = new Dunetrace();
autoInstrument({ openai: OpenAI });   // patches OpenAI + outbound fetch; add `anthropic:` / `mistral:` too, or wrap one client with dt.wrapOpenAI()

const openai = new OpenAI();          // constructed after the patch — still tracked

await dt.run("support-agent", { model: "gpt-4o" }, async (run) => {
  await openai.chat.completions.create({ model: "gpt-4o", messages });
  run.finalAnswer();                  // LLM + tool calls tracked automatically, streaming included
});
```

→ [TypeScript auto-instrumentation](docs/integrate-typescript-agent.md#auto-instrumentation)

**Try the built-in failure scenarios**

```bash
cd packages/sdk-py                                      # Python
python examples/basic_agent.py                          # No LLM calls
SCENARIO=tool_loop python examples/langchain_agent.py   # TOOL_LOOP via LangChain
SCENARIO=failures python examples/decorator_agent.py    # TOOL_LOOP, RETRY_STORM, RAG_EMPTY_RETRIEVAL

cd ../sdk-ts && npm install && ollama pull llama3.2   # TypeScript — Vercel AI SDK on local Ollama, no API key
npm run example:vercel-ai                               # Happy path
npm run example:vercel-ai:loop                          # TOOL_LOOP → detect → explain, end to end
```

Open the dashboard: **[http://localhost:3000](http://localhost:3000)**

---

## Detectors

34 detectors run on every completed run — no configuration, no LLM. A few of the main ones:

| Signal | What it catches |
|---|---|
| `TOOL_LOOP` | Same tool called repeatedly with identical args |
| `RETRY_STORM` | Tool failing, agent retrying it repeatedly |
| `COST_SPIKE` | Total token consumption unusually high vs per-agent baseline |
| `PROMPT_INJECTION_SIGNAL` | Input matched adversarial injection patterns |
| `MEMORY_POISONING` | An injection directive was written into the agent's own memory, re-steering it when read back |
| `DELEGATION_LOOP` | Agents delegate to each other in a cycle that never converges |
| `RUNAWAY_ITERATION` | Step or cost ceiling crossed with no completion signal |
| `SILENT_TRUNCATION` | A response was truncated and the agent used it without retrying |
| `MODEL_FALLBACK_DRIFT` | The run silently switched to a weaker model (e.g. under rate limiting) |

Each alert carries what fired, why it matters, a concrete fix, and rate context (first occurrence / recurring / systemic). → [docs/detectors.md](docs/detectors.md) for all 34

- **Multi-agent** — nested `dt.run()` calls auto-link into a delegation graph that `DELEGATION_LOOP` and `HANDOFF_CONTEXT_LOSS` read → [docs/multi-agent.md](docs/multi-agent.md)
- **Agent memory** — instrument memory writes/reads and `MEMORY_POISONING` flags adversarial content persisted into them → [docs/memory.md](docs/memory.md)
- **Custom detectors** — describe one in plain English; it runs in shadow mode until you approve the fire rate → [docs/detectors.md](docs/detectors.md#custom-detectors)
- **Detector packs** — opt-in bundles per org; the voice pack adds 9 detectors → [detector packs](docs/detector-packs/index.md) · [voice pack](docs/detector-packs/voice.md)

---

## Semantic evaluation

Opt-in LLM judgment for what structural checks can't see — seven [DeepEval](https://github.com/confident-ai/deepeval)-backed evaluators (hallucination, task completion, task-understanding failure, off-topic drift, user frustration, confusion loops, sycophancy). Post-hoc and sampling-based, never in your agent's request path. Its own container, off by default:

```bash
SEMANTIC_WORKER_ENABLED=true
```

→ [docs/semantic-evaluation.md](docs/semantic-evaluation.md)

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
DIGEST_ENABLED=true      # weekly digest of top failure types, Monday 9am UTC
```

→ [docs/alerts.md](docs/alerts.md)

---

## Diagnose & fix

Click **Explain +** on any alert: Dunetrace analyzes the run's own stored events (no third-party tracer) and returns a cause plus a fix — either a runtime **policy** it applies itself (tool loops, retry storms, runaway step counts) or a **prompt/code diff** you copy in or open as a draft PR. Fix effectiveness is tracked automatically.

---

## Policies

Runtime guardrails that fire mid-run — before a failure propagates. Defined in code or in the dashboard (the SDK refetches every 60s).

```python
dt.add_policy(                                  # stop, switch_model, escalate, inject, ...
    name="cap tool calls",
    condition={"trigger": "tool_call_count", "operator": "gt", "value": 5},
    action={"type": "stop"},
)
dt.add_policy(                                  # human-in-the-loop: blocks the call until
    name="approve-wires",                       # someone approves in Slack or the dashboard,
    condition={"trigger": "before_tool_call", "operator": "eq", "value": "wire_money"},
    action={"type": "require_approval", "params": {"timeout_s": 300}},   # fail-closed on timeout
)
```

→ [docs/policies.md](docs/policies.md) · [docs/approvals.md](docs/approvals.md)

---

## MCP server

Query agent signals from Claude Code, Cursor, or Codex — "what failed in the last 24 hours?" — without leaving your editor.

```bash
pip install dunetrace-mcp
```

31 tools covering agents, runs, signals, fixes, issues, policies, custom detectors and voice calls. Claude Code registers the server automatically in `~/.claude.json` (restart to load); Cursor and Codex need one config block.

<details>
<summary>A representative 10 of the 31 tools</summary>

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

→ [docs/mcp-server.md](docs/mcp-server.md)

---

## Architecture

```
Agent Code
  └─► Dunetrace SDK        (raw content → ingest events)
        └─► Ingest API      (POST /v1/ingest → Postgres)
                ├─► Detector          (poll → 34 detectors → signals)
                ├─► Semantic Worker   (optional — poll → DeepEval → signals)
                ├─► Integrations      (optional — pull Langfuse/LangSmith/Braintrust)
                ├─► Alerts            (poll → explain → Slack / webhook)
                └─► Customer API      (runs, signals, explanations → dashboard)
```

→ [docs/architecture.md](docs/architecture.md) for the full service breakdown
· [operations guide](docs/operations.md) (retention, rate limiting, quotas)

---

## Integrations

**Model providers** — OpenAI, Anthropic, Mistral and AWS Bedrock, auto-instrumented with no call-site changes.
- [Auto-instrumentation (what's patched, streaming, agent_id resolution)](docs/integrations/auto-instrumentation.md)
- [Mistral (hyperscaler-hosted clients, EU-resident evaluation)](docs/integrations/mistral.md)

**Evaluation & tracing**
- [OpenTelemetry export (Datadog, Grafana, Honeycomb, Signoz, any OTLP backend)](docs/integrations/opentelemetry.md)
- [OpenTelemetry ingestion (send your existing OTel traces to Dunetrace: OpenLIT, Traceloop, OTel contrib)](docs/integrations/otel-ingestion.md)
- [External evaluation (Langfuse, LangSmith, Braintrust, generic push)](docs/integrations/external-evaluation.md)
- [Semantic evaluation (Dunetrace's own DeepEval-backed layer)](docs/semantic-evaluation.md)

**Fix & workflow**
- [GitHub App (automated draft PRs)](docs/integrations/github-app.md)
- [Slack & Linear alerts](docs/alerts.md)
- [Coding agents — Claude Code, Cursor, Codex (MCP)](docs/integrations/coding-agents.md)

**Voice**
- [ElevenLabs (correlate TTS cost and voice choices with agent behavior)](docs/integrations/elevenlabs.md)
- [Wiring a voice framework](docs/integrations/voice-frameworks.md)
- [Voice metrics (call-level view of a voice agent)](docs/voice-metrics.md)
- [Voice detector pack (9 detectors)](docs/detector-packs/voice.md)

<details>
<summary>Agent frameworks: LangChain, CrewAI, AutoGen, Haystack, LlamaIndex, TypeScript, and more</summary>

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
- [Dify](docs/integrate-dify.md)
- [LiteLLM](docs/integrate-litellm.md)
- [Vercel AI SDK](docs/integrate-vercel-ai.md)
- [Policies](docs/policies.md)
- [Human-in-the-loop approvals](docs/approvals.md)
- [Agent state machine](docs/state-machine.md)

</details>

---

## Contributing

Fork, branch, change, `make test`, PR — open an issue first for new integrations or architecture changes. Requires Python 3.11+, Node.js 22+, Docker + Docker Compose.

→ [CONTRIBUTING.md](CONTRIBUTING.md) (setup and workflow) · [good first issues](https://github.com/dunetrace/dunetrace/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) · [adding a detector](docs/contributing/adding-a-detector.md)

## Contact

Dunetrace UG (haftungsbeschränkt) · Kolonnenstr. 8, 10827 Berlin, Germany · [vikas@dunetrace.com](mailto:vikas@dunetrace.com)

## License

[Apache 2.0](LICENSE)
