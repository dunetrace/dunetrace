# Quickstart

Two ways in. The first is faster and is the one to try.

---

## Option A: let your coding agent do it

If you use Claude Code, Cursor or Codex, the `dunetrace-setup` skill instruments
your repository for you. It finds your agent's real entry point, asks before
changing anything, and tells you which detectors are live afterwards.

```bash
npx skills add dunetrace/dunetrace-skills --skill dunetrace-setup
```

Then, in your agent:

| Agent | Invoke with |
|---|---|
| Claude Code | `/dunetrace-setup` |
| Cursor | `/dunetrace-setup` |
| Codex | `$dunetrace-setup` |

Codex uses `$` rather than `/`. You can also just say "add Dunetrace to this
repo" and the skill triggers on its own.

It handles three jobs: first-time instrumentation, debugging an integration that
produces no events, and extending an existing one to a new code path. Full
detail in [dunetrace/dunetrace-skills](https://github.com/dunetrace/dunetrace-skills).

You still need a backend for the events to land in, so do step 1 below either
way. The skill checks whether one is reachable and tells you if it is not.

---

## Option B: by hand

### 1. Start the backend

```bash
git clone https://github.com/dunetrace/dunetrace
cd dunetrace && cp .env.example .env
docker compose -f docker-compose.ghcr.yml up -d
```

Everything binds to `localhost` with authentication off. Confirm it is up:

```bash
curl -s localhost:8001/ready   # ingest
curl -s localhost:8002/ready   # customer API
```

Both should answer `{"status":"ok",...}` with `schema_version` equal to
`required`. To put this on a server, see
[Deploying](operations.md#deploying).

### 2. Install the SDK

```bash
pip install dunetrace      # Python
npm install dunetrace      # TypeScript, Node 22+
```

### 3. Instrument your agent

Pick the guide for your framework. Each one names the file the code goes in,
states whether it needs a run context, and ends with a verification step.

| | |
|---|---|
| Plain OpenAI / Anthropic / Mistral / Bedrock, Python | [integrate-custom-python-agent.md](integrate-custom-python-agent.md) |
| Plain OpenAI / Anthropic / Mistral, TypeScript | [integrate-typescript-agent.md](integrate-typescript-agent.md) |
| LangChain, LangGraph | [integrate-langchain-agent.md](integrate-langchain-agent.md) |
| CrewAI | [integrate-crewai-agent.md](integrate-crewai-agent.md) |
| Vercel AI SDK | [integrate-vercel-ai.md](integrate-vercel-ai.md) |
| AutoGen | [integrate-autogen-agent.md](integrate-autogen-agent.md) |
| Haystack | [integrate-haystack-agent.md](integrate-haystack-agent.md) |
| OpenAI Agents SDK | [integrate-openai-agents.md](integrate-openai-agents.md) |
| LlamaIndex | [integrate-llamaindex.md](integrate-llamaindex.md) |
| Pydantic AI | [integrate-pydantic-ai.md](integrate-pydantic-ai.md) |
| smolagents | [integrate-smolagents.md](integrate-smolagents.md) |
| LiteLLM | [integrate-litellm.md](integrate-litellm.md) |
| Dify | [integrate-dify.md](integrate-dify.md) |
| Already emitting OpenTelemetry | [integrations/otel-ingestion.md](integrations/otel-ingestion.md) |

The smallest possible Python version, which needs no API key and no LLM
account:

```python
# agent.py
from dunetrace import Dunetrace

dt = Dunetrace()

@dt.tool
def web_search(query: str) -> list:
    return [f"result for {query}"]

@dt.trace
def my_agent(question: str) -> str:
    return web_search(question)[0]

if __name__ == "__main__":
    print(my_agent("What is the capital of France?"))
    dt.shutdown()
```

> **Something has to open a run.** `@dt.trace`, `@dt.agent`, an explicit
> `with dt.run(...)`, or the ASGI/WSGI middleware. `dt.auto_instrument()` on its
> own records nothing: outside a run its patches are a silent no-op, with no
> error and no warning. This is the single most common reason a first
> integration shows no data.

### 4. Confirm events arrived

Run your agent once, then:

```bash
curl -s localhost:8002/v1/agents
curl -s "localhost:8002/v1/agents/<your-agent-id>/runs?limit=5"
```

A working run looks like this. `step_count` above zero is the part that matters,
because it means events were correlated into a run rather than merely accepted:

```json
{"run_id":"0a0b...","agent_id":"my-agent","exit_reason":"run.completed",
 "step_count":12,"total_tokens":720,"signal_count":1}
```

An agent that has never reported still returns `200` with an empty `runs` list,
so check the list rather than the status code.

Open the dashboard at `http://localhost:3000`.

### 5. Trigger a failure

To see detection work end to end:

```bash
SCENARIO=failures python packages/sdk-py/examples/decorator_agent.py
```

Three agents deliberately trigger `TOOL_LOOP`, `RETRY_STORM` and
`RAG_EMPTY_RETRIEVAL`. Each appears in the dashboard within about 15 seconds.

---

## Nothing showed up

Work down this list. The first two cover almost every case.

1. **Was a run open?** See the callout in step 3. Check the
   `requires_run_context` line at the top of your framework's guide.
2. **Did the process flush?** Events ship from a background thread. Call
   `dt.shutdown()` (Python) or `await dt.shutdown()` (TypeScript) before exit.
3. **Is the backend reachable?** Re-run the `/ready` checks from step 1.
4. **Turn on debug logging.** `Dunetrace(debug=True)`.

Runs appearing with no signals is usually correct rather than a fault. A healthy
run produces none, and several detectors stay dormant until the agent has enough
history to build baselines.

Each integration guide has a fuller "If nothing arrives" section.

---

## Production

Local dev needs no API key. Production does, and a key is minted in two steps.

First, bootstrap an admin key from the ingest service. This is gated on
`ADMIN_API_KEY` from your `.env`, which is your deployment secret rather than a
tenant credential:

```bash
curl -X POST http://localhost:8001/v1/keys \
  -H 'Content-Type: application/json' \
  -d '{"org_id": "my-company", "org_name": "My Company", "admin_key": "'"$ADMIN_API_KEY"'"}'
```

Then mint a narrower key for the agent itself, using that admin key. An agent
needs `ingest` and nothing else:

```bash
curl -X POST http://localhost:8002/v1/keys \
  -H "Authorization: Bearer $DUNETRACE_ADMIN_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"scopes": ["ingest"]}'
```

Give the agent that second key. An `ingest` key cannot mint keys, write
policies, or decide approvals, and `POST /v1/keys` returns 403 for any scope the
calling key does not hold, so an agent key cannot escalate.

> **Do not INSERT into `api_keys` by hand.** Keys are verified against a SHA-256
> hash. A row written with a plaintext `key` and no `key_hash` never
> authenticates.

The Python SDK reads `DUNETRACE_API_KEY` and `DUNETRACE_API_URL` from the
environment. The TypeScript SDK reads neither, so pass them to the constructor.

---

## Next

- [All 34 detectors](detectors.md)
- [Runtime policies](policies.md), for stopping a failure rather than reporting it
- [Alerts](alerts.md), Slack and webhooks
- [MCP server](mcp-server.md), query your agents from your editor
- [Operations](operations.md), deploying, retention, rate limits, metrics
