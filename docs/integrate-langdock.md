# Integrating Langdock with Dunetrace

Langdock agents can be monitored by Dunetrace with zero code changes. Langdock natively emits OpenTelemetry traces — point its "Tracing cloud URL" at the Dunetrace ingest service and all 16 detectors activate immediately.

A second, complementary path lets Langdock assistants query their own Dunetrace signals through the MCP server — so the agent can introspect its own failure history mid-conversation.

---

## Path 1 — OTel receiver (zero-code monitoring)

### How it works

Langdock emits OTLP/HTTP spans for every assistant execution. Dunetrace's ingest service accepts those spans at `POST /v1/otlp/traces` and maps them to its event model:

| Langdock span | Dunetrace events |
|---|---|
| Root assistant execution span | `run.started` + `run.completed` / `run.errored` |
| LLM call span (`gen_ai.*` attributes) | `llm.called` + `llm.responded` (model, tokens, latency) |
| Tool call span | `tool.called` + `tool.responded` (success, latency) |
| Retrieval span | `retrieval.called` + `retrieval.responded` (result count) |

All 16 detectors then run on each completed execution: tool loops, context bloat, goal abandonment, retry storms, cost spikes, and more.

### Prerequisites

- Dunetrace running locally (`docker compose up -d`)
- A Langdock workspace with admin access
- ngrok installed ([ngrok.com](https://ngrok.com)) for local testing

---

### Step 1: Verify Dunetrace is running

```bash
curl http://localhost:8001/health
# → {"status":"ok","version":"0.1.0","db":"ok"}
```

---

### Step 2: Expose the ingest service publicly

Langdock is a cloud service — its trace exporter sends spans from Langdock's servers, not from your browser. `localhost:8001` is not reachable from Langdock. You need a publicly accessible URL.

```bash
ngrok http 8001
```

ngrok will print something like:
```
Forwarding  https://abc123.ngrok-free.app → http://localhost:8001
```

Copy that `https://abc123.ngrok-free.app` URL — you'll need it in step 4.

> **Note:** The free ngrok tier generates a new URL every time you restart it. If you stop and restart ngrok, update the Tracing cloud URL in Langdock settings. ngrok's paid tier gives a fixed subdomain.

**Production:** Deploy Dunetrace behind a TLS-terminating reverse proxy (nginx, Caddy) with a fixed public hostname and skip ngrok entirely:
```
https://dunetrace.your-company.com/v1/otlp/traces  →  http://ingest:8001
```

---

### Step 3: Verify the endpoint is reachable

Before configuring Langdock, confirm the tunnel is working:

```bash
curl -X POST https://abc123.ngrok-free.app/v1/otlp/traces \
  -H "Content-Type: application/json" \
  -d '{"resourceSpans":[]}'
# → {}
```

If you get `{}` back, Langdock can reach it.

---

### Step 4: Configure Langdock

In Langdock:
```
Workspace Settings → Assistants settings
→ Enable "Allow assistant logs"  (toggle on)
→ Tracing cloud URL: https://abc123.ngrok-free.app/v1/otlp/traces
→ Save
```

Self-hosted Dunetrace runs in `AUTH_MODE=dev` by default — no API key or Authorization header needed. Langdock doesn't send one, so this just works.

---

### Step 5: Run any Langdock assistant

Trigger any assistant execution. Then check the ingest service logs:

```bash
docker compose logs ingest --tail=20
```

You should see:
```
OTLP traces received. resources=1 spans=4 batch_id=...
OTLP persisted. batch_id=... events=6 inserted=6
```

---

### Step 6: Check the dashboard

Open `http://localhost:3000` — the Langdock assistant appears as an agent under its `service.name` (whatever Langdock sets in the OTel resource attributes). Its execution is visible in the runs feed. Detectors run within ~5 seconds of the run completing — refresh the page if signals don't appear immediately.

> **Tip:** If the agent shows up as `unknown-agent`, Langdock isn't setting `service.name` in the resource attributes. Add a custom `X-Dunetrace-Agent-Id` header in Langdock's tracing configuration if that option is available, or check Langdock's docs for how to set OTel resource attributes.

---

### What you get

- **All 16 detectors** run on every Langdock execution without any code change
- **Slack / webhook alerts** fire when failure patterns are detected
- **Dashboard** shows run timelines, token usage, wasted cost per failure
- **Weekly digest** summarizes systemic patterns across all your Langdock assistants

---

## Path 2 — MCP server (in-assistant signal querying)

The Dunetrace MCP server exposes agent monitoring data as tools that any MCP-capable client can call. Langdock supports MCP servers — once connected, a Langdock assistant can query its own Dunetrace signals mid-conversation.

### Install the MCP server

```bash
cd packages/mcp-server
pip install -e .
```

### Run with SSE transport (for remote clients)

```bash
dunetrace-mcp --sse --port 8000
```

Configure the environment:

```bash
DUNETRACE_API_URL=http://localhost:8002   # Customer API
DUNETRACE_API_KEY=dt_dev_test            # Bearer token
```

### Connect in Langdock

Add the MCP server endpoint in Langdock's MCP configuration. The exact UI path depends on your Langdock version — look for "External tools" or "MCP servers" in workspace settings. Point it at:

```
http://your-dunetrace-host:8000/sse
```

### Available tools

Once connected, a Langdock assistant can call:

| Tool | What it returns |
|---|---|
| `list_agents` | All monitored agents and their health scores |
| `get_agent_signals` | Recent failure signals for a specific agent |
| `get_agent_health` | Health score, signal counts, failure breakdown |
| `get_agent_patterns` | Dominant failure types and rates over the last 7 days |
| `get_run_detail` | Full event timeline for a specific run |
| `search_signals` | Signal search by failure type, severity, or date range |
| `summarize_agent` | Natural-language summary of an agent's recent behaviour |
| `get_instrumentation_guide` | How to instrument an agent for Dunetrace |

### Example: self-monitoring assistant

With OTel (Path 1) and MCP (Path 2) both active, a Langdock assistant can answer questions about its own health:

> **User:** What's been going wrong with the research assistant this week?
>
> **Assistant:** [calls `get_agent_patterns(agent_id="research-assistant")`]
> The research assistant has triggered TOOL_LOOP in 34% of runs and GOAL_ABANDONMENT in 12% of runs over the last 7 days. The tool loop pattern appears consistently when the web_search tool returns empty results — the agent retries the same query rather than reformulating. I can suggest a prompt addition to address this.

---

## Troubleshooting

**No runs appearing in the dashboard:**
- Check ingest service logs for `OTLP traces received` — if absent, Langdock is not reaching the endpoint
- Confirm "Allow assistant logs" is enabled in Langdock settings
- Try `curl -X POST http://localhost:8001/v1/otlp/traces -H 'Content-Type: application/json' -d '{"resourceSpans":[]}'` — should return `{}`

**Runs appear but no signals:**
- The detector worker polls every 5 seconds; wait up to 10 seconds after the run appears
- Check detector logs: `docker compose logs detector --tail=50`
- Confirm the run shows LLM events (hover over the run in the dashboard to see event types)

**Agent ID shows as `unknown-agent`:**
- Langdock may not set `service.name` in the resource attributes. Use the `X-Dunetrace-Agent-Id` header override if Langdock allows custom trace headers, or set a fixed agent_id in Langdock's OTel resource configuration.

**AUTH_MODE is not dev but no API key path:**
- Set `AUTH_MODE=dev` in the ingest service environment, or deploy a reverse proxy that injects the `Authorization: Bearer <key>` header for all requests from Langdock's IP range.
