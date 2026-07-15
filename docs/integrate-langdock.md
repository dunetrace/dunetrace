# Integrating Langdock with Dunetrace

## Quick Start

Langdock emits OpenTelemetry traces natively — no code changes, just a URL:

```
Langdock → Workspace Settings → Assistants settings
→ Enable "Allow assistant logs"
→ Tracing cloud URL: https://<your-public-dunetrace-url>/v1/otlp/traces
```

Locally, expose the ingest service first with `ngrok http 8001` and use the printed `https://...ngrok-free.app/v1/otlp/traces` URL. In production, point it at a real public hostname instead.

## What this does

Langdock sends an OTLP/HTTP span for every assistant execution. Dunetrace's ingest service accepts those directly at `POST /v1/otlp/traces` and maps them onto its own event model — LLM calls, tool calls, and retrievals all become the same events a code-instrumented agent would produce. All 27 detectors run on every completed execution automatically.

## Verification

Trigger any assistant execution, then:

```bash
docker compose logs ingest --tail=20   # look for "OTLP traces received"
```

Open the dashboard at `http://localhost:3000` — the assistant appears as an agent under its `service.name`. Detectors run within ~5-10 seconds of the run completing.

---

## Advanced (optional)

### Self-monitoring via MCP

The Dunetrace MCP server exposes agent signals as tools an MCP-capable client can call. Once connected, a Langdock assistant can query its own failure history mid-conversation:

```bash
cd packages/mcp-server && pip install -e .
dunetrace-mcp --sse --port 8000
```

Add the server URL (`http://your-dunetrace-host:8000/sse`) under Langdock's "External tools"/"MCP servers" workspace setting. Available tools: `list_agents`, `get_agent_signals`, `get_agent_health`, `get_agent_patterns`, `get_run_detail`, `search_signals`, `summarize_agent`, `get_instrumentation_guide`.

### Troubleshooting

- **No runs appear** — check for `OTLP traces received` in ingest logs; confirm "Allow assistant logs" is enabled; test the endpoint with `curl -X POST .../v1/otlp/traces -d '{"resourceSpans":[]}'` (should return `{}`)
- **Runs appear but no signals** — the detector worker polls every 5s; wait a few seconds and check `docker compose logs detector`
- **Agent shows as `unknown-agent`** — Langdock isn't setting `service.name`; use the `X-Dunetrace-Agent-Id` header override if Langdock supports custom trace headers
