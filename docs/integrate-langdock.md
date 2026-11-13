# Integrating Langdock with Dunetrace

<!--dunetrace:instrument
framework: langdock
language: none
install: none
primary_symbol: POST /v1/otlp/traces
mechanism: otlp
opens_own_run: true
requires_run_context: false
target_file: no code change; a URL in Langdock workspace settings
target_example: Workspace Settings, Assistants settings, Tracing cloud URL
target_hints: []
emits: [run.started, run.completed, run.errored, llm.called, llm.responded, tool.called, tool.responded, retrieval.called, retrieval.responded]
verify_cmd: docker compose logs ingest --tail=20
-->

## Quick Start

Langdock emits OpenTelemetry traces natively — no code changes, just a URL:

```
Langdock → Workspace Settings → Assistants settings
→ Enable "Allow assistant logs"
→ Tracing cloud URL: https://<your-public-dunetrace-url>/v1/otlp/traces
```

Locally, expose the ingest service first with `ngrok http 8001` and use the printed `https://...ngrok-free.app/v1/otlp/traces` URL. In production, point it at a real public hostname instead.

## Where this goes

There is no file to edit. Langdock emits OpenTelemetry natively, and
the integration is a URL in its workspace settings pointing at Dunetrace's OTLP
receiver.

The only local requirement is that your ingest service is reachable from
Langdock's servers. A loopback-bound `localhost:8001` is not, which is why the
quick start uses `ngrok`.

## What this does

Langdock sends an OTLP/HTTP span for every assistant execution. Dunetrace's ingest service accepts those directly at `POST /v1/otlp/traces` and maps them onto its own event model — LLM calls, tool calls, and retrievals all become the same events a code-instrumented agent would produce. All 34 detectors run on every completed execution automatically.

## Verification

Trigger any assistant execution, then:

```bash
docker compose logs ingest --tail=20   # look for "OTLP traces received"
```

Open the dashboard at `http://localhost:3000` — the assistant appears as an agent under its `service.name`. Detectors run within ~5-10 seconds of the run completing.


### If nothing arrives

Work down this list. The first two cover almost every case.

1. **Is a run open?** Confirm "Allow assistant logs" is enabled and the tracing URL is reachable from Langdock's servers, not a loopback address.

2. **Did the process flush?** Events ship from a background thread. Call
   `await dt.shutdown()` before the process exits, or the buffer dies with it.

3. **Did the traces reach ingest?**

   ```bash
   docker compose logs ingest --tail=20   # look for "OTLP traces received"
   ```

**Runs appear but no signals?** That is usually correct, not a fault. The
detector polls every 5 seconds, and a healthy run produces no signals. Several
detectors also need cross-run baselines and stay dormant until the agent has
run history. Check `docker compose logs detector` if you expected one.

---

## Advanced (optional)

### Self-monitoring via MCP

The Dunetrace MCP server exposes agent signals as tools an MCP-capable client can call. Once connected, a Langdock assistant can query its own failure history mid-conversation:

```bash
cd packages/mcp-server && pip install -e .
dunetrace-mcp --sse --port 8000            # binds 127.0.0.1 only
```

Langdock is a hosted service, so it cannot reach a loopback port. The SSE
server has **no authentication of its own** — see the warning in
[docs/mcp-server.md](mcp-server.md#codex--sse-clients) — so do not simply bind
it to `0.0.0.0`. Put an authenticating reverse proxy in front of it, keep
`dunetrace-mcp` on loopback behind that proxy, and give Langdock the proxy's
URL (ending in `/sse`) under its "External tools"/"MCP servers" workspace
setting. Leave `DUNETRACE_MCP_READONLY` at its default (`true`) unless you
intend a hosted assistant to be able to write policies. Available tools: `list_agents`, `get_agent_signals`, `get_agent_health`, `get_agent_patterns`, `get_run_detail`, `search_signals`, `summarize_agent`, `get_instrumentation_guide`.

### Troubleshooting

- **No runs appear** — check for `OTLP traces received` in ingest logs; confirm "Allow assistant logs" is enabled; test the endpoint with `curl -X POST .../v1/otlp/traces -d '{"resourceSpans":[]}'` (should return `{}`)
- **Runs appear but no signals** — the detector worker polls every 5s; wait a few seconds and check `docker compose logs detector`
- **Agent shows as `unknown-agent`** — Langdock isn't setting `service.name`; use the `X-Dunetrace-Agent-Id` header override if Langdock supports custom trace headers
