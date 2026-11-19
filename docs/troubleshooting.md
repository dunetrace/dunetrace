# Troubleshooting

Most "nothing works" reports are one of five problems: the event never reached
ingest, the run never opened, the signal is in shadow, a detector is waiting for
data or a threshold, or auth/polling is in the way. Work down this page in that
order.

The commands below assume the local quickstart
(`docker compose -f docker-compose.ghcr.yml up -d`) and its default ports:
ingest `8001`, Customer API `8002`, dashboard `3000`. For a deployment, replace
`localhost` with your host and see [Operations](operations.md).

---

## 1. I instrumented my agent, but no events show up

### 1.1 Is the backend actually up?

```bash
curl -s localhost:8001/ready   # ingest
curl -s localhost:8002/ready   # Customer API
```

Both should return `{"status":"ok",...}`. If either fails, the SDK has nowhere
to send events:

```bash
docker compose -f docker-compose.ghcr.yml ps
docker compose -f docker-compose.ghcr.yml logs -f ingest
```

If you built the stack locally, drop `-f docker-compose.ghcr.yml`.

### 1.2 Is the SDK pointed at ingest, not at the Customer API?

For the core Python SDK, the ingest endpoint is `DUNETRACE_ENDPOINT` (default
`http://localhost:8001`), or the endpoint part of a DSN:

```bash
export DUNETRACE_DSN='https://<api_key>@ingest.example.com'
```

An explicit `endpoint=` / `api_key=` argument, or `DUNETRACE_ENDPOINT` /
`DUNETRACE_API_KEY`, wins over the corresponding part of the DSN.

`DUNETRACE_API_URL` is a different thing: it points at the **Customer API**
(default `http://localhost:8002`) and is used by the MCP server and pack /
approval calls. Setting `DUNETRACE_API_URL` alone will not move agent events.
(A few integrations have their own environment conventions; check that
framework's guide.) If the SDK runs in another container or on another
machine, `localhost` is that process's localhost, not the Dunetrace host — set
`DUNETRACE_ENDPOINT` to an address reachable from the SDK (the compose service
name is `ingest`).

For the TypeScript SDK, pass `endpoint` and `apiKey` to the constructor rather
than relying on the Python SDK's environment-variable names.

### 1.3 Did a run actually open?

`dt.init()` patches supported clients; it does not create a run by itself. A
patched LLM call outside a run either opens an **implicit** run (which is
shadowed and never alerts) or records nothing at all if the client was built
with `implicit_runs=False`, or if `auto_instrument()` ran without a client.

Give the call an exact boundary with one of:

```python
with dt.run("my-agent", user_input=question):
    answer = my_agent(question)
```

```python
@dt.agent()
def answer(question: str) -> str:
    ...
```

`@dt.trace` and `with dt.run(...)` do the same for non-agent entry points.
FastAPI/Flask requests, LangGraph `invoke`/`stream` and CrewAI `kickoff` already
open exact runs after `dt.init()`. See
[Auto-instrumentation › Runs that open themselves](integrations/auto-instrumentation.md#runs-that-open-themselves).

### 1.4 Did the process flush before exit?

Events ship from a background thread. A short script can exit before the batch
leaves the process. Call the shutdown hook before the interpreter exits:

```python
dt.shutdown()          # Python
await dt.shutdown()    # TypeScript
```

For a quick check, `Dunetrace(debug=True)` logs every event as it is buffered
and every batch as it is shipped.

### 1.5 Did ingest accept the batch?

```bash
docker compose -f docker-compose.ghcr.yml logs -f ingest
```

Look for the `POST /v1/ingest` result:

| Status | Meaning |
|---|---|
| `202` | The batch was accepted and committed. |
| `401` | Missing or invalid API key. See [401 / auth errors](#4-401--auth-errors). |
| `413` | The request body is larger than `INGEST_MAX_BODY_BYTES` (10 MiB by default). |
| `503` | Postgres is unavailable; the SDK buffers and retries, but events during the outage can be lost. |

`202` means "ingest accepted it", not "the events were correlated into a run".
Check the run itself:

```bash
curl -s localhost:8002/v1/agents
curl -s "localhost:8002/v1/agents/<agent-id>/runs?limit=5"
```

A working run has `step_count` above zero. An agent that has never reported
still returns `200` with an empty `runs` list, so check the list, not the
status code.

---

## 2. My run completed, but I see no signals

A clean run produces no signals. Signals are failure detectors, not a score.
Before treating "no signals" as a bug, check these four causes.

### 2.1 Is the signal in shadow mode?

A detector may be in **shadow** mode. Several ship that way by default.
Shadow signals are stored but not alerted — they are detectors in evaluation
mode, not broken detectors.

View them in either place:

```bash
curl -s "localhost:8002/v1/agents/<agent-id>/signals?include_shadow=true"
```

The dashboard fetches all signals with `?include_shadow=true` and renders
shadow signals below live alerts with a dashed border, reduced opacity and a
`SHADOW` badge. The Agents page also shows an ungraduated shadow count per
agent. See [Dashboard › Shadow signals](dashboard.md#shadow-signals).

To promote a detector, add it to `LIVE_DETECTORS` in
`packages/schemas-py/dunetrace_schemas/detector_status.py`. The file's rule is
to do that only after verifying precision above 80% on real traffic. That is a
code change: rebuild (or run from source) and restart the detector service.

### 2.2 Is the run implicit?

A run that the SDK opened and closed itself — because you called a patched LLM
client without `dt.run()` / a framework entry point — is an **implicit** run.
Every signal on it is written in shadow with confidence capped at `0.5` and
severity at `MEDIUM`, the evidence carries
`incomplete_data: {"reason": "implicit_run"}`, and policies never act on it.
Wrap the entry point (section 1.3) to turn it into an exact run that can alert.

### 2.3 Is the run incomplete?

If the SDK's outbound buffer overflows, it sheds whole runs and stamps
`dropped_events: <n>` on that run's `run.completed` / `run.errored` payload.
The detector treats any run with `dropped_events > 0` as incomplete: every
signal is shadow, confidence is capped at `0.5`, severity at `MEDIUM`, and the
evidence carries `incomplete_data: {"dropped_events": n, "reason":
"sdk_buffer_shed"}`. Such signals never feed issue tracking or baselines.
Check the run detail for `dropped_events`; if it is above zero, fix the
buffer/backpressure problem before judging the detector.

### 2.4 Does the detector need more history?

Several baseline-based detectors need at least 20 historical runs before they
can fire, including `CONTEXT_BLOAT`, `SLOW_STEP`, `STEP_COUNT_INFLATION`,
`COST_SPIKE`, `REASONING_STALL` and `SESSION_LATENCY`. Before the baseline is
ready they use a static fallback where one exists, and stay quiet where one
does not. This is expected for a new agent.

The detector worker polls every 5 seconds (`POLL_INTERVAL`). The dashboard
refreshes every 15 seconds. A signal can therefore take a few seconds to
appear after a run completes; see [Events are delayed](#5-events-are-delayed).

---

## 3. A detector isn't firing

First distinguish two different questions:

- **No signal at all** — the detector never wrote a row. Check data availability,
  thresholds and shadow mode.
- **A signal exists but no alert** — the detector fired, but its `alert_policy`
  in `detectors.yml` suppressed the alert (for example `mode: consecutive` with
  `threshold: 2`, or `mode: frequency` with `threshold: 3` and `window_runs:
  10`). One occurrence is counted and not alerted by design.

### 3.1 Is the data the detector needs actually instrumented?

Detectors are deterministic and only see instrumented events. Some need the
raw output text, not just success and length:

```python
run.tool_called("search", {"query": query})
run.tool_responded("search", success=True, output=raw_output)
```

Detectors such as `UNGROUNDED_DESTINATION`, `TOOL_ARGUMENT_FABRICATION`,
`RETRIEVED_CONTENT_INJECTION` and `AGENT_HANDOFF_FAILURE` read
`tool_responded(output=...)` / `retrieval_responded(content=...)`. If the
output text is never instrumented, they can stay silent even when the failure
is real. Auto-instrumented frameworks usually do this for you; manual
instrumentation has to pass the output.

### 3.2 Are the thresholds still the defaults?

Edit `detectors.yml` in the repo root — no code change or rebuild is needed.
The file documents every knob and its default. After editing, restart the
services that read it:

```bash
docker compose -f docker-compose.ghcr.yml restart detector alerts ingest
```

`detector` applies the thresholds server-side, `alerts` reads the same file for
per-detector `alert_policy`, and `ingest` serves the merged thresholds to the
SDK's in-path detector pass at `GET /v1/detector-config`.

### 3.3 Is an agent-category override hiding the default?

`detectors.yml` supports per-agent-category sections. The worker builds each
detector from the `default` section plus the agent's category section, falling
back to `default` when no matching section exists. If you added an override
under a category name the agent is not tagged with, it will not apply. The
commented examples in the file show `web-research` and `coding`; the full
model is in [Detectors › Tuning thresholds](detectors.md#tuning-thresholds).

### 3.4 Is the detector still shadow?

A shadow detector never alerts. See [2.1](#21-is-the-signal-in-shadow-mode) for
where to look and how to promote it.

### 3.5 Is the run complete enough to evaluate?

A run with `dropped_events > 0` is treated as incomplete: every signal is
shadow and confidence/severity are capped. Fix the buffer overflow first; see
[2.3](#23-is-the-run-incomplete).

---

## 4. 401 / auth errors

The quickstart and a production deployment use opposite auth models. Check
which one you are actually running.

| | Local quickstart | Production |
|---|---|---|
| `ENV` / `AUTH_MODE` | `dev` / `dev` — authentication off | `prod` / `prod` — every endpoint needs an API key |
| Published ports | `5432`, `8001`, `8002`, `3000`, all bound to `127.0.0.1` | dashboard on `0.0.0.0:3000`; everything else stays on loopback |
| Agent credential | none | `DUNETRACE_API_KEY` with `ingest` scope |
| Dashboard / MCP | no real key | `admin`-scoped key for the Customer API |

The service code defaults `AUTH_MODE` to `prod`; `dev` is always an explicit
opt-in. The compose files set `ENV=dev` and `AUTH_MODE=dev` for the local
quickstart. Dev mode is safe there only because every published port is bound
to `127.0.0.1`; anyone who can reach a port is an admin of the default org.
Do not widen that binding without switching to the production override.

### 4.1 Local quickstart returns 401

If the local stack is supposed to be in dev mode, confirm both values in the
running service, not just in `.env`:

```bash
docker compose -f docker-compose.ghcr.yml ps
docker compose -f docker-compose.ghcr.yml logs --tail=50 ingest api
```

A service logging `auth_mode=dev` is unauthenticated; a service logging
`auth_mode=prod` requires a key. `.env` is read by compose for interpolation,
not injected into containers as a file; a variable only reaches a service if
that service's `environment:` block names it. See [`.env.example`](../.env.example)
for the full explanation.

### 4.2 Production returns 401

`401 Missing Authorization header` means the request carried no bearer token.
`401 Trusted request missing x-org-id` means an internal service-to-service
call is missing its trusted org header. Check the agent's `DUNETRACE_API_KEY`
and that it is sent as `Authorization: Bearer <key>`.

Keys are minted in two steps. First bootstrap an admin key from ingest, using
the deployment's `ADMIN_API_KEY`:

```bash
curl -X POST http://localhost:8001/v1/keys \
  -H 'Content-Type: application/json' \
  -d '{"org_id": "my-company", "org_name": "My Company", "admin_key": "'"$ADMIN_API_KEY"'"}'
```

Then mint a narrower key for the agent:

```bash
curl -X POST http://localhost:8002/v1/keys \
  -H "Authorization: Bearer $DUNETRACE_ADMIN_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"scopes": ["ingest"]}'
```

An `ingest` key cannot mint keys, write policies or decide approvals, and
`POST /v1/keys` returns `403` for any scope the calling key does not hold.

Do **not** insert into `api_keys` by hand: keys are verified against a SHA-256
hash, and a row written with a plaintext `key` and no `key_hash` never
authenticates. See [Quickstart › Production](quickstart.md#production).

### 4.3 A service refuses to start

Every service refuses to start with `ENV=prod` and `AUTH_MODE=dev`. If a
container exits immediately, check that pair first. For a real deployment use
the production override, which pins both values as literals.

---

## 5. Events are delayed

Short delays are normal. The detector worker polls every 5 seconds
(`POLL_INTERVAL` default `5`), the alerts worker polls every 10 seconds
(`POLL_INTERVAL` default `10`), and the dashboard auto-refreshes every 15
seconds. A run can therefore take a few seconds to produce a signal and another
poll cycle to produce an alert; the dashboard can take one more refresh to
show it.

If it is longer than that:

```bash
docker compose -f docker-compose.ghcr.yml ps
docker compose -f docker-compose.ghcr.yml logs --tail=100 detector alerts
```

Check:

- both workers are running and healthy;
- the detector's `POLL_INTERVAL` has not been raised and `BATCH_SIZE` has not
  been lowered;
- there is no backlog of unprocessed runs;
- Postgres is healthy (ingest returns `503` when it is not).

If the detector worker is down, runs queue in the `events` table and are
processed when it restarts — signals are delayed, not lost. If Postgres is
down, ingest returns `503`; the SDK buffers and retries, but events during the
outage can be lost. See [Architecture › Failure modes](architecture.md#failure-modes).

If you changed `detectors.yml`, remember that only a restart applies it to the
detector, alerts and ingest services.

---

## 6. Before opening an issue

Collect these first; they answer most of the questions above in one round trip:

```bash
docker compose -f docker-compose.ghcr.yml ps
curl -s localhost:8001/ready
curl -s localhost:8002/ready
docker compose -f docker-compose.ghcr.yml logs --tail=100 ingest detector alerts
```

Also include:

- SDK language and version (`pip show dunetrace` / `npm ls dunetrace`);
- OS and Python/Node version;
- the agent ID and run ID (the dashboard's run detail has a copy-ID button);
- `step_count`, `signal_count`, `dropped_events` and whether the run is
  `implicit`;
- the `Dunetrace(debug=True)` output around the failing call.

If the run is visible but the signal is not, include the output of
`/v1/agents/<agent-id>/signals?include_shadow=true` for that run.
