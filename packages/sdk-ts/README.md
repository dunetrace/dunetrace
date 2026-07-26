# Dunetrace SDK for Node.js / TypeScript

Runtime observability for AI agents. Detects tool loops, cost spikes, context bloat, and 14 more failure patterns — automatically, on every run.

Zero runtime dependencies by default. Works with any Node.js AI framework. Node 22+. (Durable retry is opt-in and adds one optional peer dependency — see [Durable retry](#durable-retry).)

## Install

```bash
npm install dunetrace
```

## Quickstart

```typescript
import { Dunetrace } from "dunetrace";

const dt = new Dunetrace();  // default: http://localhost:8001

await dt.run("my-agent", { model: "gpt-4o", tools: ["web_search"] }, async (run) => {

  // Before + after each LLM call
  run.llmCalled("gpt-4o", promptTokens);
  const t0  = Date.now();
  const res = await openai.chat.completions.create({ /* ... */ });
  run.llmResponded({
    completionTokens: res.usage?.completion_tokens,
    latencyMs:        Date.now() - t0,
    finishReason:     res.choices[0].finish_reason ?? "stop",
    outputText:       res.choices[0].message.content ?? "",  // sent as-is
  });

  // Before + after each tool call
  const toolStart = Date.now();
  run.toolCalled("web_search", { query });          // args are sent as-is
  const results = await webSearch(query);
  run.toolResponded("web_search", true, results.length, Date.now() - toolStart);

  run.finalAnswer();
});

await dt.shutdown();
```

## Auto-wrap tools with `dt.tool()`

```typescript
// Wraps the function — emits tool.called + tool.responded automatically
const search = dt.tool(webSearch, "web_search");
// or infer the name from function.name:
const search = dt.tool(webSearch);

// Inside dt.run() it's tracked; outside a run it passes through unchanged
const results = await search(query);
```

## Auto-wrap agents with `dt.trace()`

```typescript
// Wraps the agent function — starts and ends a Dunetrace run for each call
const monitoredAgent = dt.trace(myAgent, "my-agent", { model: "gpt-4o" });

const answer = await monitoredAgent(userQuery);
```

## RAG / retrieval

```typescript
await dt.run("rag-agent", { model: "gpt-4o" }, async (run) => {
  run.retrievalCalled("product-docs", query);
  const t0   = Date.now();
  const docs = await vectorStore.search(query);
  run.retrievalResponded("product-docs", docs.length, docs[0]?.score, Date.now() - t0);

  run.llmCalled("gpt-4o", 600);
  run.llmResponded({ finishReason: "stop", completionTokens: 120 });
  run.finalAnswer();
});
```

## Rate limits and errors

```typescript
await dt.run("my-agent", { model: "gpt-4o" }, async (run) => {
  try {
    run.toolCalled("external_api");
    const result = await callExternalApi();
    run.toolResponded("external_api", true, result.length);
  } catch (err) {
    if (isRateLimitError(err)) {
      run.externalSignal("rate_limit", "external_api", { http_status: 429 });
    }
    run.toolResponded("external_api", false, 0, 0, String(err));
  }
  run.finalAnswer();
});
```

## Access the current run from nested code

```typescript
import { getCurrentRun } from "dunetrace";

function myHelper() {
  const run = getCurrentRun();  // works anywhere inside an active dt.run()
  if (run) run.externalSignal("cache_miss");
}
```

## Auto-instrument the OpenAI / Anthropic SDKs

Patch once at startup and every client instance is tracked inside a `dt.run()` — including clients constructed by code you don't control:

```typescript
import OpenAI from "openai";
import { Dunetrace, autoInstrument } from "dunetrace";

const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
autoInstrument({ openai: OpenAI });   // → ["openai"]

const openai = new OpenAI();          // constructed after the patch — still tracked
await dt.run("my-agent", {}, async () => {
  await openai.chat.completions.create({ model: "gpt-4o", messages });
});
```

`autoInstrument()` with no arguments auto-detects via `require`, which only resolves under CommonJS — **pass the imported class if you use ESM or a bundler**. It patches the resource prototype shared by every client, which works under ESM, CommonJS, and bundlers alike. Idempotent, and safe to combine with `dt.wrapOpenAI()`.

It covers three targets — `openai`, `anthropic`, and `http` — narrowable via `targets`.

**Streaming is tracked.** `llm.called` fires at call time; the stream comes back through a pass-through proxy that observes chunks as you pull them and emits `llm.responded` when it ends. Nothing is consumed on your behalf, and `.tee()` / `.controller` still work. Pass `stream_options: { include_usage: true }` for OpenAI or the API never sends token counts.

**HTTP is tracked** via the global `fetch` — `tool.called` / `tool.responded` named by hostname, the Node counterpart to the Python SDK's `httpx`/`requests` patches. Requests made *by* an instrumented LLM SDK are excluded, so one LLM call doesn't also count as a tool call; so is Dunetrace's own ingest traffic.

Emission failures never propagate into your call.

To instrument a single client rather than the whole SDK:

```typescript
const openai = dt.wrapOpenAI(new OpenAI());
const claude = dt.wrapAnthropic(new Anthropic());
```

See [docs/integrate-typescript-agent.md](../../docs/integrate-typescript-agent.md#auto-instrumentation) for the full picture, including what isn't covered.

## Deploy markers

```typescript
// Call from CI/CD or app startup — correlates signal spikes with releases
dt.markDeploy("my-agent", "v1.4.2", { env: "production", commit: "abc1234" });
```

## Vercel AI SDK integration

Wrap `generateText` / `streamText` from the `ai` package — LLM steps and tool calls are tracked automatically inside `dt.run()`:

```typescript
import { Dunetrace, wrapGenerateText } from "dunetrace";
import { generateText } from "ai";

const dt = new Dunetrace();
const instrumentedGenerateText = wrapGenerateText(generateText);

await dt.run("my-agent", { userInput: prompt, model: "gpt-4o" }, async (run) => {
  await instrumentedGenerateText({ model, prompt, tools });
  run.finalAnswer();
});
```

See [integrate-vercel-ai.md](../../docs/integrate-vercel-ai.md) for streaming, Next.js, and `traceGenerateText`. Requires the `ai` package (`npm install ai`).

## Output modes

| Mode | How to enable | Destination |
|---|---|---|
| HTTP ingest (default) | `new Dunetrace({ endpoint: "http://…" })` | Dunetrace backend → detection + alerts |
| Loki NDJSON | `new Dunetrace({ emitAsJson: true })` | stdout → Promtail / Grafana Alloy |

## Configuration

| Option | Default | Description |
|---|---|---|
| `endpoint` | `http://localhost:8001` | Ingest service URL |
| `apiKey` | `""` | API key (required for production) |
| `flushIntervalMs` | `200` | Background buffer drain interval (ms) |
| `emitAsJson` | `false` | Loki NDJSON mode |
| `emitter` | `HttpBatchEmitter` | Custom batch-shipping strategy — see [Durable retry](#durable-retry) |

## Durable retry

By default, a batch that fails to ship (backend down, network error) is dropped — same behavior as before this was made pluggable. `DurableRetryEmitter` wraps the default emitter to persist failed batches to a local SQLite queue instead, surviving a backend outage across process restarts:

```ts
import { Dunetrace, DurableRetryEmitter, HttpBatchEmitter } from "dunetrace";

const dt = new Dunetrace({
  endpoint: "https://ingest.dunetrace.com",
  apiKey:   "dt_live_...",
  emitter:  new DurableRetryEmitter(
    new HttpBatchEmitter("https://ingest.dunetrace.com", "dt_live_...")
  ),
});
```

Requires the optional peer dependency `better-sqlite3` (`npm install better-sqlite3`) — its synchronous API matches this emitter's own single-threaded access pattern. If it isn't installed, or the queue file can't be created, `DurableRetryEmitter` degrades to "failed batches are dropped" (a warning is logged once) rather than crashing the host application.

| Option | Default | Description |
|---|---|---|
| `queuePath` | `~/.dunetrace/queue-ts.db` | Also configurable via `DUNETRACE_QUEUE_PATH` env var. Deliberately a different filename from the Python SDK's `queue.db` — a mixed-language deployment sharing `~/.dunetrace/` can't have one SDK corrupt the other's queue file. |
| `maxQueueEvents` | `100_000` | Oldest batches are evicted first once exceeded |
| `maxQueueBytes` | `100 * 1024 * 1024` (100MB) | Whichever cap (events or bytes) is hit first triggers eviction |
| `retryIntervalMs` | `30_000` | How often the backlog is retried, ± jitter |
| `retryJitterMs` | `5_000` | Prevents a thundering herd if many instances reconnect together |
| `maxBatchesPerRetry` | `50` | Backlog batches retried per `ship()` call, oldest first — stops at the first failure rather than skipping ahead, since order matters more than exhausting the backlog in one pass |

Node.js only — there is no browser build of this SDK today (`engines.node >= 22`, CommonJS, no bundler target), so there's no IndexedDB fallback to reach for.

## Run API

| Method | When to call |
|---|---|
| `run.llmCalled(model, promptTokens?)` | Before each LLM API call |
| `run.llmResponded({ completionTokens?, latencyMs?, finishReason?, outputText? })` | After LLM responds |
| `run.toolCalled(toolName, args?)` | Before each tool execution |
| `run.toolResponded(toolName, success, outputLength?, latencyMs?, error?, output?)` | After tool returns |
| `run.retrievalCalled(indexName, query?)` | Before vector search |
| `run.retrievalResponded(indexName, resultCount, topScore?, latencyMs?, content?)` | After retrieval returns |
| `run.externalSignal(signalName, source?, meta?)` | Rate limits, cache misses, upstream errors |
| `run.memoryWritten(key, value, source?)` | Agent persisted something to its memory (`source`: `user_input` / `retrieval` / `tool_output` / `llm_output` / `agent_reasoning` / `external`) |
| `run.memoryRead(key)` | Agent read from its memory |
| `run.memoryCleared(key?)` | Memory cleared (a key, or all when omitted) |
| `run.finalAnswer()` | When agent produces its final output |
| `run.runId` | Read-only UUID — use to correlate with an external tracing system |

## What it detects

28 structural detectors run on every completed run — no LLM, no configuration required. Detection runs server-side (the same detector worker processes every agent's events regardless of source SDK), so additions here apply to TypeScript agents automatically. A few of the main ones:

| Category | Detectors |
|---|---|
| Loops | `TOOL_LOOP` `RETRY_STORM` `DELEGATION_LOOP` |
| Cost & latency | `COST_SPIKE` `SESSION_LATENCY` |
| Security | `PROMPT_INJECTION_SIGNAL` `MEMORY_POISONING` |
| Silent degradation | `PREMATURE_TERMINATION` `RUNAWAY_ITERATION` |

→ [docs/detectors.md](https://github.com/dunetrace/dunetrace/blob/main/docs/detectors.md) for the full list of 28 detectors

You can also define **custom detectors** in plain English from the dashboard or API. They run in shadow mode on every run and accumulate results before any alert fires. → [Custom detectors](https://github.com/dunetrace/dunetrace/blob/main/docs/detectors.md#custom-detectors)

## Backend

```bash
git clone https://github.com/dunetrace/dunetrace
cd dunetrace && cp .env.example .env && docker compose up -d
```

Dashboard → `http://localhost:3000` · Ingest → `http://localhost:8001`

## Tests

```bash
npm test
```

123 tests, all offline — no running stack required.

## Links

- [Full integration guide](https://github.com/dunetrace/dunetrace/blob/main/docs/integrate-typescript-agent.md)
- [Vercel AI SDK guide](https://github.com/dunetrace/dunetrace/blob/main/docs/integrate-vercel-ai.md)
- [MCP server](https://github.com/dunetrace/dunetrace/blob/main/docs/mcp-server.md) — query agent signals from Claude Code or Cursor
- [GitHub](https://github.com/dunetrace/dunetrace)
