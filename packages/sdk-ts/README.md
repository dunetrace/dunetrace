# Dunetrace SDK for Node.js / TypeScript

Runtime observability for AI agents. Detects tool loops, context bloat, prompt injection, and more — in real-time.

Zero runtime dependencies. Works with any Node.js AI framework.

## Install

```bash
npm install dunetrace
```

Node 18+ required (uses built-in `fetch` and `AsyncLocalStorage`).

## Quickstart

```typescript
import { Dunetrace } from "dunetrace";

const dt = new Dunetrace();   // default: http://localhost:8001, no key required

await dt.run("my-agent", {
  model:     "gpt-4o",
  tools:     ["web_search"],
  userInput: query,           // hashed before transmission — never sent raw
}, async (run) => {

  run.llmCalled("gpt-4o", 150);
  const t0  = Date.now();
  const res = await openai.chat.completions.create({ /* ... */ });
  run.llmResponded({
    completionTokens: res.usage?.completion_tokens,
    latencyMs:        Date.now() - t0,
    finishReason:     res.choices[0].finish_reason ?? "stop",
    outputText:       res.choices[0].message.content ?? "",  // hashed, not transmitted
  });

  run.toolCalled("web_search", { query });  // args are SHA-256 hashed
  const results = await webSearch(query);
  run.toolResponded("web_search", true, results.length, Date.now() - t0);

  run.finalAnswer();
});

await dt.shutdown();
```

## Auto-wrap tools with `dt.tool()`

```typescript
const search = dt.tool(webSearch, "web_search");
// or infer name from function.name:
const search = dt.tool(webSearch);

// Inside dt.run(), tool.called / tool.responded are emitted automatically.
// Outside a run, the function runs normally — dt.tool() is a no-op.
const results = await search(query);
```

## Auto-wrap agents with `dt.trace()`

```typescript
const monitoredAgent = dt.trace(myAgent, "my-agent", { model: "gpt-4o" });
// or infer agentId from function name:
const monitoredAgent = dt.trace(myAgent);

const answer = await monitoredAgent(userQuery);
```

## RAG / retrieval

```typescript
await dt.run("rag-agent", { model: "gpt-4o" }, async (run) => {
  run.retrievalCalled("product-docs", query);   // query is SHA-256 hashed
  const t0   = Date.now();
  const docs = await vectorStore.search(query);
  run.retrievalResponded("product-docs", docs.length, docs[0]?.score, Date.now() - t0);

  run.llmCalled("gpt-4o", 600);
  run.llmResponded({ finishReason: "stop", completionTokens: 120 });
  run.finalAnswer();
});
```

## Infrastructure signals

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

## Access current run from nested code

```typescript
import { getCurrentRun } from "dunetrace";

function myHelper() {
  const run = getCurrentRun();  // works inside any async code within dt.run()
  if (run) run.externalSignal("cache_miss");
}
```

## Deploy markers

```typescript
// Call from CI/CD or app startup — fire-and-forget
dt.markDeploy("my-agent", "v1.4.2", { env: "production", commit: "abc1234" });
```

## Output modes

| Mode | How to enable | Destination |
|---|---|---|
| HTTP ingest (default) | `new Dunetrace({ endpoint: "http://…" })` | Dunetrace backend → detection + alerts |
| Loki NDJSON | `new Dunetrace({ emitAsJson: true })` | stdout → Promtail / Grafana Alloy |

## Langfuse integration

Correlate a Dunetrace run with a Langfuse trace using a shared UUID. Pass the same ID to both — Dunetrace stores it as `run_id` and Langfuse stores it as the trace ID, so you can jump from a detected signal straight to the full trace.

```typescript
import { randomUUID } from "node:crypto";
import { Langfuse } from "langfuse";

const langfuse = new Langfuse({ publicKey: "pk-lf-…", secretKey: "sk-lf-…" });

const sharedId = randomUUID();

// Langfuse trace — use sharedId as the trace ID
const trace = langfuse.trace({ id: sharedId, name: "my-agent" });

// Dunetrace run — same ID links the two
await dt.run("my-agent", { runId: sharedId, model: "gpt-4o" }, async (run) => {
  // … instrument as normal …
  run.finalAnswer();
});

await langfuse.flushAsync();
```

See the full example: `examples/langfuse_agent.ts`

```bash
# Happy path
OPENAI_API_KEY=sk-… LANGFUSE_PUBLIC_KEY=pk-lf-… LANGFUSE_SECRET_KEY=sk-lf-… \
  npm run example:langfuse

# Tool loop — triggers TOOL_LOOP signal + explain
SCENARIO=tool_loop … npm run example:langfuse:loop
```

## Configuration

```typescript
const dt = new Dunetrace({
  endpoint:        "http://localhost:8001",  // ingest service URL
  apiKey:          "",                       // required for production
  flushIntervalMs: 200,                      // background drain interval
  emitAsJson:      false,                   // Loki NDJSON mode
});
```

## RunContext API

| Method | When to call |
|---|---|
| `run.llmCalled(model, promptTokens?)` | Before each LLM API call |
| `run.llmResponded({ completionTokens?, latencyMs?, finishReason?, outputText? })` | After LLM responds |
| `run.toolCalled(toolName, args?)` | Before each tool execution |
| `run.toolResponded(toolName, success, outputLength?, latencyMs?, error?)` | After tool returns |
| `run.retrievalCalled(indexName, query?)` | Before vector search |
| `run.retrievalResponded(indexName, resultCount, topScore?, latencyMs?)` | After retrieval returns |
| `run.externalSignal(signalName, source?, meta?)` | Rate limits, cache misses, upstream errors |
| `run.finalAnswer()` | When agent produces its final output |
| `run.runId` | Read-only UUID for this run — pass as Langfuse trace ID for correlation |

## Privacy

No raw content is ever transmitted. All content fields are SHA-256 hashed (16 chars) inside your process before being sent to the backend.

- User input → `input_hash`
- Tool arguments → `args_hash`
- LLM outputs → `output_hash`
- Error messages → `error_hash`
- Retrieval queries → `query_hash`

Token counts, latencies, step counts, and model names are transmitted as plain metadata.

## What it detects

17 structural detectors run on every completed run. No LLM, no configuration required.

`TOOL_LOOP` · `TOOL_THRASHING` · `RETRY_STORM` · `CONTEXT_BLOAT` · `COST_SPIKE` · `SESSION_LATENCY` · `SLOW_STEP` · `GOAL_ABANDONMENT` · `REASONING_STALL` · `LLM_TRUNCATION_LOOP` · `EMPTY_LLM_RESPONSE` · `CASCADING_TOOL_FAILURE` · `STEP_COUNT_INFLATION` · `FIRST_STEP_FAILURE` · `RAG_EMPTY_RETRIEVAL` · `TOOL_AVOIDANCE` · `PROMPT_INJECTION_SIGNAL`

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

67 tests, all offline — no running stack required.

## Links

- [Full integration guide](../../docs/integrate-typescript-agent.md)
- [GitHub](https://github.com/dunetrace/dunetrace)
