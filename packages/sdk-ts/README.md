# Dunetrace SDK for Node.js / TypeScript

Runtime observability for AI agents. Detects tool loops, cost spikes, context bloat, and 14 more failure patterns — automatically, on every run.

Zero runtime dependencies. Works with any Node.js AI framework. Node 18+.

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
    outputText:       res.choices[0].message.content ?? "",  // hashed, never sent raw
  });

  // Before + after each tool call
  const toolStart = Date.now();
  run.toolCalled("web_search", { query });          // args are SHA-256 hashed
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

## Deploy markers

```typescript
// Call from CI/CD or app startup — correlates signal spikes with releases
dt.markDeploy("my-agent", "v1.4.2", { env: "production", commit: "abc1234" });
```

## Langfuse integration

Correlate a Dunetrace run with a Langfuse trace using a shared UUID — jump straight from a detected signal to the full Langfuse trace.

```typescript
import { randomUUID } from "node:crypto";
import { Langfuse } from "langfuse";

const langfuse = new Langfuse({ publicKey: "pk-lf-…", secretKey: "sk-lf-…" });
const sharedId = randomUUID();

const trace = langfuse.trace({ id: sharedId, name: "my-agent" });

await dt.run("my-agent", { runId: sharedId, model: "gpt-4o" }, async (run) => {
  // run.runId === sharedId === Langfuse trace ID
  run.finalAnswer();
});

await langfuse.flushAsync();
```

See the full example: `examples/langfuse_agent.ts`

```bash
# Happy path
OPENAI_API_KEY=sk-… LANGFUSE_PUBLIC_KEY=pk-lf-… LANGFUSE_SECRET_KEY=sk-lf-… \
  npm run example:langfuse

# Tool loop — triggers TOOL_LOOP signal + LLM root cause analysis
SCENARIO=tool_loop … npm run example:langfuse:loop
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

## Run API

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
| `run.runId` | Read-only UUID — pass to Langfuse as the trace ID for correlation |

## Privacy

All content fields are SHA-256 hashed inside your process before transmission — raw content never leaves your agent.

| Field | Transmitted as |
|---|---|
| User input | `input_hash` (16-char hex) |
| Tool arguments | `args_hash` |
| LLM outputs | `output_hash` |
| Error messages | `error_hash` |
| Retrieval queries | `query_hash` |

Token counts, latencies, step counts, and model names are sent as plain metadata.

## What it detects

17 structural detectors run on every completed run — no LLM, no configuration required.

| Category | Detectors |
|---|---|
| Loops | `TOOL_LOOP` `TOOL_THRASHING` `RETRY_STORM` `LLM_TRUNCATION_LOOP` |
| Cost & latency | `COST_SPIKE` `SESSION_LATENCY` `CONTEXT_BLOAT` `SLOW_STEP` |
| Goal failures | `GOAL_ABANDONMENT` `TOOL_AVOIDANCE` `FIRST_STEP_FAILURE` `STEP_COUNT_INFLATION` |
| Quality | `REASONING_STALL` `EMPTY_LLM_RESPONSE` `RAG_EMPTY_RETRIEVAL` `CASCADING_TOOL_FAILURE` |
| Security | `PROMPT_INJECTION_SIGNAL` |

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

67 tests, all offline — no running stack required.

## Links

- [Full integration guide](https://github.com/dunetrace/dunetrace/blob/main/docs/integrate-typescript-agent.md)
- [Vercel AI SDK guide](https://github.com/dunetrace/dunetrace/blob/main/docs/integrate-vercel-ai.md)
- [MCP server](https://github.com/dunetrace/dunetrace/blob/main/docs/mcp-server.md) — query agent signals from Claude Code or Cursor
- [GitHub](https://github.com/dunetrace/dunetrace)
