# Integrating a TypeScript Agent with Dunetrace

This guide covers adding Dunetrace monitoring to a TypeScript or JavaScript agent using the `dunetrace` npm package.

> **Using Python?** See [integrate-custom-python-agent.md](./integrate-custom-python-agent.md).
> **Using LangChain, CrewAI, or AutoGen?** See [integrations.md](./integrations.md).

---

## How It Works

The SDK buffers events in-process and ships them to the Dunetrace ingest service in the background — a 200ms drain loop, batched at 100 events per flush. The same 17 structural detectors, dashboard, and alerts that run for Python agents apply here.

```
your TS agent  →  POST /v1/ingest  →  detector  →  dashboard + Slack alerts
```

Zero runtime dependencies. Works with any Node 18+ runtime.

---

## Prerequisites

- Dunetrace backend running (`docker compose up -d`)
- Node 18+ (built-in `fetch` and `AsyncLocalStorage` required)

> **Local dev — no API key needed.** The backend accepts requests without authentication when running in dev mode. Skip Step 1 when testing locally.

---

## Step 1: Generate an API Key (production only)

Connect to your Dunetrace Postgres instance and run:

```sql
INSERT INTO api_keys (key, agent_id, customer_id)
VALUES ('dt_live_<your-random-string>', 'my-ts-agent', 'my-company');
```

Generate a secure random suffix:

```bash
node -e "const c=require('crypto'); console.log('dt_live_'+c.randomBytes(16).toString('hex'))"
```

---

## Step 2: Set Environment Variables

```bash
# .env
DUNETRACE_ENDPOINT=http://localhost:8001   # ingest service (port 8001, not 8002)
DUNETRACE_API_KEY=                         # leave empty for local dev
```

---

## Step 3: Install

```bash
npm install dunetrace
```

---

## Step 4: Basic Usage

```typescript
import { Dunetrace } from "dunetrace";

const dt = new Dunetrace({
  endpoint: process.env.DUNETRACE_ENDPOINT,   // default: http://localhost:8001
  apiKey:   process.env.DUNETRACE_API_KEY,
});

await dt.run("my-ts-agent", {
  systemPrompt: "You are a helpful research assistant.",
  model:        "gpt-4o",
  tools:        ["web_search"],
  userInput:    query,          // hashed before transmission — never sent raw
}, async (run) => {

  // Before each LLM call
  run.llmCalled("gpt-4o", estimatedPromptTokens);
  const t0 = Date.now();

  const response = await openai.chat.completions.create({ model: "gpt-4o", messages });
  const output   = response.choices[0].message.content ?? "";

  // After LLM responds
  run.llmResponded({
    completionTokens: response.usage?.completion_tokens,
    latencyMs:        Date.now() - t0,
    finishReason:     response.choices[0].finish_reason ?? "stop",
    outputText:       output,    // hashed before transmission — never sent raw
  });

  // Tool call
  run.toolCalled("web_search", { query });
  const t1      = Date.now();
  const results = await webSearch(query);
  run.toolResponded("web_search", true, results.length, Date.now() - t1);

  run.finalAnswer();
});

await dt.shutdown();   // flush remaining events before process exits
```

`dt.run()` emits `run.started` on entry, `run.completed` on clean return, and `run.errored` if an exception escapes (re-thrown after recording).

---

## Step 5: `dt.tool()` — Auto-Wrap Tools

`dt.tool()` wraps any function to automatically emit `tool.called` / `tool.responded` events around each invocation. No-op when called outside a `dt.run()` context — the underlying function still runs normally.

```typescript
import { Dunetrace, getCurrentRun } from "dunetrace";

const dt = new Dunetrace();

// Wrap once at startup — tool name defaults to the function name
const search   = dt.tool(webSearch);
const fetchDoc = dt.tool(fetchDocument, "doc_fetch");   // explicit name

async function runAgent(query: string) {
  await dt.run("my-agent", { model: "gpt-4o", tools: ["webSearch", "doc_fetch"] }, async (run) => {
    run.llmCalled("gpt-4o", 150);
    run.llmResponded({ finishReason: "tool_calls" });

    // tool.called / tool.responded emitted automatically
    const results = await search(query);

    run.llmCalled("gpt-4o", 400);
    run.llmResponded({ finishReason: "stop", completionTokens: 80 });
    run.finalAnswer();
  });
}
```

Works for both sync and async functions. On failure, `success=false` and the error string is hashed and recorded before the exception is re-thrown.

---

## Step 6: `dt.trace()` — Function Decorator

`dt.trace()` wraps an async function so it automatically opens and closes a run each time it is called. The first argument to the wrapped function is used as `userInput`.

```typescript
// Define your agent function normally
async function myAgent(query: string): Promise<string> {
  const run = getCurrentRun()!;   // available inside because trace() sets context

  run.llmCalled("gpt-4o", 150);
  const response = await openai.chat.completions.create({ /* ... */ });
  run.llmResponded({ finishReason: "stop" });

  return response.choices[0].message.content ?? "";
}

// Wrap it — agentId defaults to the function name ("myAgent")
const agent = dt.trace(myAgent, "my-agent", {
  model: "gpt-4o",
  tools: ["web_search"],
  systemPrompt: "You are a helpful assistant.",
});

// Call normally — run.started / run.completed fire automatically
const answer = await agent("What is the capital of France?");

await dt.shutdown();
```

`dt.trace()` calls `run.finalAnswer()` automatically on clean return. If the wrapped function throws, `run.errored` is emitted and the exception propagates.

---

## Step 7: `getCurrentRun()` — Access the Run from Anywhere

`getCurrentRun()` returns the active `DunetraceRun` for the current async context, or `null` if no run is active. Uses `AsyncLocalStorage` — no prop drilling needed.

```typescript
import { Dunetrace, getCurrentRun } from "dunetrace";

const dt = new Dunetrace();

// A utility called deep in your call stack
async function dbQuery(sql: string): Promise<unknown[]> {
  const run = getCurrentRun();
  if (run) run.toolCalled("db_query", { sql });

  const t0     = Date.now();
  const result = await db.query(sql);

  if (run) run.toolResponded("db_query", true, result.length, Date.now() - t0);
  return result;
}

// Works inside dt.run(), dt.trace(), or any function they call transitively
await dt.run("my-agent", { model: "gpt-4o" }, async (run) => {
  const rows = await dbQuery("SELECT * FROM products LIMIT 10");
  run.finalAnswer();
});
```

---

## Step 8: RAG / Retrieval Agents

```typescript
await dt.run("rag-agent", { model: "gpt-4o", tools: ["vector_search"] }, async (run) => {
  run.llmCalled("gpt-4o", 200);
  run.llmResponded({ finishReason: "tool_calls" });

  // retrieval.called — query is hashed before transmission
  run.retrievalCalled("product-docs", query);
  const t0   = Date.now();
  const docs = await vectorStore.search(query, { topK: 5 });
  run.retrievalResponded("product-docs", docs.length, docs[0]?.score, Date.now() - t0);

  run.llmCalled("gpt-4o", 600);
  run.llmResponded({ finishReason: "stop", completionTokens: 120, outputText: answer });
  run.finalAnswer();
});
```

`RAG_EMPTY_RETRIEVAL` fires when `resultCount` is 0 or `topScore` is below 0.3 but the agent still produces a final answer.

---

## Step 9: Error Handling and Infrastructure Signals

```typescript
await dt.run("my-agent", { model: "gpt-4o", tools: ["external_api"] }, async (run) => {
  run.toolCalled("external_api", { endpoint: "/data" });
  const t0 = Date.now();

  try {
    const result = await callExternalApi();
    run.toolResponded("external_api", true, result.length, Date.now() - t0);
  } catch (err) {
    // Annotate with infrastructure context — does not advance the step counter
    if (isRateLimitError(err)) {
      run.externalSignal("rate_limit", "external_api", { http_status: 429 });
    }
    // Record the failure — error message is hashed before transmission
    run.toolResponded("external_api", false, 0, Date.now() - t0, String(err));
    throw err;   // re-throw — dt.run() will emit run.errored
  }

  run.finalAnswer();
});
```

`SlowStepDetector` checks for coincident external signals within a step's time window and includes them in evidence (`coincident_signals`). `RETRY_STORM` fires when the same tool fails 3+ consecutive times.

---

## Step 10: Sub-Agent Tracking

Link child agent runs to a parent using `parentRunId`. Both runs appear in the dashboard with a visual hierarchy indicator.

```typescript
await dt.run("orchestrator", { model: "gpt-4o" }, async (parentRun) => {
  // Pass the parent's run ID to child agents
  await dt.run("sub-agent", {
    model:       "gpt-4o-mini",
    parentRunId: parentRun.runId,
  }, async (childRun) => {
    childRun.llmCalled("gpt-4o-mini", 100);
    childRun.llmResponded({ finishReason: "stop" });
    childRun.finalAnswer();
  });

  parentRun.finalAnswer();
});
```

---

## Step 11: Deploy Markers

Fire-and-forget deploy markers let the dashboard overlay release boundaries on detector rate charts.

```typescript
// Call from CI/CD or app startup — does not block
dt.markDeploy("my-ts-agent", "v1.4.2", {
  commit:      "abc123f",
  environment: "production",
});
```

The `meta` object accepts any key-value pairs. Deploy markers appear as blue dashed vertical lines on the 30-day health record charts.

---

## Step 12: Loki / Grafana NDJSON

Enable `emitAsJson` to write one NDJSON line per event to stdout — useful when you already have a Promtail / Grafana Alloy pipeline.

```typescript
const dt = new Dunetrace({
  emitAsJson: true,    // writes to stdout
  endpoint:   null,    // disable HTTP ingest (or keep both active simultaneously)
});
```

Each line:

```
{"ts":"2026-05-14T10:00:00.123Z","level":"info","logger":"dunetrace",
 "event_type":"tool.called","agent_id":"my-agent","run_id":"…","step_index":3,"payload":{…}}
```

Minimal Promtail pipeline stage:

```yaml
pipeline_stages:
  - json:
      expressions: {ts: ts, event_type: event_type, agent_id: agent_id}
  - timestamp:
      source: ts
      format: RFC3339Nano
  - labels:
      agent_id:
      event_type:
```

HTTP ingest and `emitAsJson` can both be active at the same time.

---

## Step 13: Langfuse Integration

Correlate a Dunetrace run with a Langfuse trace. Because the npm package generates `run.runId` inside `dt.run()`, capture it from the callback and use it to open the Langfuse trace:

```typescript
import { randomUUID }  from "node:crypto";
import { Langfuse }    from "langfuse";
import { Dunetrace }   from "dunetrace";

const dt = new Dunetrace();
const langfuse = new Langfuse({
  publicKey: process.env.LANGFUSE_PUBLIC_KEY!,
  secretKey: process.env.LANGFUSE_SECRET_KEY!,
});

await dt.run("my-ts-agent", {
  model:        "gpt-4o",
  tools:        ["web_search"],
  userInput:    query,
  systemPrompt: SYSTEM_PROMPT,
}, async (run) => {

  // run.runId is now available — use it as the Langfuse trace ID
  const trace = langfuse.trace({ id: run.runId, name: "my-ts-agent", input: { query } });

  const messages: OpenAI.ChatCompletionMessageParam[] = [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user",   content: query },
  ];

  for (let step = 0; step < 12; step++) {
    run.llmCalled("gpt-4o", estimateTokens(messages));

    const generation = trace.generation({
      name:  `step-${step}`,
      model: "gpt-4o",
      input: messages,
    });

    const t0       = Date.now();
    const response = await openai.chat.completions.create({
      model: "gpt-4o", messages, tools: [TOOL_DEF], tool_choice: "auto",
    });
    const latencyMs = Date.now() - t0;
    const msg       = response.choices[0].message;

    run.llmResponded({
      completionTokens: response.usage?.completion_tokens,
      latencyMs,
      finishReason:     response.choices[0].finish_reason ?? "stop",
      outputText:       msg.content ?? "",
    });
    generation.end({ usage: response.usage ?? undefined });

    messages.push(msg as OpenAI.ChatCompletionMessageParam);

    if (response.choices[0].finish_reason === "stop" || !msg.tool_calls?.length) {
      run.finalAnswer();
      break;
    }

    for (const tc of msg.tool_calls ?? []) {
      const args   = JSON.parse(tc.function.arguments) as Record<string, unknown>;
      run.toolCalled(tc.function.name, args);

      const t1     = Date.now();
      const result = await webSearch(args["query"] as string ?? "");
      run.toolResponded(tc.function.name, true, result.length, Date.now() - t1);

      messages.push({ role: "tool", tool_call_id: tc.id, content: result });
    }
  }

  trace.update({ output: { answer_length: messages.at(-1)?.content?.length ?? 0 } });
});

await langfuse.flushAsync();
await dt.shutdown();
```

`run.runId` and the Langfuse trace ID are now the same UUID. When a signal fires, the dashboard **Explain with Langfuse** button passes `run.runId` (= `langfuse_trace_id`) directly to `POST /v1/signals/{id}/explain`.

For a complete working example with the explain/apply-fix flow:

```bash
cd packages/sdk-ts
OPENAI_API_KEY=sk-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
npx tsx examples/langfuse_agent.ts

# Force a tool-loop scenario to test signal detection:
SCENARIO=tool_loop npx tsx examples/langfuse_agent.ts
```

---

## Step 14: Verify the Integration

Run your agent once, then check:

1. **Dashboard** — `http://localhost:3000` — the run appears within ~15 seconds under your `agent_id`.
2. **Runs API** — `curl http://localhost:8002/v1/agents/my-ts-agent/runs`

To confirm detectors fire, send a looping run:

```typescript
await dt.run("my-ts-agent", { model: "gpt-4o", tools: ["web_search"] }, async (run) => {
  for (let i = 0; i < 5; i++) {
    run.llmCalled("gpt-4o", 200 + i * 50);
    run.llmResponded({ finishReason: "tool_calls" });
    run.toolCalled("web_search", { query: "same query every time" });
    run.toolResponded("web_search", true, 256);
  }
  run.finalAnswer();
});
await dt.shutdown();
```

`TOOL_LOOP` should appear in the dashboard within ~15 seconds.

---

## Step 15: Tune Detectors (optional)

Edit `detectors.yml` in the repo root and restart the detector container — no code changes needed:

```yaml
default:
  tool_loop:
    threshold: 5       # raise for agents that legitimately repeat queries

my-ts-agent:           # category name must match agent_id
  context_bloat:
    growth_factor: 5.0
  cost_spike:
    static_threshold_tokens: 100000   # allow up to 100k tokens before alerting
```

```bash
docker compose restart detector
```

---

## Manual Client (no npm)

If you prefer not to use npm, copy this self-contained client into your project. The npm package is recommended — it adds background buffering, `dt.tool()`, `dt.trace()`, `getCurrentRun()`, and `dt.markDeploy()`. The manual client sends all events synchronously at run completion.

Create `src/dunetrace.ts`:

```typescript
import { createHash, randomUUID } from "node:crypto";

const ENDPOINT = process.env.DUNETRACE_ENDPOINT ?? "http://localhost:8001";
const API_KEY  = process.env.DUNETRACE_API_KEY  ?? "";

function sha256hex(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

/** SHA-256, first 16 chars. Used for all content fields (args, errors, outputs). */
export function hashContent(text: string): string {
  return sha256hex(text).slice(0, 16);
}

/**
 * Stable 8-char agent version fingerprint. Any change to system prompt, model,
 * or tool list produces a new version string, preventing false positives in
 * cross-version comparisons on the dashboard.
 */
export function agentVersion(systemPrompt: string, model: string, tools: string[]): string {
  const sorted     = [...tools].sort();
  const pythonList = "[" + sorted.map(t => `'${t}'`).join(", ") + "]";
  return sha256hex(`${systemPrompt}:${model}:${pythonList}`).slice(0, 8);
}

type EventType =
  | "run.started"      | "run.completed"     | "run.errored"
  | "llm.called"       | "llm.responded"
  | "tool.called"      | "tool.responded"
  | "retrieval.called" | "retrieval.responded"
  | "external.signal";

interface AgentEvent {
  event_type:     EventType;
  run_id:         string;
  agent_id:       string;
  agent_version:  string;
  step_index:     number;
  timestamp:      number;
  payload:        Record<string, unknown>;
  parent_run_id?: string | null;
}

export class DunetraceRun {
  readonly runId: string;
  private  step         = 0;
  private  events: AgentEvent[] = [];

  constructor(
    private readonly agentId:  string,
    private readonly version:  string,
    runId?: string,           // set to match a Langfuse trace ID
  ) {
    this.runId = runId ?? randomUUID();
  }

  // advance=true for "called" events; advance=false for "responded" events.
  private emit(type: EventType, payload: Record<string, unknown>, advance = true): void {
    if (advance) this.step++;
    this.events.push({
      event_type: type, run_id: this.runId, agent_id: this.agentId,
      agent_version: this.version, step_index: this.step,
      timestamp: Date.now() / 1000, payload,
    });
  }

  llmCalled(model: string, promptTokens = 0): void {
    this.emit("llm.called", { model, prompt_tokens: promptTokens });
  }

  llmResponded(opts: {
    completionTokens?: number; latencyMs?: number;
    finishReason?: string;     outputText?: string;
  } = {}): void {
    this.emit("llm.responded", {
      completion_tokens: opts.completionTokens ?? 0,
      latency_ms:        opts.latencyMs        ?? 0,
      finish_reason:     opts.finishReason      ?? "stop",
      output_length:     opts.outputText?.length ?? 0,
      output_hash:       hashContent(opts.outputText ?? ""),
    }, false);
  }

  toolCalled(toolName: string, args: Record<string, unknown> = {}): void {
    this.emit("tool.called", {
      tool_name: toolName,
      args_hash: hashContent(JSON.stringify(args)),
    });
  }

  toolResponded(
    toolName: string, success: boolean,
    outputLength = 0, latencyMs = 0, error?: string,
  ): void {
    const payload: Record<string, unknown> = {
      tool_name: toolName, success, output_length: outputLength, latency_ms: latencyMs,
    };
    if (error) payload["error_hash"] = hashContent(error);
    this.emit("tool.responded", payload, false);
  }

  retrievalCalled(indexName: string, query = ""): void {
    this.emit("retrieval.called", {
      index_name: indexName,
      query_hash: query ? hashContent(query) : "",
    });
  }

  retrievalResponded(
    indexName: string, resultCount: number,
    topScore?: number, latencyMs = 0,
  ): void {
    this.emit("retrieval.responded", {
      index_name: indexName, result_count: resultCount,
      top_score: topScore ?? null, latency_ms: latencyMs,
    }, false);
  }

  externalSignal(signalName: string, source = "", meta: Record<string, unknown> = {}): void {
    this.events.push({
      event_type: "external.signal", run_id: this.runId, agent_id: this.agentId,
      agent_version: this.version, step_index: this.step,
      timestamp: Date.now() / 1000,
      payload: { signal_name: signalName, ...(source ? { source } : {}), ...meta },
    });
  }

  finalAnswer(): void {
    this.emit("run.completed", {
      exit_reason: "final_answer", total_steps: this.step,
      tool_call_count: this.events.filter(e => e.event_type === "tool.called").length,
    }, false);
  }

  currentStep(): number       { return this.step; }
  getEvents():   AgentEvent[] { return this.events; }
}

export class Dunetrace {
  async run(
    agentId: string,
    opts: {
      systemPrompt?: string; model?: string; tools?: string[];
      userInput?: string;    runId?: string; parentRunId?: string;
    },
    fn: (run: DunetraceRun) => Promise<void>,
  ): Promise<void> {
    const model   = opts.model ?? "unknown";
    const tools   = opts.tools ?? [];
    const version = agentVersion(opts.systemPrompt ?? "", model, tools);
    const run     = new DunetraceRun(agentId, version, opts.runId);

    const startEvent: AgentEvent = {
      event_type: "run.started", run_id: run.runId, agent_id: agentId,
      agent_version: version, step_index: 0, timestamp: Date.now() / 1000,
      payload: { input_hash: opts.userInput ? hashContent(opts.userInput) : "", model, tools },
      parent_run_id: opts.parentRunId ?? null,
    };

    try {
      await fn(run);
    } catch (err) {
      await this._flush(agentId, [
        startEvent, ...run.getEvents(),
        {
          event_type: "run.errored", run_id: run.runId, agent_id: agentId,
          agent_version: version, step_index: run.currentStep(),
          timestamp: Date.now() / 1000,
          payload: { error_type: (err as Error).name ?? "Error", error_hash: hashContent(String(err)) },
        },
      ]);
      throw err;
    }

    await this._flush(agentId, [startEvent, ...run.getEvents()]);
  }

  private async _flush(agentId: string, events: AgentEvent[]): Promise<void> {
    try {
      await fetch(`${ENDPOINT}/v1/ingest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: API_KEY, agent_id: agentId, events }),
      });
    } catch (err) {
      console.warn("[dunetrace] Failed to flush events:", err);
    }
  }
}
```

**Langfuse with the manual client** — because `DunetraceRun` accepts a `runId` parameter, you can pre-set it to match a Langfuse trace ID:

```typescript
const sharedId = randomUUID();
const trace    = langfuse.trace({ id: sharedId, name: "my-ts-agent" });

await dt.run("my-ts-agent", {
  runId:  sharedId,    // manual client only — pre-sets run_id
  model:  "gpt-4o",
}, async (run) => {
  // run.runId === sharedId
});
```

---

## API Reference

### `new Dunetrace(opts?)`

| Option | Type | Default | Description |
|---|---|---|---|
| `endpoint` | `string \| null` | `http://localhost:8001` | Ingest service URL. Pass `null` to disable HTTP (useful with `emitAsJson`) |
| `apiKey` | `string` | `""` | Bearer token for production deployments |
| `flushIntervalMs` | `number` | `200` | Background drain interval in milliseconds |
| `emitAsJson` | `boolean` | `false` | Write Loki-compatible NDJSON to stdout |

### `dt.run(agentId, opts, fn)`

Opens a run, calls `fn(run)`, and emits `run.completed` on clean return. Emits `run.errored` on exception (then re-throws).

| Option | Type | Description |
|---|---|---|
| `model` | `string` | LLM model name — used for agent version fingerprint |
| `tools` | `string[]` | Declared tool names — used for `TOOL_AVOIDANCE` detector |
| `userInput` | `string` | User query — SHA-256 hashed before transmission |
| `systemPrompt` | `string` | System prompt — used for agent version fingerprint only |
| `parentRunId` | `string` | Link to a parent run for sub-agent tracking |

### `dt.tool(fn, name?)`

Wraps a sync or async function to auto-emit `tool.called` / `tool.responded` events. The tool name defaults to `fn.name`. No-op outside a `dt.run()` context.

### `dt.trace(fn, agentId?, opts?)`

Wraps an async function to auto-open/close a run each time it is called. The first argument is used as `userInput`. Calls `run.finalAnswer()` on clean return. `agentId` defaults to `fn.name`.

### `getCurrentRun()`

Returns the active `DunetraceRun` for the current async context (via `AsyncLocalStorage`), or `null`.

### `dt.markDeploy(agentId, version, meta?)`

Fire-and-forget deploy marker. `meta` is any JSON-serialisable object (commit hash, environment, etc.).

### `dt.shutdown(timeoutMs?)`

Stops the drain timer and flushes remaining buffered events. Default timeout: 5000ms. Always call at process exit.

### `dt.flush()`

Immediately ship all buffered events. Useful in tests or when you need a synchronisation point.

### `run` methods

| Method | When to call |
|---|---|
| `run.llmCalled(model, promptTokens?)` | Before each LLM API call |
| `run.llmResponded({ completionTokens?, latencyMs?, finishReason?, outputText? })` | After LLM responds — `outputText` is hashed, never transmitted raw |
| `run.toolCalled(toolName, args?)` | Before each tool execution — `args` is SHA-256 hashed |
| `run.toolResponded(toolName, success, outputLength?, latencyMs?, error?)` | After tool returns — `error` is SHA-256 hashed |
| `run.retrievalCalled(indexName, query?)` | Before vector search — `query` is SHA-256 hashed |
| `run.retrievalResponded(indexName, resultCount, topScore?, latencyMs?)` | After retrieval returns |
| `run.externalSignal(signalName, source?, meta?)` | Rate limits, cache misses, upstream errors — does not advance step |
| `run.finalAnswer()` | When agent produces its final output |
| `run.runId` | Read-only UUID for this run |
| `run.currentStep()` | Current step index |

---

## What Is and Isn't Captured

**Transmitted (safe metadata only):**
- Model names, token counts, latencies, finish reasons
- Tool names, success/failure, output lengths
- Retrieval index names, result counts, top scores
- Signal names and sources

**Never transmitted:**
- User input text → SHA-256 hashed before transmission
- LLM prompts and completions → SHA-256 hashed before transmission
- Tool arguments and outputs → SHA-256 hashed (16 chars); raw values never leave your process
- Error messages → SHA-256 hashed before transmission

---

## Tests

The TypeScript SDK ships a full offline test suite:

```bash
cd packages/sdk-ts
npm test
```

Tests cover: event emission, step counting, privacy (no raw content in events), error paths, `dt.tool()` wrapping (sync + async), `dt.trace()` decorator, `getCurrentRun()` context propagation, `dt.markDeploy()`, background buffering and drain, `emitAsJson` output, agent version fingerprinting, and `shutdown()` flush.

---

## Troubleshooting

**No runs appear in the dashboard**
- Verify `DUNETRACE_ENDPOINT` points to port 8001 (ingest), not 8002 (customer API)
- Confirm the backend is healthy: `curl http://localhost:8001/health`
- Check the Node console for `[dunetrace] Failed to flush events:` warnings
- Make sure `dt.shutdown()` or `dt.flush()` is called before the process exits — buffered events are lost if the process exits before the drain timer fires

**Token counts are missing from signals**
- Pass `completionTokens` and `promptTokens` when your LLM client exposes them — they are optional but improve `CONTEXT_BLOAT`, `LLM_TRUNCATION_LOOP`, `COST_SPIKE`, and `REASONING_STALL` detection accuracy

**`COST_SPIKE` / `SESSION_LATENCY` not firing**
- These detectors need a P75 baseline from ≥20 historical runs. Until then they use static fallbacks: `COST_SPIKE` fires at >50,000 total tokens; `SESSION_LATENCY` fires at >5 minutes. Tune via `detectors.yml`.

**`dt.tool()` not emitting events**
- `dt.tool()` is a no-op outside a `dt.run()` context. Make sure the wrapped function is called inside the `dt.run()` callback (or inside a function called transitively from it).

**TypeScript type error on `runId` option**
- `runId` is supported by the manual client only. With the npm package, capture `run.runId` from inside the `dt.run()` callback instead.
