# Integrating a TypeScript Agent with Dunetrace

This guide covers adding Dunetrace monitoring to a TypeScript or JavaScript agent using the `dunetrace` npm package.

---

## How It Works

The SDK buffers events and POSTs them to the Dunetrace ingest service at run completion. The same detectors, dashboard, and alerts that work for Python agents apply here.

```
your TS agent  →  POST /v1/ingest  →  detector  →  dashboard + Slack alerts
```

---

## Prerequisites

- Dunetrace backend running (`docker compose up -d`)
- Node 18+ (built-in `fetch` and `AsyncLocalStorage` required)

> **Local dev — no API key needed.** The backend accepts requests without any key when running locally. API keys are only required for production — see [Step 1](#step-1-generate-an-api-key-production-only).

---

## Step 1: Generate an API Key (production only)

Skip this step when testing locally.

Connect to your Dunetrace Postgres instance and run:

```sql
INSERT INTO api_keys (key, agent_id, customer_id)
VALUES ('dt_live_<your-random-string>', 'my-ts-agent', 'my-company');
```

Generate a secure key:

```bash
node -e "const c=require('crypto'); console.log('dt_live_'+c.randomBytes(16).toString('hex'))"
```

---

## Step 2: Set Environment Variables

```bash
# .env or deployment config
DUNETRACE_ENDPOINT=http://localhost:8001   # ingest service URL
DUNETRACE_API_KEY=                         # empty for local dev
```

---

## Step 3: Install the SDK

```bash
npm install dunetrace
```

The SDK has zero runtime dependencies and works with any Node 18+ runtime.

---

## Step 3b: Manual client (copy-paste, no npm)

If you prefer not to use npm, you can copy the self-contained client below into your project. The npm package is the recommended path — it includes background buffering, `dt.tool()` auto-wrapping, and `getCurrentRun()` access.

Create `src/dunetrace.ts` in your project:

```typescript
import { createHash, randomUUID } from "node:crypto";

const ENDPOINT = process.env.DUNETRACE_ENDPOINT ?? "http://localhost:8001";
const API_KEY  = process.env.DUNETRACE_API_KEY  ?? "";

// ── Hashing — must match the Python SDK exactly ───────────────────────────────

function sha256hex(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

/** SHA-256, first 16 chars. Used for all content fields (args, errors, outputs). */
export function hashContent(text: string): string {
  return sha256hex(text).slice(0, 16);
}

/**
 * Stable 8-char fingerprint of your agent config. Any change to system prompt,
 * model, or tool list produces a new version — preventing deploy-induced false
 * positives in cross-version comparisons.
 *
 * Replicates Python: sha256(f"{systemPrompt}:{model}:{sorted(tools)}").hex()[:8]
 */
export function agentVersion(systemPrompt: string, model: string, tools: string[]): string {
  const sorted     = [...tools].sort();
  const pythonList = "[" + sorted.map(t => `'${t}'`).join(", ") + "]";
  return sha256hex(`${systemPrompt}:${model}:${pythonList}`).slice(0, 8);
}

// ── Types ─────────────────────────────────────────────────────────────────────

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
  timestamp:      number;              // Unix seconds (float)
  payload:        Record<string, unknown>;
  parent_run_id?: string | null;
}

// ── RunContext ────────────────────────────────────────────────────────────────

export class DunetraceRun {
  readonly runId: string;
  private  step         = 0;
  private  events: AgentEvent[] = [];

  constructor(
    private readonly agentId:  string,
    private readonly version:  string,
    runId?: string,
  ) {
    this.runId = runId ?? randomUUID();
  }

  // advance=true for "called" events, advance=false for "responded" events.
  // Mirrors the Python SDK: step increments when an action starts, not when it ends.
  private emit(type: EventType, payload: Record<string, unknown>, advance = true): void {
    if (advance) this.step++;
    this.events.push({
      event_type:    type,
      run_id:        this.runId,
      agent_id:      this.agentId,
      agent_version: this.version,
      step_index:    this.step,
      timestamp:     Date.now() / 1000,
      payload,
    });
  }

  /** Call before each LLM API call. */
  llmCalled(model: string, promptTokens = 0): void {
    this.emit("llm.called", { model, prompt_tokens: promptTokens });
  }

  /** Call after the LLM responds. */
  llmResponded(opts: {
    completionTokens?: number;
    latencyMs?:        number;
    finishReason?:     string;
    outputLength?:     number;
    outputText?:       string;   // hashed before transmission — never sent raw
  }): void {
    this.emit("llm.responded", {
      completion_tokens: opts.completionTokens ?? 0,
      latency_ms:        opts.latencyMs        ?? 0,
      finish_reason:     opts.finishReason      ?? "stop",
      output_length:     opts.outputLength      ?? (opts.outputText?.length ?? 0),
      output_hash:       hashContent(opts.outputText ?? ""),
    }, false);   // ← does not advance step
  }

  /** Call before each tool execution. args are SHA-256 hashed — never transmitted raw. */
  toolCalled(toolName: string, args: Record<string, unknown> = {}): void {
    this.emit("tool.called", {
      tool_name: toolName,
      args_hash: hashContent(JSON.stringify(args)),
    });
  }

  /** Call after the tool returns. */
  toolResponded(
    toolName:     string,
    success:      boolean,
    outputLength  = 0,
    latencyMs     = 0,
    error?:       string,
  ): void {
    const payload: Record<string, unknown> = {
      tool_name:     toolName,
      success,
      output_length: outputLength,
      latency_ms:    latencyMs,
    };
    if (error) payload["error_hash"] = hashContent(error);
    this.emit("tool.responded", payload, false);   // ← does not advance step
  }

  /** Call before a vector search / retrieval. */
  retrievalCalled(indexName: string, query = ""): void {
    this.emit("retrieval.called", {
      index_name: indexName,
      query_hash: query ? hashContent(query) : "",
    });
  }

  /** Call after retrieval returns. */
  retrievalResponded(
    indexName:    string,
    resultCount:  number,
    topScore?:    number,
    latencyMs     = 0,
  ): void {
    this.emit("retrieval.responded", {
      index_name:   indexName,
      result_count: resultCount,
      top_score:    topScore ?? null,
      latency_ms:   latencyMs,
    }, false);   // ← does not advance step
  }

  /**
   * Emit an infrastructure context event without advancing the step counter.
   * Use for rate limits, cache misses, upstream errors, etc.
   */
  externalSignal(signalName: string, source = "", meta: Record<string, unknown> = {}): void {
    // step_index stays at current step — this event annotates the current action
    this.events.push({
      event_type:    "external.signal",
      run_id:        this.runId,
      agent_id:      this.agentId,
      agent_version: this.version,
      step_index:    this.step,
      timestamp:     Date.now() / 1000,
      payload:       { signal_name: signalName, ...(source ? { source } : {}), ...meta },
    });
  }

  /** Call when the agent produces its final output. */
  finalAnswer(): void {
    this.emit("run.completed", {
      exit_reason:     "final_answer",
      total_steps:     this.step,
      tool_call_count: this.events.filter(e => e.event_type === "tool.called").length,
    }, false);   // ← does not advance step
  }

  currentStep(): number      { return this.step; }
  getEvents():   AgentEvent[] { return this.events; }
}

// ── Client ────────────────────────────────────────────────────────────────────

export class Dunetrace {
  /**
   * Wrap one agent invocation. Emits run.started on entry, run.completed on
   * clean exit, and run.errored if an exception escapes.
   */
  async run(
    agentId:      string,
    opts: {
      systemPrompt?: string;
      model?:        string;
      tools?:        string[];
      userInput?:    string;
      runId?:        string;   // pre-set to match a Langfuse trace ID
    },
    fn: (run: DunetraceRun) => Promise<void>,
  ): Promise<void> {
    const model   = opts.model        ?? "unknown";
    const tools   = opts.tools        ?? [];
    const version = agentVersion(opts.systemPrompt ?? "", model, tools);
    const run     = new DunetraceRun(agentId, version, opts.runId);

    const startEvent: AgentEvent = {
      event_type:    "run.started",
      run_id:        run.runId,
      agent_id:      agentId,
      agent_version: version,
      step_index:    0,
      timestamp:     Date.now() / 1000,
      payload: {
        input_hash: opts.userInput ? hashContent(opts.userInput) : "",
        model,
        tools,
      },
    };

    try {
      await fn(run);
    } catch (err) {
      await this._flush(agentId, [
        startEvent,
        ...run.getEvents(),
        {
          event_type:    "run.errored",
          run_id:        run.runId,
          agent_id:      agentId,
          agent_version: version,
          step_index:    run.currentStep(),
          timestamp:     Date.now() / 1000,
          payload: {
            error_type: (err as Error).name ?? "Error",
            error_hash: hashContent(String(err)),
          },
        },
      ]);
      throw err;
    }

    await this._flush(agentId, [startEvent, ...run.getEvents()]);
  }

  private async _flush(agentId: string, events: AgentEvent[]): Promise<void> {
    try {
      await fetch(`${ENDPOINT}/v1/ingest`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ api_key: API_KEY, agent_id: agentId, events }),
      });
    } catch (err) {
      console.warn("[dunetrace] Failed to flush events:", err);
    }
  }
}
```

---

## Step 4: Instrument Your Agent

### Basic agent

```typescript
import { Dunetrace } from "dunetrace";  // npm package

const dt = new Dunetrace();

async function runAgent(query: string): Promise<string> {
  let answer = "";

  await dt.run("my-ts-agent", {
    systemPrompt: "You are a helpful research assistant.",
    model:        "gpt-4o",
    tools:        ["web_search"],
    userInput:    query,           // hashed before transmission — never sent raw
  }, async (run) => {
    // Before LLM call
    run.llmCalled("gpt-4o", 150);
    const t0 = Date.now();

    const response = await openai.chat.completions.create({
      model:    "gpt-4o",
      messages: [{ role: "user", content: query }],
    });

    const output = response.choices[0].message.content ?? "";

    // After LLM responds
    run.llmResponded({
      completionTokens: response.usage?.completion_tokens,
      latencyMs:        Date.now() - t0,
      finishReason:     response.choices[0].finish_reason ?? "stop",
      outputText:       output,    // hashed before transmission
    });

    // Tool call
    run.toolCalled("web_search", { query });
    const t1      = Date.now();
    const results = await webSearch(query);
    run.toolResponded("web_search", true, results.length, Date.now() - t1);

    // Second LLM call with search results
    run.llmCalled("gpt-4o", 400);
    const t2   = Date.now();
    const res2 = await openai.chat.completions.create({ /* ... */ } as any);
    const out2 = res2.choices[0].message.content ?? "";
    run.llmResponded({ completionTokens: res2.usage?.completion_tokens, latencyMs: Date.now() - t2, finishReason: "stop", outputText: out2 });

    run.finalAnswer();
    answer = out2;
  });

  return answer;
}
```

### RAG agent

```typescript
await dt.run("rag-agent", { model: "gpt-4o" }, async (run) => {
  run.llmCalled("gpt-4o", 200);
  run.llmResponded({ finishReason: "tool_calls", outputText: "I will search the docs." });

  run.retrievalCalled("product-docs", query);   // query is hashed inside
  const t0   = Date.now();
  const docs = await vectorStore.search(query);
  run.retrievalResponded("product-docs", docs.length, docs[0]?.score, Date.now() - t0);

  run.llmCalled("gpt-4o", 600);
  run.llmResponded({ finishReason: "stop", completionTokens: 120, outputText: answer });
  run.finalAnswer();
});
```

### Infrastructure signals

```typescript
await dt.run("my-ts-agent", { model: "gpt-4o" }, async (run) => {
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

---

## Step 5: Verify the Integration

Run your agent once, then check:

1. **Dashboard** (`http://localhost:3000`) — the run should appear within 15 seconds under `my-ts-agent`
2. **Runs API** — `GET http://localhost:8002/v1/agents/my-ts-agent/runs`

To confirm detectors fire, send a run that loops a tool:

```typescript
await dt.run("my-ts-agent", { model: "gpt-4o", tools: ["web_search"] }, async (run) => {
  for (let i = 0; i < 5; i++) {
    run.llmCalled("gpt-4o", 200 + i * 50);
    run.llmResponded({ finishReason: "tool_calls", outputText: "search again" });
    run.toolCalled("web_search", { query: "same query every time" });
    run.toolResponded("web_search", true, 256);
  }
  run.finalAnswer();
});
```

This triggers `TOOL_LOOP` (same tool ≥3 times in a 5-call window). The signal should appear in the dashboard within ~15 seconds.

---

## Langfuse integration

Correlate a Dunetrace run with a Langfuse trace using a shared UUID. The `runId` option pre-sets the Dunetrace `run_id` to the same value you use as the Langfuse trace ID — so detected signals link directly to the full trace.

```typescript
import { randomUUID } from "node:crypto";
import { Langfuse } from "langfuse";

const langfuse = new Langfuse({
  publicKey: process.env.LANGFUSE_PUBLIC_KEY!,
  secretKey: process.env.LANGFUSE_SECRET_KEY!,
});

const sharedId = randomUUID();

// Open a Langfuse trace with the shared ID first
const trace = langfuse.trace({ id: sharedId, name: "my-ts-agent" });

await dt.run("my-ts-agent", {
  runId:  sharedId,    // ← links Dunetrace run to Langfuse trace
  model:  "gpt-4o",
  tools:  ["web_search"],
}, async (run) => {
  // instrument as normal …
  run.finalAnswer();
});

await langfuse.flushAsync();
// Langfuse trace URL: https://cloud.langfuse.com/trace/<sharedId>
// Dunetrace signals will carry run_id = sharedId
```

For a complete working example including the explain/autofix flow, see [`packages/sdk-ts/examples/langfuse_agent.ts`](../packages/sdk-ts/examples/langfuse_agent.ts).

---

## RunContext API Reference

`dt.run()` accepts an options object:

| Option | Type | Description |
|---|---|---|
| `model` | `string` | LLM model name — used for agent version fingerprint |
| `tools` | `string[]` | Declared tool names — used for agent version fingerprint |
| `userInput` | `string` | User query — SHA-256 hashed before transmission |
| `systemPrompt` | `string` | System prompt — used for agent version fingerprint only |
| `runId` | `string` | Pre-set the run UUID (e.g. to match a Langfuse trace ID) |
| `parentRunId` | `string` | Link to a parent run for sub-agent tracking |

`run` methods inside the callback:

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

---

## What Is and Isn't Captured

**Transmitted (safe metadata only):**
- Model names, token counts, latencies, finish reasons
- Tool names, success/failure, output lengths
- Retrieval index names, result counts, top scores
- Signal names and sources

**Never transmitted (privacy):**
- User input text
- LLM prompts and completions
- Tool arguments and outputs — SHA-256 hashed (16 chars) before transmission; raw values never leave your process
- Error messages — SHA-256 hashed before transmission

---

## Troubleshooting

**No runs appear in the dashboard**
- Check `DUNETRACE_ENDPOINT` points to the ingest service (port 8001, not 8002)
- Confirm the backend is healthy: `curl http://localhost:8001/health`
- Check the Node console for `[dunetrace] Failed to flush events` warnings

**Token counts are missing**
- Pass `completionTokens` and `promptTokens` if your LLM client exposes them — they are optional but improve `CONTEXT_BLOAT` and `LLM_TRUNCATION_LOOP` detection accuracy

**Detectors fire too aggressively**
- Tune thresholds in `detectors.yml` on the server — see the [Python integration guide](./integrate-custom-python-agent.md#step-6-tune-detectors-optional) for the format
