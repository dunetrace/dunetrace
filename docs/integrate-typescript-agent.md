# Integrating a TypeScript Agent with Dunetrace

> **Using Python?** See [integrate-custom-python-agent.md](./integrate-custom-python-agent.md).
> **Using LangChain, CrewAI, AutoGen, or the Vercel AI SDK?** Those have dedicated guides — see [integrate-langchain-agent.md](./integrate-langchain-agent.md), [integrate-crewai-agent.md](./integrate-crewai-agent.md), [integrate-autogen-agent.md](./integrate-autogen-agent.md), [integrate-vercel-ai.md](./integrate-vercel-ai.md).

## Quick Start

```bash
npm install dunetrace
```

```typescript
import { Dunetrace } from "dunetrace";

const dt = new Dunetrace();          // local dev, no API key needed
const search = dt.tool(webSearch);   // wrap a tool function once

await dt.run("my-agent", { model: "gpt-4o" }, async (run) => {
  const results = await search("capital of France");
  run.finalAnswer();
});

await dt.shutdown();
```

Start the backend once, locally, before running this: `docker compose up -d`. Requires Node 22+.

## What this does

Wrap your agent's entry point in `dt.run(...)`. Everything called inside that callback — LLM calls, tool calls — is auto-traced and shipped to the backend in the background. Dunetrace detects structural failures (tool loops, retry storms, cost spikes, and 26 more) within ~15 seconds — no other code changes.

## Recommended usage pattern

Patch your LLM client once with `dt.wrapOpenAI()` / `dt.wrapAnthropic()`, and wrap tool functions with `dt.tool()`. Every call made inside a `dt.run()` context is then tracked automatically:

```typescript
import { Dunetrace } from "dunetrace";
import OpenAI from "openai";

const dt     = new Dunetrace();
const openai = dt.wrapOpenAI(new OpenAI());
const search = dt.tool(webSearch);

await dt.run("my-agent", { model: "gpt-4o", tools: ["web_search"] }, async (run) => {
  const response = await openai.chat.completions.create({ model: "gpt-4o", messages });
  const results  = await search(query);
  run.finalAnswer();
});

await dt.shutdown();
```

`dt.wrapAnthropic(new Anthropic())` works the same way for Anthropic. Both skip streamed calls (`stream: true`) — use `run.llmCalled()` / `run.llmResponded()` manually for those.

## Initialization (optional)

```typescript
const dt = new Dunetrace({ endpoint: "http://localhost:8001" });   // default — local dev, no key needed
```

**Production** needs an API key:

```typescript
const dt = new Dunetrace({ endpoint: "https://your-ingest", apiKey: "dt_live_..." });
```

Generate the first key directly in Postgres (there's no UI for this yet):

```sql
INSERT INTO organizations (id, name) VALUES ('my-company', 'My Company') ON CONFLICT (id) DO NOTHING;
INSERT INTO api_keys (key, org_id) VALUES ('dt_live_<random-string>', 'my-company');
```

```bash
node -e "const c=require('crypto'); console.log('dt_live_'+c.randomBytes(16).toString('hex'))"
```

Always call `await dt.shutdown()` before your process exits — this flushes any buffered events.

## Auto instrumentation

`autoInstrument()` patches the OpenAI and Anthropic SDKs once, globally. Every client instance is then tracked inside a `dt.run()` — including clients you never touch, such as ones constructed inside a library:

```typescript
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import { Dunetrace, autoInstrument } from "dunetrace";

const dt = new Dunetrace({ endpoint: "http://localhost:8001" });

// Pass the imported classes. Returns the targets it patched, e.g. ["openai", "anthropic"].
autoInstrument({ openai: OpenAI, anthropic: Anthropic });

// No wrapping at the call site — this is tracked because it runs inside dt.run().
const openai = new OpenAI();
await dt.run("my-agent", {}, async () => {
  await openai.chat.completions.create({ model: "gpt-4o", messages });
});
```

Called with no arguments it tries to `require` each SDK and patches whatever it finds:

```typescript
autoInstrument();  // CommonJS only — see below
```

**Pass the imports explicitly if you use ESM or a bundler.** The zero-argument form depends on `require`, which doesn't resolve under ESM. Passing the class works everywhere, so prefer it unless you know you're on CommonJS. Add `strict: true` to throw rather than skip silently when a requested SDK can't be found.

**How it works, and why it's shaped this way.** The Python SDK rebinds module attributes and every existing reference sees the patch. Node has no portable equivalent: under ESM an imported binding is read-only, and `require`-cache interception only covers CommonJS consumers. But ESM freezes the *binding*, not the *object* — and both SDKs define `create` on a resource-class prototype shared by every client. So `autoInstrument()` patches that prototype, which behaves identically under ESM, CommonJS, and bundlers, with no loader hook.

`autoInstrument()` is idempotent, and safe to combine with the per-client wrappers below — calling both won't double-count events.

### Streaming

Streamed calls are tracked. They can't be measured at call time — usage and the finish reason only exist once the stream is drained, and draining it to read them would consume the iterator out from under you. So `llm.called` is emitted immediately, and the stream is handed back through a pass-through proxy that observes chunks **as you pull them**, emitting `llm.responded` when it ends:

```typescript
await dt.run("my-agent", {}, async () => {
  const stream = await openai.chat.completions.create({
    model: "gpt-4o",
    messages,
    stream: true,
    stream_options: { include_usage: true },   // otherwise token counts are absent
  });
  for await (const chunk of stream) {          // llm.responded fires when this ends
    process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
  }
});
```

Notes:

- **Pass `stream_options: { include_usage: true }`** for OpenAI, or the API never sends token counts and `completion_tokens` will be absent. Anthropic reports usage in its stream events natively.
- **Breaking out early still reports** what arrived — `break` reaches the iterator as `return()`.
- **A stream you never consume emits no `llm.responded`.** There's nothing to report; `llm.called` still stands, because the call did happen.
- The proxy only intercepts iteration. Other members — OpenAI's `.tee()`, `.controller`, `.toReadableStream()` — pass straight through.

### HTTP

`autoInstrument()` also patches the global `fetch`, emitting `tool.called` / `tool.responded` per request. This is the Node counterpart to the Python SDK's `httpx` and `requests` patches; `fetch` is what nearly all Node HTTP goes through.

Requests are named by hostname (`api.example.com`), which keeps full URLs out of the event while staying stable enough to group on. Two categories are deliberately excluded:

- **Requests made by an instrumented LLM SDK.** Both vendors call `fetch` internally, so without this a single `chat.completions.create()` would register as an LLM call *and* a tool call to `api.openai.com` — inflating tool counts and tripping tool-loop detection on one ordinary call.
- **Dunetrace's own ingest traffic.** Instrumenting the batch POST would emit an event describing the POST, which buffers another event, which POSTs.

Disable it while keeping the LLM patches with `targets`:

```typescript
autoInstrument({ openai: OpenAI, targets: ["openai"] });   // no HTTP instrumentation
```

**Not covered:**

| | |
|---|---|
| Errors | A failed call propagates unchanged and emits no `llm.responded` — there's no usage or output to describe. HTTP failures *are* recorded, as `tool.responded` with `success: false`. |
| LangChain.js | Needs a callback-handler integration rather than a patch — LangChain.js has no global handler registry equivalent to the Python `register_configure_hook` the Python SDK relies on. Use `dt.tool()` / `dt.trace()`, or the [Vercel AI SDK integration](#vercel-ai-sdk-integration). |
| CrewAI | Python-only — there is no JavaScript port to instrument. |

A bug in event emission can never fail your LLM call: emission is wrapped so it warns and continues rather than throwing into your call path.

### Per-client wrapping

To instrument one client instead of the whole SDK, the wrappers shown above (`dt.wrapOpenAI`, `dt.wrapAnthropic`, `dt.tool`, `dt.trace`) remain available. Each is a no-op outside a `dt.run()` context, so it's safe to wrap once at startup and call normally everywhere else.

`dt.trace()` wraps an entire agent function so it opens and closes its own run automatically:

```typescript
async function myAgent(query: string): Promise<string> {
  const response = await openai.chat.completions.create({ /* ... */ });
  return response.choices[0].message.content ?? "";
}

const agent = dt.trace(myAgent, "my-agent", { model: "gpt-4o" });
const answer = await agent("What is the capital of France?");
```

## Verification

Run your agent once, then check:

1. **Dashboard** — `http://localhost:3000` — the run appears within ~15 seconds
2. **Runs API** — `curl http://localhost:8002/v1/agents/my-agent/runs`

To confirm detectors fire, send a looping run:

```typescript
await dt.run("my-agent", { model: "gpt-4o", tools: ["web_search"] }, async (run) => {
  for (let i = 0; i < 5; i++) {
    run.llmCalled("gpt-4o");
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

## Advanced (optional)

### Manual tracking (full control)

```typescript
await dt.run("my-agent", { model: "gpt-4o" }, async (run) => {
  run.llmCalled("gpt-4o", estimatedPromptTokens);
  const response = await openai.chat.completions.create({ model: "gpt-4o", messages });
  run.llmResponded({
    completionTokens: response.usage?.completion_tokens,
    finishReason:     response.choices[0].finish_reason ?? "stop",
    outputText:       response.choices[0].message.content ?? "",
  });
  run.finalAnswer();
});
```

Full `run` API: `llmCalled` / `llmResponded`, `toolCalled` / `toolResponded`, `retrievalCalled` / `retrievalResponded`, `externalSignal`, `memoryWritten` / `memoryRead` / `memoryCleared` (the agent memory channel — feeds `MEMORY_POISONING`), `finalAnswer`, `run.llm(model, promise)` (auto-extracts tokens from an OpenAI/Anthropic response).

### `getCurrentRun()`

Access the active run from any function without threading it through your call stack (works via `AsyncLocalStorage`):

```typescript
import { getCurrentRun } from "dunetrace";

async function dbQuery(sql: string) {
  const run = getCurrentRun();
  if (run) run.toolCalled("db_query", { sql });
  const result = await db.query(sql);
  if (run) run.toolResponded("db_query", true, result.length);
  return result;
}
```

### RAG / retrieval agents

```typescript
run.retrievalCalled("product-docs", query);
const docs = await vectorStore.search(query, { topK: 5 });
run.retrievalResponded("product-docs", docs.length, docs[0]?.score);
```

`RAG_EMPTY_RETRIEVAL` fires when `resultCount` is 0 or `topScore` is below 0.3 but the agent still answers.

### Sub-agent tracking

A run opened inside another run **auto-inherits** the active run's id as its
`parent_run_id` — no manual threading. Both appear in the dashboard with a
hierarchy indicator, and the parent/child graph feeds the `DELEGATION_LOOP` and
`HANDOFF_CONTEXT_LOSS` detectors:

```typescript
await dt.run("orchestrator", { model: "gpt-4o" }, async (parentRun) => {
  // Nested run: parent_run_id is set to orchestrator's id automatically.
  await dt.run("sub-agent", { model: "gpt-4o-mini" }, async (childRun) => {
    childRun.finalAnswer();
  });
  parentRun.finalAnswer();
});
```

Threading rides `AsyncLocalStorage`, so it survives `await`s within the same
async context. Pass `parentRunId` explicitly if you ever need to link across a
boundary the async context can't cross (it always wins over the inherited id).

### Deploy markers

```typescript
dt.markDeploy("my-agent", "v1.4.2", { commit: "abc123f", environment: "production" });
```

### Grafana / Loki (no HTTP ingest)

```typescript
const dt = new Dunetrace({ emitAsJson: true, endpoint: null });
```

### Tuning detectors

Edit `detectors.yml` on the server, then `docker compose restart detector`:

```yaml
default:
  tool_loop:
    threshold: 5
my-agent:              # category name must match agent_id
  context_bloat:
    growth_factor: 5.0
```

### Manual client (no npm package)

If you can't add the `dunetrace` npm dependency, a self-contained client can be copied directly into your project — it sends events synchronously at run completion instead of background-buffering. Ask in the repo's discussions for the current reference implementation, or read `packages/sdk-ts/src/client.ts` and port the parts you need.

### Data handling

User input, tool arguments, retrieved content, and completions are sent to the backend over TLS as-is — content-aware detectors need to see what the agent actually said and did. Self-host for an air-gapped deployment. Full detector list: [docs/detectors.md](detectors.md).

### Tests

```bash
cd packages/sdk-ts && npm test
```
