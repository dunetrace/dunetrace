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

Wrap your agent's entry point in `dt.run(...)`. Everything called inside that callback — LLM calls, tool calls — is auto-traced and shipped to the backend in the background. Dunetrace detects behavioral failures (tool loops, retry storms, cost spikes, and 20 more) within ~15 seconds — no other code changes.

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

There's no global auto-patch in the TS SDK — instrumentation is opt-in per client, via the wrappers shown above (`dt.wrapOpenAI`, `dt.wrapAnthropic`, `dt.tool`, `dt.trace`). Each is a no-op outside a `dt.run()` context, so it's safe to wrap once at startup and call normally everywhere else.

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
