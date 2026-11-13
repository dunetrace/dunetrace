# Integrating a Vercel AI SDK Agent with Dunetrace

<!--dunetrace:instrument
framework: vercel-ai
language: typescript
install: npm install dunetrace ai
primary_symbol: dunetrace.wrapGenerateText
mechanism: option-injection
opens_own_run: false
requires_run_context: true
target_file: the module that calls generateText or streamText
target_example: `app/api/chat/route.ts`, `src/agent.ts`
target_hints: ["generateText(", "streamText(", "from 'ai'"]
emits: [run.started, run.completed, run.errored, llm.called, llm.responded, tool.called, tool.responded]
verify_cmd: curl http://localhost:8002/v1/agents/my-agent/runs
-->

> **Using plain OpenAI/Anthropic clients?** See [integrate-typescript-agent.md](./integrate-typescript-agent.md). **Using LangChain?** See [integrate-langchain-agent.md](./integrate-langchain-agent.md).

## Quick Start

```bash
npm install dunetrace ai
```

```typescript
import { Dunetrace, wrapGenerateText } from "dunetrace";
import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";

const dt = new Dunetrace();   // local dev, no API key needed
const instrumentedGenerateText = wrapGenerateText(generateText);   // patch once, at startup

await dt.run("my-agent", { model: "gpt-4o" }, async (run) => {
  const result = await instrumentedGenerateText({ model: openai("gpt-4o"), prompt: "Hi" });
  run.finalAnswer();
});

await dt.shutdown();
```

Start the backend once, locally, before running this: `docker compose up -d`. Requires Node 22+ and Vercel AI SDK v7+.

## Where this goes

`wrapGenerateText` / `wrapStreamText` are applied **once at startup**,
to the imported function. The `dt.run()` goes around the call site.

Find it with:

```bash
grep -rn "generateText(\|streamText(" --include=*.ts --include=*.tsx src app
```

In a Next.js project that is almost always a route handler under `app/api/`.
Wrap the import at module scope and open the run inside the handler.

For `streamText` the run must stay open until the stream drains, because
`onStepEnd` fires as the stream is consumed. See [Next.js App
Router](#nextjs-app-router) for the shape that gets this right.

## What this does

`wrapGenerateText`/`wrapStreamText` hook into the AI SDK's step lifecycle (`onStepEnd`) — every LLM call and tool call inside a `dt.run()` context is captured automatically, including multi-step tool loops. Your own `onStepStart`/`onStepEnd`/`onEnd` callbacks are preserved and still run.

## Streaming

`wrapStreamText` works the same way — events fire as the stream is consumed:

```typescript
const instrumentedStreamText = wrapStreamText(streamText);
const result = instrumentedStreamText({ model: openai("gpt-4o"), prompt });
for await (const chunk of result.textStream) {
  process.stdout.write(chunk);
}
```

## Verification

Run your agent, then open the dashboard at `http://localhost:3000` — the run should appear within ~15 seconds. To trigger a detector signal without a real LLM call:

```bash
SCENARIO=failures python packages/sdk-py/examples/decorator_agent.py
```


### If nothing arrives

Work down this list. The first two cover almost every case.

1. **Is a run open?** This integration attaches to a run that is already open. Outside one it emits nothing, silently. Confirm your top-level call is inside `dt.run(...)` (or a decorator that opens one).

2. **Did the process flush?** Events ship from a background thread. Call
   `await dt.shutdown()` before the process exits, or the buffer dies with it.

3. **Is the backend reachable?** Both should return `{"status":"ok",...}`:

   ```bash
   curl -s localhost:8001/ready   # ingest
   curl -s localhost:8002/ready   # customer API
   ```

4. **Did anything land?** If your agent id is listed here, instrumentation is
   working and the problem is downstream:

   ```bash
   curl -s localhost:8002/v1/agents
   curl -s "localhost:8002/v1/agents/<your-agent-id>/runs?limit=5"
   ```

5. **Turn on debug logging.** `new Dunetrace({ debug: true })` logs every event as it is buffered and
   every batch as it ships.

**Runs appear but no signals?** That is usually correct, not a fault. The
detector polls every 5 seconds, and a healthy run produces no signals. Several
detectors also need cross-run baselines and stay dormant until the agent has
run history. Check `docker compose logs detector` if you expected one.

---

## Advanced (optional)

### Manual option injection

If you'd rather not wrap the imports, merge Dunetrace's callbacks into your own options:

```typescript
import { instrumentGenerateTextOptions } from "dunetrace";
const result = await generateText(instrumentGenerateTextOptions({ model: openai("gpt-4o"), prompt }));
```

### Full run wrapper

`traceGenerateText`/`traceStreamText` open and close the run for you — useful for a single call with no other run-scoped logic:

```typescript
import { traceGenerateText } from "dunetrace";
const result = await traceGenerateText(dt, "my-agent", { userInput: prompt }, generateText, {
  model: openai("gpt-4o"), prompt,
});
```

`traceStreamText` drains the stream internally before closing the run — use it when you only need the final `result.text`/`result.usage`. For incremental streaming to a client, use `wrapStreamText` inside an explicit `dt.run()` instead.

### Next.js App Router

`onStepEnd` fires only while the stream is being consumed, so the run has to
stay open until the stream drains. Returning the stream out of the `dt.run()`
callback closes the run first, and the `llm.*` / `tool.*` events then land after
`run.completed`, where the detector has usually already claimed the run and will
not look at them again.

Keep the run open across the response by holding the `dt.run()` callback until
the stream's own `onFinish` fires. `onFinish` is preserved untouched by the
instrumentation, so it is safe to use for this:

```typescript
export async function POST(req: Request) {
  const { prompt } = await req.json();

  let resolveStream: (r: ReturnType<typeof streamText>) => void;
  const streamReady = new Promise<ReturnType<typeof streamText>>((r) => {
    resolveStream = r;
  });

  // Deliberately not awaited. The run stays open until onFinish fires, which
  // happens once the client has drained the stream, so the per-step llm.* and
  // tool.* events land inside the run boundary.
  void dt
    .run("chat-api", { model: "gpt-4o", userInput: prompt }, async (run) => {
      await new Promise<void>((done) => {
        const result = instrumentedStreamText({
          model: openai("gpt-4o"),
          prompt,
          onFinish: () => {
            run.finalAnswer();
            done();
          },
        });
        resolveStream(result);
      });
    })
    .catch((err) => console.error("dunetrace run failed", err));

  return (await streamReady).toTextStreamResponse();
}
```

If the client disconnects before draining, `onFinish` may never fire and that
run stays open until the process exits. Add an `AbortSignal` or a timeout that
calls `done()` if your traffic makes that likely.

If that shape does not suit your route, `traceStreamText` drains the stream
internally and is simpler, at the cost of no longer streaming incrementally to
the client:

```typescript
const result = await traceStreamText(dt, "chat-api", { userInput: prompt }, streamText, {
  model: openai("gpt-4o"), prompt,
});
return new Response(result.text);   // already drained
```

### Troubleshooting

- **No events in the dashboard** — confirm the wrapped call happens inside `dt.run()` (or use `traceGenerateText`); call `await dt.shutdown()` before exit
- **No events from a streamed run** — `onStepEnd` only fires once the stream is consumed; make sure you read `result.textStream`
- **Type errors** — install a matching `ai` peer dependency (`npm install ai@^7`)
