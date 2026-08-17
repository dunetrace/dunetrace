# Integrating a Vercel AI SDK Agent with Dunetrace

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

```typescript
export async function POST(req: Request) {
  const { prompt } = await req.json();
  const result = await dt.run("chat-api", { model: "gpt-4o", userInput: prompt }, async () =>
    instrumentedStreamText({ model: openai("gpt-4o"), prompt })
  );
  return result.toTextStreamResponse();
}
```

### Troubleshooting

- **No events in the dashboard** — confirm the wrapped call happens inside `dt.run()` (or use `traceGenerateText`); call `await dt.shutdown()` before exit
- **No events from a streamed run** — `onStepEnd` only fires once the stream is consumed; make sure you read `result.textStream`
- **Type errors** — install a matching `ai` peer dependency (`npm install ai@^7`)
