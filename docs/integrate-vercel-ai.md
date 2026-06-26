# Integrating a Vercel AI SDK Agent with Dunetrace

This guide covers adding Dunetrace monitoring to a TypeScript agent built with the [Vercel AI SDK](https://ai-sdk.dev/) (`ai` package) — common in Next.js App Router and edge-deployed agents.

> **Using plain OpenAI/Anthropic clients?** See [integrate-typescript-agent.md](./integrate-typescript-agent.md).
> **Using LangChain?** See [integrate-langchain-agent.md](./integrate-langchain-agent.md).

---

## How It Works

The integration hooks into Vercel AI SDK lifecycle callbacks and translates them into Dunetrace events:

| Vercel AI SDK callback | Dunetrace event |
| --- | --- |
| `onStepFinish` (each LLM step) | `llm.called` + `llm.responded` |
| `toolCalls` in step result | `tool.called` |
| `toolResults` in step result | `tool.responded` |
| `traceGenerateText` / `traceStreamText` wrapper | `run.started` + `run.completed` |

Both `generateText` and `streamText` fire `onStepFinish` once per step, so a single hook captures every LLM call and tool invocation. Your own `onStepFinish` is preserved and called after Dunetrace's; your `onFinish` is left untouched.

No changes to your tool definitions or model providers are required.

---

## Prerequisites

- Dunetrace backend running (`docker compose up -d`)
- Node 18+
- Vercel AI SDK (`npm install ai`)

> **Local dev — no API key needed.** The backend accepts requests without authentication when running in dev mode.

---

## Install

```bash
npm install dunetrace ai
```

The Vercel AI SDK (`ai`) is an optional peer dependency — install it when using this integration.

---

## Option A: Wrap inside an existing `dt.run()` (recommended)

Patch `generateText` / `streamText` once at startup. Events are emitted automatically whenever you call them inside a Dunetrace run context.

```typescript
import { Dunetrace, wrapGenerateText, wrapStreamText } from "dunetrace";
import { generateText, streamText, stepCountIs } from "ai";
import { openai } from "@ai-sdk/openai";

const dt = new Dunetrace();

// Patch once — no dependency on `ai` is added to dunetrace itself
const instrumentedGenerateText = wrapGenerateText(generateText);
const instrumentedStreamText   = wrapStreamText(streamText);

export async function runAgent(prompt: string) {
  await dt.run("my-vercel-agent", {
    model: "gpt-4o",
    tools: ["weather"],
    userInput: prompt,
  }, async (run) => {
    const result = await instrumentedGenerateText({
      model: openai("gpt-4o"),
      prompt,
      tools: { weather: weatherTool },
      stopWhen: stepCountIs(5),   // AI SDK v5: enable multi-step tool loops
    });
    // ↑ each LLM step + tool call is tracked automatically

    run.finalAnswer();
    return result.text;
  });

  await dt.shutdown();
}
```

### Streaming

`wrapStreamText` instruments per-step callbacks during streaming. `streamText` returns synchronously (you consume the stream afterwards), so the wrapper does too — use it the same way:

```typescript
const result = instrumentedStreamText({
  model: openai("gpt-4o"),
  prompt,
  tools: { weather: weatherTool },
});

for await (const chunk of result.textStream) {
  process.stdout.write(chunk);
}
// ↑ llm.* / tool.* events fire as the stream is consumed
```

---

## Option B: Manual option injection

If you prefer not to wrap the imports, pass your options through `instrumentGenerateTextOptions` / `instrumentStreamTextOptions`:

```typescript
import { Dunetrace, instrumentGenerateTextOptions } from "dunetrace";
import { generateText } from "ai";

const dt = new Dunetrace();

await dt.run("my-vercel-agent", { userInput: prompt, model: "gpt-4o" }, async (run) => {
  const result = await generateText(instrumentGenerateTextOptions({
    model: openai("gpt-4o"),
    prompt,
    tools: { weather: weatherTool },
  }));

  run.finalAnswer();
  return result.text;
});
```

Your existing `onStepFinish` and `onFinish` callbacks are preserved — Dunetrace chains its handlers before yours.

---

## Option C: Full run wrapper

Use `traceGenerateText` or `traceStreamText` when you want Dunetrace to open and close the run boundary for you:

```typescript
import { Dunetrace, traceGenerateText } from "dunetrace";
import { generateText } from "ai";

const dt = new Dunetrace();

const result = await traceGenerateText(
  dt,
  "my-vercel-agent",
  { userInput: "What's the weather in Paris?", tools: ["weather"] },
  generateText,
  {
    model: openai("gpt-4o"),
    prompt: "What's the weather in Paris?",
    tools: { weather: weatherTool },
  },
);

await dt.shutdown();
console.log(result.text);
```

> **Streaming with `traceStreamText`:** because `onStepFinish` only fires while a stream is consumed, `traceStreamText` drains the stream internally before closing the run so events stay inside the run boundary. The returned result is therefore already consumed — use it when you just need `result.text` / `result.usage`. For incremental streaming to a client, use `wrapStreamText` (Option A) inside an explicit `dt.run()` and call `run.finalAnswer()` yourself.

---

## Next.js App Router example

```typescript
// app/api/chat/route.ts
import { Dunetrace, wrapStreamText } from "dunetrace";
import { streamText } from "ai";
import { openai } from "@ai-sdk/openai";

const dt = new Dunetrace();
const instrumentedStreamText = wrapStreamText(streamText);

export async function POST(req: Request) {
  const { prompt } = await req.json();

  const result = await dt.run("chat-api", {
    model: "gpt-4o",
    userInput: prompt,
  }, async () => {
    return instrumentedStreamText({
      model: openai("gpt-4o"),
      prompt,
    });
  });

  return result.toTextStreamResponse();
}
```

> Call `dt.shutdown()` on process exit in long-running servers, or rely on the background flush loop for short-lived serverless invocations.

---

## Verify

1. Start the backend: `docker compose up -d`
2. Run your agent
3. Open the dashboard: [http://localhost:3000](http://localhost:3000)

Runs should appear within ~15 seconds. To trigger a detector signal without a real LLM, use the built-in Python examples:

```bash
SCENARIO=failures python packages/sdk-py/examples/decorator_agent.py
```

---

## API reference

| Export | Description |
|---|---|
| `wrapGenerateText(fn)` | Returns a wrapped `generateText` that auto-instruments inside `dt.run()` |
| `wrapStreamText(fn)` | Returns a wrapped `streamText` that auto-instruments inside `dt.run()` |
| `instrumentGenerateTextOptions(opts)` | Merge Dunetrace callbacks into generateText options |
| `instrumentStreamTextOptions(opts)` | Merge Dunetrace callbacks into streamText options |
| `traceGenerateText(dt, agentId, runOpts, fn, textOpts)` | Open run → generateText → close run |
| `traceStreamText(dt, agentId, runOpts, fn, textOpts)` | Open run → streamText → drain stream → close run (returns a consumed result) |
| `modelId(model)` | Extract model name from AI SDK provider objects |
| `toolNames(tools)` | Extract tool name list from a tools record |

---

## Troubleshooting

**No events in the dashboard**

- Confirm `DUNETRACE_ENDPOINT` points at the ingest service (`http://localhost:8001`, not `:8002`)
- Ensure `wrapGenerateText` / `instrumentGenerateTextOptions` is called inside `dt.run()`, or use `traceGenerateText`
- Call `await dt.shutdown()` before process exit to flush buffered events

**No events from a streamed run**

- `onStepFinish` only fires once the stream is consumed. Make sure you read `result.textStream` (or return a `toTextStreamResponse()`); an abandoned stream emits nothing.

**Type errors with AI SDK versions**

- Install a matching `ai` peer dependency (`npm install ai@^5`). Types are taken directly from the Vercel AI SDK.
