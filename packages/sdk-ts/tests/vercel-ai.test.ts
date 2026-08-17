import { describe, it, expect, vi } from "vitest";
import type { StepResult, ToolSet, LanguageModelUsage } from "ai";
import { Dunetrace } from "../src/client.js";
import {
  instrumentGenerateTextOptions,
  instrumentStreamTextOptions,
  wrapGenerateText,
  wrapStreamText,
  traceGenerateText,
  traceStreamText,
  modelId,
  toolNames,
  type GenerateTextOptions,
  type StreamTextOptions,
} from "../src/integrations/vercel-ai.js";
import type { AgentEvent } from "../src/models.js";

function captureEvents(dt: Dunetrace): AgentEvent[] {
  const events: AgentEvent[] = [];
  dt._emit = (e) => events.push(e);
  return events;
}

const FAKE_MODEL = { modelId: "gpt-4o" };

/** Build generateText options that satisfy the SDK's prompt/messages requirement. */
function genOpts(extra: Partial<GenerateTextOptions> = {}): GenerateTextOptions {
  return { model: FAKE_MODEL as never, prompt: "test prompt", ...extra } as GenerateTextOptions;
}

/** Build streamText options that satisfy the SDK's prompt/messages requirement. */
function streamOpts(extra: Partial<StreamTextOptions> = {}): StreamTextOptions {
  return { model: FAKE_MODEL as never, prompt: "test prompt", ...extra } as StreamTextOptions;
}

function makeUsage(usage: Partial<LanguageModelUsage> = {}): LanguageModelUsage {
  return {
    inputTokens:        usage.inputTokens,
    outputTokens:       usage.outputTokens,
    totalTokens:        usage.totalTokens,
    inputTokenDetails:  usage.inputTokenDetails ?? {
      noCacheTokens:    undefined,
      cacheReadTokens:  undefined,
      cacheWriteTokens: undefined,
    },
    outputTokenDetails: usage.outputTokenDetails ?? {
      textTokens:       undefined,
      reasoningTokens:  undefined,
    },
  };
}

// StepResult carries runtime-only fields (callId, performance, toolsContext, …)
// that this integration never reads. The fixture sets the fields the code under
// test inspects, validates them against the SDK type, and casts the rest.
function makeStep(overrides: Partial<StepResult<ToolSet>> = {}): StepResult<ToolSet> {
  const step: Partial<StepResult<ToolSet>> = {
    content:           [],
    text:              "",
    reasoning:         [],
    reasoningText:     undefined,
    files:             [],
    sources:           [],
    toolCalls:         [],
    staticToolCalls:   [],
    dynamicToolCalls:  [],
    toolResults:       [],
    staticToolResults: [],
    dynamicToolResults: [],
    finishReason:      "stop",
    usage:             makeUsage(),
    warnings:          undefined,
    request:           {},
    response:          { id: "resp-1", timestamp: new Date(0), modelId: "gpt-4o", messages: [] },
    providerMetadata:  undefined,
    ...overrides,
  };
  return step as unknown as StepResult<ToolSet>;
}

describe("vercel-ai helpers", () => {
  it("modelId extracts modelId from provider objects", () => {
    expect(modelId(FAKE_MODEL as never)).toBe("gpt-4o");
    expect(modelId(undefined)).toBe("unknown");
  });

  it("toolNames extracts keys from a tools record", () => {
    expect(toolNames({ weather: {} as never, search: {} as never })).toEqual(["weather", "search"]);
    expect(toolNames(undefined)).toEqual([]);
  });
});

describe("instrumentGenerateTextOptions()", () => {
  it("is a no-op outside a run context", async () => {
    const userStep = vi.fn();
    const opts = instrumentGenerateTextOptions(genOpts({ onStepEnd: userStep }));
    await opts.onStepEnd?.(makeStep({ text: "hi", usage: makeUsage({ inputTokens: 10, outputTokens: 2, totalTokens: 12 }) }));
    expect(userStep).toHaveBeenCalledOnce();
  });

  it("emits llm.called + llm.responded on onStepEnd", async () => {
    const dt     = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events = captureEvents(dt);

    await dt.run("vercel-agent", { model: "gpt-4o" }, async () => {
      const opts = instrumentGenerateTextOptions(genOpts());
      await opts.onStepEnd?.(makeStep({
        text: "Paris",
        finishReason: "stop",
        usage: makeUsage({ inputTokens: 60, outputTokens: 5, totalTokens: 65 }),
      }));
    });

    const llmCalled    = events.find(e => e.event_type === "llm.called");
    const llmResponded = events.find(e => e.event_type === "llm.responded");

    expect(llmCalled).toBeDefined();
    expect(llmResponded).toBeDefined();
    expect(llmCalled!.payload["model"]).toBe("gpt-4o");
    expect(llmCalled!.payload["prompt_tokens"]).toBe(60);
    expect(llmResponded!.payload["completion_tokens"]).toBe(5);
    expect(llmResponded!.payload["finish_reason"]).toBe("stop");

    await dt.shutdown();
  });

  it("emits tool.called + tool.responded for step toolCalls/toolResults", async () => {
    const dt     = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events = captureEvents(dt);

    await dt.run("vercel-agent", { model: "gpt-4o", tools: ["weather"] }, async () => {
      const opts = instrumentGenerateTextOptions(genOpts());
      await opts.onStepEnd?.(makeStep({
        finishReason: "tool-calls",
        usage: makeUsage({ inputTokens: 40, outputTokens: 10, totalTokens: 50 }),
        toolCalls: [{
          type: "tool-call",
          toolCallId: "tc-1",
          toolName: "weather",
          input: { city: "Paris" },
        }],
        toolResults: [{
          type: "tool-result",
          toolCallId: "tc-1",
          toolName: "weather",
          input: { city: "Paris" },
          output: { temp: 18 },
        }],
      }));
    });

    expect(events.some(e => e.event_type === "tool.called")).toBe(true);
    expect(events.some(e => e.event_type === "tool.responded")).toBe(true);

    const toolCalled = events.find(e => e.event_type === "tool.called");
    expect(toolCalled!.payload["tool_name"]).toBe("weather");
    expect(toolCalled!.payload["args"]).toBe(JSON.stringify({ city: "Paris" }));

    await dt.shutdown();
  });

  it("emits tool.responded with success=false for tool-error content parts", async () => {
    const dt     = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events = captureEvents(dt);

    await dt.run("vercel-agent", { model: "gpt-4o", tools: ["weather"] }, async () => {
      const opts = instrumentGenerateTextOptions(genOpts());
      await opts.onStepEnd?.(makeStep({
        finishReason: "tool-calls",
        usage: makeUsage({ inputTokens: 40, outputTokens: 10, totalTokens: 50 }),
        toolCalls: [{
          type: "tool-call",
          toolCallId: "tc-err",
          toolName: "weather",
          input: { city: "Paris" },
        }],
        content: [{
          type: "tool-error",
          toolCallId: "tc-err",
          toolName: "weather",
          input: { city: "Paris" },
          error: new Error("upstream 500"),
        }] as never,
      }));
    });

    const toolResponded = events.find(e => e.event_type === "tool.responded");
    expect(toolResponded).toBeDefined();
    expect(toolResponded!.payload["success"]).toBe(false);
    expect(toolResponded!.payload["error"]).toBe("Error: upstream 500");

    await dt.shutdown();
  });

  it("records per-step latency on llm.responded", async () => {
    const dt     = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events = captureEvents(dt);

    await dt.run("vercel-agent", {}, async () => {
      const opts = instrumentGenerateTextOptions(genOpts());
      await new Promise(r => setTimeout(r, 5));
      await opts.onStepEnd?.(makeStep({
        usage: makeUsage({ inputTokens: 1, outputTokens: 1, totalTokens: 2 }),
      }));
    });

    const llmResponded = events.find(e => e.event_type === "llm.responded");
    expect(llmResponded!.payload["latency_ms"]).toBeGreaterThan(0);
    await dt.shutdown();
  });

  it("chains user onStepEnd after emitting Dunetrace events", async () => {
    const dt     = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events = captureEvents(dt);
    const userCb = vi.fn();

    await dt.run("vercel-agent", {}, async () => {
      const opts = instrumentGenerateTextOptions(genOpts({ onStepEnd: userCb }));
      await opts.onStepEnd?.(makeStep({
        usage: makeUsage({ inputTokens: 1, outputTokens: 1, totalTokens: 2 }),
      }));
    });

    expect(events.some(e => e.event_type === "llm.called")).toBe(true);
    expect(userCb).toHaveBeenCalledOnce();
    await dt.shutdown();
  });

  it("chains user onStepStart and measures latency from it", async () => {
    const dt        = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events    = captureEvents(dt);
    const userStart = vi.fn();

    await dt.run("vercel-agent", {}, async () => {
      const opts = instrumentGenerateTextOptions(genOpts({ onStepStart: userStart as never }));
      // Step boundary: onStepStart fires first (resets the latency clock),
      // then the provider call elapses, then onStepEnd emits the events.
      await opts.onStepStart?.({} as never);
      await new Promise(r => setTimeout(r, 5));
      await opts.onStepEnd?.(makeStep({
        usage: makeUsage({ inputTokens: 1, outputTokens: 1, totalTokens: 2 }),
      }));
    });

    expect(userStart).toHaveBeenCalledOnce();
    const llmResponded = events.find(e => e.event_type === "llm.responded");
    expect(llmResponded!.payload["latency_ms"]).toBeGreaterThan(0);
    await dt.shutdown();
  });
});

describe("wrapGenerateText()", () => {
  it("instruments options before delegating to the original function", async () => {
    const dt     = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events = captureEvents(dt);

    const fakeGenerateText = vi.fn(async (opts: { onStepEnd?: (step: StepResult<ToolSet>) => Promise<void> }) => {
      await opts.onStepEnd?.(makeStep({
        text: "done",
        finishReason: "stop",
        usage: makeUsage({ inputTokens: 20, outputTokens: 4, totalTokens: 24 }),
      }));
      return { text: "done" };
    });

    const wrapped = wrapGenerateText(fakeGenerateText as never);

    await dt.run("vercel-agent", {}, async () => {
      await wrapped(genOpts());
    });

    expect(fakeGenerateText).toHaveBeenCalledOnce();
    expect(events.filter(e => e.event_type === "llm.called")).toHaveLength(1);
    await dt.shutdown();
  });
});

describe("instrumentStreamTextOptions()", () => {
  it("emits llm events once per step from onStepEnd during streaming", async () => {
    const dt     = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events = captureEvents(dt);

    await dt.run("vercel-agent", {}, async () => {
      const opts = instrumentStreamTextOptions(streamOpts());
      await opts.onStepEnd?.(makeStep({
        text: "streamed answer",
        finishReason: "stop",
        usage: makeUsage({ inputTokens: 30, outputTokens: 8, totalTokens: 38 }),
      }));
    });

    expect(events.filter(e => e.event_type === "llm.responded")).toHaveLength(1);
    await dt.shutdown();
  });

  it("does not double-emit by also wrapping onEnd", async () => {
    const dt     = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events = captureEvents(dt);

    await dt.run("vercel-agent", {}, async () => {
      const opts = instrumentStreamTextOptions(streamOpts());
      // The AI SDK fires onStepEnd per step; onEnd fires once at the end.
      // Only onStepEnd should be instrumented, so onEnd stays undefined here.
      expect(opts.onEnd).toBeUndefined();
      await opts.onStepEnd?.(makeStep({
        finishReason: "stop",
        usage: makeUsage({ inputTokens: 10, outputTokens: 2, totalTokens: 12 }),
      }));
    });

    expect(events.filter(e => e.event_type === "llm.called")).toHaveLength(1);
    expect(events.filter(e => e.event_type === "llm.responded")).toHaveLength(1);
    await dt.shutdown();
  });

  it("preserves a user-supplied onEnd callback", async () => {
    const dt     = new Dunetrace({ endpoint: "http://localhost:8001" });
    captureEvents(dt);
    const useronEnd = vi.fn();

    await dt.run("vercel-agent", {}, async () => {
      const opts = instrumentStreamTextOptions(streamOpts({ onEnd: useronEnd as never }));
       (opts.onEnd as ((event: unknown) => void) | undefined)?.({});
    });

    expect(useronEnd).toHaveBeenCalledOnce();
    await dt.shutdown();
  });
});

describe("wrapStreamText()", () => {
  it("preserves the synchronous return type of streamText (not a Promise)", () => {
    const fakeStreamText = (_opts: unknown) => ({ textStream: (async function* () {})() });
    const wrapped = wrapStreamText(fakeStreamText as never);
    const result  = wrapped({ model: FAKE_MODEL as never, prompt: "hi" }) as { then?: unknown };
    expect(typeof result.then).not.toBe("function");
  });

  it("instruments streamText calls", async () => {
    const dt     = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events = captureEvents(dt);

    const fakeStreamText = vi.fn(async (opts: { onStepEnd?: (step: StepResult<ToolSet>) => Promise<void> }) => {
      await opts.onStepEnd?.(makeStep({
        finishReason: "stop",
        usage: makeUsage({ inputTokens: 15, outputTokens: 3, totalTokens: 18 }),
        text: "chunk",
      }));
      return { textStream: (async function* () { yield "chunk"; })() };
    });

    const wrapped = wrapStreamText(fakeStreamText as never);

    await dt.run("vercel-agent", {}, async () => {
      wrapped({ model: FAKE_MODEL as never, prompt: "hello" });
    });

    expect(fakeStreamText).toHaveBeenCalledOnce();
    expect(events.some(e => e.event_type === "llm.called")).toBe(true);
    await dt.shutdown();
  });
});

describe("traceGenerateText()", () => {
  it("opens and closes a run around generateText", async () => {
    const dt     = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events = captureEvents(dt);

    const fakeGenerateText = async (opts: { onStepEnd?: (step: StepResult<ToolSet>) => Promise<void> }) => {
      await opts.onStepEnd?.(makeStep({
        text: "answer",
        finishReason: "stop",
        usage: makeUsage({ inputTokens: 10, outputTokens: 2, totalTokens: 12 }),
      }));
      return { text: "answer" };
    };

    const result = await traceGenerateText(
      dt,
      "my-vercel-agent",
      { userInput: "What is 2+2?", tools: ["calculator"] },
      fakeGenerateText as never,
      { model: FAKE_MODEL as never, prompt: "What is 2+2?" },
    );

    expect(result).toEqual({ text: "answer" });
    expect(events[0].event_type).toBe("run.started");
    expect(events.some(e => e.event_type === "run.completed")).toBe(true);
    expect(events.find(e => e.event_type === "run.started")!.agent_id).toBe("my-vercel-agent");

    await dt.shutdown();
  });

  it("derives distinct raw input_text from an array-form prompt", async () => {
    const dt     = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events = captureEvents(dt);

    const fakeGenerateText = async () => ({ text: "ok" });

    const inputTextFor = async (content: string): Promise<string> => {
      await traceGenerateText(
        dt,
        "array-prompt-agent",
        {},
        fakeGenerateText as never,
        { model: FAKE_MODEL as never, prompt: [{ role: "user", content }] } as never,
      );
      return events.filter(e => e.event_type === "run.started").at(-1)!.payload["input_text"] as string;
    };

    const inputA = await inputTextFor("What is 2+2?");
    const inputB = await inputTextFor("What is 9+9?");

    expect(inputA).toBe("What is 2+2?");
    expect(inputB).toBe("What is 9+9?");

    await dt.shutdown();
  });

  it("derives input_text from messages when prompt and userInput are absent", async () => {
    const dt     = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events = captureEvents(dt);

    const fakeGenerateText = async () => ({ text: "ok" });

    await traceGenerateText(
      dt,
      "chat-agent",
      {},
      fakeGenerateText as never,
      { model: FAKE_MODEL as never, messages: [{ role: "user", content: "What is 2+2?" }] } as never,
    );

    const started = events.find(e => e.event_type === "run.started")!;
    expect(started.payload["input_text"]).toBe("What is 2+2?");

    await dt.shutdown();
  });
});

describe("traceStreamText()", () => {
  it("opens and closes a run around streamText", async () => {
    const dt     = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events = captureEvents(dt);

    const fakeStreamText = async (opts: { onStepEnd?: (step: StepResult<ToolSet>) => Promise<void> }) => {
      await opts.onStepEnd?.(makeStep({
        finishReason: "stop",
        usage: makeUsage({ inputTokens: 12, outputTokens: 4, totalTokens: 16 }),
        text: "stream",
      }));
      return {
        toTextStreamResponse: () => new Response(),
        consumeStream: async () => {},
      };
    };

    await traceStreamText(
      dt,
      "stream-agent",
      { userInput: "hello" },
      fakeStreamText as never,
      { model: FAKE_MODEL as never, prompt: "hello" },
    );

    expect(events.some(e => e.event_type === "run.started")).toBe(true);
    expect(events.some(e => e.event_type === "run.completed")).toBe(true);
    await dt.shutdown();
  });
});
