import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import { Dunetrace } from "../src/client.js";
import {
  autoInstrument,
  wrapAnthropicClient,
  wrapOpenAIClient,
  _resetAutoInstrumentState,
} from "../src/auto.js";
import type { AgentEvent } from "../src/models.js";

/**
 * Auto-instrumentation tests.
 *
 * The mechanism under test is prototype patching: both vendor SDKs define
 * `create` on a resource-class prototype shared by every client instance, so
 * patching it once covers clients this code never sees — including ones built
 * after the patch, which is the whole difference between `autoInstrument()` and
 * `dt.wrapOpenAI(client)`.
 *
 * The fakes below mirror that layout (class + prototype method) rather than the
 * vendor packages themselves. What matters for correctness is the shape, and
 * faking it keeps `openai` / `@anthropic-ai/sdk` out of devDependencies.
 */

// ── Fakes shaped like the real SDKs ───────────────────────────────────────────

class FakeCompletions {
  public calls: unknown[] = [];
  async create(opts: Record<string, unknown>): Promise<unknown> {
    this.calls.push(opts);
    if (opts["stream"]) return { __stream: true };
    if (opts["__throw"]) throw new Error("upstream 500");
    return {
      choices: [{ message: { content: "hello there" }, finish_reason: "stop" }],
      usage: { prompt_tokens: 11, completion_tokens: 7 },
    };
  }
}

class FakeChat {
  completions = new FakeCompletions();
}

class FakeOpenAI {
  static Chat = { Completions: FakeCompletions };
  chat = new FakeChat();
}

class FakeMessages {
  async create(opts: Record<string, unknown>): Promise<unknown> {
    if (opts["stream"]) return { __stream: true };
    return {
      content: [{ text: "claude says hi" }],
      stop_reason: "end_turn",
      usage: { input_tokens: 21, output_tokens: 5 },
    };
  }
}

class FakeAnthropic {
  static Messages = FakeMessages;
  messages = new FakeMessages();
}

/**
 * Mistral's layout differs from the other two in three ways that the spec
 * machinery had to grow for, all verified against @mistralai/mistralai:
 *   - the Chat class is NOT exported, so there is no static path to walk;
 *     `chat` is a lazily-cached getter on Mistral.prototype;
 *   - the methods are `complete`/`stream`, not a single flagged `create`;
 *   - responses deserialise to camelCase (promptTokens, finishReason).
 */
class FakeMistralChat {
  async complete(opts: Record<string, unknown>): Promise<unknown> {
    if (opts["__throw"]) throw new Error("upstream 500");
    return {
      choices: [{ message: { content: "bonjour" }, finishReason: "stop" }],
      usage: { promptTokens: 31, completionTokens: 9 },
    };
  }
  async stream(_opts: Record<string, unknown>): Promise<unknown> {
    return { __stream: true };
  }
}

class FakeMistral {
  private _chat?: FakeMistralChat;
  constructor(_opts: unknown) {}
  get chat(): FakeMistralChat {
    if (!this._chat) this._chat = new FakeMistralChat();
    return this._chat;
  }
}

// ── Harness ───────────────────────────────────────────────────────────────────

const captured: AgentEvent[] = [];

function newClient(): Dunetrace {
  captured.length = 0;
  // `exporter` is a synchronous per-event sink, so events are observable without
  // waiting on the batch flush.
  return new Dunetrace({
    exporter: { handle: (event: AgentEvent) => { captured.push(event); } },
  });
}

function eventsOfType(type: string): AgentEvent[] {
  return captured.filter(e => e.event_type === type);
}

/** Restore a patched prototype so tests don't leak into one another. */
function restore<T extends object>(proto: T, key: string, original: unknown): void {
  (proto as Record<string, unknown>)[key] = original;
}

beforeEach(() => {
  _resetAutoInstrumentState();
});

afterEach(() => {
  vi.restoreAllMocks();
});

// ── Prototype patching ────────────────────────────────────────────────────────

describe("autoInstrument — prototype patching", () => {
  it("patches the shared prototype, so instances created AFTER the call are covered", async () => {
    const original = FakeCompletions.prototype.create;
    try {
      const dt = newClient();
      expect(autoInstrument({ openai: FakeOpenAI, targets: ["openai"] })).toEqual(["openai"]);

      // Constructed after patching — the point of the feature.
      const client = new FakeOpenAI();
      await dt.run("agent-a", {}, async () => {
        await client.chat.completions.create({ model: "gpt-4o", messages: [] });
      });

      expect(eventsOfType("llm.called")).toHaveLength(1);
      expect(eventsOfType("llm.responded")).toHaveLength(1);
    } finally {
      restore(FakeCompletions.prototype, "create", original);
    }
  });

  it("covers a client the caller never handed us", async () => {
    const original = FakeCompletions.prototype.create;
    try {
      const dt = newClient();
      autoInstrument({ openai: FakeOpenAI, targets: ["openai"] });

      // Stands in for a client constructed deep inside a third-party library.
      const buriedClient = (() => new FakeOpenAI())();
      await dt.run("agent-a", {}, async () => {
        await buriedClient.chat.completions.create({ model: "gpt-4o", messages: [] });
      });

      expect(eventsOfType("llm.called")).toHaveLength(1);
    } finally {
      restore(FakeCompletions.prototype, "create", original);
    }
  });

  it("preserves `this`, so the original method still reaches its own instance state", async () => {
    const original = FakeCompletions.prototype.create;
    try {
      const dt = newClient();
      autoInstrument({ openai: FakeOpenAI, targets: ["openai"] });

      const client = new FakeOpenAI();
      await dt.run("agent-a", {}, async () => {
        await client.chat.completions.create({ model: "gpt-4o", messages: [] });
      });

      // A wrapper that bound `this` to the prototype would record nothing here.
      expect(client.chat.completions.calls).toHaveLength(1);
    } finally {
      restore(FakeCompletions.prototype, "create", original);
    }
  });

  it("resolves the prototype from a live instance, not just a class", async () => {
    const original = FakeCompletions.prototype.create;
    try {
      const dt = newClient();
      const probe = new FakeOpenAI();
      expect(autoInstrument({ openai: probe, targets: ["openai"] })).toEqual(["openai"]);

      const other = new FakeOpenAI();
      await dt.run("agent-a", {}, async () => {
        await other.chat.completions.create({ model: "gpt-4o", messages: [] });
      });

      expect(eventsOfType("llm.called")).toHaveLength(1);
    } finally {
      restore(FakeCompletions.prototype, "create", original);
    }
  });

  it("unwraps a module namespace object", async () => {
    const original = FakeCompletions.prototype.create;
    try {
      const dt = newClient();
      const moduleNamespace = { OpenAI: FakeOpenAI, default: FakeOpenAI };
      expect(autoInstrument({ openai: moduleNamespace, targets: ["openai"] })).toEqual(["openai"]);

      await dt.run("agent-a", {}, async () => {
        await new FakeOpenAI().chat.completions.create({ model: "gpt-4o", messages: [] });
      });
      expect(eventsOfType("llm.called")).toHaveLength(1);
    } finally {
      restore(FakeCompletions.prototype, "create", original);
    }
  });
});

// ── Event content ─────────────────────────────────────────────────────────────

describe("autoInstrument — emitted events", () => {
  it("maps OpenAI usage and finish reason onto the events", async () => {
    const original = FakeCompletions.prototype.create;
    try {
      const dt = newClient();
      autoInstrument({ openai: FakeOpenAI, targets: ["openai"] });
      await dt.run("agent-a", {}, async () => {
        await new FakeOpenAI().chat.completions.create({ model: "gpt-4o-mini", messages: [] });
      });

      const called = eventsOfType("llm.called")[0];
      const responded = eventsOfType("llm.responded")[0];
      expect(called?.payload["model"]).toBe("gpt-4o-mini");
      expect(called?.payload["prompt_tokens"]).toBe(11);
      expect(responded?.payload["completion_tokens"]).toBe(7);
      expect(responded?.payload["finish_reason"]).toBe("stop");
    } finally {
      restore(FakeCompletions.prototype, "create", original);
    }
  });

  it("maps Anthropic's differently-named usage fields", async () => {
    const original = FakeMessages.prototype.create;
    try {
      const dt = newClient();
      autoInstrument({ anthropic: FakeAnthropic, targets: ["anthropic"] });
      await dt.run("agent-a", {}, async () => {
        await new FakeAnthropic().messages.create({ model: "claude-opus-4", messages: [] });
      });

      const called = eventsOfType("llm.called")[0];
      const responded = eventsOfType("llm.responded")[0];
      expect(called?.payload["prompt_tokens"]).toBe(21);      // input_tokens
      expect(responded?.payload["completion_tokens"]).toBe(5); // output_tokens
      expect(responded?.payload["finish_reason"]).toBe("end_turn");
    } finally {
      restore(FakeMessages.prototype, "create", original);
    }
  });

  it("emits nothing outside a run, and still returns the response", async () => {
    const original = FakeCompletions.prototype.create;
    try {
      newClient();
      autoInstrument({ openai: FakeOpenAI, targets: ["openai"] });
      const resp = await new FakeOpenAI().chat.completions.create({ model: "gpt-4o", messages: [] });
      expect(captured).toHaveLength(0);
      expect(resp).toBeTruthy();
    } finally {
      restore(FakeCompletions.prototype, "create", original);
    }
  });
});

// ── Pass-through behaviour ────────────────────────────────────────────────────

describe("autoInstrument — never breaks the host call", () => {
  it("hands back a non-iterable stream response untouched", async () => {
    const original = FakeCompletions.prototype.create;
    try {
      const dt = newClient();
      autoInstrument({ openai: FakeOpenAI, targets: ["openai"] });
      await dt.run("agent-a", {}, async () => {
        const resp = await new FakeOpenAI().chat.completions.create({
          model: "gpt-4o", messages: [], stream: true,
        });
        // Nothing to observe on a non-async-iterable: return it as-is rather
        // than wrapping it in a proxy that can never fire.
        expect(resp).toEqual({ __stream: true });
      });
      // The call still happened, so llm.called stands; there is no stream to
      // drain, so no llm.responded follows.
      expect(eventsOfType("llm.called")).toHaveLength(1);
      expect(eventsOfType("llm.responded")).toHaveLength(0);
    } finally {
      restore(FakeCompletions.prototype, "create", original);
    }
  });

  it("propagates the original error and emits no llm.responded", async () => {
    const original = FakeCompletions.prototype.create;
    try {
      const dt = newClient();
      autoInstrument({ openai: FakeOpenAI, targets: ["openai"] });
      await dt.run("agent-a", {}, async () => {
        await expect(
          new FakeOpenAI().chat.completions.create({ model: "gpt-4o", __throw: true }),
        ).rejects.toThrow("upstream 500");
      });
      expect(eventsOfType("llm.responded")).toHaveLength(0);
    } finally {
      restore(FakeCompletions.prototype, "create", original);
    }
  });

  it("swallows an emission failure rather than failing the LLM call", async () => {
    const original = FakeCompletions.prototype.create;
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const dt = newClient();
      autoInstrument({ openai: FakeOpenAI, targets: ["openai"] });

      await dt.run("agent-a", {}, async (run) => {
        // A bug inside our own event emission must not surface to the caller.
        vi.spyOn(run, "llmCalled").mockImplementation(() => {
          throw new Error("instrumentation bug");
        });
        const resp = await new FakeOpenAI().chat.completions.create({
          model: "gpt-4o", messages: [],
        });
        expect(resp).toBeTruthy();
      });

      expect(warn).toHaveBeenCalled();
    } finally {
      restore(FakeCompletions.prototype, "create", original);
      warn.mockRestore();
    }
  });
});

// ── Idempotency and configuration ─────────────────────────────────────────────

describe("autoInstrument — idempotency and options", () => {
  it("does not double-count when called twice", async () => {
    const original = FakeCompletions.prototype.create;
    try {
      const dt = newClient();
      autoInstrument({ openai: FakeOpenAI, targets: ["openai"] });
      autoInstrument({ openai: FakeOpenAI, targets: ["openai"] });

      await dt.run("agent-a", {}, async () => {
        await new FakeOpenAI().chat.completions.create({ model: "gpt-4o", messages: [] });
      });
      expect(eventsOfType("llm.called")).toHaveLength(1);
    } finally {
      restore(FakeCompletions.prototype, "create", original);
    }
  });

  it("does not double-count when combined with dt.wrapOpenAI on the same client", async () => {
    const original = FakeCompletions.prototype.create;
    try {
      const dt = newClient();
      autoInstrument({ openai: FakeOpenAI, targets: ["openai"] });
      const client = dt.wrapOpenAI(new FakeOpenAI());

      await dt.run("agent-a", {}, async () => {
        await client.chat.completions.create({ model: "gpt-4o", messages: [] });
      });
      expect(eventsOfType("llm.called")).toHaveLength(1);
    } finally {
      restore(FakeCompletions.prototype, "create", original);
    }
  });

  it("honours the targets filter", () => {
    const original = FakeMessages.prototype.create;
    try {
      const patched = autoInstrument({
        openai: FakeOpenAI, anthropic: FakeAnthropic, targets: ["anthropic"],
      });
      expect(patched).toEqual(["anthropic"]);
    } finally {
      restore(FakeMessages.prototype, "create", original);
    }
  });

  it("skips silently when an SDK isn't installed", () => {
    expect(autoInstrument({ targets: ["openai"] })).toEqual([]);
  });

  it("throws in strict mode when a requested SDK is missing", () => {
    expect(() => autoInstrument({ targets: ["openai"], strict: true })).toThrow(/could not load/);
  });

  it("warns on an unrecognised target rather than throwing", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(autoInstrument({ targets: ["nope"] })).toEqual([]);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it("warns when the SDK layout is unrecognisable", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(autoInstrument({ openai: { nothing: true }, targets: ["openai"] })).toEqual([]);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });
});

// ── Instance wrappers ─────────────────────────────────────────────────────────

describe("per-instance wrappers", () => {
  it("wrapOpenAIClient instruments only the instance given", async () => {
    const dt = newClient();
    const wrapped = wrapOpenAIClient(new FakeOpenAI());
    const untouched = new FakeOpenAI();

    await dt.run("agent-a", {}, async () => {
      await wrapped.chat.completions.create({ model: "gpt-4o", messages: [] });
      await untouched.chat.completions.create({ model: "gpt-4o", messages: [] });
    });

    expect(eventsOfType("llm.called")).toHaveLength(1);
  });

  it("wrapAnthropicClient instruments only the instance given", async () => {
    const dt = newClient();
    const wrapped = wrapAnthropicClient(new FakeAnthropic());

    await dt.run("agent-a", {}, async () => {
      await wrapped.messages.create({ model: "claude-opus-4", messages: [] });
    });

    expect(eventsOfType("llm.called")).toHaveLength(1);
  });

  it("warns and returns the object unchanged when the shape is wrong", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const notAClient = { nope: true };
    expect(wrapOpenAIClient(notAClient)).toBe(notAClient);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });
});

// ── Streaming ─────────────────────────────────────────────────────────────────

/** OpenAI-shaped stream: an object that is async-iterable and has extra members. */
function fakeOpenAIStream(chunks: unknown[]) {
  return {
    controller: { abort: () => {} },
    tee: () => "teed",
    async *[Symbol.asyncIterator]() {
      for (const c of chunks) yield c;
    },
  };
}

const OPENAI_CHUNKS = [
  { choices: [{ delta: { content: "Hel" } }] },
  { choices: [{ delta: { content: "lo" } }] },
  { choices: [{ delta: {}, finish_reason: "stop" }] },
  { choices: [], usage: { prompt_tokens: 12, completion_tokens: 3 } },
];

const ANTHROPIC_CHUNKS = [
  { type: "message_start", message: { usage: { input_tokens: 31 } } },
  { type: "content_block_delta", delta: { text: "Hi " } },
  { type: "content_block_delta", delta: { text: "there" } },
  { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 9 } },
  { type: "message_stop" },
];

class StreamingCompletions {
  async create(opts: Record<string, unknown>): Promise<unknown> {
    if (opts["stream"]) return fakeOpenAIStream(OPENAI_CHUNKS);
    return { choices: [{ message: { content: "x" }, finish_reason: "stop" }], usage: {} };
  }
}
class StreamingOpenAI {
  static Chat = { Completions: StreamingCompletions };
  chat = { completions: new StreamingCompletions() };
}

class StreamingMessages {
  async create(opts: Record<string, unknown>): Promise<unknown> {
    if (opts["stream"]) return fakeOpenAIStream(ANTHROPIC_CHUNKS);
    return { content: [{ text: "x" }], usage: {} };
  }
}
class StreamingAnthropic {
  static Messages = StreamingMessages;
  messages = new StreamingMessages();
}

describe("autoInstrument — streaming", () => {
  it("emits llm.responded with text and usage accumulated from the stream", async () => {
    const original = StreamingCompletions.prototype.create;
    try {
      const dt = newClient();
      autoInstrument({ openai: StreamingOpenAI, targets: ["openai"] });

      const seen: string[] = [];
      await dt.run("agent-a", {}, async () => {
        const stream = await new StreamingOpenAI().chat.completions.create({
          model: "gpt-4o", stream: true,
        }) as AsyncIterable<Record<string, unknown>>;
        for await (const chunk of stream) seen.push(JSON.stringify(chunk));
      });

      // The caller still received every chunk — we observe, never consume.
      expect(seen).toHaveLength(OPENAI_CHUNKS.length);

      const responded = eventsOfType("llm.responded")[0];
      expect(eventsOfType("llm.called")).toHaveLength(1);
      expect(responded?.payload["output"]).toBe("Hello");
      expect(responded?.payload["finish_reason"]).toBe("stop");
      expect(responded?.payload["completion_tokens"]).toBe(3);
    } finally {
      restore(StreamingCompletions.prototype, "create", original);
    }
  });

  it("accumulates Anthropic's event-typed stream", async () => {
    const original = StreamingMessages.prototype.create;
    try {
      const dt = newClient();
      autoInstrument({ anthropic: StreamingAnthropic, targets: ["anthropic"] });

      await dt.run("agent-a", {}, async () => {
        const stream = await new StreamingAnthropic().messages.create({
          model: "claude-opus-4", stream: true,
        }) as AsyncIterable<unknown>;
        for await (const _ of stream) { /* drain */ }
      });

      const responded = eventsOfType("llm.responded")[0];
      expect(responded?.payload["output"]).toBe("Hi there");
      expect(responded?.payload["finish_reason"]).toBe("end_turn");
      expect(responded?.payload["completion_tokens"]).toBe(9);
    } finally {
      restore(StreamingMessages.prototype, "create", original);
    }
  });

  it("preserves non-iterator members of the stream object", async () => {
    const original = StreamingCompletions.prototype.create;
    try {
      const dt = newClient();
      autoInstrument({ openai: StreamingOpenAI, targets: ["openai"] });
      await dt.run("agent-a", {}, async () => {
        const stream = await new StreamingOpenAI().chat.completions.create({
          model: "gpt-4o", stream: true,
        }) as { tee: () => string; controller: unknown };
        // A bare async generator would have dropped these.
        expect(stream.tee()).toBe("teed");
        expect(stream.controller).toBeDefined();
      });
    } finally {
      restore(StreamingCompletions.prototype, "create", original);
    }
  });

  it("still reports when the caller breaks out of the stream early", async () => {
    const original = StreamingCompletions.prototype.create;
    try {
      const dt = newClient();
      autoInstrument({ openai: StreamingOpenAI, targets: ["openai"] });

      await dt.run("agent-a", {}, async () => {
        const stream = await new StreamingOpenAI().chat.completions.create({
          model: "gpt-4o", stream: true,
        }) as AsyncIterable<unknown>;
        for await (const _ of stream) break;   // reaches the iterator as return()
      });

      const responded = eventsOfType("llm.responded")[0];
      expect(responded).toBeDefined();
      expect(responded?.payload["output"]).toBe("Hel");  // only what arrived
    } finally {
      restore(StreamingCompletions.prototype, "create", original);
    }
  });

  it("emits llm.responded exactly once even if iterated to completion then returned", async () => {
    const original = StreamingCompletions.prototype.create;
    try {
      const dt = newClient();
      autoInstrument({ openai: StreamingOpenAI, targets: ["openai"] });
      await dt.run("agent-a", {}, async () => {
        const stream = await new StreamingOpenAI().chat.completions.create({
          model: "gpt-4o", stream: true,
        }) as AsyncIterable<unknown>;
        for await (const _ of stream) { /* full drain */ }
      });
      expect(eventsOfType("llm.responded")).toHaveLength(1);
    } finally {
      restore(StreamingCompletions.prototype, "create", original);
    }
  });

  it("emits nothing for a stream that is never consumed", async () => {
    const original = StreamingCompletions.prototype.create;
    try {
      const dt = newClient();
      autoInstrument({ openai: StreamingOpenAI, targets: ["openai"] });
      await dt.run("agent-a", {}, async () => {
        await new StreamingOpenAI().chat.completions.create({ model: "gpt-4o", stream: true });
      });
      // llm.called is recorded (the call did happen); there is no response to report.
      expect(eventsOfType("llm.called")).toHaveLength(1);
      expect(eventsOfType("llm.responded")).toHaveLength(0);
    } finally {
      restore(StreamingCompletions.prototype, "create", original);
    }
  });
});

// ── HTTP ──────────────────────────────────────────────────────────────────────

describe("instrumentHttp", () => {
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  function stubFetch(status = 200, length = "42") {
    globalThis.fetch = (async () =>
      new Response("ok", { status, headers: { "content-length": length } })
    ) as typeof fetch;
  }

  it("emits tool.called / tool.responded named after the host", async () => {
    stubFetch();
    const dt = newClient();
    expect(autoInstrument({ targets: ["http"] })).toEqual(["http"]);

    await dt.run("agent-a", {}, async () => {
      await fetch("https://api.example.com/v1/things?q=1");
    });

    const called = eventsOfType("tool.called")[0];
    const responded = eventsOfType("tool.responded")[0];
    expect(called?.payload["tool_name"]).toBe("api.example.com");
    expect(responded?.payload["success"]).toBe(true);
    expect(responded?.payload["output_length"]).toBe(42);
  });

  it("marks non-2xx/3xx responses as failures", async () => {
    stubFetch(503);
    const dt = newClient();
    autoInstrument({ targets: ["http"] });
    await dt.run("agent-a", {}, async () => { await fetch("https://api.example.com/x"); });

    const responded = eventsOfType("tool.responded")[0];
    expect(responded?.payload["success"]).toBe(false);
    expect(responded?.payload["error"]).toBe("503");
  });

  it("records a network failure and rethrows", async () => {
    globalThis.fetch = (async () => { throw new Error("ECONNREFUSED"); }) as typeof fetch;
    const dt = newClient();
    autoInstrument({ targets: ["http"] });

    await dt.run("agent-a", {}, async () => {
      await expect(fetch("https://api.example.com/x")).rejects.toThrow("ECONNREFUSED");
    });
    const responded = eventsOfType("tool.responded")[0];
    expect(responded?.payload["success"]).toBe(false);
    expect(responded?.payload["error"]).toContain("ECONNREFUSED");
  });

  it("ignores Dunetrace's own ingest traffic", async () => {
    stubFetch();
    _resetAutoInstrumentState();
    const dt = new Dunetrace({
      endpoint: "http://ingest.internal:8001",
      exporter: { handle: (e: AgentEvent) => { captured.push(e); } },
    });
    captured.length = 0;
    autoInstrument({ targets: ["http"] });

    await dt.run("agent-a", {}, async () => {
      await fetch("http://ingest.internal:8001/v1/ingest", { method: "POST" });
    });
    // Instrumenting our own shipping would emit an event describing the ship,
    // which buffers another event, which ships.
    expect(eventsOfType("tool.called")).toHaveLength(0);
  });

  it("does not double-count the HTTP an instrumented LLM call makes", async () => {
    const original = FakeCompletions.prototype.create;
    try {
      const dt = newClient();
      autoInstrument({ openai: FakeOpenAI, targets: ["openai", "http"] });

      // The SDK's own request, issued from inside create().
      globalThis.fetch = (async () =>
        new Response("{}", { status: 200 })) as typeof fetch;
      const proto = FakeCompletions.prototype as unknown as Record<string, unknown>;
      const patchedCreate = proto["create"] as (o: Record<string, unknown>) => Promise<unknown>;
      proto["create"] = async function (o: Record<string, unknown>) {
        await fetch("https://api.openai.com/v1/chat/completions");
        return patchedCreate.call(this, o);
      };

      await dt.run("agent-a", {}, async () => {
        await new FakeOpenAI().chat.completions.create({ model: "gpt-4o" });
      });

      // One LLM call, not one LLM call plus a tool call to api.openai.com.
      expect(eventsOfType("llm.called")).toHaveLength(1);
      expect(eventsOfType("tool.called")).toHaveLength(0);
    } finally {
      restore(FakeCompletions.prototype, "create", original);
    }
  });

  it("emits nothing outside a run", async () => {
    stubFetch();
    newClient();
    autoInstrument({ targets: ["http"] });
    await fetch("https://api.example.com/x");
    expect(captured).toHaveLength(0);
  });

  it("is idempotent", async () => {
    stubFetch();
    const dt = newClient();
    autoInstrument({ targets: ["http"] });
    autoInstrument({ targets: ["http"] });
    await dt.run("agent-a", {}, async () => { await fetch("https://api.example.com/x"); });
    expect(eventsOfType("tool.called")).toHaveLength(1);
  });
});

// ── Mistral ───────────────────────────────────────────────────────────────────

describe("autoInstrument — mistral", () => {
  it("resolves the Chat prototype by constructing a throwaway client", () => {
    // The class is unexported and `chat` is an instance getter, so neither the
    // static path nor a bare module namespace can reach it.
    expect(autoInstrument({ mistral: FakeMistral, targets: ["mistral"] })).toEqual(["mistral"]);
    restore(FakeMistralChat.prototype, "complete", FakeMistralChat.prototype.complete);
    restore(FakeMistralChat.prototype, "stream", FakeMistralChat.prototype.stream);
  });

  it("patches every client, including ones built after the call", async () => {
    const origComplete = FakeMistralChat.prototype.complete;
    try {
      const dt = newClient();
      autoInstrument({ mistral: FakeMistral, targets: ["mistral"] });
      const client = new FakeMistral({ apiKey: "k" });

      await dt.run("agent", {}, async () => {
        await client.chat.complete({ model: "mistral-large-latest", messages: [] });
      });

      const called = eventsOfType("llm.called");
      const responded = eventsOfType("llm.responded");
      expect(called).toHaveLength(1);
      expect(responded).toHaveLength(1);
      expect(called[0]!.payload["model"]).toBe("mistral-large-latest");
      // camelCase usage, which the other two SDKs do not use.
      expect(called[0]!.payload["prompt_tokens"]).toBe(31);
      expect(responded[0]!.payload["completion_tokens"]).toBe(9);
      expect(responded[0]!.payload["output"]).toBe("bonjour");
    } finally {
      restore(FakeMistralChat.prototype, "complete", origComplete);
    }
  });

  it("treats stream() as always streaming, with no stream flag to read", async () => {
    const origStream = FakeMistralChat.prototype.stream;
    try {
      const dt = newClient();
      autoInstrument({ mistral: FakeMistral, targets: ["mistral"] });
      const client = new FakeMistral({ apiKey: "k" });

      await dt.run("agent", {}, async () => {
        await client.chat.stream({ model: "mistral-large-latest", messages: [] });
      });

      // llm.called fires immediately; llm.responded waits for the caller to
      // drain, exactly as for the other two SDKs' streaming path.
      expect(eventsOfType("llm.called")).toHaveLength(1);
      expect(eventsOfType("llm.responded")).toHaveLength(0);
    } finally {
      restore(FakeMistralChat.prototype, "stream", origStream);
    }
  });

  it("never breaks the host call when the provider throws", async () => {
    const origComplete = FakeMistralChat.prototype.complete;
    try {
      const dt = newClient();
      autoInstrument({ mistral: FakeMistral, targets: ["mistral"] });
      const client = new FakeMistral({ apiKey: "k" });

      await expect(
        dt.run("agent", {}, async () => {
          await client.chat.complete({ model: "m", messages: [], __throw: true });
        }),
      ).rejects.toThrow("upstream 500");
    } finally {
      restore(FakeMistralChat.prototype, "complete", origComplete);
    }
  });

  it("is idempotent", () => {
    const origComplete = FakeMistralChat.prototype.complete;
    try {
      autoInstrument({ mistral: FakeMistral, targets: ["mistral"] });
      const afterFirst = FakeMistralChat.prototype.complete;
      _resetAutoInstrumentState();
      autoInstrument({ mistral: FakeMistral, targets: ["mistral"] });
      expect(FakeMistralChat.prototype.complete).toBe(afterFirst);
    } finally {
      restore(FakeMistralChat.prototype, "complete", origComplete);
    }
  });

  it("is a known target", () => {
    const dt = newClient();
    void dt;
    expect(autoInstrument({ targets: ["mistral"], mistral: FakeMistral })).toEqual(["mistral"]);
    restore(FakeMistralChat.prototype, "complete", FakeMistralChat.prototype.complete);
    restore(FakeMistralChat.prototype, "stream", FakeMistralChat.prototype.stream);
  });
});
