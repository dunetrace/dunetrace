import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { Dunetrace, getCurrentRun } from "../src/client.js";
import type { AgentEvent } from "../src/models.js";

// ── Fetch mock ────────────────────────────────────────────────────────────────

const capturedBodies: Array<{ url: string; body: unknown }> = [];

function mockFetch(ok = true) {
  vi.stubGlobal("fetch", async (url: string, init?: RequestInit) => {
    capturedBodies.push({
      url,
      body: init?.body ? JSON.parse(init.body as string) : null,
    });
    if (!ok) throw new Error("network error");
    return new Response(JSON.stringify({ ok: true }), { status: 200 });
  });
}

function lastBatch(): AgentEvent[] {
  return (capturedBodies.at(-1)?.body as { events: AgentEvent[] })?.events ?? [];
}

beforeEach(() => {
  capturedBodies.length = 0;
  vi.useFakeTimers();
  mockFetch();
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

// ── run() lifecycle ────────────────────────────────────────────────────────────

describe("Dunetrace.run() — lifecycle events", () => {
  it("emits run.started on entry", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events: AgentEvent[] = [];
    dt._emit = (e) => events.push(e);

    await dt.run("my-agent", { model: "gpt-4o" }, async () => {});
    expect(events[0].event_type).toBe("run.started");
    expect(events[0].agent_id).toBe("my-agent");
    await dt.shutdown();
  });

  it("emits run.completed on clean exit", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events: AgentEvent[] = [];
    dt._emit = (e) => events.push(e);

    await dt.run("my-agent", {}, async () => {});
    expect(events.at(-1)!.event_type).toBe("run.completed");
    await dt.shutdown();
  });

  it("emits run.errored when fn throws", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events: AgentEvent[] = [];
    dt._emit = (e) => events.push(e);

    await expect(
      dt.run("my-agent", {}, async () => { throw new Error("boom"); }),
    ).rejects.toThrow("boom");

    expect(events.at(-1)!.event_type).toBe("run.errored");
    await dt.shutdown();
  });

  it("re-throws the original exception", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
    const err = new TypeError("specific error");
    await expect(
      dt.run("my-agent", {}, async () => { throw err; }),
    ).rejects.toThrow(err);
    await dt.shutdown();
  });

  it("run.errored includes error_type", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events: AgentEvent[] = [];
    dt._emit = (e) => events.push(e);

    await expect(
      dt.run("my-agent", {}, async () => { throw new RangeError("out of range"); }),
    ).rejects.toThrow();

    const errEv = events.find(e => e.event_type === "run.errored")!;
    expect(errEv.payload["error_type"]).toBe("RangeError");
    expect(errEv.payload).not.toHaveProperty("error_message");
    await dt.shutdown();
  });

  it("run.started hashes userInput — never stores raw", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events: AgentEvent[] = [];
    dt._emit = (e) => events.push(e);

    await dt.run("my-agent", { userInput: "secret user query" }, async () => {});
    const started = events.find(e => e.event_type === "run.started")!;
    expect(started.payload).not.toHaveProperty("input");
    expect(typeof started.payload["input_hash"]).toBe("string");
    expect((started.payload["input_hash"] as string)).toHaveLength(16);
    await dt.shutdown();
  });

  it("run.started records model and tools", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events: AgentEvent[] = [];
    dt._emit = (e) => events.push(e);

    await dt.run("my-agent", { model: "gpt-4o", tools: ["search"] }, async () => {});
    const started = events.find(e => e.event_type === "run.started")!;
    expect(started.payload["model"]).toBe("gpt-4o");
    expect(started.payload["tools"]).toEqual(["search"]);
    await dt.shutdown();
  });

  it("sets consistent agent_version across all events", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events: AgentEvent[] = [];
    dt._emit = (e) => events.push(e);

    await dt.run("agent", { model: "gpt-4o" }, async (run) => {
      run.llmCalled("gpt-4o");
      run.llmResponded({});
    });
    const versions = events.map(e => e.agent_version);
    expect(new Set(versions).size).toBe(1);
    await dt.shutdown();
  });

  it("run provides a working DunetraceRun context", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events: AgentEvent[] = [];
    dt._emit = (e) => events.push(e);

    await dt.run("agent", {}, async (run) => {
      run.toolCalled("search", { q: "test" });
      run.toolResponded("search", true, 128, 50);
      run.finalAnswer();
    });

    const types = events.map(e => e.event_type);
    expect(types).toContain("tool.called");
    expect(types).toContain("tool.responded");
    await dt.shutdown();
  });

  it("multiple sequential runs don't share state", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
    const runIds = new Set<string>();

    for (let i = 0; i < 3; i++) {
      await dt.run("agent", {}, async (run) => {
        runIds.add(run.runId);
      });
    }
    expect(runIds.size).toBe(3);
    await dt.shutdown();
  });

  it("preset runId is used on all events (Langfuse correlation)", async () => {
    const dt     = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events: AgentEvent[] = [];
    dt._emit = (e) => events.push(e);

    const presetId = "550e8400-e29b-41d4-a716-446655440000";
    await dt.run("agent", { runId: presetId }, async (run) => {
      expect(run.runId).toBe(presetId);
      run.llmCalled("gpt-4o");
    });

    expect(events.every(e => e.run_id === presetId)).toBe(true);
    await dt.shutdown();
  });

  it("omitting runId still generates a UUID", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
    await dt.run("agent", {}, async (run) => {
      expect(run.runId).toMatch(
        /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/,
      );
    });
    await dt.shutdown();
  });
});

// ── tool() wrapper ────────────────────────────────────────────────────────────

describe("Dunetrace.tool()", () => {
  it("auto-emits tool.called and tool.responded for sync function", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events: AgentEvent[] = [];
    dt._emit = (e) => events.push(e);

    const greet = dt.tool((name: unknown) => `Hello ${name}`, "greet");

    await dt.run("agent", {}, async () => {
      greet("world");
    });

    const types = events.map(e => e.event_type);
    expect(types).toContain("tool.called");
    expect(types).toContain("tool.responded");
    await dt.shutdown();
  });

  it("auto-emits for async function", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events: AgentEvent[] = [];
    dt._emit = (e) => events.push(e);

    const search = dt.tool(async (_q: unknown) => ["result1", "result2"], "search");

    await dt.run("agent", {}, async () => {
      await search("query");
    });

    expect(events.some(e => e.event_type === "tool.called")).toBe(true);
    expect(events.some(e => e.event_type === "tool.responded")).toBe(true);
    const resp = events.find(e => e.event_type === "tool.responded")!;
    expect(resp.payload["success"]).toBe(true);
    await dt.shutdown();
  });

  it("emits success=false on error and re-throws", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events: AgentEvent[] = [];
    dt._emit = (e) => events.push(e);

    const broken = dt.tool(async (_: unknown) => { throw new Error("fail"); }, "broken");

    await expect(
      dt.run("agent", {}, async () => { await broken("x"); }),
    ).rejects.toThrow("fail");

    const resp = events.find(e => e.event_type === "tool.responded")!;
    expect(resp.payload["success"]).toBe(false);
    expect(resp.payload).toHaveProperty("error_hash");
    await dt.shutdown();
  });

  it("is a no-op outside a run context (function still runs)", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events: AgentEvent[] = [];
    dt._emit = (e) => events.push(e);

    const greet = dt.tool((name: unknown) => `Hello ${name}`, "greet");
    const result = greet("world");
    expect(result).toBe("Hello world");
    expect(events.filter(e => e.event_type === "tool.called")).toHaveLength(0);
    await dt.shutdown();
  });

  it("uses provided name over function name", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events: AgentEvent[] = [];
    dt._emit = (e) => events.push(e);

    function myInternalName(_x: unknown) { return "ok"; }
    const wrapped = dt.tool(myInternalName, "public_name");

    await dt.run("agent", {}, async () => { wrapped("x"); });

    const called = events.find(e => e.event_type === "tool.called")!;
    expect(called.payload["tool_name"]).toBe("public_name");
    await dt.shutdown();
  });

  it("uses function.name when no name provided", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events: AgentEvent[] = [];
    dt._emit = (e) => events.push(e);

    function webSearch(_q: unknown) { return "result"; }
    const wrapped = dt.tool(webSearch);

    await dt.run("agent", {}, async () => { wrapped("q"); });

    const called = events.find(e => e.event_type === "tool.called")!;
    expect(called.payload["tool_name"]).toBe("webSearch");
    await dt.shutdown();
  });
});

// ── trace() wrapper ────────────────────────────────────────────────────────────

describe("Dunetrace.trace()", () => {
  it("wraps an async function with a run context", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events: AgentEvent[] = [];
    dt._emit = (e) => events.push(e);

    async function myAgent(_query: string) { return "answer"; }
    const wrapped = dt.trace(myAgent, "test-agent");
    await wrapped("hello");

    expect(events[0].event_type).toBe("run.started");
    expect(events.at(-1)!.event_type).toBe("run.completed");
    await dt.shutdown();
  });

  it("uses function name as agentId when not provided", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events: AgentEvent[] = [];
    dt._emit = (e) => events.push(e);

    async function researchAgent(_q: string) { return "ok"; }
    const wrapped = dt.trace(researchAgent);
    await wrapped("q");

    expect(events[0].agent_id).toBe("researchAgent");
    await dt.shutdown();
  });

  it("hashes first arg as userInput", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
    const events: AgentEvent[] = [];
    dt._emit = (e) => events.push(e);

    async function agent(_q: string) { return "ok"; }
    const wrapped = dt.trace(agent, "a");
    await wrapped("sensitive input");

    const started = events.find(e => e.event_type === "run.started")!;
    expect(started.payload).not.toHaveProperty("input");
    expect(typeof started.payload["input_hash"]).toBe("string");
    await dt.shutdown();
  });
});

// ── getCurrentRun() ────────────────────────────────────────────────────────────

describe("getCurrentRun()", () => {
  it("returns null outside a run", () => {
    expect(getCurrentRun()).toBeNull();
  });

  it("returns the active run inside dt.run()", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
    let capturedRun = null;
    await dt.run("agent", {}, async (run) => {
      capturedRun = getCurrentRun();
      expect(capturedRun).toBe(run);
    });
    expect(capturedRun).not.toBeNull();
    await dt.shutdown();
  });

  it("returns null again after the run completes", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
    await dt.run("agent", {}, async () => {});
    expect(getCurrentRun()).toBeNull();
    await dt.shutdown();
  });
});

// ── markDeploy ────────────────────────────────────────────────────────────────

describe("Dunetrace.markDeploy()", () => {
  it("fires a POST to /v1/deploy without blocking", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
    dt.markDeploy("my-agent", "v1.0.0", { env: "production" });
    // Allow microtasks to run
    await Promise.resolve();
    const deploy = capturedBodies.find(b => (b.url as string).includes("/v1/deploy"));
    expect(deploy).toBeDefined();
    expect((deploy!.body as Record<string, unknown>)["agent_id"]).toBe("my-agent");
    expect((deploy!.body as Record<string, unknown>)["version"]).toBe("v1.0.0");
    await dt.shutdown();
  });
});

// ── emitAsJson ────────────────────────────────────────────────────────────────

describe("Dunetrace — emitAsJson mode", () => {
  it("writes NDJSON to stdout for each event", async () => {
    const written: string[] = [];
    vi.spyOn(process.stdout, "write").mockImplementation((data: unknown) => {
      written.push(data as string);
      return true;
    });

    const dt = new Dunetrace({ endpoint: null as unknown as string, emitAsJson: true });
    await dt.run("agent", { model: "gpt-4o" }, async () => {});
    await dt.shutdown();

    expect(written.length).toBeGreaterThanOrEqual(2);
    const parsed = JSON.parse(written[0]);
    expect(parsed).toHaveProperty("event_type");
    expect(parsed).toHaveProperty("ts");
    expect(parsed["event_type"]).toBe("run.started");
    expect(parsed).not.toHaveProperty("raw_input");
  });
});

// ── flush and drain ────────────────────────────────────────────────────────────

describe("Dunetrace — flush / shutdown", () => {
  it("ships buffered events on shutdown", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001", flushIntervalMs: 60000 });
    const events: AgentEvent[] = [];
    dt._emit = (e) => { events.push(e); (dt as unknown as { _buffer: AgentEvent[] })._buffer.push(e); };

    await dt.run("agent", {}, async () => {});
    // Don't advance timers — drain only happens on shutdown
    await dt.shutdown();
    // After shutdown, buffer should be empty
    expect((dt as unknown as { _buffer: AgentEvent[] })._buffer.length).toBe(0);
  });
});
