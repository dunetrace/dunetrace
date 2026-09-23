/**
 * Runs that start and end without dt.run(): the DSN, implicit runs and how
 * they close (idle, explicit run, process exit, crash), and the Vercel AI
 * entry points that open exact runs.
 *
 * Events are read through the `exporter` sink; nothing is shipped.
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { Dunetrace, getCurrentRun, resolveRun, openEntryRun, parseDsn, dsnHost } from "../src/index.js";
import { setRunOpener, runStorage } from "../src/context.js";
import { wrapOpenAIClient, _resetAutoInstrumentState } from "../src/auto.js";
import { wrapGenerateText, wrapStreamText } from "../src/integrations/vercel-ai.js";
import { NoopBatchEmitter } from "../src/emitters.js";
import type { AgentEvent, ClientOptions } from "../src/models.js";

const ENV = ["DUNETRACE_DSN", "DUNETRACE_ENDPOINT", "DUNETRACE_API_KEY", "DUNETRACE_IMPLICIT_RUNS", "DUNETRACE_IMPLICIT_RUN_IDLE_S", "DUNETRACE_AGENT_ID"];
const saved: Record<string, string | undefined> = {};

beforeEach(() => {
  for (const k of ENV) { saved[k] = process.env[k]; delete process.env[k]; }
  setRunOpener(null);
  _resetAutoInstrumentState();
  // enterWith() from a test at its top level leaks the run into the runner's
  // own async context, which the next test would inherit. Clear it.
  runStorage.enterWith(undefined as never);
});
afterEach(() => {
  for (const k of ENV) { if (saved[k] === undefined) delete process.env[k]; else process.env[k] = saved[k]; }
  setRunOpener(null);
});

function client(opts: ClientOptions = {}): { dt: Dunetrace; events: AgentEvent[] } {
  const events: AgentEvent[] = [];
  const dt = new Dunetrace({
    emitter: new NoopBatchEmitter(),
    flushOnExit: false,
    exporter: { handle: (e) => { events.push(e); } },
    ...opts,
  });
  return { dt, events };
}

function fakeOpenAI() {
  return {
    chat: {
      completions: {
        create: async (opts: { model: string }) => ({
          model: opts.model,
          choices: [{ finish_reason: "stop", message: { content: "hi" } }],
          usage: { prompt_tokens: 3, completion_tokens: 2 },
        }),
      },
    },
  };
}

const started   = (events: AgentEvent[]) => events.filter((e) => e.event_type === "run.started");
const completed = (events: AgentEvent[]) => events.filter((e) => e.event_type === "run.completed");
const errored   = (events: AgentEvent[]) => events.filter((e) => e.event_type === "run.errored");
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// ── DSN ──────────────────────────────────────────────────────────────────────

describe("DSN", () => {
  it("splits endpoint and key", () => {
    expect(parseDsn("https://dt_live_abc@ingest.example.com")).toEqual({ endpoint: "https://ingest.example.com", apiKey: "dt_live_abc" });
    expect(parseDsn("http://k%40ey@localhost:8001/base/")).toEqual({ endpoint: "http://localhost:8001/base", apiKey: "k@ey" });
    expect(dsnHost("https://secret@ingest.example.com/x")).toBe("ingest.example.com");
  });

  it("rejects a missing key or a bad scheme", () => {
    for (const bad of ["https://ingest.example.com", "ftp://k@host", "not a url", "https://@host"]) {
      expect(() => parseDsn(bad), bad).toThrow();
    }
  });

  it("configures the client from the environment, with explicit settings winning", async () => {
    process.env.DUNETRACE_DSN = "https://dt_live_env@ingest.example.com";
    const a = new Dunetrace({ emitter: new NoopBatchEmitter(), flushOnExit: false });
    expect((a as unknown as { _ingestUrl: string })._ingestUrl).toBe("https://ingest.example.com/v1/ingest");
    expect((a as unknown as { _apiKey: string })._apiKey).toBe("dt_live_env");
    const b = new Dunetrace({ emitter: new NoopBatchEmitter(), flushOnExit: false, endpoint: "http://localhost:9001", apiKey: "k2" });
    expect((b as unknown as { _ingestUrl: string })._ingestUrl).toBe("http://localhost:9001/v1/ingest");
    expect((b as unknown as { _apiKey: string })._apiKey).toBe("k2");
    await a.shutdown(); await b.shutdown();
  });

  it("ignores a malformed DSN and keeps the defaults", async () => {
    const dt = new Dunetrace({ emitter: new NoopBatchEmitter(), flushOnExit: false, dsn: "https://ingest.example.com" });
    expect((dt as unknown as { _ingestUrl: string })._ingestUrl).toBe("http://localhost:8001/v1/ingest");
    await dt.shutdown();
  });
});

// ── Implicit runs ────────────────────────────────────────────────────────────

describe("implicit runs", () => {
  it("a patched call with no run open gets a marked run, and the next call attaches", async () => {
    const { dt, events } = client({ defaultAgentId: "script-agent" });
    const openai = wrapOpenAIClient(fakeOpenAI());
    await openai.chat.completions.create({ model: "gpt-4o" });
    await openai.chat.completions.create({ model: "gpt-4o" });

    const runs = started(events);
    expect(runs).toHaveLength(1);
    expect(runs[0].agent_id).toBe("script-agent");
    expect(runs[0].payload["implicit"]).toBe(true);
    expect(runs[0].payload["opened_by"]).toBe("openai.chat.completions.create");
    expect(events.filter((e) => e.event_type === "llm.called")).toHaveLength(2);
    expect(events.filter((e) => e.event_type === "llm.responded")).toHaveLength(2);
    const run = getCurrentRun();
    expect(run?.implicit).toBe(true);
    expect(dt._implicitOpenCount()).toBe(1);
    await dt.shutdown();
  });

  it("closes on idle with exit_reason idle, and a later call opens a fresh run", async () => {
    const { dt, events } = client({ implicitRunIdleS: 0.05 });
    const openai = wrapOpenAIClient(fakeOpenAI());
    await openai.chat.completions.create({ model: "gpt-4o" });
    const first = getCurrentRun()!;
    await sleep(120);
    expect(first.closed).toBe(true);
    expect(completed(events).map((e) => e.payload["exit_reason"])).toEqual(["idle"]);
    expect(dt._implicitOpenCount()).toBe(0);

    await openai.chat.completions.create({ model: "gpt-4o" });
    expect(started(events)).toHaveLength(2);
    expect(getCurrentRun()).not.toBe(first);
    await dt.shutdown();
  });

  it("activity pushes the idle deadline out", async () => {
    const { dt, events } = client({ implicitRunIdleS: 0.08 });
    const openai = wrapOpenAIClient(fakeOpenAI());
    await openai.chat.completions.create({ model: "gpt-4o" });
    await sleep(50);
    await openai.chat.completions.create({ model: "gpt-4o" });
    await sleep(50);
    expect(completed(events)).toHaveLength(0); // 100ms elapsed, but only 50ms idle
    await sleep(70);
    expect(completed(events)).toHaveLength(1);
    await dt.shutdown();
  });

  it("a declared dt.run() closes the implicit run and does not nest under it", async () => {
    const { dt, events } = client();
    const openai = wrapOpenAIClient(fakeOpenAI());
    await openai.chat.completions.create({ model: "gpt-4o" });
    const implicit = getCurrentRun()!;
    await dt.run("declared", { model: "gpt-4o" }, async (run) => {
      expect(run.implicit).toBe(false);
      expect(getCurrentRun()).toBe(run);
    });
    const done = completed(events);
    expect(done.map((e) => e.run_id)).toEqual([implicit.runId, expect.any(String)]);
    expect(done[0].payload["exit_reason"]).toBe("explicit_run_opened");
    const declared = started(events).find((e) => e.agent_id === "declared")!;
    expect(declared.parent_run_id).toBeNull();
    await dt.shutdown();
  });

  it("shutdown closes open implicit runs with process_exit before draining", async () => {
    const { dt, events } = client();
    const openai = wrapOpenAIClient(fakeOpenAI());
    await openai.chat.completions.create({ model: "gpt-4o" });
    await dt.shutdown();
    expect(completed(events).map((e) => e.payload["exit_reason"])).toEqual(["process_exit"]);
    expect(dt._implicitOpenCount()).toBe(0);
  });

  it("the exit flush closes them too", async () => {
    const { dt, events } = client();
    const openai = wrapOpenAIClient(fakeOpenAI());
    await openai.chat.completions.create({ model: "gpt-4o" });
    dt._flushBeforeExit();
    expect(completed(events).map((e) => e.payload["exit_reason"])).toEqual(["process_exit"]);
    await dt.shutdown();
  });

  it("can be turned off, by option or by env, and the call still works", async () => {
    const byOption = client({ implicitRuns: false });
    const openai = wrapOpenAIClient(fakeOpenAI());
    const resp = await openai.chat.completions.create({ model: "gpt-4o" });
    expect(resp.choices[0].message.content).toBe("hi");
    expect(byOption.events).toHaveLength(0);
    expect(resolveRun("x")).toBeNull();
    await byOption.dt.shutdown();

    process.env.DUNETRACE_IMPLICIT_RUNS = "0";
    const byEnv = client();
    await openai.chat.completions.create({ model: "gpt-4o" });
    expect(byEnv.events).toHaveLength(0);
    await byEnv.dt.shutdown();
  });

  it("agent id comes from the option, then the env, then the script name", async () => {
    process.env.DUNETRACE_AGENT_ID = "from-env";
    const a = client();
    resolveRun("x");
    expect(started(a.events)[0].agent_id).toBe("from-env");
    await a.dt.shutdown();

    const b = client({ defaultAgentId: "from-option" });
    resolveRun("x");
    expect(started(b.events)[0].agent_id).toBe("from-option");
    await b.dt.shutdown();
  });
});

// ── Crash ────────────────────────────────────────────────────────────────────

describe("crash recording", () => {
  it("an uncaught exception errors the open implicit run", async () => {
    const { dt, events } = client();
    const openai = wrapOpenAIClient(fakeOpenAI());
    await openai.chat.completions.create({ model: "gpt-4o" });
    const err = new Error("boom");
    expect(dt._recordCrash(err)).toBe(1);
    expect(errored(events).map((e) => e.payload["error_type"])).toEqual(["Error"]);
    expect(errored(events)[0].payload["error"]).toBe("Error: boom");
    // Recorded once: the same error reaching the hook again is ignored.
    expect(dt._recordCrash(err)).toBe(0);
    await dt.shutdown();
  });

  it("with no run open, records a one-event run", async () => {
    const { dt, events } = client();
    expect(dt._recordCrash(new TypeError("bad"))).toBe(1);
    expect(events.map((e) => e.event_type)).toEqual(["run.started", "run.errored"]);
    expect(events[0].payload["opened_by"]).toBe("uncaughtException");
    expect(events[0].payload["implicit"]).toBe(true);
    await dt.shutdown();
  });

  it("an error dt.run() already recorded is not recorded twice", async () => {
    const { dt, events } = client();
    const err = new Error("inside");
    await expect(dt.run("a", { model: "m" }, async () => { throw err; })).rejects.toBe(err);
    expect(errored(events)).toHaveLength(1);
    expect(dt._recordCrash(err)).toBe(0);
    expect(errored(events)).toHaveLength(1);
    await dt.shutdown();
  });
});

// ── Vercel AI entry points ───────────────────────────────────────────────────

function fakeStep(overrides: Record<string, unknown> = {}) {
  return {
    content: [], text: "answer", reasoning: [], files: [], sources: [],
    toolCalls: [], staticToolCalls: [], dynamicToolCalls: [],
    toolResults: [], staticToolResults: [], dynamicToolResults: [],
    finishReason: "stop", usage: { inputTokens: 5, outputTokens: 7 },
    ...overrides,
  };
}

describe("Vercel AI entry points", () => {
  it("wrapGenerateText opens an exact run around the call and closes it on return", async () => {
    const { dt, events } = client({ defaultAgentId: "assistant" });
    const generateText = wrapGenerateText((async (opts: { onStepEnd?: (s: unknown) => unknown }) => {
      await opts.onStepEnd?.(fakeStep());
      return { text: "answer" };
    }) as never);
    const result = await generateText({ model: "gpt-4o" } as never);
    expect((result as { text: string }).text).toBe("answer");

    const runs = started(events);
    expect(runs).toHaveLength(1);
    expect(runs[0].agent_id).toBe("assistant");
    expect(runs[0].payload["opened_by"]).toBe("ai.generateText");
    expect(runs[0].payload["implicit"]).toBeUndefined();
    expect(events.map((e) => e.event_type)).toEqual(["run.started", "llm.called", "llm.responded", "run.completed"]);
    expect(completed(events)[0].payload["exit_reason"]).toBe("final_answer");
    expect(getCurrentRun()).toBeNull();
    await dt.shutdown();
  });

  it("wrapGenerateText inside a declared run attaches to it", async () => {
    const { dt, events } = client();
    const generateText = wrapGenerateText((async (opts: { onStepEnd?: (s: unknown) => unknown }) => {
      await opts.onStepEnd?.(fakeStep());
      return {};
    }) as never);
    await dt.run("outer", { model: "gpt-4o" }, async () => { await generateText({ model: "gpt-4o" } as never); });
    expect(started(events).map((e) => e.agent_id)).toEqual(["outer"]);
    await dt.shutdown();
  });

  it("wrapGenerateText errors the run when the call throws", async () => {
    const { dt, events } = client();
    const generateText = wrapGenerateText((async () => { throw new Error("provider down"); }) as never);
    await expect(generateText({ model: "gpt-4o" } as never)).rejects.toThrow("provider down");
    expect(errored(events)).toHaveLength(1);
    expect(errored(events)[0].payload["error"]).toBe("Error: provider down");
    await dt.shutdown();
  });

  it("wrapStreamText closes the run from onFinish, after the caller's own hook", async () => {
    const { dt, events } = client();
    const order: string[] = [];
    const streamText = wrapStreamText(((opts: { onFinish?: (e: unknown) => unknown; onStepEnd?: (s: unknown) => unknown }) => {
      // Simulate the SDK finishing the stream later.
      setTimeout(async () => { await opts.onStepEnd?.(fakeStep()); await opts.onFinish?.({}); }, 10);
      return { textStream: [] };
    }) as never);
    streamText({ model: "gpt-4o", onFinish: async () => { order.push("user"); } } as never);
    expect(completed(events)).toHaveLength(0);
    await sleep(40);
    expect(order).toEqual(["user"]);
    expect(completed(events)).toHaveLength(1);
    expect(started(events)[0].payload["opened_by"]).toBe("ai.streamText");
    await dt.shutdown();
  });

  it("openEntryRun returns null inside a declared run and closes an implicit one", async () => {
    const { dt, events } = client();
    await dt.run("a", { model: "m" }, async () => { expect(openEntryRun("x")).toBeNull(); });
    resolveRun("openai.chat.completions.create");
    const handle = openEntryRun("ai.generateText")!;
    expect(handle).not.toBeNull();
    expect(completed(events).map((e) => e.payload["exit_reason"])).toEqual([expect.any(String), "explicit_run_opened"]);
    runStorage.run(handle.run, () => handle.finish());
    await dt.shutdown();
  });
});
