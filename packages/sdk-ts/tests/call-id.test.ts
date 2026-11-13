/**
 * `call_id` — pairing llm.called with its llm.responded.
 *
 * Ordering cannot do it. A streamed response lands whenever the *caller* drains
 * the stream, so two overlapping streams emit
 *     called(A), called(B), responded(A), responded(B)
 * and positional pairing hands B's response to A — swapping model against
 * tokens, latency and cost. The Python SDK stamps a per-run sequence number on
 * both payloads (run_context.py) and the shared server-side builder
 * (run_builder.py) pairs on it; this is the TypeScript half of that wire
 * contract.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { Dunetrace } from "../src/client.js";
import { DunetraceRun } from "../src/run.js";
import { autoInstrument, _resetAutoInstrumentState } from "../src/auto.js";
import type { AgentEvent } from "../src/models.js";

function makeRun() {
  const emitted: AgentEvent[] = [];
  const run = new DunetraceRun("agent", "v1", { _emit: (e) => emitted.push(e) });
  return { run, emitted };
}

describe("DunetraceRun — call_id on the wire", () => {
  it("numbers calls from 0, in call order", () => {
    const { run, emitted } = makeRun();
    run.llmCalled("gpt-4o");
    run.llmCalled("claude-opus-4");
    run.llmCalled("mistral-large");
    expect(emitted.map((e) => e.payload["call_id"])).toEqual([0, 1, 2]);
  });

  it("returns the id from llmCalled so a caller can hold it", () => {
    const { run } = makeRun();
    expect(run.llmCalled("gpt-4o")).toBe(0);
    expect(run.llmCalled("gpt-4o")).toBe(1);
  });

  it("echoes the most recent call's id when called/responded are adjacent", () => {
    const { run, emitted } = makeRun();
    run.llmCalled("gpt-4o");
    run.llmResponded({ completionTokens: 5 });
    expect(emitted[0].payload["call_id"]).toBe(0);
    expect(emitted[1].payload["call_id"]).toBe(0);
  });

  it("honours an explicit callId over the most recent call", () => {
    const { run, emitted } = makeRun();
    const a = run.llmCalled("model-a");
    const b = run.llmCalled("model-b");
    run.llmResponded({ completionTokens: 1, callId: a });
    run.llmResponded({ completionTokens: 2, callId: b });

    const responses = emitted.filter((e) => e.event_type === "llm.responded");
    expect(responses.map((e) => e.payload["call_id"])).toEqual([0, 1]);
    // The wrong answer this exists to prevent: both responses keyed to call 1.
    expect(responses.map((e) => e.payload["completion_tokens"])).toEqual([1, 2]);
  });

  it("omits call_id when there is no call to name", () => {
    const { run, emitted } = makeRun();
    run.llmResponded({ completionTokens: 3 });
    expect(emitted[0].payload).not.toHaveProperty("call_id");
  });

  it("omits call_id rather than pointing at a different call when the id is out of range", () => {
    // An out-of-range id means llm.called never landed (its emit was swallowed).
    const { run, emitted } = makeRun();
    run.llmCalled("gpt-4o");
    run.llmResponded({ callId: 99 });
    expect(emitted[1].payload).not.toHaveProperty("call_id");
  });

  it("run.llm() threads the id through its own called/responded pair", async () => {
    const { run, emitted } = makeRun();
    await run.llm("gpt-4o", Promise.resolve({
      choices: [{ message: { content: "hi" }, finish_reason: "stop" }],
      usage: { prompt_tokens: 4, completion_tokens: 2 },
    }));
    expect(emitted[0].payload["call_id"]).toBe(0);
    expect(emitted[1].payload["call_id"]).toBe(0);
  });
});

// ── Overlapping streams, through the real auto-instrumentation ────────────────

/** A stream the test drives chunk by chunk, so two can be interleaved. */
function manualStream(chunks: unknown[]) {
  let resolveGate: (() => void) | null = null;
  const gate = new Promise<void>((r) => { resolveGate = r; });
  return {
    release: () => resolveGate?.(),
    stream: {
      async *[Symbol.asyncIterator]() {
        await gate;
        for (const c of chunks) yield c;
      },
    },
  };
}

const captured: AgentEvent[] = [];

class InterleavedCompletions {
  static pending: Array<{ release: () => void; stream: AsyncIterable<unknown> }> = [];
  async create(opts: Record<string, unknown>): Promise<unknown> {
    const tokens = opts["__tokens"] as number;
    const handle = manualStream([
      { choices: [{ delta: { content: `out-${tokens}` } }] },
      { choices: [{ delta: {}, finish_reason: "stop" }] },
      { choices: [], usage: { prompt_tokens: tokens, completion_tokens: tokens } },
    ]);
    InterleavedCompletions.pending.push(handle);
    return handle.stream;
  }
}
class InterleavedOpenAI {
  static Chat = { Completions: InterleavedCompletions };
  chat = { completions: new InterleavedCompletions() };
}

describe("overlapping streams get their own responses", () => {
  beforeEach(() => {
    captured.length = 0;
    InterleavedCompletions.pending.length = 0;
    _resetAutoInstrumentState();
  });
  afterEach(() => { vi.restoreAllMocks(); });

  it("called(A), called(B), responded(A), responded(B) stays correctly paired", async () => {
    const original = InterleavedCompletions.prototype.create;
    try {
      const dt = new Dunetrace({
        exporter: { handle: (e: AgentEvent) => { captured.push(e); } },
      });
      autoInstrument({ openai: InterleavedOpenAI, targets: ["openai"] });

      await dt.run("agent", {}, async () => {
        const client = new InterleavedOpenAI();
        // Both calls start before either response is known — the exact shape
        // that defeats ordering.
        const a = await client.chat.completions.create({
          model: "model-a", stream: true, __tokens: 11,
        }) as AsyncIterable<unknown>;
        const b = await client.chat.completions.create({
          model: "model-b", stream: true, __tokens: 22,
        }) as AsyncIterable<unknown>;

        // Drain A first, then B — responses land in call order here, but the
        // point is that each one names its own call rather than relying on it.
        InterleavedCompletions.pending[0].release();
        for await (const _ of a) { /* drain A */ }
        InterleavedCompletions.pending[1].release();
        for await (const _ of b) { /* drain B */ }
      });

      const called = captured.filter((e) => e.event_type === "llm.called");
      const responded = captured.filter((e) => e.event_type === "llm.responded");
      expect(called.map((e) => e.payload["model"])).toEqual(["model-a", "model-b"]);
      expect(called.map((e) => e.payload["call_id"])).toEqual([0, 1]);

      // Each response carries the id of the call it actually belongs to, so a
      // server-side builder maps model-a to 11 completion tokens and model-b
      // to 22 — never the other way round.
      const byCallId = new Map(responded.map((e) => [e.payload["call_id"], e.payload]));
      expect(byCallId.get(0)?.["completion_tokens"]).toBe(11);
      expect(byCallId.get(1)?.["completion_tokens"]).toBe(22);
    } finally {
      (InterleavedCompletions.prototype as Record<string, unknown>)["create"] = original;
    }
  });

  it("a response drained out of order still names its own call", async () => {
    const original = InterleavedCompletions.prototype.create;
    try {
      const dt = new Dunetrace({
        exporter: { handle: (e: AgentEvent) => { captured.push(e); } },
      });
      autoInstrument({ openai: InterleavedOpenAI, targets: ["openai"] });

      await dt.run("agent", {}, async () => {
        const client = new InterleavedOpenAI();
        const a = await client.chat.completions.create({
          model: "model-a", stream: true, __tokens: 11,
        }) as AsyncIterable<unknown>;
        const b = await client.chat.completions.create({
          model: "model-b", stream: true, __tokens: 22,
        }) as AsyncIterable<unknown>;

        // B finishes FIRST — with positional pairing, B's 22 tokens would be
        // back-filled onto model-a.
        InterleavedCompletions.pending[1].release();
        for await (const _ of b) { /* drain B */ }
        InterleavedCompletions.pending[0].release();
        for await (const _ of a) { /* drain A */ }
      });

      const responded = captured.filter((e) => e.event_type === "llm.responded");
      expect(responded[0].payload["call_id"]).toBe(1);
      expect(responded[0].payload["completion_tokens"]).toBe(22);
      expect(responded[1].payload["call_id"]).toBe(0);
      expect(responded[1].payload["completion_tokens"]).toBe(11);
    } finally {
      (InterleavedCompletions.prototype as Record<string, unknown>)["create"] = original;
    }
  });
});
