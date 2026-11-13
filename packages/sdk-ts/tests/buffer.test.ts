/**
 * Run-aware outbound buffer.
 *
 * The contract, ported from packages/sdk-py/dunetrace/buffer.py:
 *   1. run terminals are NEVER shed,
 *   2. a full buffer sheds the OLDEST WHOLE RUN, not the newest event,
 *   3. the shed run's terminal carries `dropped_events: <n>`,
 *   4. one rate-limited warning, not one per dropped event.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { EventBuffer } from "../src/buffer.js";
import { Dunetrace } from "../src/client.js";
import type { AgentEvent, EventType } from "../src/models.js";

let seq = 0;
function ev(runId: string, type: EventType = "tool.called"): AgentEvent {
  return {
    event_type: type,
    run_id: runId,
    agent_id: "agent",
    agent_version: "v1",
    step_index: seq++,
    timestamp: Date.now() / 1000,
    payload: { n: seq },
  };
}

const runIds = (batch: AgentEvent[]): string[] => batch.map((e) => e.run_id);

// Shedding is meant to warn; keep it out of the test runner's stderr and make
// the warning itself assertable.
let warn: ReturnType<typeof vi.spyOn>;
beforeEach(() => { warn = vi.spyOn(console, "warn").mockImplementation(() => {}); });
afterEach(() => { warn.mockRestore(); });

describe("EventBuffer — shedding the oldest run", () => {
  it("sheds every buffered event of the oldest run, not just one", () => {
    const buf = new EventBuffer(6);
    for (let i = 0; i < 3; i++) buf.push(ev("a"));
    for (let i = 0; i < 3; i++) buf.push(ev("b"));
    expect(buf.length).toBe(6);

    buf.push(ev("c")); // full — run "a" goes, all three of its events
    expect(buf.shedRunsTotal).toBe(1);
    expect(buf.droppedCount("a")).toBe(3);
    expect(runIds(buf.drainAll())).toEqual(["b", "b", "b", "c"]);
  });

  it("keeps the surviving runs WHOLE — no run is left half-buffered", () => {
    const buf = new EventBuffer(4);
    buf.push(ev("old"));
    buf.push(ev("old"));
    buf.push(ev("keep"));
    buf.push(ev("keep"));
    buf.push(ev("keep")); // sheds "old" entirely to make room

    const drained = buf.drainAll();
    expect(runIds(drained)).toEqual(["keep", "keep", "keep"]);
    expect(drained.every((e) => e.run_id === "keep")).toBe(true);
  });

  it("counts later events of an already-shed run instead of buffering them", () => {
    const buf = new EventBuffer(2);
    buf.push(ev("a"));
    buf.push(ev("a"));
    buf.push(ev("b")); // sheds "a" (2 events)
    expect(buf.droppedCount("a")).toBe(2);

    expect(buf.push(ev("a"))).toBe(false); // a straggler from the cut run
    expect(buf.push(ev("a"))).toBe(false);
    expect(buf.droppedCount("a")).toBe(4);
    expect(buf.length).toBe(1);
  });

  it("sheds the requesting run whole when only terminals remain", () => {
    const buf = new EventBuffer(2);
    buf.push(ev("x", "run.completed"));
    buf.push(ev("y", "run.errored"));
    expect(buf.length).toBe(2);

    // Nothing sheddable is left; the new run is cut whole rather than
    // piecemeal, and both terminals survive.
    expect(buf.push(ev("z"))).toBe(false);
    expect(buf.droppedCount("z")).toBe(1);
    expect(runIds(buf.drainAll())).toEqual(["x", "y"]);
  });
});

describe("EventBuffer — terminals are never shed", () => {
  it("enqueues a terminal even when the buffer is full of terminals", () => {
    const buf = new EventBuffer(2);
    buf.push(ev("a", "run.completed"));
    buf.push(ev("b", "run.completed"));
    expect(buf.push(ev("c", "run.errored"))).toBe(true);
    expect(runIds(buf.drainAll())).toEqual(["a", "b", "c"]);
  });

  it("enqueues a terminal for a run that was itself shed", () => {
    const buf = new EventBuffer(2);
    buf.push(ev("a"));
    buf.push(ev("a"));
    buf.push(ev("b")); // sheds "a"

    const terminal = ev("a", "run.completed");
    expect(buf.push(terminal)).toBe(true);
    const drained = buf.drainAll();
    expect(drained.map((e) => e.event_type)).toContain("run.completed");
  });

  it("never sheds a terminal to make room for a non-terminal", () => {
    const buf = new EventBuffer(3);
    buf.push(ev("done", "run.completed"));
    for (let i = 0; i < 10; i++) buf.push(ev(`r${i}`));
    const drained = buf.drainAll();
    expect(drained[0].event_type).toBe("run.completed");
    expect(drained[0].run_id).toBe("done");
  });
});

describe("EventBuffer — dropped_events on the terminal", () => {
  it("stamps the drop count at drain even when the client stamped nothing", () => {
    const buf = new EventBuffer(2);
    buf.push(ev("a"));
    buf.push(ev("a"));
    const terminal = ev("a", "run.completed");
    buf.push(terminal); // terminals are forced through
    buf.push(ev("b")); // full of one terminal + 2 live "a" events -> sheds "a"

    const drained = buf.drainAll();
    const out = drained.find((e) => e.event_type === "run.completed")!;
    expect(out.payload["dropped_events"]).toBe(2);
  });

  it("reserveTerminal makes room first and reports the run's loss", () => {
    const buf = new EventBuffer(4);
    for (let i = 0; i < 4; i++) buf.push(ev("a"));
    buf.push(ev("b")); // sheds "a" (4 events)

    // What the client does just before building run.completed.
    const dropped = buf.reserveTerminal("a");
    expect(dropped).toBe(4);

    const terminal = ev("a", "run.completed");
    terminal.payload["dropped_events"] = dropped;
    buf.push(terminal);
    const out = buf.drainAll().find((e) => e.event_type === "run.completed")!;
    expect(out.payload["dropped_events"]).toBe(4);
  });

  it("reserveTerminal returns 0 for a run nothing was dropped from", () => {
    const buf = new EventBuffer(10);
    buf.push(ev("clean"));
    expect(buf.reserveTerminal("clean")).toBe(0);
  });

  it("never lowers a count the client already stamped", () => {
    const buf = new EventBuffer(10);
    const terminal = ev("a", "run.completed");
    terminal.payload["dropped_events"] = 7;
    buf.push(terminal);
    const out = buf.drainAll()[0];
    expect(out.payload["dropped_events"]).toBe(7);
  });
});

describe("EventBuffer — accounting stays consistent", () => {
  it("length never exceeds capacity while non-terminals keep arriving", () => {
    const buf = new EventBuffer(20);
    for (let i = 0; i < 500; i++) {
      buf.push(ev(`run-${i % 37}`));
      expect(buf.length).toBeLessThanOrEqual(20);
    }
  });

  it("drains exactly what it holds, with no tombstones leaking out", () => {
    const buf = new EventBuffer(10);
    for (let i = 0; i < 400; i++) buf.push(ev(`run-${i % 13}`));
    const held = buf.length;
    const drained = buf.drainAll();
    expect(drained.length).toBe(held);
    expect(buf.length).toBe(0);
    expect(buf.drainAll()).toEqual([]);
    // No shed run's non-terminal events made it into the batch.
    for (const e of drained) expect(buf.droppedCount(e.run_id)).toBe(0);
  });

  it("compacts rather than growing the physical array without bound", () => {
    const buf = new EventBuffer(8);
    for (let i = 0; i < 5000; i++) buf.push(ev(`run-${i % 50}`));
    const physical = (buf as unknown as { _buf: unknown[] })._buf.length;
    expect(physical).toBeLessThanOrEqual(2 * 8 + 1);
    expect(buf.length).toBeLessThanOrEqual(8);
  });

  it("drains in FIFO order within a run", () => {
    const buf = new EventBuffer(100);
    const pushed = [ev("a"), ev("a"), ev("a")];
    for (const e of pushed) buf.push(e);
    expect(buf.drainAll().map((e) => e.step_index)).toEqual(pushed.map((e) => e.step_index));
  });

  it("partial drains leave the rest in order", () => {
    const buf = new EventBuffer(100);
    for (let i = 0; i < 10; i++) buf.push(ev("a"));
    const first = buf.drain(4);
    expect(first).toHaveLength(4);
    expect(buf.length).toBe(6);
    const rest = buf.drainAll();
    expect(rest).toHaveLength(6);
    expect(rest[0].step_index).toBeGreaterThan(first[3].step_index);
  });
});

describe("EventBuffer — warning", () => {
  it("warns once and reports the run it cut", () => {
    const buf = new EventBuffer(2);
    buf.push(ev("victim"));
    buf.push(ev("victim"));
    buf.push(ev("next"));
    expect(warn.mock.calls.some((c) => String(c[0]).includes("shed run victim"))).toBe(true);
  });

  it("rate-limits to one line a minute and reports how many were suppressed", () => {
    const buf = new EventBuffer(1);
    for (let i = 0; i < 50; i++) buf.push(ev(`run-${i}`));
    // 49 sheds, one warning: the drop rate must not become the log volume.
    expect(warn.mock.calls.length).toBe(1);
    expect(buf.shedRunsTotal).toBeGreaterThan(1);
  });

  it("emits the FIRST warning of a freshly started process (no 0 sentinel)", () => {
    // process.hrtime.bigint() starts near zero, so a `now - 0 < 60s` guard
    // would swallow the very first warning of every process's lifetime.
    const buf = new EventBuffer(1);
    buf.push(ev("a"));
    buf.push(ev("b"));
    expect(warn).toHaveBeenCalled();
  });
});

describe("Dunetrace — a run that overflows the buffer still reports itself", () => {
  it("run.completed survives and carries dropped_events", async () => {
    const dt = new Dunetrace({ bufferSize: 8, flushIntervalMs: 99_999 });
    const noisy = dt.tool(async (_i: unknown) => "ok", "noisy");

    await dt.run("agent", { model: "gpt-4o" }, async () => {
      // Far more events than the buffer can hold: the run is shed, but the
      // terminal must still get through, and it must say how much was lost.
      for (let i = 0; i < 60; i++) await noisy(i);
    });

    const buf = (dt as unknown as { _buffer: EventBuffer })._buffer;
    const drained = buf.drainAll();
    const completed = drained.find((e) => e.event_type === "run.completed");
    expect(completed, "run.completed must never be shed").toBeDefined();
    expect(completed!.payload["dropped_events"]).toBeGreaterThan(0);
  });

  it("run.errored survives too — a failed run is still reported as failed", async () => {
    const dt = new Dunetrace({ bufferSize: 4, flushIntervalMs: 99_999 });
    const noisy = dt.tool(async (_i: unknown) => "ok", "noisy");

    await expect(
      dt.run("agent", {}, async () => {
        for (let i = 0; i < 40; i++) await noisy(i);
        throw new Error("agent blew up");
      }),
    ).rejects.toThrow("agent blew up");

    const buf = (dt as unknown as { _buffer: EventBuffer })._buffer;
    const errored = buf.drainAll().find((e) => e.event_type === "run.errored");
    expect(errored, "run.errored must never be shed").toBeDefined();
    expect(errored!.payload["dropped_events"]).toBeGreaterThan(0);
  });

  it("a healthy run carries no dropped_events key at all", async () => {
    const dt = new Dunetrace({ bufferSize: 1000, flushIntervalMs: 99_999 });
    await dt.run("agent", {}, async () => {});
    const buf = (dt as unknown as { _buffer: EventBuffer })._buffer;
    const completed = buf.drainAll().find((e) => e.event_type === "run.completed")!;
    expect(completed.payload).not.toHaveProperty("dropped_events");
  });
});
