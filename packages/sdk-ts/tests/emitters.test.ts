/**
 * Tests for DurableRetryEmitter — the disk-backed queue wrapping any
 * BatchEmitter so failed batches survive a backend outage across process
 * restarts, rather than being dropped. Ported from
 * packages/sdk-py/tests/test_durable_retry_emitter.py — same coverage,
 * adapted for ship() being async here (Node's fetch has no synchronous form).
 *
 * No network required. Uses a real SQLite file per test (a temp path), not a
 * mock — this is exactly the kind of persistence-correctness logic that's
 * worth verifying against the real thing.
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  DurableRetryEmitter,
  DEFAULT_QUEUE_PATH,
  HttpBatchEmitter,
  NoopBatchEmitter,
  type BatchEmitter,
} from "../src/emitters.js";
import type { AgentEvent } from "../src/models.js";

function event(runId = "run-1"): AgentEvent {
  return {
    event_type: "run.started",
    run_id: runId,
    agent_id: "agent-1",
    agent_version: "v1",
    step_index: 0,
    timestamp: Date.now() / 1000,
    payload: { k: "v" },
  };
}

/** Returns canned true/false results in order; records every batch it saw. */
class ScriptedEmitter implements BatchEmitter {
  private _results: boolean[];
  received: string[][] = [];

  constructor(results: boolean[]) {
    this._results = [...results];
  }

  async ship(batch: AgentEvent[]): Promise<boolean> {
    this.received.push(batch.map((e) => e.run_id));
    if (this._results.length === 0) return true;
    return this._results.shift()!;
  }
}

class AlwaysFails implements BatchEmitter {
  callCount = 0;
  async ship(_batch: AgentEvent[]): Promise<boolean> {
    this.callCount += 1;
    return false;
  }
}

class AlwaysSucceeds implements BatchEmitter {
  received: AgentEvent[] = [];
  async ship(batch: AgentEvent[]): Promise<boolean> {
    this.received.push(...batch);
    return true;
  }
}

let queuePath: string;

beforeEach(() => {
  queuePath = path.join(os.tmpdir(), `dunetrace-test-queue-${process.pid}-${Math.random()}.db`);
});

afterEach(() => {
  for (const p of [queuePath, queuePath + "-journal", queuePath + "-wal", queuePath + "-shm"]) {
    if (fs.existsSync(p)) fs.unlinkSync(p);
  }
});

function rowCount(dbPath: string): number {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const Database = require("better-sqlite3");
  const db = new Database(dbPath);
  try {
    return (db.prepare("SELECT COUNT(*) as c FROM queue").get() as { c: number }).c;
  } finally {
    db.close();
  }
}

// ── Basic ship() delegation and queuing on failure ──────────────────────────

describe("DurableRetryEmitter — ship() delegation", () => {
  it("passes through inner success", async () => {
    const inner = new AlwaysSucceeds();
    const emitter = new DurableRetryEmitter(inner, { queuePath });
    const result = await emitter.ship([event()]);
    expect(result).toBe(true);
    expect(inner.received.length).toBe(1);
    expect(rowCount(queuePath)).toBe(0); // nothing queued — it succeeded
  });

  it("queues inner failure and ship() still resolves true", async () => {
    // Once durably queued, ship() reports success — the caller's ring buffer
    // doesn't need to hold onto it anymore.
    const inner = new AlwaysFails();
    const emitter = new DurableRetryEmitter(inner, { queuePath });
    const result = await emitter.ship([event()]);
    expect(result).toBe(true);
    expect(rowCount(queuePath)).toBe(1);
  });

  it("persists the queue across emitter instances (simulated restart)", async () => {
    const inner1 = new AlwaysFails();
    const emitter1 = new DurableRetryEmitter(inner1, { queuePath });
    await emitter1.ship([event("r1")]);
    expect(rowCount(queuePath)).toBe(1);

    const inner2 = new AlwaysSucceeds();
    const emitter2 = new DurableRetryEmitter(inner2, { queuePath });
    (emitter2 as any)._nextRetryAt = 0; // force the backlog check to run immediately
    await emitter2.ship([event("r2")]);

    expect(rowCount(queuePath)).toBe(0); // backlog drained
    const runIds = inner2.received.map((e) => e.run_id);
    expect(runIds).toContain("r1");
    expect(runIds).toContain("r2");
  });
});

// ── Backlog retry ordering and cadence ──────────────────────────────────────

describe("DurableRetryEmitter — backlog retry", () => {
  it("drains the backlog oldest first", async () => {
    const inner = new AlwaysFails();
    const emitter = new DurableRetryEmitter(inner, { queuePath });
    await emitter.ship([event("r1")]);
    await emitter.ship([event("r2")]);
    await emitter.ship([event("r3")]);
    expect(rowCount(queuePath)).toBe(3);

    const recording = new ScriptedEmitter([true, true, true]);
    (emitter as any)._inner = recording;
    (emitter as any)._nextRetryAt = 0;
    await emitter.ship([event("r4")]); // triggers backlog drain, then ships r4

    // First three ship() calls on the inner emitter are the backlog, in order.
    expect(recording.received[0]).toEqual(["r1"]);
    expect(recording.received[1]).toEqual(["r2"]);
    expect(recording.received[2]).toEqual(["r3"]);
    expect(rowCount(queuePath)).toBe(0);
  });

  it("stops retrying at the first failure, preserving order", async () => {
    const inner = new AlwaysFails();
    const emitter = new DurableRetryEmitter(inner, { queuePath });
    await emitter.ship([event("r1")]);
    await emitter.ship([event("r2")]);
    expect(rowCount(queuePath)).toBe(2);

    // r1 (backlog) succeeds, r2 (backlog) fails again -> retry stops, doesn't
    // skip ahead. r3 (the new ship() call's own batch) also fails -> queued.
    const recording = new ScriptedEmitter([true, false, false]);
    (emitter as any)._inner = recording;
    (emitter as any)._nextRetryAt = 0;
    await emitter.ship([event("r3")]);

    expect(rowCount(queuePath)).toBe(2); // r2 (still queued) and r3 (newly queued)
  });

  it("does not retry before the interval elapses", async () => {
    const inner = new AlwaysFails();
    const emitter = new DurableRetryEmitter(inner, {
      queuePath,
      retryIntervalMs: 30_000,
      retryJitterMs: 5_000,
    });
    await emitter.ship([event("r1")]);
    expect(rowCount(queuePath)).toBe(1);

    const recording = new AlwaysSucceeds();
    (emitter as any)._inner = recording;
    // _nextRetryAt was set in the future by the first ship() call — don't reset it.
    await emitter.ship([event("r2")]);

    // r1 must still be queued — the retry interval hasn't elapsed.
    expect(rowCount(queuePath)).toBe(1);
    const runIds = recording.received.map((e) => e.run_id);
    expect(runIds).toEqual(["r2"]);
  });

  it("jitters the next retry time around the configured interval", async () => {
    const emitter = new DurableRetryEmitter(new AlwaysSucceeds(), {
      queuePath,
      retryIntervalMs: 30_000,
      retryJitterMs: 5_000,
    });
    const before = Number(process.hrtime.bigint() / 1_000_000n);
    await emitter.ship([event()]);
    const after = Number(process.hrtime.bigint() / 1_000_000n);
    // nextRetryAt should land within [25s, 35s] from "now"
    expect((emitter as any)._nextRetryAt).toBeGreaterThanOrEqual(before + 25_000);
    expect((emitter as any)._nextRetryAt).toBeLessThanOrEqual(after + 35_000);
  });
});

// ── Bounded queue + eviction ─────────────────────────────────────────────────

describe("DurableRetryEmitter — bounded queue eviction", () => {
  it("evicts the oldest batch when the event cap is exceeded", async () => {
    const inner = new AlwaysFails();
    const emitter = new DurableRetryEmitter(inner, { queuePath, maxQueueEvents: 2 });
    await emitter.ship([event("r1")]);
    await emitter.ship([event("r2")]);
    await emitter.ship([event("r3")]); // should evict r1

    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const Database = require("better-sqlite3");
    const db = new Database(queuePath);
    const payloads = (
      db.prepare("SELECT payload FROM queue ORDER BY id").all() as { payload: string }[]
    ).map((r) => r.payload);
    db.close();

    expect(payloads.length).toBe(2);
    expect(payloads[0]).not.toContain('"r1"');
    expect(payloads[0]).toContain("r2");
    expect(payloads[1]).toContain("r3");
  });

  it("evicts the oldest batch when the byte cap is exceeded", async () => {
    const oneBatchSize = Buffer.byteLength(JSON.stringify([event("r1")]), "utf8");
    const inner = new AlwaysFails();
    // Cap large enough for exactly one batch, too small for two.
    const emitter = new DurableRetryEmitter(inner, {
      queuePath,
      maxQueueBytes: oneBatchSize + 10,
    });
    await emitter.ship([event("r1")]);
    expect(rowCount(queuePath)).toBe(1);
    await emitter.ship([event("r2")]);
    expect(rowCount(queuePath)).toBe(1); // r1 evicted, only r2 (the newest) survives
  });

  it("logs a warning on eviction", async () => {
    const inner = new AlwaysFails();
    const emitter = new DurableRetryEmitter(inner, { queuePath, maxQueueEvents: 1 });
    await emitter.ship([event("r1")]);
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    await emitter.ship([event("r2")]);
    expect(warnSpy).toHaveBeenCalled();
    expect(warnSpy.mock.calls.some((c) => String(c[0]).includes("evicted"))).toBe(true);
    warnSpy.mockRestore();
  });

  it("still warns when hrtime starts near zero (freshly started process)", async () => {
    // Regression for the same bug class fixed in emitters.py this sprint:
    // comparing "now" against a 0 sentinel for "never warned" silently
    // suppresses the very first warning when the clock itself starts near
    // zero — which for process.hrtime.bigint() is the *normal* case at
    // process start, not a rare edge case like Python's boot-time uptime.
    const realHrtime = process.hrtime.bigint;
    let callCount = 0;
    vi.spyOn(process.hrtime, "bigint").mockImplementation(() => {
      callCount += 1;
      return BigInt(5_000_000); // 5ms since an arbitrary origin — deliberately tiny
    });
    try {
      const inner = new AlwaysFails();
      const emitter = new DurableRetryEmitter(inner, { queuePath, maxQueueEvents: 1 });
      await emitter.ship([event("r1")]);
      const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
      await emitter.ship([event("r2")]);
      expect(warnSpy.mock.calls.some((c) => String(c[0]).includes("evicted"))).toBe(true);
      warnSpy.mockRestore();
    } finally {
      process.hrtime.bigint = realHrtime;
      expect(callCount).toBeGreaterThan(0);
    }
  });

  it("rate-limits the eviction warning to once per minute", async () => {
    const inner = new AlwaysFails();
    const emitter = new DurableRetryEmitter(inner, { queuePath, maxQueueEvents: 1 });
    await emitter.ship([event("r1")]); // fills the single slot, no eviction yet

    // Simulate the rate-limit window already being "fresh" (just warned).
    (emitter as any)._lastEvictionWarningNs = process.hrtime.bigint();

    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    await emitter.ship([event("r2")]); // evicts r1 — but warning is rate-limited
    expect(warnSpy.mock.calls.some((c) => String(c[0]).includes("evicted"))).toBe(false);
    warnSpy.mockRestore();

    // The eviction still happened even though the warning was suppressed.
    expect(rowCount(queuePath)).toBe(1);
  });
});

// ── Graceful degradation ─────────────────────────────────────────────────────

describe("DurableRetryEmitter — graceful degradation", () => {
  it("does not crash when the queue path is unwritable", async () => {
    const inner = new AlwaysFails();
    const emitter = new DurableRetryEmitter(inner, {
      queuePath: "/nonexistent-root-xyz/queue.db",
    });
    expect((emitter as any)._dbOk).toBe(false);
    const result = await emitter.ship([event()]);
    expect(result).toBe(false); // can't queue, can't deliver — honest failure, no crash
  });
});

// ── Path resolution ───────────────────────────────────────────────────────────

describe("DurableRetryEmitter — queue path resolution", () => {
  it("prefers the explicit constructor path", () => {
    const emitter = new DurableRetryEmitter(new AlwaysSucceeds(), { queuePath });
    expect((emitter as any)._path).toBe(queuePath);
  });

  it("falls back to DUNETRACE_QUEUE_PATH when no explicit path is given", () => {
    const prev = process.env.DUNETRACE_QUEUE_PATH;
    process.env.DUNETRACE_QUEUE_PATH = queuePath;
    try {
      const emitter = new DurableRetryEmitter(new AlwaysSucceeds());
      expect((emitter as any)._path).toBe(queuePath);
    } finally {
      if (prev === undefined) delete process.env.DUNETRACE_QUEUE_PATH;
      else process.env.DUNETRACE_QUEUE_PATH = prev;
    }
  });

  it("defaults to ~/.dunetrace/queue-ts.db", () => {
    expect(DEFAULT_QUEUE_PATH).toBe(path.join(os.homedir(), ".dunetrace", "queue-ts.db"));
  });

  it("uses a filename distinct from the Python SDK's queue.db", () => {
    // The whole point of the distinct name — a mixed Python+TS deployment
    // sharing ~/.dunetrace/ must not have one SDK's queue corrupt the other's.
    expect(path.basename(DEFAULT_QUEUE_PATH)).not.toBe("queue.db");
  });
});

// ── Missing optional dependency (TS-specific — Python has no equivalent   ──
// since sqlite3 is stdlib there, not an optional peer dependency here)     ──

describe("DurableRetryEmitter — missing better-sqlite3", () => {
  it("degrades gracefully when better-sqlite3 cannot be required", async () => {
    // Patching the resolved better-sqlite3 module's own export to throw when
    // constructed reaches the same code path (_initDb's try/catch around the
    // require + `new Database(...)` call) as the module being entirely
    // absent, without fighting Vite's module graph over what "not installed"
    // means under its own SSR transform.
    const betterSqlite3 = require("better-sqlite3");
    const realDefault = betterSqlite3;
    const throwingCtor = function () {
      throw new Error("Cannot find module 'better-sqlite3'");
    };
    require.cache[require.resolve("better-sqlite3")]!.exports = throwingCtor;
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const inner = new AlwaysFails();
      const emitter = new DurableRetryEmitter(inner, { queuePath });
      expect((emitter as any)._dbOk).toBe(false);
      const result = await emitter.ship([event()]);
      expect(result).toBe(false); // honest failure, no crash
      expect(
        warnSpy.mock.calls.some((c) => String(c[0]).toLowerCase().includes("better-sqlite3")),
      ).toBe(true);
    } finally {
      require.cache[require.resolve("better-sqlite3")]!.exports = realDefault;
      warnSpy.mockRestore();
    }
  });
});

// ── HttpBatchEmitter / NoopBatchEmitter — needed as the "inner" for the   ──
// tests above to be meaningful, and to confirm the extraction from        ──
// client.ts's old private _ship() preserved behavior.                     ──

describe("HttpBatchEmitter", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("resolves true on a 2xx response", async () => {
    vi.stubGlobal("fetch", async () => new Response("{}", { status: 200 }));
    const emitter = new HttpBatchEmitter("http://localhost:8001", "key");
    expect(await emitter.ship([event()])).toBe(true);
  });

  it("resolves false on a non-2xx response", async () => {
    vi.stubGlobal("fetch", async () => new Response("{}", { status: 500 }));
    const emitter = new HttpBatchEmitter("http://localhost:8001", "key");
    expect(await emitter.ship([event()])).toBe(false);
  });

  it("resolves false, never throws, on a network error", async () => {
    vi.stubGlobal("fetch", async () => {
      throw new Error("network error");
    });
    const emitter = new HttpBatchEmitter("http://localhost:8001", "key");
    await expect(emitter.ship([event()])).resolves.toBe(false);
  });
});

describe("NoopBatchEmitter", () => {
  it("always resolves true and ships nowhere", async () => {
    const emitter = new NoopBatchEmitter();
    expect(await emitter.ship([event()])).toBe(true);
  });
});
