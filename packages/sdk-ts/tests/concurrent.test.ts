/**
 * T5: Concurrent run isolation
 * T6: Error scenarios (network errors, 500s, buffer cap)
 *
 * Run:
 *   cd packages/sdk-ts && npm test
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { Dunetrace, getCurrentRun } from "../src/client.js";
import type { AgentEvent } from "../src/models.js";

// ── Fetch mock helpers ────────────────────────────────────────────────────────

function mockFetchOk() {
  vi.stubGlobal("fetch", async (_url: string, _init?: RequestInit) => {
    return new Response(JSON.stringify({ ok: true }), { status: 200 });
  });
}

function mockFetchNetworkError() {
  vi.stubGlobal("fetch", async (_url: string, _init?: RequestInit) => {
    throw new Error("network error");
  });
}

function mockFetchStatus(status: number) {
  vi.stubGlobal("fetch", async (_url: string, _init?: RequestInit) => {
    return new Response(JSON.stringify({ error: "server error" }), { status });
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  mockFetchOk();
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

// ── T5: Concurrent run isolation ──────────────────────────────────────────────

describe("T5 — Concurrent run isolation", () => {
  it("each concurrent run sees its own context via getCurrentRun()", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });

    const runIds: string[] = [];
    const currentRunIds: Array<string | null> = [];

    // Start 3 runs concurrently; each captures its own getCurrentRun() result
    await Promise.all([
      dt.run("agent", {}, async (run) => {
        runIds.push(run.runId);
        // Yield to let other coroutines run
        await Promise.resolve();
        currentRunIds.push(getCurrentRun()?.runId ?? null);
      }),
      dt.run("agent", {}, async (run) => {
        runIds.push(run.runId);
        await Promise.resolve();
        currentRunIds.push(getCurrentRun()?.runId ?? null);
      }),
      dt.run("agent", {}, async (run) => {
        runIds.push(run.runId);
        await Promise.resolve();
        currentRunIds.push(getCurrentRun()?.runId ?? null);
      }),
    ]);

    // All 3 run IDs are unique
    expect(new Set(runIds).size).toBe(3);

    // Each context captured the correct run (not someone else's)
    for (let i = 0; i < 3; i++) {
      expect(currentRunIds[i]).toBe(runIds[i]);
    }

    await dt.shutdown();
  });

  it("events from each run carry only that run's run_id", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
    const allEvents: AgentEvent[] = [];
    dt._emit = (e) => allEvents.push(e);

    await Promise.all([
      dt.run("agent", {}, async (run) => {
        await Promise.resolve();
        run.llmCalled("gpt-4o");
      }),
      dt.run("agent", {}, async (run) => {
        await Promise.resolve();
        run.toolCalled("search", {});
      }),
      dt.run("agent", {}, async (run) => {
        await Promise.resolve();
        run.llmCalled("gpt-4o");
      }),
    ]);

    // Group events by run_id
    const byRunId = new Map<string, AgentEvent[]>();
    for (const e of allEvents) {
      const bucket = byRunId.get(e.run_id) ?? [];
      bucket.push(e);
      byRunId.set(e.run_id, bucket);
    }

    // We should have exactly 3 distinct run IDs
    expect(byRunId.size).toBe(3);

    // Within each run, all events share the same run_id and agent_version
    for (const [runId, events] of byRunId) {
      for (const e of events) {
        expect(e.run_id).toBe(runId);
      }
      // All events within a run should have the same agent_version
      const versions = new Set(events.map((e) => e.agent_version));
      expect(versions.size).toBe(1);
    }

    await dt.shutdown();
  });

  it("events from 3 concurrent runs do not mix tool calls across runs", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
    const allEvents: AgentEvent[] = [];
    dt._emit = (e) => allEvents.push(e);

    const runIdByTool = new Map<string, string>(); // tool_name -> run_id

    await Promise.all([
      dt.run("agent", {}, async (run) => {
        await Promise.resolve();
        run.toolCalled("tool-A", {});
        runIdByTool.set("tool-A", run.runId);
      }),
      dt.run("agent", {}, async (run) => {
        await Promise.resolve();
        run.toolCalled("tool-B", {});
        runIdByTool.set("tool-B", run.runId);
      }),
      dt.run("agent", {}, async (run) => {
        await Promise.resolve();
        run.toolCalled("tool-C", {});
        runIdByTool.set("tool-C", run.runId);
      }),
    ]);

    // Each tool.called event should have the run_id registered for that tool
    const toolCalledEvents = allEvents.filter((e) => e.event_type === "tool.called");
    for (const e of toolCalledEvents) {
      const toolName = e.payload["tool_name"] as string;
      const expectedRunId = runIdByTool.get(toolName);
      if (expectedRunId !== undefined) {
        expect(e.run_id).toBe(expectedRunId);
      }
    }

    await dt.shutdown();
  });

  it("getCurrentRun() returns null after a concurrent run completes", async () => {
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });

    await Promise.all([
      dt.run("agent", {}, async () => { await Promise.resolve(); }),
      dt.run("agent", {}, async () => { await Promise.resolve(); }),
    ]);

    // After all runs finish, the global context is null
    expect(getCurrentRun()).toBeNull();
    await dt.shutdown();
  });
});

// ── T6: Error scenarios ────────────────────────────────────────────────────────

describe("T6 — Error scenarios", () => {
  it("network error in fetch does NOT throw from run()", async () => {
    mockFetchNetworkError();
    const dt = new Dunetrace({ endpoint: "http://localhost:8001", flushIntervalMs: 60000 });

    // run() itself must complete without throwing
    await expect(
      dt.run("agent", {}, async () => { /* no-op */ }),
    ).resolves.toBeUndefined();

    // Flush manually — this triggers the network call; must not throw
    await expect(dt.flush()).resolves.toBeUndefined();
    await dt.shutdown();
  });

  it("network error is written to stderr, not thrown", async () => {
    mockFetchNetworkError();
    const stderrWrites: string[] = [];
    vi.spyOn(process.stderr, "write").mockImplementation((data: unknown) => {
      stderrWrites.push(String(data));
      return true;
    });

    const dt = new Dunetrace({ endpoint: "http://localhost:8001", flushIntervalMs: 60000 });
    await dt.run("agent", {}, async () => {});
    await dt.flush();

    expect(stderrWrites.some((s) => s.includes("[dunetrace]"))).toBe(true);
    await dt.shutdown();
  });

  it("500 response from server does NOT throw from flush()", async () => {
    mockFetchStatus(500);
    const dt = new Dunetrace({ endpoint: "http://localhost:8001", flushIntervalMs: 60000 });

    await dt.run("agent", {}, async () => {});
    await expect(dt.flush()).resolves.toBeUndefined();
    await dt.shutdown();
  });

  it("503 response does NOT throw from run()", async () => {
    mockFetchStatus(503);
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });

    await expect(
      dt.run("agent", {}, async () => "result"),
    ).resolves.toBe("result");

    await dt.shutdown();
  });

  it("buffer stays at bufferSize after pushing bufferSize + 10 events", async () => {
    const BUFFER_SIZE = 50;
    const dt = new Dunetrace({
      endpoint: "http://localhost:8001",
      bufferSize: BUFFER_SIZE,
      flushIntervalMs: 99999, // prevent auto-drain
    });

    // Push BUFFER_SIZE + 10 events via _emit directly
    for (let i = 0; i < BUFFER_SIZE + 10; i++) {
      dt._emit({
        event_type: "tool.called",
        run_id: `run-${i}`,
        agent_id: "agent",
        agent_version: "v1",
        step_index: i,
        timestamp: Date.now() / 1000,
        payload: { tool_name: "search", args: "aa" },
      });
    }

    const buffer = (dt as unknown as { _buffer: AgentEvent[] })._buffer;
    expect(buffer.length).toBeLessThanOrEqual(BUFFER_SIZE);

    await dt.shutdown();
  });

  it("buffer cap is enforced — first BUFFER_SIZE events fill the buffer, extras are dropped", async () => {
    const BUFFER_SIZE = 10;
    const dt = new Dunetrace({
      endpoint: "http://localhost:8001",
      bufferSize: BUFFER_SIZE,
      flushIntervalMs: 99999,
    });

    const TOTAL = BUFFER_SIZE + 5;
    for (let i = 0; i < TOTAL; i++) {
      dt._emit({
        event_type: "llm.called",
        run_id: "run-cap",
        agent_id: "agent",
        agent_version: "v1",
        step_index: i,
        timestamp: Date.now() / 1000,
        payload: { model: "gpt-4o", prompt_tokens: i },
      });
    }

    const buffer = (dt as unknown as { _buffer: AgentEvent[] })._buffer;
    // Buffer must not exceed the configured cap
    expect(buffer.length).toBe(BUFFER_SIZE);

    await dt.shutdown();
  });

  it("run() still returns the user's value when the ingest endpoint returns 500", async () => {
    mockFetchStatus(500);
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });

    const value = await dt.run("agent", {}, async () => 42);
    expect(value).toBe(42);

    await dt.shutdown();
  });

  it("run() still propagates exceptions from user code even when fetch fails", async () => {
    mockFetchNetworkError();
    const dt = new Dunetrace({ endpoint: "http://localhost:8001" });

    await expect(
      dt.run("agent", {}, async () => {
        throw new Error("user error");
      }),
    ).rejects.toThrow("user error");

    await dt.shutdown();
  });
});
