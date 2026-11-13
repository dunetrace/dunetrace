/**
 * Automatic flush when the process exits.
 *
 * The bug: the client's only drain path is a 200ms `setInterval` that is
 * deliberately `unref()`'d, and nothing registered a process hook. A script
 * that awaited one `dt.run()` and returned exited with the emitter having
 * received ZERO events — the run finished well inside one tick and the unref'd
 * timer never fired. That is the normal shape of a CLI agent, a cron job or a
 * Lambda handler, and the loss was total and silent. The Python client has
 * always registered a weakref'd `atexit` flush; this is its counterpart.
 *
 * The subprocess test below is the one that actually reproduces it: only a real
 * process exiting can prove `beforeExit` does the job, and it is also what
 * proves the flush does not hold the process open (the child must still exit,
 * and the test would time out if it did not).
 */

import { describe, it, expect } from "vitest";
import { execFileSync } from "node:child_process";
import { writeFileSync, rmSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { Dunetrace } from "../src/client.js";
import type { AgentEvent } from "../src/models.js";

const PKG_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

/** Collects shipped batches instead of POSTing them. */
function recordingEmitter(): { batches: AgentEvent[][]; ship: (b: AgentEvent[]) => Promise<boolean> } {
  const batches: AgentEvent[][] = [];
  return {
    batches,
    ship: async (batch: AgentEvent[]) => { batches.push(batch); return true; },
  };
}

describe("exit flush — in process", () => {
  it("ships the buffer when beforeExit fires", async () => {
    const emitter = recordingEmitter();
    const dt = new Dunetrace({ emitter, flushIntervalMs: 60_000 });
    try {
      await dt.run("exit-agent", {}, async () => { /* one run, no shutdown */ });
      expect(emitter.batches).toHaveLength(0);   // the timer has not fired

      process.emit("beforeExit", 0);
      await new Promise((r) => setImmediate(r));

      expect(emitter.batches).toHaveLength(1);
      expect(emitter.batches[0].map(e => e.event_type)).toEqual(["run.started", "run.completed"]);
    } finally {
      await dt.shutdown();
    }
  });

  it("does not flush a second time when the buffer is already empty", async () => {
    const emitter = recordingEmitter();
    const dt = new Dunetrace({ emitter, flushIntervalMs: 60_000 });
    try {
      await dt.run("exit-agent", {}, async () => {});
      process.emit("beforeExit", 0);
      await new Promise((r) => setImmediate(r));
      // beforeExit fires again once the flush settles — that pass must be a
      // no-op, or the hook would spin instead of letting the process exit.
      process.emit("beforeExit", 0);
      await new Promise((r) => setImmediate(r));
      expect(emitter.batches).toHaveLength(1);
    } finally {
      await dt.shutdown();
    }
  });

  it("stops flushing after an explicit shutdown()", async () => {
    const emitter = recordingEmitter();
    const dt = new Dunetrace({ emitter, flushIntervalMs: 60_000 });
    await dt.run("exit-agent", {}, async () => {});
    await dt.shutdown();
    expect(emitter.batches).toHaveLength(1);   // shutdown flushed it

    await dt.run("exit-agent", {}, async () => {});
    process.emit("beforeExit", 0);
    await new Promise((r) => setImmediate(r));
    // Deregistered, so the second run stays buffered rather than being flushed
    // twice by two different paths.
    expect(emitter.batches).toHaveLength(1);
  });

  it("honours flushOnExit: false", async () => {
    const emitter = recordingEmitter();
    const dt = new Dunetrace({ emitter, flushIntervalMs: 60_000, flushOnExit: false });
    try {
      await dt.run("exit-agent", {}, async () => {});
      process.emit("beforeExit", 0);
      await new Promise((r) => setImmediate(r));
      expect(emitter.batches).toHaveLength(0);
    } finally {
      await dt.shutdown();
    }
  });
});

describe("exit flush — real process", () => {
  const tsx = join(PKG_ROOT, "node_modules", ".bin", "tsx");
  // A .ts probe, not .mts: this package is "type": "commonjs", so a real
  // consumer's entry point is CJS. An .mts probe cannot name-import the CJS
  // build ("does not provide an export named 'Dunetrace'"), which is a
  // property of the harness, not of the behaviour under test.
  const probe = join(PKG_ROOT, "exit-flush-probe.ts");

  // The reported repro, verbatim: await one dt.run() and return. No shutdown(),
  // no explicit flush, no keep-alive. Before the fix this printed nothing.
  const SOURCE = `
import { Dunetrace } from "./src/client.js";
const dt = new Dunetrace({
  flushIntervalMs: 60000,
  emitter: { ship: async (batch: any) => { console.log("SHIPPED:" + batch.map((e: any) => e.event_type).join(",")); return true; } },
});
async function main() { await dt.run("cli-agent", {}, async () => {}); }
void main();
`;

  it("a script that returns without shutdown() still ships its events", () => {
    if (!existsSync(tsx)) return;   // devDependency missing; nothing to assert
    writeFileSync(probe, SOURCE);
    try {
      // A timeout here is also the assertion that the flush does not keep the
      // process alive: the child has to exit on its own.
      const out = execFileSync(tsx, [probe], {
        cwd: PKG_ROOT,
        encoding: "utf8",
        timeout: 60_000,
      });
      expect(out).toContain("SHIPPED:run.started,run.completed");
    } finally {
      rmSync(probe, { force: true });
    }
  });
});
