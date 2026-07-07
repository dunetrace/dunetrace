/**
 * Pluggable batch-shipping strategies. Port of packages/sdk-py/dunetrace/emitters.py —
 * see that file's module docstring for the full design rationale; this keeps the same
 * shape (BatchEmitter interface, HttpBatchEmitter default, DurableRetryEmitter wrapper)
 * so behavior stays comparable across both SDKs, adjusted only where a language
 * difference forces it (ship() is async here — Node's fetch has no synchronous
 * equivalent to Python's urllib.request.urlopen; better-sqlite3 is a peer dependency,
 * not a stdlib module like Python's sqlite3, so it's loaded lazily and optionally —
 * see DurableRetryEmitter's constructor).
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import type { AgentEvent } from "./models.js";

/**
 * Strategy for delivering one drained batch of events.
 *
 * ship() must never throw — catch and log internally, resolving false on
 * failure so a caller (e.g. DurableRetryEmitter) can react without needing to
 * parse errors. Resolving true means "handled" — for NoopBatchEmitter that
 * means "intentionally discarded," not "delivered."
 */
export interface BatchEmitter {
  ship(batch: AgentEvent[]): Promise<boolean>;
}

/** POSTs each batch to the ingest API. The default emitter — same behavior
 * client.ts's private _ship() had before this became pluggable. */
export class HttpBatchEmitter implements BatchEmitter {
  private readonly _ingestUrl: string;
  private readonly _apiKey: string;
  private readonly _timeoutMs: number;

  constructor(endpoint: string, apiKey = "", timeoutMs = 5000) {
    this._ingestUrl = endpoint.replace(/\/$/, "") + "/v1/ingest";
    this._apiKey = apiKey;
    this._timeoutMs = timeoutMs;
  }

  async ship(batch: AgentEvent[]): Promise<boolean> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this._timeoutMs);
    try {
      const resp = await fetch(this._ingestUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Dunetrace-Agent": batch[0]?.agent_id ?? "",
          ...(this._apiKey ? { Authorization: `Bearer ${this._apiKey}` } : {}),
        },
        body: JSON.stringify({
          api_key: this._apiKey,
          agent_id: batch[0]?.agent_id ?? "",
          events: batch,
        }),
        signal: controller.signal,
      });
      return resp.ok;
    } catch (err) {
      const msg = String(err);
      const isConnRefused = msg.includes("ECONNREFUSED") || msg.includes("fetch failed");
      if (isConnRefused) {
        process.stderr.write(
          `[dunetrace] Backend unreachable at ${this._ingestUrl} — is it running? (docker compose up -d)\n`,
        );
      } else {
        process.stderr.write(`[dunetrace] Failed to ship ${batch.length} events: ${err}\n`);
      }
      return false;
    } finally {
      clearTimeout(timer);
    }
  }
}

/** Discards every batch. The explicit way to disable HTTP shipping. */
export class NoopBatchEmitter implements BatchEmitter {
  async ship(_batch: AgentEvent[]): Promise<boolean> {
    return true;
  }
}

function batchToJson(batch: AgentEvent[]): string {
  return JSON.stringify(batch);
}

function batchFromJson(payload: string): AgentEvent[] {
  return JSON.parse(payload) as AgentEvent[];
}

export const DEFAULT_QUEUE_PATH = path.join(os.homedir(), ".dunetrace", "queue-ts.db");

interface QueueRow {
  id: number;
  enqueued_at: number;
  size_bytes: number;
  payload: string;
}

// better-sqlite3's own type (`Database.Database`). Kept as an alias rather than
// importing the type unconditionally — the import itself is dynamic (see
// _initDb) so the whole package still loads without better-sqlite3 installed
// for anyone who never constructs a DurableRetryEmitter.
type SqliteDatabase = {
  prepare(sql: string): {
    run(...params: unknown[]): unknown;
    get(...params: unknown[]): unknown;
    all(...params: unknown[]): unknown[];
  };
  close(): void;
};

/**
 * Wraps any BatchEmitter to survive a backend outage across process restarts.
 * On ship() failure, the batch is persisted to a local SQLite queue instead of
 * dropped; on every ship() call, any due backlog is retried first (oldest
 * batch first), before the current new batch.
 *
 * Requires the optional peer dependency `better-sqlite3` (synchronous API —
 * matches this class's own single-threaded, lock-guarded access pattern, the
 * same tradeoff Python's threading.Lock-guarded synchronous sqlite3 makes).
 * Not a required dependency of the `dunetrace` package: the default emitter
 * (HttpBatchEmitter alone) has zero runtime dependencies, and durable retry is
 * opt-in, same as Python's design (see client.py — DurableRetryEmitter is
 * never the default there either). If better-sqlite3 isn't installed, or the
 * queue file can't be created, this degrades to "failed batches are dropped,
 * same as no retry at all" rather than throwing — a missing optional
 * dependency should never crash the host application.
 *
 * Queue location: `queuePath` constructor option, else `DUNETRACE_QUEUE_PATH`
 * env var, else `~/.dunetrace/queue-ts.db` — deliberately a different filename
 * from Python's `queue.db`, so a mixed Python+TypeScript deployment sharing
 * `~/.dunetrace/` (e.g. a monorepo running both SDKs) can't have one process
 * corrupt the other's queue file.
 *
 * Bounded: evicts the oldest queued batch(es) once either maxQueueEvents or
 * maxQueueBytes is exceeded. Eviction is logged at warning level, rate-limited
 * to once per minute (with the count evicted since the last warning).
 *
 * Retry cadence is decoupled from the normal flush interval (default every
 * 200ms) — draining the backlog every cycle would hammer a still-down backend
 * with a full-timeout connection attempt every 200ms. Default: every 30s ± 5s
 * jitter (prevents a thundering herd if many SDK instances come back online
 * together after a shared outage). A retry attempt stops at the first failure
 * rather than skipping ahead — preserving order matters more than attempting
 * all of them once the backend is confirmed still down.
 */
export class DurableRetryEmitter implements BatchEmitter {
  private readonly _inner: BatchEmitter;
  private readonly _path: string;
  private readonly _maxQueueEvents: number;
  private readonly _maxQueueBytes: number;
  private readonly _retryIntervalMs: number;
  private readonly _retryJitterMs: number;
  private readonly _maxBatchesPerRetry: number;

  private _db: SqliteDatabase | null = null;
  private _dbOk = false;
  private _nextRetryAt = 0; // process.hrtime.bigint() ns; 0 == due immediately
  private _evictedSinceWarning = 0;
  // Never a 0 sentinel for "not yet warned" — process.hrtime.bigint() starts
  // near zero at process start (it measures time since an arbitrary point,
  // typically process start on Node), so comparing against 0 would make the
  // very first eviction warning of every process's lifetime silently
  // disappear for the first ~60s, every time. Same bug class fixed in
  // emitters.py this same sprint (there via time.monotonic()'s epoch being
  // boot-time on Linux) — null here for the identical reason, made worse by
  // hrtime's guaranteed-near-zero start.
  private _lastEvictionWarningNs: bigint | null = null;

  constructor(
    inner: BatchEmitter,
    opts: {
      queuePath?: string;
      maxQueueEvents?: number;
      maxQueueBytes?: number;
      retryIntervalMs?: number;
      retryJitterMs?: number;
      maxBatchesPerRetry?: number;
    } = {},
  ) {
    this._inner = inner;
    this._path = opts.queuePath ?? process.env.DUNETRACE_QUEUE_PATH ?? DEFAULT_QUEUE_PATH;
    this._maxQueueEvents = opts.maxQueueEvents ?? 100_000;
    this._maxQueueBytes = opts.maxQueueBytes ?? 100 * 1024 * 1024;
    this._retryIntervalMs = opts.retryIntervalMs ?? 30_000;
    this._retryJitterMs = opts.retryJitterMs ?? 5_000;
    this._maxBatchesPerRetry = opts.maxBatchesPerRetry ?? 50;

    this._dbOk = this._initDb();
  }

  private _initDb(): boolean {
    let Database: new (path: string) => SqliteDatabase;
    try {
      // Dynamic require, not a static import: the whole emitters module (and
      // therefore the whole dunetrace package) must still load for consumers
      // who never construct a DurableRetryEmitter and never installed this
      // optional peer dependency.
      Database = require("better-sqlite3");
    } catch {
      console.warn(
        "[dunetrace] DurableRetryEmitter requires the optional peer dependency " +
          "'better-sqlite3' (npm install better-sqlite3). Failed batches will be " +
          "dropped instead of queued until it's installed.",
      );
      return false;
    }

    try {
      const dir = path.dirname(this._path);
      if (dir) fs.mkdirSync(dir, { recursive: true });
      this._db = new Database(this._path);
      this._db.prepare(
        `CREATE TABLE IF NOT EXISTS queue (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          enqueued_at REAL NOT NULL,
          size_bytes INTEGER NOT NULL,
          payload TEXT NOT NULL
        )`,
      ).run();
      return true;
    } catch (err) {
      console.warn(
        `[dunetrace] DurableRetryEmitter: could not initialize queue at ${this._path}: ${err}. ` +
          "Failed batches will be dropped instead of queued.",
      );
      return false;
    }
  }

  async ship(batch: AgentEvent[]): Promise<boolean> {
    await this._maybeRetryBacklog();
    if (await this._inner.ship(batch)) return true;
    return this._enqueue(batch);
  }

  private _enqueue(batch: AgentEvent[]): boolean {
    if (!this._dbOk || !this._db) return false;
    const payload = batchToJson(batch);
    const size = Buffer.byteLength(payload, "utf8");
    try {
      this._db.prepare(
        "INSERT INTO queue (enqueued_at, size_bytes, payload) VALUES (?, ?, ?)",
      ).run(Date.now() / 1000, size, payload);
      this._evictIfOverBounds();
      return true;
    } catch (err) {
      console.warn(`[dunetrace] DurableRetryEmitter: failed to queue batch: ${err}`);
      return false;
    }
  }

  private _evictIfOverBounds(): void {
    if (!this._db) return;
    const row = this._db
      .prepare("SELECT COUNT(*) as count, COALESCE(SUM(size_bytes), 0) as total FROM queue")
      .get() as { count: number; total: number };
    let count = row.count;
    let totalBytes = row.total;
    let evicted = 0;

    while (count > this._maxQueueEvents || totalBytes > this._maxQueueBytes) {
      const oldest = this._db
        .prepare("SELECT id, size_bytes FROM queue ORDER BY id ASC LIMIT 1")
        .get() as { id: number; size_bytes: number } | undefined;
      if (!oldest) break;
      this._db.prepare("DELETE FROM queue WHERE id = ?").run(oldest.id);
      count -= 1;
      totalBytes -= oldest.size_bytes;
      evicted += 1;
    }

    if (evicted > 0) {
      this._evictedSinceWarning += evicted;
      const now = process.hrtime.bigint();
      const sixtySecondsNs = 60_000_000_000n;
      if (
        this._lastEvictionWarningNs === null ||
        now - this._lastEvictionWarningNs >= sixtySecondsNs
      ) {
        console.warn(
          `[dunetrace] DurableRetryEmitter: evicted ${this._evictedSinceWarning} oldest ` +
            `batch(es) from the disk queue at ${this._path} (over the ` +
            `${this._maxQueueEvents}-event / ${this._maxQueueBytes}-byte cap) — this is ` +
            "silent data loss during a long outage.",
        );
        this._evictedSinceWarning = 0;
        this._lastEvictionWarningNs = now;
      }
    }
  }

  /** Retries due backlog, oldest first, stopping at the first failure — same
   * ordering guarantee as Python's synchronous version. Awaited by ship()
   * before it attempts the new batch, matching Python's "retry backlog first,
   * then ship the new batch" sequencing; the only difference forced by
   * TypeScript is that this is a Promise instead of a blocking call, since
   * inner.ship() has no synchronous form (fetch is async-only). */
  private async _maybeRetryBacklog(): Promise<void> {
    const now = Number(process.hrtime.bigint() / 1_000_000n); // ms
    if (now < this._nextRetryAt) return;
    const jitter = (Math.random() * 2 - 1) * this._retryJitterMs;
    this._nextRetryAt = now + this._retryIntervalMs + jitter;

    if (!this._dbOk || !this._db) return;

    try {
      const rows = this._db
        .prepare("SELECT id, payload FROM queue ORDER BY id ASC LIMIT ?")
        .all(this._maxBatchesPerRetry) as QueueRow[];
      for (const row of rows) {
        const batch = batchFromJson(row.payload);
        const ok = await this._inner.ship(batch);
        if (ok) {
          this._db.prepare("DELETE FROM queue WHERE id = ?").run(row.id);
        } else {
          break; // still down — stop rather than skip ahead out of order
        }
      }
    } catch (err) {
      console.warn(`[dunetrace] DurableRetryEmitter: retry attempt failed: ${err}`);
    }
  }
}
