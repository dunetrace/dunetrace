/**
 * Ring buffer for the outbound event queue.
 *
 * Fixed capacity, never blocks the agent. When full it sheds the OLDEST RUN —
 * every buffered event of that run and every later non-terminal event it emits
 * — rather than whatever happened to arrive next. A plain newest-drops ring
 * (`if (buffer.length >= size) return;`) sheds run terminals, so a run that
 * overflowed the buffer never reached the detector as *finished* at all; and
 * dropping single events produced partial runs the detector could not tell
 * apart from complete ones — a tool loop with its early calls missing looks
 * clean, and a run missing its `run.started` has no tools list.
 *
 * Shedding by run keeps every run that does reach the detector whole, and the
 * runs that were cut carry a visible `dropped_events` count on their terminal
 * event (see {@link EventBuffer.reserveTerminal}) so the detector can mark
 * their verdicts as drawn from incomplete data —
 * `services/detector/tests/test_dropped_events.py` is the server side.
 *
 * Terminal events (`run.completed` / `run.errored`) are never shed; they are
 * always enqueued, evicting another run if that is what it takes.
 *
 * This is the TypeScript port of `packages/sdk-py/dunetrace/buffer.py`, with
 * the same guarantees and a simpler implementation of them: a run's shed
 * events are left in place as tombstones and skipped on drain (so a shed is
 * O(1)), but tombstones are tracked by a *separate* `_dead` map rather than
 * being inferred from the drop-count record. That is what lets a terminal
 * release its drop record without an in-drain compaction — the buffer is
 * only ever rebuilt from `push`, never from underneath an in-progress drain.
 */

import type { AgentEvent } from "./models.js";

/**
 * Wire names, not the EventType union, so this module keeps working for
 * anything shaped like an event.
 */
const TERMINAL_EVENT_TYPES = new Set<string>(["run.completed", "run.errored"]);

const WARN_INTERVAL_NS = 60_000_000_000n; // 60s

function isTerminal(event: AgentEvent): boolean {
  return TERMINAL_EVENT_TYPES.has(event.event_type as string);
}

/**
 * Stamp `dropped_events` onto a terminal payload. Only ever raises the number:
 * the client stamps a count at build time and the buffer re-stamps at drain,
 * when more of the run may have been shed in between.
 */
function stampTerminal(event: AgentEvent, dropped: number): void {
  const payload = event.payload;
  if (payload === null || typeof payload !== "object") return;
  const current = Number(payload["dropped_events"] ?? 0);
  if (dropped > (Number.isFinite(current) ? current : 0)) {
    payload["dropped_events"] = dropped;
  }
}

export class EventBuffer {
  private readonly _maxSize: number;
  private readonly _maxShedRuns: number;

  // Physical storage: an array plus a head index, so the FIFO pop is O(1) and
  // the array is rebuilt (amortised) only when it grows past twice capacity.
  private _buf: (AgentEvent | null)[] = [];
  private _head = 0;

  /** Items a drain will ship — the physical array also holds tombstones. */
  private _live = 0;
  /** run_id -> live non-terminal events of that run currently buffered. */
  private _pending = new Map<string, number>();
  /** run_id -> tombstones of that run still physically in the array. */
  private _dead = new Map<string, number>();
  /**
   * run_id -> events dropped for that run (buffered at shed time plus later
   * arrivals). Insertion order is shed order, so eviction is oldest-first.
   */
  private _shed = new Map<string, number>();

  private _shedRunsTotal = 0;
  private _shedsSinceWarning = 0;
  // Never a 0 sentinel for "not yet warned": process.hrtime.bigint() starts
  // near zero at process start, so comparing against 0 would silently swallow
  // the first warning of every process's lifetime for ~60s. Same reason
  // DurableRetryEmitter uses null — see emitters.ts.
  private _lastWarningNs: bigint | null = null;

  constructor(maxSize = 10_000, opts: { maxShedRuns?: number } = {}) {
    if (!Number.isInteger(maxSize) || maxSize < 1) {
      throw new Error("EventBuffer: maxSize must be an integer >= 1");
    }
    this._maxSize = maxSize;
    this._maxShedRuns = Math.max(1, opts.maxShedRuns ?? 4096);
  }

  // ── Producer side ──────────────────────────────────────────────────────────

  /**
   * Append `event`. Returns true if it was enqueued, false if it was dropped.
   *
   * A non-terminal event of a run that has already been shed is counted against
   * that run and dropped. When the buffer is full the oldest run is shed to make
   * room. Terminal events are always enqueued, even when their run is (or later
   * becomes) shed: they carry the drop count to the server, so losing one would
   * hide the loss.
   */
  push(event: AgentEvent): boolean {
    const key = event.run_id;
    const terminal = isTerminal(event);

    if (!terminal && this._shed.has(key)) {
      this._shed.set(key, (this._shed.get(key) ?? 0) + 1);
      return false;
    }

    while (this._live >= this._maxSize && this._shedOldest()) {
      /* shed until there is room, or until only terminals are left */
    }

    if (!terminal) {
      if (this._shed.has(key)) {
        // The eviction above just shed this event's own run.
        this._shed.set(key, (this._shed.get(key) ?? 0) + 1);
        return false;
      }
      if (this._live >= this._maxSize) {
        // Nothing sheddable left — the buffer is all terminals. Drop this
        // event, and shed its run so the run is cut whole, not piecemeal.
        this._shedRun(key);
        this._shed.set(key, (this._shed.get(key) ?? 0) + 1);
        return false;
      }
    }

    this._buf.push(event);
    this._live += 1;
    if (!terminal) this._pending.set(key, (this._pending.get(key) ?? 0) + 1);
    if (this._buf.length - this._head > 2 * this._maxSize) this._compact();
    return true;
  }

  /**
   * Make room for `runId`'s terminal event and return its drop count.
   *
   * Called by the client just before it builds `run.completed` /
   * `run.errored`: any shedding needed to fit the terminal happens *now*, so
   * the count it returns already includes it and every consumer of the event
   * (the JSON log line, an OTel exporter) sees the same stamped payload the
   * server does. Returns 0 for a run nothing was dropped from.
   */
  reserveTerminal(runId: string): number {
    while (this._live >= this._maxSize && this._shedOldest()) {
      /* make room for the terminal before its payload is built */
    }
    return this._shed.get(runId) ?? 0;
  }

  /** Events dropped so far for `runId` (0 if the run was never shed). */
  droppedCount(runId: string): number {
    return this._shed.get(runId) ?? 0;
  }

  // ── Consumer side ──────────────────────────────────────────────────────────

  /** Up to `n` live events from the front, removed from the buffer. */
  drain(n = 100): AgentEvent[] {
    const batch: AgentEvent[] = [];
    while (this._head < this._buf.length && batch.length < n) {
      const event = this._buf[this._head];
      this._buf[this._head] = null;
      this._head += 1;
      if (event === null) continue;

      const key = event.run_id;
      const terminal = isTerminal(event);
      if (!terminal && this._dead.has(key)) {
        this._decrement(this._dead, key);
        continue;
      }
      if (terminal && this._shed.has(key)) {
        // The run was shed after this terminal was already buffered, so the
        // count the client stamped is stale (or absent). The wire must say how
        // much is missing, or the detector reads a partial run as a whole one.
        stampTerminal(event, this._shed.get(key) ?? 0);
        this._shed.delete(key);
      }
      this._live -= 1;
      if (!terminal) this._decrement(this._pending, key);
      batch.push(event);
    }
    if (this._head >= this._buf.length) {
      this._buf = [];
      this._head = 0;
    }
    return batch;
  }

  /** Every live event currently in the buffer. */
  drainAll(): AgentEvent[] {
    return this.drain(this._buf.length - this._head);
  }

  /** Live events — the ones a drain will ship. */
  get length(): number {
    return this._live;
  }

  /** Runs shed since construction (for tests and diagnostics). */
  get shedRunsTotal(): number {
    return this._shedRunsTotal;
  }

  // ── Internals ──────────────────────────────────────────────────────────────

  /** Evict the oldest still-live run. Returns false if nothing can be shed. */
  private _shedOldest(): boolean {
    // Tombstones at the head are already dead — clear them first, so the victim
    // is the oldest run that is still live.
    while (this._head < this._buf.length) {
      const head = this._buf[this._head];
      if (head !== null && !isTerminal(head) && this._dead.has(head.run_id)) {
        this._buf[this._head] = null;
        this._head += 1;
        this._decrement(this._dead, head.run_id);
        continue;
      }
      if (head === null) {
        this._head += 1;
        continue;
      }
      break;
    }
    if (this._head >= this._buf.length) return false;

    const head = this._buf[this._head]!;
    let victim: string | null = null;
    if (!isTerminal(head)) {
      victim = head.run_id;
    } else {
      // Terminals can never be shed. Scan past them for the oldest run that
      // still has sheddable events — a scan bounded by the number of protected
      // heads, not the buffer, since every shed removes the run behind them.
      for (let i = this._head; i < this._buf.length; i++) {
        const event = this._buf[i];
        if (event === null || isTerminal(event) || this._dead.has(event.run_id)) continue;
        victim = event.run_id;
        break;
      }
      if (victim === null) return false;
    }
    this._shedRun(victim);
    return true;
  }

  /** Tombstone every buffered event of `key` and start counting drops for it. */
  private _shedRun(key: string): void {
    const count = this._pending.get(key) ?? 0;
    this._pending.delete(key);
    this._live -= count;
    if (count > 0) this._dead.set(key, (this._dead.get(key) ?? 0) + count);
    this._shed.set(key, (this._shed.get(key) ?? 0) + count);
    this._shedRunsTotal += 1;
    // A shed run whose terminal never comes must not pin memory forever.
    // Its tombstones stay identified by _dead, so dropping the record here is
    // safe — it only stops counting further drops for a run nobody will report.
    while (this._shed.size > this._maxShedRuns) {
      const oldest = this._shed.keys().next().value as string | undefined;
      if (oldest === undefined) break;
      this._shed.delete(oldest);
    }
    this._warn(key, count);
  }

  /** Physically remove every tombstone. O(n), amortised by the 2x trigger. */
  private _compact(): void {
    const keep: AgentEvent[] = [];
    for (let i = this._head; i < this._buf.length; i++) {
      const event = this._buf[i];
      if (event === null) continue;
      if (!isTerminal(event) && this._dead.has(event.run_id)) continue;
      keep.push(event);
    }
    this._buf = keep;
    this._head = 0;
    this._dead.clear();
  }

  private _decrement(map: Map<string, number>, key: string): void {
    const left = (map.get(key) ?? 0) - 1;
    if (left > 0) map.set(key, left);
    else map.delete(key);
  }

  /** One WARNING per shed run, rate-limited to one line a minute. */
  private _warn(key: string, count: number): void {
    this._shedsSinceWarning += 1;
    const now = process.hrtime.bigint();
    if (this._lastWarningNs !== null && now - this._lastWarningNs < WARN_INTERVAL_NS) return;
    const suppressed = this._shedsSinceWarning - 1;
    this._lastWarningNs = now;
    this._shedsSinceWarning = 0;
    console.warn(
      `[dunetrace] event buffer full (${this._maxSize}) — shed run ${key} ` +
        `(${count} buffered event(s) dropped); ${suppressed} other run(s) shed since the ` +
        "last warning. The run will reach the server with dropped_events set and its " +
        "signals held in shadow.",
    );
  }
}
