/**
 * Ambient run context.
 *
 * Lives in its own module so instrumentation can read the active run without
 * importing the client. `auto.ts` needs `getCurrentRun()`, and `client.ts` calls
 * into `auto.ts` for its wrap helpers — routing both through here keeps that from
 * becoming an import cycle.
 */

import { AsyncLocalStorage } from "node:async_hooks";

import type { DunetraceRun } from "./run.js";

/**
 * The active run for the current async context. `dt.run()` enters it; anything
 * awaited inside — including code several libraries deep — reads the same run
 * back out, which is what lets auto-instrumentation attach events without the
 * call site passing a run around.
 */
export const runStorage = new AsyncLocalStorage<DunetraceRun>();

/** Return the current DunetraceRun from async context, or null. */
export function getCurrentRun(): DunetraceRun | null {
  return runStorage.getStore() ?? null;
}

/**
 * Set while an instrumented LLM call is in flight.
 *
 * Both vendor SDKs issue their requests through `fetch`, so with HTTP
 * instrumentation enabled a single `chat.completions.create()` would otherwise
 * emit `llm.called` *and* a `tool.called` for api.openai.com — inflating the tool
 * count and tripping tool-loop detection on what is really one LLM call. The
 * fetch wrapper reads this and stays out of the way.
 *
 * Mirrors the Python SDK's `_in_framework_call` guard.
 */
export const httpSuppression = new AsyncLocalStorage<boolean>();

/** True when the current async context is inside an instrumented LLM call. */
export function httpInstrumentationSuppressed(): boolean {
  return httpSuppression.getStore() === true;
}

// ── Runs that open themselves ─────────────────────────────────────────────────
//
// Every event belongs to a run, and a run needs a start and an end. The client
// registers an opener here so instrumentation that finds no active run can ask
// for one without importing the client (which would be an import cycle). How
// the run it gets will END is decided by which opener is used:
//
//   openEntryRun()  — a framework entry point with a natural end (Vercel AI's
//                     generateText, an HTTP request). The caller closes it when
//                     the call returns. Exact boundary; an ordinary run.
//   resolveRun()    — a bare LLM call with no boundary in sight. The client
//                     opens an *implicit* run, attaches the calls that follow in
//                     this async context, and closes it after an idle window or
//                     at process exit. A guess, marked as such on the wire and
//                     shadowed by the detector. See client.ts.

export interface EntryRunHandle {
  run: DunetraceRun;
  /** Emit the terminal event: run.completed, or run.errored when `err` is given. */
  finish(err?: unknown): void;
}

export interface RunOpener {
  implicit(openedBy: string): DunetraceRun | null;
  entry(openedBy: string, agentId?: string): EntryRunHandle | null;
}

let _opener: RunOpener | null = null;

/** Called by the client constructor; the most recent client wins. */
export function setRunOpener(opener: RunOpener | null): void {
  _opener = opener;
}

/** The active run when there is a live one, else an implicit run opened by
 *  the registered client, else null (no client, or implicit runs disabled).
 *  A closed run left in the async context reads as none. */
export function resolveRun(openedBy: string): DunetraceRun | null {
  const current = runStorage.getStore();
  if (current && !current.closed) return current;
  return _opener?.implicit(openedBy) ?? null;
}

/** An exact run for a framework entry point, or null when a real run is
 *  already active (the entry point then attaches to it) or no client is
 *  registered. An implicit run active here is closed by the client first. */
export function openEntryRun(openedBy: string, agentId?: string): EntryRunHandle | null {
  const current = runStorage.getStore();
  if (current && !current.closed && !current.implicit) return null;
  return _opener?.entry(openedBy, agentId) ?? null;
}
