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
