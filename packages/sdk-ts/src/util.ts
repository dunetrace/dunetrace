import { safeJsonStringify } from "./redaction.js";

/**
 * A value's string form: the string itself, else its JSON form.
 *
 * Serialisation goes through safeJsonStringify, so a tool that returns a
 * circular object or a BigInt still reports its output instead of reporting an
 * empty one (and never throws — see safeEmit below for why that matters).
 */
export function resultText(v: unknown): string {
  if (v == null) return "";
  if (typeof v === "string") return v;
  return safeJsonStringify(v);
}

/** Length of a value's string form. */
export function resultLength(v: unknown): number {
  if (v == null) return 0;
  if (typeof v === "string") return v.length;
  return safeJsonStringify(v).length;
}

/**
 * Run an emission and swallow anything it throws.
 *
 * Instrumentation sits in the middle of a call the host application depends on.
 * A bug in our event emission — or a tool argument JSON cannot serialise, or a
 * customer redact hook that throws — must never turn a working call into a
 * failed one, so every emit on the customer path goes through here. Same
 * guarantee as the Python SDK's `_safe_emit`.
 */
export function safeEmit(emit: () => void, context = "instrumentation"): void {
  try {
    emit();
  } catch (err) {
    const suffix = err instanceof Error ? `: ${err.message}` : err ? `: ${String(err)}` : "";
    console.warn(
      `[dunetrace] ${context} failed to emit an event (call itself is unaffected)${suffix}`,
    );
  }
}
