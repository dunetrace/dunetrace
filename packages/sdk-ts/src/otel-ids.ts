/**
 * OTel correlation ids derived from a Dunetrace run id.
 *
 * Kept in a module with no imports so the core client can stamp a run with its
 * trace id without loading `@opentelemetry/api`. `integrations/otel.ts`
 * re-exports these, so the public surface is unchanged.
 */

/** True when s is a usable 32-hex-char trace id (not the all-zero invalid one). */
export function isTraceIdHex(s: string): boolean {
  return /^[0-9a-f]{32}$/.test(s) && s !== "00000000000000000000000000000000";
}

/** 32-hex-char trace ID for run_id (W3C traceparent format). A Dunetrace run UUID
 *  maps to a stable trace both a backend and the dashboard can address. Returns
 *  "" when run_id is not UUID-shaped. */
export function traceIdHex(runId: string): string {
  const hex = runId.replace(/-/g, "").toLowerCase();
  return isTraceIdHex(hex) ? hex : "";
}

/** 16-hex-char root span ID for run_id (its lower 64 bits). "" when not UUID-shaped. */
export function rootSpanIdHex(runId: string): string {
  const hex = traceIdHex(runId);
  return hex ? hex.slice(16) : "";
}
