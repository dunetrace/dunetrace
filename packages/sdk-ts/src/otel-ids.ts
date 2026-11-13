/**
 * OTel correlation ids and the per-run sampling decision, derived from a
 * Dunetrace run id.
 *
 * Kept free of OpenTelemetry imports so the core client can stamp a run with
 * its trace id without loading `@opentelemetry/api`. `integrations/otel.ts`
 * re-exports these, so the public surface is unchanged.
 */

import { createHash } from "node:crypto";

const TWO_POW_64 = 2 ** 64;

/**
 * Whether a run's spans are exported at this sampling ratio. The same function
 * as the Python SDK's `dunetrace.otel.run_sampled`, so both SDKs and the
 * detector agree on every run: the first 8 bytes of SHA-256(runId), read
 * big-endian, compared against `ratio * 2**64`.
 *
 * Hashed rather than read off the trace id: the trace id is the run UUID, and
 * the low 64 bits a ratio sampler compares start with the fixed UUID variant
 * bits `10`, so every UUID lands in [0.5, 0.75) of the range and the ratio
 * stops being a probability.
 */
export function runSampled(runId: string, ratio: number): boolean {
  if (ratio >= 1) return true;
  if (!(ratio > 0)) return false; // 0, negative, NaN
  const head = createHash("sha256").update(runId, "utf8").digest().readBigUInt64BE(0);
  // Float multiply then truncate, exactly as Python's int(ratio * 2**64).
  return head < BigInt(Math.floor(ratio * TWO_POW_64));
}

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
