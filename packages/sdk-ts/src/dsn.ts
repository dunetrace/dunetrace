/**
 * DSN parsing: one value that carries the ingest endpoint and the API key.
 *
 *     DUNETRACE_DSN=https://dt_live_abc@ingest.example.com
 *
 * Mirrors the Python SDK's dunetrace.implicit.parse_dsn. The key still
 * travels as a bearer token on every request; the DSN is a convenience for
 * copying one value from the dashboard, not a security mechanism, and must
 * never be logged whole. Use dsnHost() in log lines.
 */

export interface ParsedDsn {
  endpoint: string;
  apiKey: string;
}

export function parseDsn(dsn: string): ParsedDsn {
  let url: URL;
  try {
    url = new URL(dsn.trim());
  } catch {
    throw new Error("DUNETRACE_DSN must look like https://<api_key>@host[:port][/path]");
  }
  if ((url.protocol !== "http:" && url.protocol !== "https:") || !url.hostname) {
    throw new Error("DUNETRACE_DSN must look like https://<api_key>@host[:port][/path]");
  }
  const apiKey = decodeURIComponent(url.username);
  if (!apiKey) throw new Error("DUNETRACE_DSN has no api key before the '@'");
  const path = url.pathname.replace(/\/+$/, "");
  return { endpoint: `${url.protocol}//${url.host}${path}`, apiKey };
}

/** The host part of a DSN, for log lines that must not carry the key. */
export function dsnHost(dsn: string): string {
  try {
    return new URL(dsn.trim()).hostname || "?";
  } catch {
    return "?";
  }
}
