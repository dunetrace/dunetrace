/**
 * OTel export bootstrap for Dunetrace.
 *
 * Reads config from DUNETRACE_OTEL_* env vars, builds a TracerProvider with a
 * bounded async export pipeline, and hands back the tracer the client's
 * DunetraceOtelExporter emits spans through. Opt-in: does nothing unless
 * DUNETRACE_OTEL_ENABLED is set, so the SDK behaves exactly as before for anyone
 * who doesn't configure it. Mirrors `packages/sdk-py/dunetrace/otel.py`.
 *
 * Failure isolation is the whole point of this module. A bad endpoint, a slow
 * collector, or a missing OpenTelemetry install must never throw into agent
 * code or block the event loop:
 *   - export runs on the BatchSpanProcessor's timer, off the agent's call path
 *   - the queue is bounded and drops spans on overflow
 *   - a circuit breaker stops export attempts for a cooldown after repeated
 *     failures, so a dead collector isn't retried on every batch
 *   - every entry point catches and degrades to "OTel disabled"
 *
 * The OpenTelemetry packages are optional peer dependencies and are loaded with
 * `require` only when export is enabled, so this module can be imported by the
 * core client without pulling them in.
 *
 * Config (all optional except when enabling):
 *   DUNETRACE_OTEL_ENABLED          "1"/"true" to turn export on. Default off.
 *   DUNETRACE_OTEL_ENDPOINT         OTLP endpoint URL. Required when enabled.
 *   DUNETRACE_OTEL_HEADERS          "k1=v1,k2=v2" auth headers (e.g. "DD-API-KEY=xxx").
 *   DUNETRACE_OTEL_PROTOCOL         "grpc" (default) or "http/protobuf".
 *   DUNETRACE_OTEL_SERVICE_NAME     service.name resource attr. Default "dunetrace".
 *   DUNETRACE_OTEL_SAMPLING_RATIO   0.0-1.0 share of runs exported. Default 1.0.
 *                                   Decided per run (runSampled), the same
 *                                   function as the Python SDK.
 *   DUNETRACE_OTEL_CAPTURE_CONTENT  "false" to drop content attributes. Default on.
 *   DUNETRACE_ORG_ID                optional dunetrace.org_id resource attr. The
 *                                   authoritative org is resolved server-side from
 *                                   the API key; this is only a convenience label.
 *
 * Standard OTEL_* env vars the exporter reads natively (e.g. OTEL_EXPORTER_OTLP_
 * INSECURE for plaintext gRPC to a local collector) still apply on top of this.
 */

import { createRequire } from "node:module";
import type { Tracer } from "@opentelemetry/api";
import type {
  BasicTracerProvider,
  ReadableSpan,
  SpanExporter,
} from "@opentelemetry/sdk-trace-base";
import type { AgentEvent, EventSink } from "./models.js";

const DEFAULT_SERVICE_NAME = "dunetrace";
const VALID_PROTOCOLS = ["grpc", "http/protobuf"] as const;

/** Bounded export queue. BatchSpanProcessor drops spans once the queue is full,
 *  which is the backpressure we want: never grow memory unbounded, never block
 *  the agent to wait on a slow collector. Same limits as the Python SDK. */
const MAX_QUEUE_SIZE = 2048;
const MAX_EXPORT_BATCH_SIZE = 512;
/** Upper bound on one export attempt, applied to the batch processor and to
 *  the OTLP exporter's own request. The OpenTelemetry defaults are 30s and 10s,
 *  which is how long a collector that accepts the connection and never answers
 *  would hold shutdown() or the exit flush: the pending request keeps the event
 *  loop alive until the exporter gives up. forceFlush() is bounded separately. */
const EXPORT_TIMEOUT_MS = 5_000;

// ExportResultCode from @opentelemetry/core. Spelled out here so this module
// does not import the package at load time; a test pins them to the enum.
const EXPORT_SUCCESS = 0;
const EXPORT_FAILED = 1;

/** Structural ExportResult, so this module's declarations never reference
 *  @opentelemetry/core (not a declared peer). */
export interface ExportResult {
  code: number;
  error?: Error;
}

export type OtelProtocol = (typeof VALID_PROTOCOLS)[number];

/** Resolved OTel export configuration. Build from env with configFromEnv(). */
export interface OtelConfig {
  enabled: boolean;
  endpoint: string;
  headers: Record<string, string>;
  protocol: OtelProtocol;
  serviceName: string;
  serviceVersion: string;
  samplingRatio: number;
  orgId: string;
  /** PII control for span content. Dunetrace is raw-by-default, so this is true
   *  unless DUNETRACE_OTEL_CAPTURE_CONTENT is turned off. When false the exporter
   *  drops content-bearing span attributes (tool args, request URL, retrieval
   *  query) and keeps only metadata. */
  captureContent: boolean;
}

/** Builds the OTLP exporter. Injectable so tests can substitute an in-memory one. */
export type ExporterFactory = (config: OtelConfig) => SpanExporter;

export interface InitOptions {
  exporterFactory?: ExporterFactory;
  /** Clock for the circuit breaker, in milliseconds. Monotonic by default
   *  (performance.now), so a wall-clock step cannot hold the circuit open or
   *  close it early. Tests drive it. */
  now?: () => number;
}

// ── Logging ──────────────────────────────────────────────────────────────────

const _warned = new Set<string>();

function warn(message: string): void {
  process.stderr.write(`[dunetrace.otel] ${message}\n`);
}

/** Log a warning at most once per process, so a persistently misconfigured
 *  deployment doesn't repeat the same line on every client construction. */
function warnOnce(message: string): void {
  if (_warned.has(message)) return;
  _warned.add(message);
  warn(message);
}

// ── Env parsing helpers ──────────────────────────────────────────────────────

function sdkVersion(): string {
  try {
    // Resolves to the package root from both dist/ and src/.
    const pkg = require("../package.json") as { version?: unknown };
    return typeof pkg.version === "string" ? pkg.version : "0.0.0";
  } catch {
    return "0.0.0";
  }
}

function envBool(raw: string | undefined, fallback: boolean): boolean {
  if (raw === undefined) return fallback;
  return ["1", "true", "yes", "on"].includes(raw.trim().toLowerCase());
}

/** Parse "k1=v1,k2=v2" into a record. Tolerant: whitespace is stripped, entries
 *  without a '=' or an empty key are skipped rather than throwing, so a
 *  malformed header string can't take down export init. */
export function parseHeaders(raw: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const part of raw.split(",")) {
    const entry = part.trim();
    if (!entry || !entry.includes("=")) continue;
    const idx = entry.indexOf("=");
    const key = entry.slice(0, idx).trim();
    if (key) out[key] = entry.slice(idx + 1).trim();
  }
  return out;
}

/** Parse a sampling ratio, clamped to [0.0, 1.0]. A non-numeric value falls
 *  back to the default (full sampling) rather than throwing. */
export function parseRatio(raw: string | undefined, fallback = 1.0): number {
  if (raw === undefined || raw.trim() === "") return fallback;
  const ratio = Number(raw);
  if (!Number.isFinite(ratio)) {
    warnOnce(`DUNETRACE_OTEL_SAMPLING_RATIO=${JSON.stringify(raw)} is not a number, using ${fallback}`);
    return fallback;
  }
  return Math.min(1.0, Math.max(0.0, ratio));
}

// ── Config ───────────────────────────────────────────────────────────────────

/** Read config from DUNETRACE_OTEL_* env vars. Overrides win over the
 *  environment, so callers (and tests) can force any field. */
export function configFromEnv(
  overrides: Partial<OtelConfig> = {},
  env: NodeJS.ProcessEnv = process.env,
): OtelConfig {
  let protocol = (env.DUNETRACE_OTEL_PROTOCOL ?? "grpc").trim().toLowerCase();
  if (!(VALID_PROTOCOLS as readonly string[]).includes(protocol)) {
    warnOnce(
      `DUNETRACE_OTEL_PROTOCOL=${JSON.stringify(protocol)} not one of ${VALID_PROTOCOLS.join(", ")}, using 'grpc'`,
    );
    protocol = "grpc";
  }
  const serviceName = (env.DUNETRACE_OTEL_SERVICE_NAME ?? "").trim() || DEFAULT_SERVICE_NAME;
  return {
    enabled: envBool(env.DUNETRACE_OTEL_ENABLED, false),
    endpoint: (env.DUNETRACE_OTEL_ENDPOINT ?? "").trim(),
    headers: parseHeaders(env.DUNETRACE_OTEL_HEADERS ?? ""),
    protocol: protocol as OtelProtocol,
    serviceName,
    serviceVersion: sdkVersion(),
    samplingRatio: parseRatio(env.DUNETRACE_OTEL_SAMPLING_RATIO),
    orgId: (env.DUNETRACE_ORG_ID ?? "").trim(),
    captureContent: envBool(env.DUNETRACE_OTEL_CAPTURE_CONTENT, true),
    ...overrides,
  };
}

// ── Circuit breaker ──────────────────────────────────────────────────────────

/**
 * Wraps a SpanExporter so a dead or slow collector can't turn into an endless
 * stream of failing export attempts.
 *
 * After FAILURE_THRESHOLD failed exports inside WINDOW_MS the circuit opens:
 * batches are dropped without touching the wrapped exporter for COOLDOWN_MS,
 * then it tries again. A single successful export closes the circuit and
 * clears the failure count. `now` is injectable so tests can drive the clock.
 */
export class CircuitBreakerExporter implements SpanExporter {
  static FAILURE_THRESHOLD = 5;
  static WINDOW_MS = 60_000;
  static COOLDOWN_MS = 60_000;

  private readonly _wrapped: SpanExporter;
  private readonly _now: () => number;
  private _failures: number[] = [];
  private _openUntil = 0;
  // -Infinity, not 0: performance.now() starts near 0 at process start, so a
  // 0 here silenced every failure in a short-lived process's first minute.
  private _lastWarn = -Infinity;

  constructor(wrapped: SpanExporter, now: () => number = () => performance.now()) {
    this._wrapped = wrapped;
    this._now = now;
  }

  /** True while the circuit is open and batches are being dropped. */
  isOpen(): boolean {
    return this._now() < this._openUntil;
  }

  export(spans: ReadableSpan[], resultCallback: (result: ExportResult) => void): void {
    const now = this._now();
    if (now < this._openUntil) {
      resultCallback({ code: EXPORT_FAILED }); // circuit open, drop this batch
      return;
    }
    let settled = false;
    const done = (result: ExportResult): void => {
      if (settled) return;
      settled = true;
      if (result.code === EXPORT_SUCCESS) {
        this._failures = [];
      } else {
        this._recordFailure(now, result.error?.message ?? "exporter returned FAILURE");
      }
      resultCallback(result);
    };
    try {
      this._wrapped.export(spans, done);
    } catch (err) {
      // A throwing exporter must not reach agent code. done() records the
      // failure (once), and is a no-op if the exporter already called back.
      done({ code: EXPORT_FAILED, error: err instanceof Error ? err : new Error(String(err)) });
    }
  }

  private _recordFailure(now: number, reason: string): void {
    const { FAILURE_THRESHOLD, WINDOW_MS, COOLDOWN_MS } = CircuitBreakerExporter;
    this._failures.push(now);
    this._failures = this._failures.filter((t) => now - t <= WINDOW_MS);
    const tripped = this._failures.length >= FAILURE_THRESHOLD;
    if (tripped) {
      this._openUntil = now + COOLDOWN_MS;
      this._failures = [];
    }
    // Throttled to at most once per window so a persistently down collector
    // doesn't flood the log.
    if (tripped || now - this._lastWarn > WINDOW_MS) {
      this._lastWarn = now;
      if (tripped) {
        warn(
          `export failing (${reason}). Circuit open for ${Math.round(COOLDOWN_MS / 1000)}s; ` +
            "spans dropped meanwhile. Agent unaffected.",
        );
      } else {
        warn(`span export failed (${reason}). Continuing; agent unaffected.`);
      }
    }
  }

  shutdown(): Promise<void> {
    return this._wrapped.shutdown().catch(() => undefined);
  }

  forceFlush(): Promise<void> {
    const fn = this._wrapped.forceFlush;
    if (typeof fn !== "function") return Promise.resolve();
    return fn.call(this._wrapped).catch(() => undefined);
  }
}

// ── Provider construction ────────────────────────────────────────────────────

const INSTALL_HINT =
  "npm install @opentelemetry/api @opentelemetry/sdk-trace-base @opentelemetry/resources " +
  "and @opentelemetry/exporter-trace-otlp-grpc or @opentelemetry/exporter-trace-otlp-proto";

/** Construct the OTLP exporter for the configured protocol. Throws on a
 *  genuinely broken setup (missing exporter package); the caller turns that into
 *  "OTel disabled", never a crash in agent code. */
/** The OTLP/HTTP traces URL for an endpoint. The JS exporter posts to `url`
 *  verbatim, so a bare collector or vendor base URL (no path) gets the
 *  standard `/v1/traces` path, as OTEL_EXPORTER_OTLP_ENDPOINT would. An
 *  endpoint that already has a path is left alone. */
export function httpTracesUrl(endpoint: string): string {
  try {
    const url = new URL(endpoint);
    if (url.pathname === "" || url.pathname === "/") {
      url.pathname = "/v1/traces";
      return url.toString();
    }
  } catch {
    // Not a URL; let the exporter report it.
  }
  return endpoint;
}

function buildSpanExporter(config: OtelConfig): SpanExporter {
  if (config.protocol === "http/protobuf") {
    const { OTLPTraceExporter } = require("@opentelemetry/exporter-trace-otlp-proto") as
      typeof import("@opentelemetry/exporter-trace-otlp-proto");
    return new OTLPTraceExporter({
      url: httpTracesUrl(config.endpoint),
      headers: config.headers,
      timeoutMillis: EXPORT_TIMEOUT_MS,
    });
  }
  const { OTLPTraceExporter } = require("@opentelemetry/exporter-trace-otlp-grpc") as
    typeof import("@opentelemetry/exporter-trace-otlp-grpc");
  let metadata: import("@grpc/grpc-js").Metadata | undefined;
  if (Object.keys(config.headers).length > 0) {
    // @grpc/grpc-js is the gRPC exporter's dependency, not ours: resolve it
    // from the exporter's location, so strict layouts (pnpm, Yarn PnP) find
    // it and we use the same copy the exporter does.
    const fromExporter = createRequire(require.resolve("@opentelemetry/exporter-trace-otlp-grpc"));
    const { Metadata } = fromExporter("@grpc/grpc-js") as typeof import("@grpc/grpc-js");
    metadata = new Metadata();
    for (const [key, value] of Object.entries(config.headers)) metadata.set(key, value);
  }
  return new OTLPTraceExporter({ url: config.endpoint, metadata, timeoutMillis: EXPORT_TIMEOUT_MS });
}

/** A Resource on either OpenTelemetry line: 2.x removed the `Resource` class
 *  in favour of `resourceFromAttributes` / `defaultResource`. Taking the module
 *  as an argument keeps one code path testable against both shapes. */
export function buildResource(resources: unknown, attrs: Record<string, string>): unknown {
  const r = resources as {
    resourceFromAttributes?: (a: Record<string, string>) => { merge(o: unknown): unknown };
    defaultResource?: () => { merge(o: unknown): unknown };
    Resource?: { new (a: Record<string, string>): unknown; default(): { merge(o: unknown): unknown } };
  };
  if (typeof r.resourceFromAttributes === "function" && typeof r.defaultResource === "function") {
    return r.defaultResource().merge(r.resourceFromAttributes(attrs));
  }
  if (r.Resource) return r.Resource.default().merge(new r.Resource(attrs));
  throw new Error("unrecognised @opentelemetry/resources API");
}

/**
 * Build a TracerProvider from config, or return null when export should stay
 * off (disabled, OpenTelemetry not installed, no endpoint, or init failed).
 * Never throws: a broken export pipeline degrades to null so the SDK keeps
 * running with OTel simply disabled.
 */
export function buildTracerProvider(
  config: OtelConfig,
  opts: InitOptions = {},
): BasicTracerProvider | null {
  if (!config.enabled) return null;
  if (!config.endpoint) {
    // Once per process: every client construction retries init(), and a
    // misconfigured deployment should not repeat this line for each one.
    warnOnce("DUNETRACE_OTEL_ENABLED is set but DUNETRACE_OTEL_ENDPOINT is empty; OTel export disabled.");
    return null;
  }
  let sdk: typeof import("@opentelemetry/sdk-trace-base");
  let resources: typeof import("@opentelemetry/resources");
  try {
    sdk = require("@opentelemetry/sdk-trace-base");
    resources = require("@opentelemetry/resources");
  } catch {
    warnOnce(
      "DUNETRACE_OTEL_ENABLED is set but OpenTelemetry is not installed; OTel export disabled. " +
        `Install with: ${INSTALL_HINT}`,
    );
    return null;
  }
  try {
    const exporter = new CircuitBreakerExporter(
      opts.exporterFactory ? opts.exporterFactory(config) : buildSpanExporter(config),
      opts.now,
    );
    const attrs: Record<string, string> = {
      "service.name": config.serviceName,
      // The SDK's version, not the customer's service version: service.version
      // belongs to whoever names the service. It also rides on the tracer's
      // instrumentation scope.
      "dunetrace.sdk.version": config.serviceVersion,
    };
    if (config.orgId) attrs["dunetrace.org_id"] = config.orgId;
    // Dunetrace's own run spans are sampled per run by the exporter
    // (runSampled) and parented on a sampled synthetic root, so this sampler
    // keeps exactly what the exporter chose to emit. The ratio here only
    // governs root spans opened with no parent; children follow their parent,
    // per standard OTel. Same as the Python SDK.
    const provider = new sdk.BasicTracerProvider({
      resource: buildResource(resources, attrs) as
        import("@opentelemetry/sdk-trace-base").TracerConfig["resource"],
      sampler: new sdk.ParentBasedSampler({
        root: new sdk.TraceIdRatioBasedSampler(config.samplingRatio),
      }),
      spanProcessors: [
        new sdk.BatchSpanProcessor(exporter, {
          maxQueueSize: MAX_QUEUE_SIZE,
          maxExportBatchSize: MAX_EXPORT_BATCH_SIZE,
          exportTimeoutMillis: EXPORT_TIMEOUT_MS,
        }),
      ],
    });
    return provider;
  } catch (err) {
    warnOnce(`failed to initialize export pipeline (${String(err)}); OTel export disabled.`);
    return null;
  }
}

// ── Module-global tracer ─────────────────────────────────────────────────────
//
// One provider/tracer per process. The client reads the tracer through
// getTracer(); a null tracer means "OTel off". The provider is deliberately NOT
// registered as the global OTel provider, so it never collides with tracing the
// application has set up for itself.

let _provider: BasicTracerProvider | null = null;
let _tracer: Tracer | null = null;
let _config: OtelConfig | null = null;
/** Bumped by shutdown(). A sink built for an earlier pipeline goes inert, so a
 *  client that outlives otel.shutdown() stops stamping OTel ids and emitting
 *  into a provider that drops everything. */
let _generation = 0;
/** An env-driven init that already failed. Env does not change under a
 *  running process, so every later client construction skips the retry (and
 *  its filesystem probes for missing packages). An explicit config always
 *  retries. */
let _envInitFailed = false;

/**
 * Idempotent bootstrap of the process-global tracer. Returns true when OTel
 * export is active after the call, false otherwise.
 *
 * Reads config from env when none is passed. Never throws: any failure degrades
 * to disabled so agent code is unaffected. Calling it again after a successful
 * init is a no-op (the first provider wins).
 */
export function init(config?: OtelConfig, opts: InitOptions = {}): boolean {
  if (_provider !== null) return true;
  if (config === undefined && _envInitFailed) return false;
  let cfg: OtelConfig;
  try {
    cfg = config ?? configFromEnv();
  } catch (err) {
    warn(`could not read config (${String(err)}); OTel export disabled.`);
    return false;
  }
  const provider = buildTracerProvider(cfg, opts);
  if (provider === null) {
    if (config === undefined) _envInitFailed = true;
    return false;
  }
  _provider = provider;
  _tracer = provider.getTracer(DEFAULT_SERVICE_NAME, cfg.serviceVersion);
  _config = cfg;
  return true;
}

/** The config OTel export was initialized with, or null when disabled. */
export function activeConfig(): OtelConfig | null {
  return _config;
}

/** The active Dunetrace tracer, or null when OTel export is disabled. */
export function getTracer(): Tracer | null {
  return _tracer;
}

/** True when OTel export is active (a provider was successfully built). */
export function isEnabled(): boolean {
  return _tracer !== null;
}

/** The active TracerProvider, or null when disabled. Mostly for tests. */
export function getTracerProvider(): BasicTracerProvider | null {
  return _provider;
}

/** Push buffered spans to the collector now, waiting at most `timeoutMs`.
 *  Safe to call when disabled and never rejects; the client calls it from
 *  shutdown() and before exit. The wait is bounded because a collector that
 *  accepts the connection and never answers would otherwise hold the caller
 *  for the full export timeout. The timer is unref'd so the bound itself never
 *  keeps the process alive. */
export function forceFlush(timeoutMs = EXPORT_TIMEOUT_MS): Promise<void> {
  const provider = _provider;
  if (provider === null) return Promise.resolve();
  const flush = provider.forceFlush().catch(() => undefined);
  // No budget left: the flush is started, but the caller does not wait on it.
  if (!(timeoutMs > 0)) return Promise.resolve();
  let timer: ReturnType<typeof setTimeout> | null = null;
  const deadline = new Promise<void>((resolve) => {
    timer = setTimeout(resolve, timeoutMs);
    if (typeof timer.unref === "function") timer.unref();
  });
  return Promise.race([flush, deadline]).finally(() => {
    if (timer !== null) clearTimeout(timer);
  });
}

/**
 * The flush the client runs from Node's `beforeExit`.
 *
 * Runs on every `beforeExit`, not once per process: work done after the
 * first exit event (a listener that starts another run) produces spans the
 * first flush never saw. Flushing an empty batch queue schedules no I/O, so
 * this does not keep the loop alive. Concurrent callers (one per client)
 * share the flush already in flight. Returns null when export is off.
 */
let _exitFlush: Promise<void> | null = null;
export function flushForExit(): Promise<void> | null {
  if (_provider === null) return null;
  if (_exitFlush === null) {
    _exitFlush = forceFlush().finally(() => {
      _exitFlush = null;
    });
  }
  return _exitFlush;
}

/** Flush and tear down the export pipeline. Safe to call when disabled. */
export function shutdown(): Promise<void> {
  const provider = _provider;
  _provider = null;
  _tracer = null;
  _config = null;
  _envInitFailed = false;
  _generation += 1;
  if (provider === null) return Promise.resolve();
  return provider.shutdown().catch(() => undefined);
}

/**
 * Build the client-side EventSink that turns AgentEvents into spans on the
 * active tracer, or null when export is off. Loads `integrations/otel.js` on
 * demand so `@opentelemetry/api` is only required once export is enabled.
 */
export function createEventSink(): EventSink | null {
  if (_tracer === null) return null;
  try {
    const mod = _exporterModule ?? (require(EXPORTER_MODULE) as ExporterModule);
    const inner = new mod.DunetraceOtelExporter({
      tracer: _tracer,
      captureContent: _config?.captureContent ?? true,
      samplingRatio: _config?.samplingRatio ?? 1,
    });
    return new PipelineSink(inner, _generation);
  } catch (err) {
    // Without a sink the tracer has no producer, so leaving it up would only
    // make isEnabled() report an export that never happens.
    warn(`exporter init failed (${String(err)}); OTel export disabled.`);
    void shutdown();
    return null;
  }
}

/** Loaded on demand so @opentelemetry/api is only required once export is
 *  enabled. A test pins this specifier to a real source file, so a rename
 *  cannot silently turn env-driven export off in the built package. */
export const EXPORTER_MODULE = "./integrations/otel.js";

interface ExporterModule {
  DunetraceOtelExporter: new (opts: {
    tracer: Tracer;
    captureContent: boolean;
    samplingRatio: number;
  }) => EventSink;
}

/** The env-built sink, bound to the pipeline generation it was built for.
 *  After otel.shutdown() it stops forwarding and reports no run as sampled,
 *  so its client stops stamping OTel ids for spans that would be dropped. A
 *  client constructed after a re-init gets a new sink on the new pipeline. */
class PipelineSink implements EventSink {
  readonly exportsOtelSpans = true;
  constructor(
    private readonly _inner: EventSink,
    private readonly _gen: number,
  ) {}

  private get _live(): boolean {
    return this._gen === _generation;
  }

  shouldSample(runId: string, parentRunId: string | null): boolean {
    if (!this._live) return false;
    return this._inner.shouldSample ? this._inner.shouldSample(runId, parentRunId) : true;
  }

  handle(event: AgentEvent): void {
    if (this._live) this._inner.handle(event);
  }
}

let _exporterModule: ExporterModule | null = null;

/**
 * Drop the global provider/tracer and warning state without flushing the
 * pipeline. Test hook only.
 * @internal
 */
export function _resetForTests(exporterModule: ExporterModule | null = null): void {
  _provider = null;
  _tracer = null;
  _config = null;
  _envInitFailed = false;
  _exitFlush = null;
  _generation += 1;
  _warned.clear();
  _exporterModule = exporterModule;
}
