/**
 * Tests for src/otel.ts, the env-driven OTel export bootstrap.
 *
 * Mirrors the Python SDK's tests for dunetrace/otel.py: config parsing never
 * throws, a missing endpoint or a disabled flag yields no provider, init is
 * idempotent, the circuit breaker opens and closes on an injectable clock, and
 * a client built with export enabled stamps runs and emits spans without the
 * caller wiring an exporter.
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { InMemorySpanExporter, type ReadableSpan, type SpanExporter } from "@opentelemetry/sdk-trace-base";
import { ExportResultCode, type ExportResult } from "@opentelemetry/core";
import * as otel from "../src/otel.js";
import * as otelIntegration from "../src/integrations/otel.js";
import { Dunetrace } from "../src/client.js";
import { NoopBatchEmitter } from "../src/emitters.js";
import type { AgentEvent, EventType } from "../src/models.js";
import { runSampled } from "../src/otel-ids.js";
import { randomUUID } from "node:crypto";

const ENV_KEYS = [
  "DUNETRACE_OTEL_ENABLED",
  "DUNETRACE_OTEL_ENDPOINT",
  "DUNETRACE_OTEL_HEADERS",
  "DUNETRACE_OTEL_PROTOCOL",
  "DUNETRACE_OTEL_SERVICE_NAME",
  "DUNETRACE_OTEL_SAMPLING_RATIO",
  "DUNETRACE_OTEL_CAPTURE_CONTENT",
  "DUNETRACE_ORG_ID",
];

const savedEnv: Record<string, string | undefined> = {};

beforeEach(() => {
  for (const k of ENV_KEYS) {
    savedEnv[k] = process.env[k];
    delete process.env[k];
  }
  // Under vitest the src tree has no compiled integrations/otel.js to require,
  // so hand the bootstrap the module directly. Production loads it on demand.
  otel._resetForTests(otelIntegration);
});

afterEach(async () => {
  await otel.shutdown();
  otel._resetForTests();
  for (const k of ENV_KEYS) {
    if (savedEnv[k] === undefined) delete process.env[k];
    else process.env[k] = savedEnv[k];
  }
});

function enabledConfig(overrides: Partial<otel.OtelConfig> = {}): otel.OtelConfig {
  return otel.configFromEnv({ enabled: true, endpoint: "http://collector:4318/v1/traces", ...overrides }, {});
}

const RUN_ID = "1b4e28ba-2fa1-11d2-883f-0016d3cca427";
let clock = 1_700_000_000;

/** An AgentEvent for RUN_ID, timestamps advancing so spans have durations. */
function runEvent(type: EventType, payload: Record<string, unknown>): AgentEvent {
  clock += 0.5;
  return { event_type: type, run_id: RUN_ID, agent_id: "agent", agent_version: "v1", step_index: 1, timestamp: clock, payload };
}

// ── Config ───────────────────────────────────────────────────────────────────

describe("configFromEnv", () => {
  it("is disabled with defaults when nothing is set", () => {
    const cfg = otel.configFromEnv({}, {});
    expect(cfg.enabled).toBe(false);
    expect(cfg.endpoint).toBe("");
    expect(cfg.protocol).toBe("grpc");
    expect(cfg.serviceName).toBe("dunetrace");
    expect(cfg.samplingRatio).toBe(1.0);
    expect(cfg.captureContent).toBe(true);
    expect(cfg.headers).toEqual({});
    expect(cfg.serviceVersion).toMatch(/^\d+\.\d+\.\d+/);
  });

  it("reads every DUNETRACE_OTEL_* variable", () => {
    const cfg = otel.configFromEnv({}, {
      DUNETRACE_OTEL_ENABLED: "true",
      DUNETRACE_OTEL_ENDPOINT: " http://localhost:4318/v1/traces ",
      DUNETRACE_OTEL_HEADERS: "DD-API-KEY=abc, x-team = infra ,broken,=novalue",
      DUNETRACE_OTEL_PROTOCOL: "HTTP/protobuf",
      DUNETRACE_OTEL_SERVICE_NAME: "billing-agent",
      DUNETRACE_OTEL_SAMPLING_RATIO: "0.25",
      DUNETRACE_OTEL_CAPTURE_CONTENT: "0",
      DUNETRACE_ORG_ID: "org-1",
    });
    expect(cfg.enabled).toBe(true);
    expect(cfg.endpoint).toBe("http://localhost:4318/v1/traces");
    expect(cfg.headers).toEqual({ "DD-API-KEY": "abc", "x-team": "infra" });
    expect(cfg.protocol).toBe("http/protobuf");
    expect(cfg.serviceName).toBe("billing-agent");
    expect(cfg.samplingRatio).toBe(0.25);
    expect(cfg.captureContent).toBe(false);
    expect(cfg.orgId).toBe("org-1");
  });

  it("falls back instead of throwing on bad values", () => {
    const cfg = otel.configFromEnv({}, {
      DUNETRACE_OTEL_PROTOCOL: "thrift",
      DUNETRACE_OTEL_SAMPLING_RATIO: "lots",
      DUNETRACE_OTEL_SERVICE_NAME: "   ",
    });
    expect(cfg.protocol).toBe("grpc");
    expect(cfg.samplingRatio).toBe(1.0);
    expect(cfg.serviceName).toBe("dunetrace");
    expect(otel.parseRatio("7")).toBe(1.0);
    expect(otel.parseRatio("-1")).toBe(0.0);
  });

  it("lets overrides win over the environment", () => {
    const cfg = otel.configFromEnv({ enabled: true, samplingRatio: 0.5 }, { DUNETRACE_OTEL_ENABLED: "0" });
    expect(cfg.enabled).toBe(true);
    expect(cfg.samplingRatio).toBe(0.5);
  });

  it("pins the export result codes it spells out to @opentelemetry/core", () => {
    // otel.ts avoids importing core at load time and hardcodes the codes; if
    // the enum ever changes, the breaker would misread results.
    expect(ExportResultCode.SUCCESS).toBe(0);
    expect(ExportResultCode.FAILED).toBe(1);
    const breaker = new otel.CircuitBreakerExporter(
      new ScriptedExporter(() => ({ code: ExportResultCode.SUCCESS })),
    );
    expect(attempt(breaker).code).toBe(ExportResultCode.SUCCESS);
  });
});

// ── Provider ─────────────────────────────────────────────────────────────────

describe("buildTracerProvider / init", () => {
  it("returns null when disabled", () => {
    expect(otel.buildTracerProvider(otel.configFromEnv({}, {}))).toBeNull();
    expect(otel.init(otel.configFromEnv({}, {}))).toBe(false);
    expect(otel.isEnabled()).toBe(false);
    expect(otel.getTracer()).toBeNull();
  });

  it("returns null when enabled without an endpoint", () => {
    expect(otel.buildTracerProvider(otel.configFromEnv({ enabled: true }, {}))).toBeNull();
  });

  it("builds a provider with a real OTLP exporter for each protocol", () => {
    for (const protocol of ["http/protobuf", "grpc"] as const) {
      const provider = otel.buildTracerProvider(enabledConfig({ protocol, headers: { "x-key": "v" } }));
      expect(provider, protocol).not.toBeNull();
      // Nothing was exported, so shutting down must not try the network.
      void provider!.shutdown();
    }
  });

  it("degrades to null when the exporter cannot be built", () => {
    const provider = otel.buildTracerProvider(enabledConfig(), {
      exporterFactory: () => { throw new Error("no exporter for you"); },
    });
    expect(provider).toBeNull();
  });

  it("init is idempotent and the first provider wins", () => {
    const memory = new InMemorySpanExporter();
    expect(otel.init(enabledConfig(), { exporterFactory: () => memory })).toBe(true);
    const first = otel.getTracerProvider();
    expect(otel.init(enabledConfig({ serviceName: "other" }), { exporterFactory: () => new InMemorySpanExporter() })).toBe(true);
    expect(otel.getTracerProvider()).toBe(first);
    expect(otel.activeConfig()!.serviceName).toBe("dunetrace");
    expect(otel.isEnabled()).toBe(true);
  });

  it("stamps service, version, and org on the resource", () => {
    const memory = new InMemorySpanExporter();
    otel.init(enabledConfig({ orgId: "org-9", serviceName: "svc" }), { exporterFactory: () => memory });
    const attrs = otel.getTracerProvider()!.resource.attributes;
    expect(attrs["service.name"]).toBe("svc");
    expect(attrs["dunetrace.org_id"]).toBe("org-9");
    // The SDK's own version, not service.version: that belongs to whoever
    // names the service, often the customer via DUNETRACE_OTEL_SERVICE_NAME.
    expect(String(attrs["dunetrace.sdk.version"])).toMatch(/^\d+\.\d+\.\d+/);
    expect(attrs["service.version"]).toBeUndefined();
  });

  it("applies the sampling ratio to run spans, whose synthetic parent is already flagged sampled", async () => {
    // A dunetrace.run span starts under a remote parent context that carries
    // TraceFlags.SAMPLED. A parent-based sampler would honour that flag and
    // never consult the ratio, so the check has to go through the exporter,
    // not a bare tracer.startSpan().
    const memory = new InMemorySpanExporter();
    otel.init(enabledConfig({ samplingRatio: 0 }), { exporterFactory: () => memory });
    const sink = otel.createEventSink()!;
    sink.handle(runEvent("run.started", { model: "gpt-4o", tools: [] }));
    sink.handle(runEvent("tool.called", { tool_name: "lookup", args: "{}" }));
    sink.handle(runEvent("tool.responded", { tool_name: "lookup", success: true }));
    sink.handle(runEvent("run.completed", { total_steps: 1, exit_reason: "final_answer", tool_call_count: 1 }));
    await otel.forceFlush();
    expect(memory.getFinishedSpans()).toHaveLength(0);

    // Ratio 1 keeps the whole trace.
    await otel.shutdown();
    otel._resetForTests(otelIntegration);
    const kept = new InMemorySpanExporter();
    otel.init(enabledConfig({ samplingRatio: 1 }), { exporterFactory: () => kept });
    const sink2 = otel.createEventSink()!;
    sink2.handle(runEvent("run.started", { model: "gpt-4o", tools: [] }));
    sink2.handle(runEvent("run.completed", { total_steps: 0, exit_reason: "final_answer", tool_call_count: 0 }));
    await otel.forceFlush();
    expect(kept.getFinishedSpans().map((s) => s.name)).toEqual(["dunetrace.run"]);
  });

  it("forceFlush returns at once when the budget is already spent", async () => {
    const stuck: SpanExporter = {
      export: () => { /* never calls back */ },
      shutdown: () => Promise.resolve(),
      forceFlush: () => Promise.resolve(),
    };
    otel.init(enabledConfig(), { exporterFactory: () => stuck });
    otel.getTracer()!.startSpan("pending").end();
    const t0 = Date.now();
    await otel.forceFlush(0);
    await otel.forceFlush(-5);
    expect(Date.now() - t0).toBeLessThan(100);
  });

  it("warns once per process on a repeated misconfiguration", () => {
    const lines: string[] = [];
    const orig = process.stderr.write;
    process.stderr.write = ((chunk: string | Uint8Array) => { lines.push(String(chunk)); return true; }) as typeof process.stderr.write;
    try {
      // Every client construction retries init(); an enabled-but-empty
      // endpoint must not produce one line per client.
      const cfg = otel.configFromEnv({ enabled: true }, {});
      expect(otel.init(cfg)).toBe(false);
      expect(otel.init(cfg)).toBe(false);
      expect(otel.buildTracerProvider(enabledConfig(), { exporterFactory: () => { throw new Error("x"); } })).toBeNull();
      expect(otel.buildTracerProvider(enabledConfig(), { exporterFactory: () => { throw new Error("x"); } })).toBeNull();
    } finally {
      process.stderr.write = orig;
    }
    expect(lines.filter((l) => l.includes("DUNETRACE_OTEL_ENDPOINT is empty"))).toHaveLength(1);
    expect(lines.filter((l) => l.includes("failed to initialize export pipeline"))).toHaveLength(1);
  });

  it("forceFlush gives up after its timeout when the exporter never answers", async () => {
    const stuck: SpanExporter = {
      export: () => { /* never calls back */ },
      shutdown: () => Promise.resolve(),
      forceFlush: () => Promise.resolve(),
    };
    otel.init(enabledConfig(), { exporterFactory: () => stuck });
    otel.getTracer()!.startSpan("pending").end();
    const t0 = Date.now();
    await otel.forceFlush(50);
    expect(Date.now() - t0).toBeLessThan(1000);
  });

  it("tears the pipeline down when the event sink cannot be built", () => {
    otel._resetForTests({
      DunetraceOtelExporter: class { constructor() { throw new Error("no sink"); } } as never,
    });
    otel.init(enabledConfig(), { exporterFactory: () => new InMemorySpanExporter() });
    expect(otel.isEnabled()).toBe(true);
    expect(otel.createEventSink()).toBeNull();
    expect(otel.isEnabled()).toBe(false);
    expect(otel.getTracerProvider()).toBeNull();
  });

  it("forceFlush delivers queued spans; shutdown clears state and is safe to repeat", async () => {
    const memory = new InMemorySpanExporter();
    otel.init(enabledConfig(), { exporterFactory: () => memory });
    otel.getTracer()!.startSpan("kept").end();
    expect(memory.getFinishedSpans()).toHaveLength(0); // still in the batch queue
    await otel.forceFlush();
    expect(memory.getFinishedSpans().map((s) => s.name)).toEqual(["kept"]);
    // InMemorySpanExporter.shutdown() resets its buffer, so assert state only.
    await otel.shutdown();
    expect(otel.isEnabled()).toBe(false);
    expect(otel.activeConfig()).toBeNull();
    await expect(otel.shutdown()).resolves.toBeUndefined();
    await expect(otel.forceFlush()).resolves.toBeUndefined();
  });
});

// ── Circuit breaker ──────────────────────────────────────────────────────────

class ScriptedExporter implements SpanExporter {
  calls = 0;
  constructor(private readonly outcome: () => ExportResult | Error) {}
  export(_spans: ReadableSpan[], cb: (r: ExportResult) => void): void {
    this.calls += 1;
    const o = this.outcome();
    if (o instanceof Error) throw o;
    cb(o);
  }
  shutdown(): Promise<void> { return Promise.resolve(); }
  forceFlush(): Promise<void> { return Promise.resolve(); }
}

function attempt(breaker: otel.CircuitBreakerExporter): ExportResult {
  let seen: ExportResult | null = null;
  breaker.export([], (r) => { seen = r; });
  return seen!;
}

describe("CircuitBreakerExporter", () => {
  let tick = 0;
  const now = () => tick;
  beforeEach(() => { tick = 1_000_000; });

  it("opens after the failure threshold and stops calling the exporter", () => {
    const inner = new ScriptedExporter(() => ({ code: ExportResultCode.FAILED, error: new Error("refused") }));
    const breaker = new otel.CircuitBreakerExporter(inner, now);
    for (let i = 0; i < otel.CircuitBreakerExporter.FAILURE_THRESHOLD; i++) {
      tick += 1000;
      expect(attempt(breaker).code).toBe(ExportResultCode.FAILED);
    }
    expect(breaker.isOpen()).toBe(true);
    const callsAtTrip = inner.calls;
    tick += 1000;
    expect(attempt(breaker).code).toBe(ExportResultCode.FAILED);
    expect(inner.calls).toBe(callsAtTrip); // dropped without touching the exporter
  });

  it("retries after the cooldown and closes on the first success", () => {
    let fail = true;
    const inner = new ScriptedExporter(() => (fail ? new Error("boom") : { code: ExportResultCode.SUCCESS }));
    const breaker = new otel.CircuitBreakerExporter(inner, now);
    for (let i = 0; i < otel.CircuitBreakerExporter.FAILURE_THRESHOLD; i++) attempt(breaker);
    expect(breaker.isOpen()).toBe(true);

    tick += otel.CircuitBreakerExporter.COOLDOWN_MS + 1;
    fail = false;
    expect(breaker.isOpen()).toBe(false);
    expect(attempt(breaker).code).toBe(ExportResultCode.SUCCESS);
    // Failures were cleared on success: a single new failure must not trip it.
    fail = true;
    attempt(breaker);
    expect(breaker.isOpen()).toBe(false);
  });

  it("only counts failures inside the window", () => {
    const inner = new ScriptedExporter(() => ({ code: ExportResultCode.FAILED }));
    const breaker = new otel.CircuitBreakerExporter(inner, now);
    for (let i = 0; i < otel.CircuitBreakerExporter.FAILURE_THRESHOLD - 1; i++) attempt(breaker);
    tick += otel.CircuitBreakerExporter.WINDOW_MS + 1; // the earlier failures age out
    attempt(breaker);
    expect(breaker.isOpen()).toBe(false);
  });

  it("counts a throwing exporter's failure once, so it trips at the threshold", () => {
    const inner = new ScriptedExporter(() => new Error("exporter bug"));
    const breaker = new otel.CircuitBreakerExporter(inner, now);
    for (let i = 0; i < otel.CircuitBreakerExporter.FAILURE_THRESHOLD - 1; i++) attempt(breaker);
    expect(breaker.isOpen()).toBe(false); // used to open after 3 of 5
    attempt(breaker);
    expect(breaker.isOpen()).toBe(true);
  });

  it("warns about a failure in the first minute of process uptime", () => {
    // performance.now() starts near 0; the throttle used to start at 0 too, so
    // a short-lived process never logged a failing export.
    tick = 5;
    const lines: string[] = [];
    const orig = process.stderr.write;
    process.stderr.write = ((chunk: string | Uint8Array) => { lines.push(String(chunk)); return true; }) as typeof process.stderr.write;
    try {
      const breaker = new otel.CircuitBreakerExporter(
        new ScriptedExporter(() => ({ code: ExportResultCode.FAILED, error: new Error("refused") })),
        now,
      );
      attempt(breaker);
    } finally {
      process.stderr.write = orig;
    }
    expect(lines.some((l) => l.includes("span export failed (refused)"))).toBe(true);
  });

  it("turns a throwing exporter into a FAILED result instead of propagating", () => {
    const inner = new ScriptedExporter(() => new Error("exporter bug"));
    const breaker = new otel.CircuitBreakerExporter(inner, now);
    const result = attempt(breaker);
    expect(result.code).toBe(ExportResultCode.FAILED);
    expect(result.error?.message).toBe("exporter bug");
  });
});

// ── Client wiring ────────────────────────────────────────────────────────────

describe("Dunetrace client auto-wiring", () => {
  it("leaves runs unstamped and emits no spans when OTel is off", async () => {
    const dt = new Dunetrace({ emitter: new NoopBatchEmitter(), flushOnExit: false });
    await dt.run("agent", { model: "gpt-4o" }, async (run) => {
      expect(run.otelTraceId).toBeNull();
      expect(run.otelParentSpanId).toBeNull();
    });
    await dt.shutdown();
  });

  it("builds the exporter from env and emits spans on the shared tracer", async () => {
    const memory = new InMemorySpanExporter();
    process.env.DUNETRACE_OTEL_ENABLED = "1";
    process.env.DUNETRACE_OTEL_ENDPOINT = "http://collector:4318/v1/traces";
    process.env.DUNETRACE_OTEL_CAPTURE_CONTENT = "false";
    // Same env-driven path the client takes, with the network exporter swapped out.
    expect(otel.init(undefined, { exporterFactory: () => memory })).toBe(true);

    const dt = new Dunetrace({ emitter: new NoopBatchEmitter(), flushOnExit: false });
    let runId = "";
    await dt.run("billing-agent", { model: "gpt-4o" }, async (run) => {
      runId = run.runId;
      expect(run.otelTraceId).toBe(runId.replace(/-/g, ""));
      expect(run.otelParentSpanId).toBe(run.otelTraceId!.slice(16));
      run.toolCalled("lookup", { url: "https://crm.internal/acct/42" });
      run.toolResponded("lookup", true, 12, 8);
      run.finalAnswer();
    });
    await dt.shutdown(); // flushes the batch processor too

    const spans = memory.getFinishedSpans();
    const names = spans.map((s) => s.name).sort();
    expect(names).toEqual(["dunetrace.run", "dunetrace.tool.lookup"]);
    const root = spans.find((s) => s.name === "dunetrace.run")!;
    expect(root.spanContext().traceId).toBe(runId.replace(/-/g, ""));
    // The run span's own id is SDK-assigned; what the run carries is the id of
    // the synthetic parent it hangs under, which is how a backend finds it.
    expect(root.parentSpanId).toBe(runId.replace(/-/g, "").slice(16));
    expect(root.spanContext().spanId).not.toBe(root.parentSpanId);
    // captureContent=false came through from env: no URL on the tool span.
    const tool = spans.find((s) => s.name === "dunetrace.tool.lookup")!;
    expect(tool.attributes["url.full"]).toBeUndefined();
    expect(tool.attributes["server.address"]).toBe("crm.internal");
  });

  it("every exit flush pushes what is queued, including spans from after the first", async () => {
    const memory = new InMemorySpanExporter();
    process.env.DUNETRACE_OTEL_ENABLED = "1";
    process.env.DUNETRACE_OTEL_ENDPOINT = "http://collector:4318/v1/traces";
    otel.init(undefined, { exporterFactory: () => memory });

    const dt = new Dunetrace({ emitter: new NoopBatchEmitter(), flushOnExit: false });
    await dt.run("agent", { model: "gpt-4o" }, async (run) => { run.finalAnswer(); });
    expect(memory.getFinishedSpans()).toHaveLength(0); // sitting in the batch queue

    await dt._flushBeforeExit();
    expect(memory.getFinishedSpans().map((s) => s.name)).toEqual(["dunetrace.run"]);

    // Work done after the first beforeExit (a listener that starts another
    // run) used to be lost: the flush only ever ran once per process.
    otel.getTracer()!.startSpan("after-exit").end();
    await dt._flushBeforeExit();
    expect(memory.getFinishedSpans().map((s) => s.name)).toEqual(["dunetrace.run", "after-exit"]);
    await dt.shutdown();
  });

  it("an explicit exporter option wins over the env bootstrap", async () => {
    const memory = new InMemorySpanExporter();
    process.env.DUNETRACE_OTEL_ENABLED = "1";
    process.env.DUNETRACE_OTEL_ENDPOINT = "http://collector:4318/v1/traces";
    otel.init(undefined, { exporterFactory: () => memory });

    const handled: string[] = [];
    const dt = new Dunetrace({
      emitter: new NoopBatchEmitter(),
      flushOnExit: false,
      exporter: { handle: (e) => { handled.push(e.event_type); } },
    });
    await dt.run("agent", { model: "gpt-4o" }, async (run) => {
      // A sink that is not an OTel exporter gets no OTel correlation ids,
      // even though export is enabled in the environment.
      expect(run.otelTraceId).toBeNull();
      expect(run.otelParentSpanId).toBeNull();
      run.finalAnswer();
    });
    await dt.shutdown();
    expect(handled).toEqual(["run.started", "run.completed"]);
    expect(memory.getFinishedSpans()).toHaveLength(0);
  });

  it("an explicit DunetraceOtelExporter on the caller's own tracer still gets the ids", async () => {
    // No env, no bootstrap: the caller wired the exporter by hand.
    const { BasicTracerProvider, SimpleSpanProcessor } = await import("@opentelemetry/sdk-trace-base");
    const memory = new InMemorySpanExporter();
    const provider = new BasicTracerProvider({ spanProcessors: [new SimpleSpanProcessor(memory)] });
    const dt = new Dunetrace({
      emitter: new NoopBatchEmitter(),
      flushOnExit: false,
      exporter: new otelIntegration.DunetraceOtelExporter({ tracer: provider.getTracer("mine") }),
    });
    expect(otel.isEnabled()).toBe(false);
    let traceId = "";
    await dt.run("agent", { model: "gpt-4o" }, async (run) => {
      traceId = run.otelTraceId ?? "";
      expect(traceId).toBe(run.runId.replace(/-/g, ""));
      run.finalAnswer();
    });
    await dt.shutdown();
    expect(memory.getFinishedSpans()[0].spanContext().traceId).toBe(traceId);
  });

  it("shutdown honours its deadline even when the ingest drain has spent it and the collector is silent", async () => {
    // The HTTP drain and the OTel flush share one budget. If the drain uses
    // all of it, the OTel flush gets a zero budget and must not add the
    // processor's own export timeout on top.
    const stuck: SpanExporter = {
      export: () => { /* never calls back */ },
      shutdown: () => Promise.resolve(),
      forceFlush: () => Promise.resolve(),
    };
    process.env.DUNETRACE_OTEL_ENABLED = "1";
    process.env.DUNETRACE_OTEL_ENDPOINT = "http://collector:4318/v1/traces";
    otel.init(undefined, { exporterFactory: () => stuck });

    const budgetMs = 150;
    const slowEmitter = {
      // Each batch takes the whole budget to ship, so the drain loop exits on
      // the deadline with the OTel flush owed nothing.
      ship: () => new Promise<void>((r) => setTimeout(r, budgetMs)),
    };
    const dt = new Dunetrace({ emitter: slowEmitter, flushOnExit: false });
    await dt.run("agent", { model: "gpt-4o" }, async (run) => { run.finalAnswer(); });

    const t0 = Date.now();
    await dt.shutdown(budgetMs);
    const took = Date.now() - t0;
    expect(took).toBeGreaterThanOrEqual(budgetMs - 5);
    expect(took).toBeLessThan(budgetMs * 3);
  });

  it("concurrent exit flushes from several clients share one flush", async () => {
    const memory = new InMemorySpanExporter();
    process.env.DUNETRACE_OTEL_ENABLED = "1";
    process.env.DUNETRACE_OTEL_ENDPOINT = "http://collector:4318/v1/traces";
    otel.init(undefined, { exporterFactory: () => memory });
    const a = new Dunetrace({ emitter: new NoopBatchEmitter(), flushOnExit: false });
    const b = new Dunetrace({ emitter: new NoopBatchEmitter(), flushOnExit: false });
    await a.run("a", { model: "m" }, async (run) => { run.finalAnswer(); });
    await b.run("b", { model: "m" }, async (run) => { run.finalAnswer(); });

    const first = otel.flushForExit();
    expect(otel.flushForExit()).toBe(first); // the second client joins, not repeats
    await Promise.all([a._flushBeforeExit(), b._flushBeforeExit()]);
    expect(memory.getFinishedSpans()).toHaveLength(2);
    await a.shutdown();
    await b.shutdown();
  });
});

// ── Per-run sampling, shared with the Python SDK ─────────────────────────────

describe("runSampled", () => {
  it("matches the Python SDK's known vectors", () => {
    // Pinned in packages/sdk-py/tests/test_otel_config.py::TestRunSampled.
    expect(runSampled("00000000-0000-4000-8000-000000000000", 0.5)).toBe(false);
    expect(runSampled("00000000-0000-4000-8000-000000000001", 0.5)).toBe(true);
  });

  it("is a real probability over UUID run ids", () => {
    const ids = Array.from({ length: 4000 }, () => randomUUID());
    for (const ratio of [0.1, 0.25, 0.5, 0.9]) {
      const kept = ids.filter((id) => runSampled(id, ratio)).length / ids.length;
      expect(Math.abs(kept - ratio)).toBeLessThan(0.04);
    }
  });

  it("handles the edges", () => {
    expect(runSampled("x", 1)).toBe(true);
    expect(runSampled("x", 0)).toBe(false);
    expect(runSampled("x", Number.NaN)).toBe(false);
  });
});

function idsByDecision(ratio: number): { kept: string; dropped: string } {
  let kept = "";
  let dropped = "";
  while (!kept || !dropped) {
    const id = randomUUID();
    if (runSampled(id, ratio)) kept ||= id;
    else dropped ||= id;
  }
  return { kept, dropped };
}

describe("per-run sampling through the client", () => {
  async function startedPayloadFor(ratio: number, runId: string) {
    const memory = new InMemorySpanExporter();
    otel.init(enabledConfig({ samplingRatio: ratio }), { exporterFactory: () => memory });
    const sink = otel.createEventSink()!;
    const seen: AgentEvent[] = [];
    const spy = { exportsOtelSpans: true as const, shouldSample: sink.shouldSample!.bind(sink), handle: (e: AgentEvent) => { seen.push(e); sink.handle(e); } };
    const dt = new Dunetrace({ emitter: new NoopBatchEmitter(), flushOnExit: false, exporter: spy });
    let ids: [string | null, string | null] = [null, null];
    await dt.run("agent", { model: "m", runId }, async (run) => {
      ids = [run.otelTraceId, run.otelParentSpanId];
      run.finalAnswer();
    });
    await dt.shutdown();
    const started = seen.find((e) => e.event_type === "run.started")!;
    return { payload: started.payload, ids, spans: memory.getFinishedSpans() };
  }

  it("a dropped run gets no trace id, records the decision, and exports nothing", async () => {
    const { dropped } = idsByDecision(0.5);
    const r = await startedPayloadFor(0.5, dropped);
    expect(r.payload["otel_sampled"]).toBe(false);
    expect(r.ids).toEqual([null, null]);
    expect(r.spans).toHaveLength(0);
  });

  it("a kept run gets its trace id and records the decision", async () => {
    const { kept } = idsByDecision(0.5);
    const r = await startedPayloadFor(0.5, kept);
    expect(r.payload["otel_sampled"]).toBe(true);
    expect(r.ids[0]).toBe(kept.replace(/-/g, ""));
    expect(r.spans.map((s) => s.name)).toEqual(["dunetrace.run"]);
  });

  it("a nested child inherits its parent's decision", () => {
    const exp = new otelIntegration.DunetraceOtelExporter({ tracer: otel.getTracer() ?? undefined, samplingRatio: 0.5 });
    const { kept, dropped } = idsByDecision(0.5);
    expect(exp.shouldSample(kept)).toBe(true);
    expect(exp.shouldSample(dropped, kept)).toBe(true);
    expect(exp.shouldSample(randomUUID(), dropped)).toBe(true); // dropped is now kept-by-inheritance
  });
});

describe("pipeline lifecycle", () => {
  it("a sink built before shutdown goes inert", async () => {
    const memory = new InMemorySpanExporter();
    otel.init(enabledConfig(), { exporterFactory: () => memory });
    const sink = otel.createEventSink()!;
    await otel.shutdown();
    expect(sink.shouldSample!(RUN_ID, null)).toBe(false);
    sink.handle(runEvent("run.started", { model: "m", tools: [] }));
    expect(memory.getFinishedSpans()).toHaveLength(0);
  });

  it("remembers a failed env init instead of retrying it per client", () => {
    process.env.DUNETRACE_OTEL_ENABLED = "1"; // no endpoint
    const lines: string[] = [];
    const orig = process.stderr.write;
    process.stderr.write = ((chunk: string | Uint8Array) => { lines.push(String(chunk)); return true; }) as typeof process.stderr.write;
    try {
      expect(otel.init()).toBe(false);
      process.env.DUNETRACE_OTEL_ENDPOINT = "http://collector:4318/v1/traces";
      expect(otel.init()).toBe(false); // remembered; env doesn't change under a process
    } finally {
      process.stderr.write = orig;
    }
  });

  it("the on-demand exporter module specifier names a real source file", async () => {
    const path = await import("node:path");
    const fs = await import("node:fs");
    const src = path.resolve(__dirname, "../src", otel.EXPORTER_MODULE.replace(/\.js$/, ".ts"));
    expect(fs.existsSync(src)).toBe(true);
    expect(typeof otelIntegration.DunetraceOtelExporter).toBe("function");
  });
});

describe("OpenTelemetry 1.x and 2.x resource APIs", () => {
  it("uses resourceFromAttributes/defaultResource when present (2.x)", () => {
    const calls: string[] = [];
    const fake2x = {
      defaultResource: () => ({ merge: (o: unknown) => { calls.push("merge"); return o; } }),
      resourceFromAttributes: (a: Record<string, string>) => { calls.push("fromAttributes"); return a; },
    };
    expect(otel.buildResource(fake2x, { "service.name": "s" })).toEqual({ "service.name": "s" });
    expect(calls).toEqual(["fromAttributes", "merge"]);
  });

  it("falls back to the Resource class (1.x)", async () => {
    const resources = await import("@opentelemetry/resources");
    const r = otel.buildResource(resources, { "service.name": "s" }) as { attributes: Record<string, unknown> };
    expect(r.attributes["service.name"]).toBe("s");
  });
});

describe("httpTracesUrl", () => {
  it("adds /v1/traces to a bare base URL and keeps an explicit path", () => {
    expect(otel.httpTracesUrl("https://ingest.eu.signoz.cloud:443")).toBe("https://ingest.eu.signoz.cloud/v1/traces");
    expect(otel.httpTracesUrl("http://collector:4318/")).toBe("http://collector:4318/v1/traces");
    expect(otel.httpTracesUrl("http://c:4318/v1/traces")).toBe("http://c:4318/v1/traces");
    expect(otel.httpTracesUrl("not a url")).toBe("not a url");
  });
});

describe("Greptile review: sampling hook isolation and cross-client trees", () => {
  it("a throwing shouldSample never stops the run", async () => {
    const seen: AgentEvent[] = [];
    const sink = {
      exportsOtelSpans: true as const,
      shouldSample: () => { throw new Error("sink bug"); },
      handle: (e: AgentEvent) => { seen.push(e); },
    };
    const dt = new Dunetrace({ emitter: new NoopBatchEmitter(), flushOnExit: false, exporter: sink });
    let ran = false;
    await dt.run("agent", { model: "m" }, async (run) => {
      ran = true;
      expect(run.otelTraceId).toBeNull();
      run.finalAnswer();
    });
    await dt.shutdown();
    expect(ran).toBe(true);
    const started = seen.find((e) => e.event_type === "run.started")!;
    expect("otel_sampled" in started.payload).toBe(false);
  });

  it("a child opened by another client's exporter inherits the parent's decision", () => {
    const a = new otelIntegration.DunetraceOtelExporter({ tracer: undefined, samplingRatio: 0.5 });
    const b = new otelIntegration.DunetraceOtelExporter({ tracer: undefined, samplingRatio: 0.5 });
    const { kept, dropped } = idsByDecision(0.5);
    // Root decided by client A; the child B opens under it re-hashes nothing.
    expect(a.shouldSample(dropped)).toBe(false);
    expect(b.shouldSample(kept, dropped)).toBe(false);
    a.handle(runEvent("run.completed", {}));
  });
});

