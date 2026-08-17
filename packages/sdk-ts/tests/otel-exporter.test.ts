import { describe, it, expect, beforeEach } from "vitest";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  SimpleSpanProcessor,
  type ReadableSpan,
} from "@opentelemetry/sdk-trace-base";
import { SpanKind, SpanStatusCode } from "@opentelemetry/api";
import { DunetraceOtelExporter, traceIdHex, rootSpanIdHex } from "../src/integrations/otel.js";
import type { AgentEvent, EventType } from "../src/models.js";

// A fixed run UUID so the derived trace id is stable and assertable.
const RUN_ID = "1b4e28ba-2fa1-11d2-883f-0016d3cca427";

function setup(captureContent = true) {
  const memory = new InMemorySpanExporter();
  const provider = new BasicTracerProvider({ spanProcessors: [new SimpleSpanProcessor(memory)] });
  const tracer = provider.getTracer("test");
  const exporter = new DunetraceOtelExporter({ tracer, captureContent });
  return { memory, exporter };
}

let clock = 1_700_000_000; // seconds; advance per event so spans have real durations

function ev(
  type: EventType,
  payload: Record<string, unknown>,
  extra: Partial<AgentEvent> = {},
): AgentEvent {
  clock += 0.5;
  return {
    event_type: type,
    run_id: RUN_ID,
    agent_id: "billing-agent",
    agent_version: "v1",
    step_index: 1,
    timestamp: clock,
    payload,
    ...extra,
  };
}

function spanByName(spans: ReadableSpan[], name: string): ReadableSpan | undefined {
  return spans.find((s) => s.name === name);
}

beforeEach(() => {
  clock = 1_700_000_000;
});

describe("DunetraceOtelExporter — run span", () => {
  it("opens a dunetrace.run span on a deterministic trace and closes it completed", () => {
    const { memory, exporter } = setup();
    exporter.handle(ev("run.started", { model: "gpt-4o", tools: ["search"] }));
    exporter.handle(ev("run.completed", { total_steps: 3, exit_reason: "final_answer", tool_call_count: 1 }));

    const spans = memory.getFinishedSpans();
    const run = spanByName(spans, "dunetrace.run");
    expect(run).toBeDefined();
    expect(run!.kind).toBe(SpanKind.INTERNAL);
    // trace id is derived from run_id, so a run maps to a stable, addressable trace
    expect(run!.spanContext().traceId).toBe(traceIdHex(RUN_ID));
    expect(traceIdHex(RUN_ID)).toHaveLength(32);
    expect(rootSpanIdHex(RUN_ID)).toHaveLength(16);
    expect(run!.attributes["dunetrace.run.id"]).toBe(RUN_ID);
    expect(run!.attributes["dunetrace.run.agent_id"]).toBe("billing-agent");
    expect(run!.attributes["dunetrace.run.status"]).toBe("completed");
    expect(run!.attributes["dunetrace.run.total_steps"]).toBe(3);
    expect(run!.attributes["dunetrace.run.exit_reason"]).toBe("final_answer");
    expect(run!.attributes["dunetrace.run.tools"]).toBe("search");
    expect(run!.status.code).toBe(SpanStatusCode.UNSET);
  });

  it("marks the run span ERROR on run.errored", () => {
    const { memory, exporter } = setup();
    exporter.handle(ev("run.started", {}));
    exporter.handle(ev("run.errored", { error_type: "ToolLoop" }));

    const run = spanByName(memory.getFinishedSpans(), "dunetrace.run")!;
    expect(run.attributes["dunetrace.run.status"]).toBe("failed");
    expect(run.attributes["dunetrace.run.error_type"]).toBe("ToolLoop");
    expect(run.status.code).toBe(SpanStatusCode.ERROR);
  });

  it("links a child run back to its parent run's root span", () => {
    const { memory, exporter } = setup();
    const parentId = "2c5f39cb-3fb2-22e3-994a-1127e4ddb538";
    exporter.handle(ev("run.started", {}, { parent_run_id: parentId }));
    exporter.handle(ev("run.completed", {}));

    const run = spanByName(memory.getFinishedSpans(), "dunetrace.run")!;
    expect(run.attributes["dunetrace.run.parent_run_id"]).toBe(parentId);
    expect(run.links).toHaveLength(1);
    expect(run.links[0].context.traceId).toBe(traceIdHex(parentId));
    expect(run.links[0].context.spanId).toBe(rootSpanIdHex(parentId));
  });
});

describe("DunetraceOtelExporter — LLM span (GenAI conventions)", () => {
  it("emits a chat span with gen_ai.* attributes, provider, tokens and cost", () => {
    const { memory, exporter } = setup();
    exporter.handle(ev("run.started", {}));
    exporter.handle(ev("llm.called", { model: "gpt-4o", prompt_tokens: 1000 }));
    exporter.handle(ev("llm.responded", { completion_tokens: 500, finish_reason: "stop", latency_ms: 800 }));
    exporter.handle(ev("run.completed", {}));

    const spans = memory.getFinishedSpans();
    const llm = spanByName(spans, "chat gpt-4o")!;
    const run = spanByName(spans, "dunetrace.run")!;
    expect(llm).toBeDefined();
    expect(llm.kind).toBe(SpanKind.CLIENT);
    // parented onto the run span, same trace
    expect(llm.spanContext().traceId).toBe(run.spanContext().traceId);
    expect(llm.parentSpanId).toBe(run.spanContext().spanId);
    // current GenAI conventions, not deprecated gen_ai.system / prompt_tokens
    expect(llm.attributes["gen_ai.operation.name"]).toBe("chat");
    expect(llm.attributes["gen_ai.provider.name"]).toBe("openai");
    expect(llm.attributes["gen_ai.request.model"]).toBe("gpt-4o");
    expect(llm.attributes["gen_ai.usage.input_tokens"]).toBe(1000);
    expect(llm.attributes["gen_ai.usage.output_tokens"]).toBe(500);
    expect(llm.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
    // cost matches the SDK pricing table: 1000*5e-6 + 500*15e-6
    expect(llm.attributes["dunetrace.llm.cost_usd"]).toBeCloseTo(1000 * 5.0e-6 + 500 * 15.0e-6, 10);
    expect(llm.attributes["dunetrace.llm.output_truncated"]).toBe(false);
  });

  it("infers anthropic provider and flags truncation on finish_reason=length", () => {
    const { memory, exporter } = setup();
    exporter.handle(ev("run.started", {}));
    exporter.handle(ev("llm.called", { model: "claude-sonnet-4-20250514", prompt_tokens: 10 }));
    exporter.handle(ev("llm.responded", { completion_tokens: 20, finish_reason: "length" }));
    exporter.handle(ev("run.completed", {}));

    const llm = spanByName(memory.getFinishedSpans(), "chat claude-sonnet-4-20250514")!;
    expect(llm.attributes["gen_ai.provider.name"]).toBe("anthropic");
    expect(llm.attributes["dunetrace.llm.output_truncated"]).toBe(true);
  });

  it("back-fills input tokens when only known on the response", () => {
    const { memory, exporter } = setup();
    exporter.handle(ev("run.started", {}));
    exporter.handle(ev("llm.called", { model: "gpt-4o", prompt_tokens: 0 }));
    exporter.handle(ev("llm.responded", { completion_tokens: 5, prompt_tokens: 42 }));
    exporter.handle(ev("run.completed", {}));

    const llm = spanByName(memory.getFinishedSpans(), "chat gpt-4o")!;
    expect(llm.attributes["gen_ai.usage.input_tokens"]).toBe(42);
  });
});

describe("DunetraceOtelExporter — tool span", () => {
  it("maps an HTTP-shaped tool call to the current stable HTTP conventions", () => {
    const { memory, exporter } = setup();
    exporter.handle(ev("run.started", {}));
    exporter.handle(ev("tool.called", {
      tool_name: "fetch",
      args: JSON.stringify({ url: "https://api.example.com/v1/orders", method: "get" }),
    }));
    exporter.handle(ev("tool.responded", { success: false, error: "503", latency_ms: 120 }));
    exporter.handle(ev("run.completed", {}));

    const tool = spanByName(memory.getFinishedSpans(), "dunetrace.tool.fetch")!;
    expect(tool.kind).toBe(SpanKind.CLIENT);
    expect(tool.attributes["server.address"]).toBe("api.example.com");
    expect(tool.attributes["http.request.method"]).toBe("GET");
    expect(tool.attributes["url.full"]).toBe("https://api.example.com/v1/orders");
    // a bare numeric error on an HTTP tool becomes the response status code
    expect(tool.attributes["http.response.status_code"]).toBe(503);
    expect(tool.attributes["dunetrace.tool.result_status"]).toBe("error");
    expect(tool.status.code).toBe(SpanStatusCode.ERROR);
    // deprecated keys must be absent
    expect(tool.attributes["http.url"]).toBeUndefined();
    expect(tool.attributes["http.method"]).toBeUndefined();
    expect(tool.attributes["http.status_code"]).toBeUndefined();
  });

  it("treats a non-HTTP tool as an INTERNAL span", () => {
    const { memory, exporter } = setup();
    exporter.handle(ev("run.started", {}));
    exporter.handle(ev("tool.called", { tool_name: "calc", args: JSON.stringify({ a: 1, b: 2 }) }));
    exporter.handle(ev("tool.responded", { success: true }));
    exporter.handle(ev("run.completed", {}));

    const tool = spanByName(memory.getFinishedSpans(), "dunetrace.tool.calc")!;
    expect(tool.kind).toBe(SpanKind.INTERNAL);
    expect(tool.attributes["server.address"]).toBeUndefined();
    expect(tool.attributes["dunetrace.tool.result_status"]).toBe("success");
  });
});

describe("DunetraceOtelExporter — retrieval span", () => {
  it("emits document_count and top_score", () => {
    const { memory, exporter } = setup();
    exporter.handle(ev("run.started", {}));
    exporter.handle(ev("retrieval.called", { index_name: "kb", query: "refund policy" }));
    exporter.handle(ev("retrieval.responded", { result_count: 4, top_score: 0.82, latency_ms: 30 }));
    exporter.handle(ev("run.completed", {}));

    const r = spanByName(memory.getFinishedSpans(), "dunetrace.retrieval")!;
    expect(r.kind).toBe(SpanKind.CLIENT);
    expect(r.attributes["dunetrace.retrieval.vector_store"]).toBe("kb");
    expect(r.attributes["dunetrace.retrieval.query"]).toBe("refund policy");
    expect(r.attributes["dunetrace.retrieval.document_count"]).toBe(4);
    expect(r.attributes["dunetrace.retrieval.top_score"]).toBeCloseTo(0.82, 10);
  });
});

describe("DunetraceOtelExporter — voice", () => {
  it("emits transcription/tts point spans with counts only, never raw text", () => {
    const { memory, exporter } = setup();
    exporter.handle(ev("run.started", {}));
    exporter.handle(ev("transcription.received", { text: "where is my order", confidence: 0.9, latency_ms: 100 }));
    exporter.handle(ev("tts.generated", { text: "it ships today", model: "eleven_turbo_v2", voice_id: "rachel" }));
    exporter.handle(ev("run.completed", {}));

    const spans = memory.getFinishedSpans();
    const stt = spanByName(spans, "dunetrace.voice.transcription")!;
    const tts = spanByName(spans, "dunetrace.voice.tts")!;
    expect(stt.attributes["dunetrace.voice.char_count"]).toBe("where is my order".length);
    expect(stt.attributes["dunetrace.voice.confidence"]).toBeCloseTo(0.9, 10);
    expect(tts.attributes["dunetrace.voice.char_count"]).toBe("it ships today".length);
    expect(tts.attributes["dunetrace.voice.model"]).toBe("eleven_turbo_v2");
    // no attribute should carry the raw transcript / TTS text
    const allValues = [...Object.values(stt.attributes), ...Object.values(tts.attributes)];
    expect(allValues).not.toContain("where is my order");
    expect(allValues).not.toContain("it ships today");
  });

  it("records a VAD transition as a span event on the run span", () => {
    const { memory, exporter } = setup();
    exporter.handle(ev("run.started", {}));
    exporter.handle(ev("voice_activity.detected", { type: "barge_in", duration_ms: 200 }));
    exporter.handle(ev("run.completed", {}));

    const run = spanByName(memory.getFinishedSpans(), "dunetrace.run")!;
    expect(run.events).toHaveLength(1);
    expect(run.events[0].name).toBe("dunetrace.voice.barge_in");
    expect(run.events[0].attributes?.["dunetrace.voice.duration_ms"]).toBe(200);
  });
});

describe("DunetraceOtelExporter — content gating & resilience", () => {
  it("drops content-bearing attributes when captureContent is false", () => {
    const { memory, exporter } = setup(false);
    exporter.handle(ev("run.started", {}));
    exporter.handle(ev("tool.called", {
      tool_name: "fetch",
      args: JSON.stringify({ url: "https://secret.example.com/x" }),
    }));
    exporter.handle(ev("tool.responded", { success: true }));
    exporter.handle(ev("retrieval.called", { index_name: "kb", query: "sensitive question" }));
    exporter.handle(ev("retrieval.responded", { result_count: 1 }));
    exporter.handle(ev("run.completed", {}));

    const spans = memory.getFinishedSpans();
    const tool = spanByName(spans, "dunetrace.tool.fetch")!;
    const r = spanByName(spans, "dunetrace.retrieval")!;
    // host is metadata (kept); url.full and args are content (dropped)
    expect(tool.attributes["server.address"]).toBe("secret.example.com");
    expect(tool.attributes["url.full"]).toBeUndefined();
    expect(tool.attributes["dunetrace.tool.args"]).toBeUndefined();
    expect(r.attributes["dunetrace.retrieval.query"]).toBeUndefined();
  });

  it("handle() never throws on an event for an unknown run", () => {
    const { exporter } = setup();
    // no run.started first: every handler no-ops on a missing run
    expect(() => exporter.handle(ev("llm.called", { model: "gpt-4o" }))).not.toThrow();
    expect(() => exporter.handle(ev("tts.generated", { text: "hi" }))).not.toThrow();
  });

  it("closes an orphaned child span as ERROR when a response event is dropped", () => {
    const { memory, exporter } = setup();
    exporter.handle(ev("run.started", {}));
    exporter.handle(ev("llm.called", { model: "gpt-4o", prompt_tokens: 5 }));
    // no llm.responded — the run ends with the child still open
    exporter.handle(ev("run.completed", {}));

    const llm = spanByName(memory.getFinishedSpans(), "chat gpt-4o")!;
    expect(llm.status.code).toBe(SpanStatusCode.ERROR);
  });
});
