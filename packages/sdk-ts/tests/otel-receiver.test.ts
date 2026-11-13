import { describe, it, expect } from "vitest";
import { BasicTracerProvider, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { trace, context, SpanKind, SpanStatusCode, type Tracer, type Span } from "@opentelemetry/api";
import { Dunetrace } from "../src/client.js";
import { NoopBatchEmitter } from "../src/emitters.js";
import { DunetraceOtelReceiver } from "../src/integrations/otel-receiver.js";
import type { AgentEvent } from "../src/models.js";

function setup(agentId = "") {
  const captured: AgentEvent[] = [];
  // The EventSink captures every event as the client emits it — including the
  // ones the receiver produces from translated spans. NoopBatchEmitter keeps the
  // test from shipping anywhere.
  const dt = new Dunetrace({
    exporter: { handle: (e) => captured.push(e) },
    emitter: new NoopBatchEmitter(),
  });
  const receiver = new DunetraceOtelReceiver(dt, agentId);
  const provider = new BasicTracerProvider({ spanProcessors: [new SimpleSpanProcessor(receiver)] });
  const tracer = provider.getTracer("emitter-sim");
  return { captured, tracer };
}

/** dt.run() is async; span-emit calls inside it run synchronously, but the
 *  closing run.completed lands a microtask later. Let the queue drain. */
const flush = () => new Promise((r) => setTimeout(r, 0));

function child(tracer: Tracer, root: Span, name: string, attributes: Record<string, unknown>, error = false): void {
  const s = tracer.startSpan(name, { kind: SpanKind.CLIENT, attributes: attributes as never }, trace.setSpan(context.active(), root));
  if (error) s.setStatus({ code: SpanStatusCode.ERROR });
  s.end();
}

function types(captured: AgentEvent[]): string[] {
  return captured.map((e) => e.event_type);
}
function ofType(captured: AgentEvent[], t: string): AgentEvent[] {
  return captured.filter((e) => e.event_type === t);
}

describe("DunetraceOtelReceiver — LLM spans", () => {
  it("translates an OpenLLMetry-style (legacy naming) LLM span into a Dunetrace run", async () => {
    const { captured, tracer } = setup("billing-agent");
    const root = tracer.startSpan("agent.workflow");
    child(tracer, root, "chat gpt-4o", {
      "gen_ai.request.model": "gpt-4o",
      "gen_ai.usage.prompt_tokens": 1200,
      "gen_ai.usage.completion_tokens": 300,
      "gen_ai.completion": "here is your refund status",
      "gen_ai.completion.0.finish_reason": "stop",
    });
    root.end();
    await flush();

    expect(types(captured)).toEqual(["run.started", "llm.called", "llm.responded", "run.completed"]);
    expect(captured[0].agent_id).toBe("billing-agent");
    expect(captured[0].payload["model"]).toBe("gpt-4o");
    const called = ofType(captured, "llm.called")[0];
    const responded = ofType(captured, "llm.responded")[0];
    expect(called.payload).toMatchObject({ model: "gpt-4o", prompt_tokens: 1200 });
    expect(responded.payload).toMatchObject({ completion_tokens: 300, finish_reason: "stop" });
    expect(responded.payload["output"]).toBe("here is your refund status");
    expect(responded.payload["output_length"]).toBe("here is your refund status".length);
    expect(typeof responded.payload["latency_ms"]).toBe("number");
  });

  it("reads current GenAI naming and the structured gen_ai.output.messages output", async () => {
    const { captured, tracer } = setup("agent");
    const root = tracer.startSpan("agent.workflow");
    child(tracer, root, "chat claude-sonnet-4", {
      "gen_ai.request.model": "claude-sonnet-4-20250514",
      "gen_ai.usage.input_tokens": 800,
      "gen_ai.usage.output_tokens": 150,
      "gen_ai.usage.reasoning_tokens": 64,
      "gen_ai.output.messages": JSON.stringify([
        { role: "assistant", parts: [{ type: "text", content: "the answer" }] },
      ]),
    });
    root.end();
    await flush();

    const called = ofType(captured, "llm.called")[0];
    const responded = ofType(captured, "llm.responded")[0];
    expect(called.payload).toMatchObject({ model: "claude-sonnet-4-20250514", prompt_tokens: 800 });
    expect(responded.payload).toMatchObject({ completion_tokens: 150, reasoning_tokens: 64 });
    expect(responded.payload["output"]).toBe("the answer");
  });

  it("marks an errored LLM span with finish_reason=error", async () => {
    const { captured, tracer } = setup();
    const root = tracer.startSpan("agent.workflow");
    child(tracer, root, "chat gpt-4o", { "gen_ai.request.model": "gpt-4o" }, true);
    root.end();
    await flush();

    expect(ofType(captured, "llm.responded")[0].payload["finish_reason"]).toBe("error");
  });

  it("translates a single-span trace where the root IS the LLM span", async () => {
    // OpenLIT / some OpenLLMetry setups emit one span per call with no wrapper.
    const { captured, tracer } = setup("solo-agent");
    const root = tracer.startSpan("chat gpt-4o-mini", {
      attributes: {
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.usage.input_tokens": 50,
        "gen_ai.usage.output_tokens": 20,
      },
    });
    root.end();
    await flush();

    expect(types(captured)).toEqual(["run.started", "llm.called", "llm.responded", "run.completed"]);
    expect(ofType(captured, "llm.called")[0].payload["model"]).toBe("gpt-4o-mini");
  });
});

describe("DunetraceOtelReceiver — tool & retrieval spans", () => {
  it("translates a tool span, carrying arguments and result through", async () => {
    const { captured, tracer } = setup("agent");
    const root = tracer.startSpan("agent.workflow");
    child(tracer, root, "search", {
      "gen_ai.tool.name": "web_search",
      "gen_ai.tool.call.arguments": JSON.stringify({ query: "refund policy" }),
      "gen_ai.tool.call.result": "10 results",
    });
    root.end();
    await flush();

    const called = ofType(captured, "tool.called")[0];
    const responded = ofType(captured, "tool.responded")[0];
    expect(called.payload["tool_name"]).toBe("web_search");
    expect(JSON.parse(called.payload["args"] as string)).toMatchObject({ query: "refund policy" });
    expect(responded.payload).toMatchObject({ tool_name: "web_search", success: true });
    expect(responded.payload["output"]).toBe("10 results");
  });

  it("marks a failed tool span success=false", async () => {
    const { captured, tracer } = setup("agent");
    const root = tracer.startSpan("agent.workflow");
    child(tracer, root, "charge", { "tool.name": "charge_card" }, true);
    root.end();
    await flush();

    expect(ofType(captured, "tool.responded")[0].payload["success"]).toBe(false);
  });

  it("translates a retrieval span with document count and top score", async () => {
    const { captured, tracer } = setup("agent");
    const root = tracer.startSpan("agent.workflow");
    child(tracer, root, "vector.query", {
      "vector_db.collection_name": "kb",
      "retrieval.result_count": 5,
      "retrieval.top_score": 0.77,
      "retrieval.documents": "doc-a doc-b",
    });
    root.end();
    await flush();

    const called = ofType(captured, "retrieval.called")[0];
    const responded = ofType(captured, "retrieval.responded")[0];
    expect(called.payload["index_name"]).toBe("kb");
    expect(responded.payload).toMatchObject({ index_name: "kb", result_count: 5 });
    expect(responded.payload["top_score"]).toBeCloseTo(0.77, 10);
  });
});

describe("DunetraceOtelReceiver — trace assembly", () => {
  it("holds spans until the root arrives, then emits one ordered run", async () => {
    const { captured, tracer } = setup("agent");
    const root = tracer.startSpan("agent.workflow");
    // two LLM calls + a tool call, children end before the root
    child(tracer, root, "chat gpt-4o", { "gen_ai.request.model": "gpt-4o", "gen_ai.usage.input_tokens": 10 });
    child(tracer, root, "search", { "tool.name": "web_search", "tool.arguments": "{}" });
    child(tracer, root, "chat gpt-4o", { "gen_ai.request.model": "gpt-4o", "gen_ai.usage.input_tokens": 12 });

    // No root yet: nothing translated, spans are pending.
    await flush();
    expect(captured).toHaveLength(0);

    root.end();
    await flush();

    expect(types(captured)).toEqual([
      "run.started",
      "llm.called", "llm.responded",
      "tool.called", "tool.responded",
      "llm.called", "llm.responded",
      "run.completed",
    ]);
    // every event belongs to the same run
    const runIds = new Set(captured.map((e) => e.run_id));
    expect(runIds.size).toBe(1);
  });

  it("defaults the agent id to the root span name", async () => {
    const { captured, tracer } = setup(); // no agentId
    const root = tracer.startSpan("my-cool-agent");
    child(tracer, root, "chat gpt-4o", { "gen_ai.request.model": "gpt-4o" });
    root.end();
    await flush();

    expect(captured[0].agent_id).toBe("my-cool-agent");
  });
});
