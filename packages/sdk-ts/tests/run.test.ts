import { describe, it, expect, vi, beforeEach } from "vitest";
import { DunetraceRun } from "../src/run.js";
import type { AgentEvent } from "../src/models.js";

function makeRun() {
  const emitted: AgentEvent[] = [];
  const emitter = { _emit: (e: AgentEvent) => emitted.push(e) };
  const run = new DunetraceRun("test-agent", "abc12345", emitter);
  return { run, emitted };
}

describe("DunetraceRun — step counter", () => {
  it("starts at 0", () => {
    const { run } = makeRun();
    expect(run.currentStep()).toBe(0);
  });

  it("llmCalled advances step", () => {
    const { run } = makeRun();
    run.llmCalled("gpt-4o");
    expect(run.currentStep()).toBe(1);
  });

  it("llmResponded does NOT advance step", () => {
    const { run } = makeRun();
    run.llmCalled("gpt-4o");
    run.llmResponded({ finishReason: "stop" });
    expect(run.currentStep()).toBe(1);
  });

  it("toolCalled advances step", () => {
    const { run } = makeRun();
    run.toolCalled("search");
    expect(run.currentStep()).toBe(1);
  });

  it("toolResponded does NOT advance step", () => {
    const { run } = makeRun();
    run.toolCalled("search");
    run.toolResponded("search", true);
    expect(run.currentStep()).toBe(1);
  });

  it("retrievalCalled advances step", () => {
    const { run } = makeRun();
    run.retrievalCalled("docs");
    expect(run.currentStep()).toBe(1);
  });

  it("retrievalResponded does NOT advance step", () => {
    const { run } = makeRun();
    run.retrievalCalled("docs");
    run.retrievalResponded("docs", 5);
    expect(run.currentStep()).toBe(1);
  });

  it("externalSignal does NOT advance step", () => {
    const { run } = makeRun();
    run.llmCalled("gpt-4o");
    run.externalSignal("rate_limit", "openai");
    expect(run.currentStep()).toBe(1);
  });

  it("finalAnswer does NOT advance step", () => {
    const { run } = makeRun();
    run.toolCalled("search");
    run.finalAnswer();
    expect(run.currentStep()).toBe(1);
  });

  it("multiple calls accumulate correctly", () => {
    const { run } = makeRun();
    run.llmCalled("gpt-4o");          // step 1
    run.llmResponded({});              // no advance
    run.toolCalled("search");          // step 2
    run.toolResponded("search", true); // no advance
    run.llmCalled("gpt-4o");          // step 3
    run.llmResponded({});              // no advance
    expect(run.currentStep()).toBe(3);
  });
});

describe("DunetraceRun — event payloads", () => {
  it("llmCalled emits correct event type and model", () => {
    const { run, emitted } = makeRun();
    run.llmCalled("gpt-4o", 150);
    expect(emitted[0].event_type).toBe("llm.called");
    expect(emitted[0].payload["model"]).toBe("gpt-4o");
    expect(emitted[0].payload["prompt_tokens"]).toBe(150);
  });

  it("llmResponded emits finish_reason", () => {
    const { run, emitted } = makeRun();
    run.llmCalled("gpt-4o");
    run.llmResponded({ finishReason: "length", completionTokens: 50 });
    const resp = emitted.find(e => e.event_type === "llm.responded")!;
    expect(resp.payload["finish_reason"]).toBe("length");
    expect(resp.payload["completion_tokens"]).toBe(50);
  });

  it("llmResponded hashes outputText — does not include raw text", () => {
    const { run, emitted } = makeRun();
    run.llmCalled("gpt-4o");
    run.llmResponded({ outputText: "secret output" });
    const resp = emitted.find(e => e.event_type === "llm.responded")!;
    expect(resp.payload).not.toHaveProperty("output_text");
    expect(typeof resp.payload["output_hash"]).toBe("string");
    expect((resp.payload["output_hash"] as string)).toHaveLength(16);
  });

  it("toolCalled hashes args", () => {
    const { run, emitted } = makeRun();
    run.toolCalled("search", { query: "secret" });
    const ev = emitted.find(e => e.event_type === "tool.called")!;
    expect(ev.payload["tool_name"]).toBe("search");
    expect(ev.payload).not.toHaveProperty("query");
    expect(ev.payload).not.toHaveProperty("args");
    expect(typeof ev.payload["args_hash"]).toBe("string");
    expect((ev.payload["args_hash"] as string)).toHaveLength(16);
  });

  it("toolResponded includes error_hash when error provided", () => {
    const { run, emitted } = makeRun();
    run.toolCalled("search");
    run.toolResponded("search", false, 0, 100, "Connection refused");
    const ev = emitted.find(e => e.event_type === "tool.responded")!;
    expect(ev.payload["success"]).toBe(false);
    expect(typeof ev.payload["error_hash"]).toBe("string");
    expect(ev.payload).not.toHaveProperty("error");
  });

  it("toolResponded success=true has no error_hash", () => {
    const { run, emitted } = makeRun();
    run.toolCalled("search");
    run.toolResponded("search", true, 256);
    const ev = emitted.find(e => e.event_type === "tool.responded")!;
    expect(ev.payload["success"]).toBe(true);
    expect(ev.payload).not.toHaveProperty("error_hash");
  });

  it("retrievalCalled hashes query", () => {
    const { run, emitted } = makeRun();
    run.retrievalCalled("docs", "sensitive query");
    const ev = emitted.find(e => e.event_type === "retrieval.called")!;
    expect(ev.payload["index_name"]).toBe("docs");
    expect(ev.payload).not.toHaveProperty("query");
    expect(typeof ev.payload["query_hash"]).toBe("string");
  });

  it("retrievalCalled without query sets empty hash", () => {
    const { run, emitted } = makeRun();
    run.retrievalCalled("docs");
    const ev = emitted.find(e => e.event_type === "retrieval.called")!;
    expect(ev.payload["query_hash"]).toBe("");
  });

  it("retrievalResponded includes result_count", () => {
    const { run, emitted } = makeRun();
    run.retrievalCalled("docs", "q");
    run.retrievalResponded("docs", 3, 0.92, 45);
    const ev = emitted.find(e => e.event_type === "retrieval.responded")!;
    expect(ev.payload["result_count"]).toBe(3);
    expect(ev.payload["top_score"]).toBe(0.92);
    expect(ev.payload["latency_ms"]).toBe(45);
  });

  it("externalSignal emits signal_name and source", () => {
    const { run, emitted } = makeRun();
    run.externalSignal("rate_limit", "openai", { http_status: 429 });
    const ev = emitted.find(e => e.event_type === "external.signal")!;
    expect(ev.payload["signal_name"]).toBe("rate_limit");
    expect(ev.payload["source"]).toBe("openai");
    expect(ev.payload["http_status"]).toBe(429);
  });

  it("externalSignal emitted at current step index", () => {
    const { run, emitted } = makeRun();
    run.toolCalled("api");              // step 1
    run.externalSignal("timeout");
    const ev = emitted.find(e => e.event_type === "external.signal")!;
    expect(ev.step_index).toBe(1);
  });

  it("finalAnswer sets exitReason", () => {
    const { run } = makeRun();
    run.finalAnswer();
    expect(run.exitReason()).toBe("final_answer");
  });

  it("getEvents returns all events in order", () => {
    const { run } = makeRun();
    run.llmCalled("gpt-4o");
    run.llmResponded({});
    run.toolCalled("search");
    run.toolResponded("search", true);
    run.finalAnswer();
    const types = run.getEvents().map(e => e.event_type);
    expect(types).toEqual(["llm.called", "llm.responded", "tool.called", "tool.responded"]);
  });

  it("all events share the same run_id", () => {
    const { run } = makeRun();
    run.llmCalled("gpt-4o");
    run.toolCalled("search");
    const ids = run.getEvents().map(e => e.run_id);
    expect(ids.every(id => id === run.runId)).toBe(true);
  });

  it("run_id is a UUID", () => {
    const { run } = makeRun();
    expect(run.runId).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/,
    );
  });

  it("events include agent_version", () => {
    const { run } = makeRun();
    run.llmCalled("gpt-4o");
    expect(run.getEvents()[0].agent_version).toBe("abc12345");
  });
});

// ── run.llm() helper ──────────────────────────────────────────────────────────

describe("DunetraceRun.llm() — OpenAI format", () => {
  it("emits llm.called then llm.responded", async () => {
    const { run, emitted } = makeRun();
    const fakeResp = {
      choices: [{ finish_reason: "stop", message: { content: "Paris" } }],
      usage: { prompt_tokens: 50, completion_tokens: 10 },
    };
    await run.llm("gpt-4o", Promise.resolve(fakeResp as unknown as Record<string, unknown>));
    expect(emitted[0].event_type).toBe("llm.called");
    expect(emitted[1].event_type).toBe("llm.responded");
  });

  it("extracts prompt_tokens from usage", async () => {
    const { run, emitted } = makeRun();
    const fakeResp = {
      choices: [{ finish_reason: "stop", message: { content: "" } }],
      usage: { prompt_tokens: 120, completion_tokens: 40 },
    };
    await run.llm("gpt-4o", Promise.resolve(fakeResp as unknown as Record<string, unknown>));
    expect(emitted[0].payload["prompt_tokens"]).toBe(120);
    expect(emitted[1].payload["completion_tokens"]).toBe(40);
  });

  it("extracts finish_reason and output_length", async () => {
    const { run, emitted } = makeRun();
    const fakeResp = {
      choices: [{ finish_reason: "tool_calls", message: { content: "Hello" } }],
      usage: { prompt_tokens: 0, completion_tokens: 5 },
    };
    await run.llm("gpt-4o", Promise.resolve(fakeResp as unknown as Record<string, unknown>));
    expect(emitted[1].payload["finish_reason"]).toBe("tool_calls");
    expect(emitted[1].payload["output_length"]).toBe(5);
  });

  it("returns the original response", async () => {
    const { run } = makeRun();
    const fakeResp = {
      choices: [{ finish_reason: "stop", message: { content: "result" } }],
      usage: { prompt_tokens: 10, completion_tokens: 3 },
    };
    const result = await run.llm("gpt-4o", Promise.resolve(fakeResp as unknown as Record<string, unknown>));
    expect(result).toBe(fakeResp);
  });
});

describe("DunetraceRun.llm() — Anthropic format", () => {
  it("extracts Anthropic usage fields", async () => {
    const { run, emitted } = makeRun();
    const fakeResp = {
      content: [{ type: "text", text: "Bonjour" }],
      stop_reason: "end_turn",
      usage: { input_tokens: 80, output_tokens: 20 },
    };
    await run.llm("claude-3-5-haiku-20241022", Promise.resolve(fakeResp as unknown as Record<string, unknown>));
    expect(emitted[0].payload["prompt_tokens"]).toBe(80);
    expect(emitted[1].payload["completion_tokens"]).toBe(20);
    expect(emitted[1].payload["finish_reason"]).toBe("end_turn");
  });
});
