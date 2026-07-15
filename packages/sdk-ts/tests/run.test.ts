import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
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

  it("llmResponded transmits raw outputText", () => {
    const { run, emitted } = makeRun();
    run.llmCalled("gpt-4o");
    run.llmResponded({ outputText: "secret output" });
    const resp = emitted.find(e => e.event_type === "llm.responded")!;
    expect(resp.payload["output"]).toBe("secret output");
  });

  it("toolCalled transmits raw args", () => {
    const { run, emitted } = makeRun();
    run.toolCalled("search", { query: "secret" });
    const ev = emitted.find(e => e.event_type === "tool.called")!;
    expect(ev.payload["tool_name"]).toBe("search");
    expect(ev.payload["args"]).toBe(JSON.stringify({ query: "secret" }));
  });

  it("toolResponded includes raw error when error provided", () => {
    const { run, emitted } = makeRun();
    run.toolCalled("search");
    run.toolResponded("search", false, 0, 100, "Connection refused");
    const ev = emitted.find(e => e.event_type === "tool.responded")!;
    expect(ev.payload["success"]).toBe(false);
    expect(ev.payload["error"]).toBe("Connection refused");
  });

  it("toolResponded success=true has no error field", () => {
    const { run, emitted } = makeRun();
    run.toolCalled("search");
    run.toolResponded("search", true, 256);
    const ev = emitted.find(e => e.event_type === "tool.responded")!;
    expect(ev.payload["success"]).toBe(true);
    expect(ev.payload).not.toHaveProperty("error");
  });

  it("toolResponded transmits raw output text", () => {
    const { run, emitted } = makeRun();
    run.toolCalled("search");
    run.toolResponded("search", true, 20, 50, undefined, "search result body");
    const ev = emitted.find(e => e.event_type === "tool.responded")!;
    expect(ev.payload["output"]).toBe("search result body");
  });

  it("toolResponded defaults output to empty string", () => {
    const { run, emitted } = makeRun();
    run.toolCalled("search");
    run.toolResponded("search", true);
    const ev = emitted.find(e => e.event_type === "tool.responded")!;
    expect(ev.payload["output"]).toBe("");
  });

  it("retrievalCalled transmits raw query", () => {
    const { run, emitted } = makeRun();
    run.retrievalCalled("docs", "sensitive query");
    const ev = emitted.find(e => e.event_type === "retrieval.called")!;
    expect(ev.payload["index_name"]).toBe("docs");
    expect(ev.payload["query"]).toBe("sensitive query");
  });

  it("retrievalCalled without query sets empty string", () => {
    const { run, emitted } = makeRun();
    run.retrievalCalled("docs");
    const ev = emitted.find(e => e.event_type === "retrieval.called")!;
    expect(ev.payload["query"]).toBe("");
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

  it("retrievalResponded transmits raw content", () => {
    const { run, emitted } = makeRun();
    run.retrievalCalled("docs", "q");
    run.retrievalResponded("docs", 1, 0.9, 10, "the retrieved text");
    const ev = emitted.find(e => e.event_type === "retrieval.responded")!;
    expect(ev.payload["content"]).toBe("the retrieved text");
  });

  it("retrievalResponded defaults content to empty string", () => {
    const { run, emitted } = makeRun();
    run.retrievalCalled("docs", "q");
    run.retrievalResponded("docs", 1);
    const ev = emitted.find(e => e.event_type === "retrieval.responded")!;
    expect(ev.payload["content"]).toBe("");
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

describe("DunetraceRun — memory channel", () => {
  it("memoryWritten emits memory.written with key/value/source", () => {
    const { run, emitted } = makeRun();
    run.memoryWritten("user_prefs", "prefers dark mode", "user_input");
    expect(emitted).toHaveLength(1);
    expect(emitted[0].event_type).toBe("memory.written");
    expect(emitted[0].payload).toEqual({
      key: "user_prefs",
      value: "prefers dark mode",
      source: "user_input",
    });
  });

  it("memoryWritten omits source when not given", () => {
    const { run, emitted } = makeRun();
    run.memoryWritten("note", "some text");
    expect(emitted[0].payload).toEqual({ key: "note", value: "some text" });
    expect("source" in (emitted[0].payload as object)).toBe(false);
  });

  it("memoryWritten rejects an invalid source", () => {
    const { run } = makeRun();
    expect(() => run.memoryWritten("k", "v", "not_a_source" as never)).toThrow(/source must be one of/);
  });

  it("memoryWritten accepts every documented source", () => {
    const { run, emitted } = makeRun();
    for (const s of ["user_input", "retrieval", "tool_output", "llm_output", "agent_reasoning", "external"] as const) {
      run.memoryWritten(`k_${s}`, "v", s);
    }
    expect(emitted).toHaveLength(6);
  });

  it("memoryRead and memoryCleared emit the right payloads", () => {
    const { run, emitted } = makeRun();
    run.memoryRead("user_prefs");
    run.memoryCleared("user_prefs");
    run.memoryCleared(); // clear all
    expect(emitted[0]).toMatchObject({ event_type: "memory.read", payload: { key: "user_prefs" } });
    expect(emitted[1]).toMatchObject({ event_type: "memory.cleared", payload: { key: "user_prefs" } });
    expect(emitted[2]).toMatchObject({ event_type: "memory.cleared", payload: { key: null } });
  });

  it("memory events do NOT advance the step counter", () => {
    const { run } = makeRun();
    run.toolCalled("search");
    const stepAfterTool = run.currentStep();
    run.memoryWritten("k", "v", "tool_output");
    run.memoryRead("k");
    run.memoryCleared();
    expect(run.currentStep()).toBe(stepAfterTool);
  });
});

describe("DunetraceRun — DUNETRACE_OMIT_LLM_OUTPUT_TEXT opt-out", () => {
  const KEY = "DUNETRACE_OMIT_LLM_OUTPUT_TEXT";
  let saved: string | undefined;
  beforeEach(() => { saved = process.env[KEY]; delete process.env[KEY]; });
  afterEach(() => { if (saved === undefined) delete process.env[KEY]; else process.env[KEY] = saved; });

  it("transmits output by default", () => {
    const { run, emitted } = makeRun();
    run.llmResponded({ outputText: "hello world", finishReason: "stop" });
    expect(emitted[0].payload["output"]).toBe("hello world");
    expect(emitted[0].payload["output_length"]).toBe(11);
  });

  it("omits output when opted out, but still sends output_length", () => {
    process.env[KEY] = "1";
    const { run, emitted } = makeRun();
    run.llmResponded({ outputText: "hello world", finishReason: "stop" });
    expect("output" in (emitted[0].payload as object)).toBe(false);
    expect(emitted[0].payload["output_length"]).toBe(11);
  });

  it("accepts true / yes (case-insensitive) as opt-out values", () => {
    for (const v of ["true", "YES", "True"]) {
      process.env[KEY] = v;
      const { run, emitted } = makeRun();
      run.llmResponded({ outputText: "x", finishReason: "stop" });
      expect("output" in (emitted[0].payload as object)).toBe(false);
    }
  });

  it("a non-truthy value keeps transmitting output", () => {
    process.env[KEY] = "0";
    const { run, emitted } = makeRun();
    run.llmResponded({ outputText: "keep me", finishReason: "stop" });
    expect(emitted[0].payload["output"]).toBe("keep me");
  });
});
