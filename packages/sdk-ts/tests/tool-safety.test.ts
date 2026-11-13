/**
 * Instrumentation must never break the thing it instruments, and must never
 * ship a secret.
 *
 * `dt.tool()` used to call `run.toolCalled()` OUTSIDE the try that protects the
 * customer's function, and `toolCalled` did an unguarded `JSON.stringify(args)`
 * — so a circular object (an ORM entity, an Express req/res, a DB client) or a
 * BigInt argument threw before the tool body ran. Adding Dunetrace turned a
 * working call into a hard failure.
 *
 * And nothing capped or redacted what it shipped: the server never redacts and
 * caps only the OTLP path, so an Authorization header in tool args reached
 * Postgres verbatim.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { Dunetrace } from "../src/client.js";
import { REDACTED } from "../src/redaction.js";
import type { AgentEvent } from "../src/models.js";

let captured: AgentEvent[] = [];

function newClient(opts: Record<string, unknown> = {}): Dunetrace {
  captured = [];
  return new Dunetrace({
    exporter: { handle: (e: AgentEvent) => { captured.push(e); } },
    ...opts,
  });
}

const ofType = (t: string): AgentEvent[] => captured.filter((e) => e.event_type === t);

let warn: ReturnType<typeof vi.spyOn>;
beforeEach(() => { warn = vi.spyOn(console, "warn").mockImplementation(() => {}); });
afterEach(() => { warn.mockRestore(); });

// ── Finding 3: instrumentation never throws into customer code ────────────────

describe("dt.tool() — a hostile argument cannot fail the customer's call", () => {
  it("runs the tool body when an argument is circular", async () => {
    const dt = newClient();
    const circular: Record<string, unknown> = { name: "orm-entity" };
    circular["self"] = circular;

    let ran = false;
    const search = dt.tool(async (_entity: unknown) => { ran = true; return "ok"; }, "search");

    const out = await dt.run("agent", {}, async () => search(circular));
    expect(ran).toBe(true);
    expect(out).toBe("ok");
  });

  it("runs the tool body when an argument is a BigInt", async () => {
    const dt = newClient();
    const charge = dt.tool(async (_amount: unknown) => "charged", "charge");
    await expect(dt.run("agent", {}, async () => charge(42n))).resolves.toBe("charged");
  });

  it("works for synchronous tools too", () => {
    const dt = newClient();
    const circular: Record<string, unknown> = {};
    circular["loop"] = circular;
    const compute = dt.tool((_x: unknown) => 99, "compute");

    return dt.run("agent", {}, async () => {
      expect(compute(circular)).toBe(99);
    });
  });

  it("still emits a usable tool.called for the circular argument", async () => {
    const dt = newClient();
    const circular: Record<string, unknown> = { id: 7 };
    circular["self"] = circular;
    const search = dt.tool(async (_e: unknown) => "ok", "search");
    await dt.run("agent", {}, async () => search(circular));

    const called = ofType("tool.called")[0];
    expect(called.payload["tool_name"]).toBe("search");
    expect(String(called.payload["args"])).toContain("[Circular]");
    expect(() => JSON.parse(String(called.payload["args"]))).not.toThrow();
  });

  it("a throwing emit is swallowed, and the tool's own error still propagates", async () => {
    const dt = newClient();
    const boom = dt.tool(async () => { throw new Error("tool failed"); }, "boom");
    // Break the emit path itself; the customer's error must be what surfaces.
    (dt as unknown as { _emit: (e: AgentEvent) => void })._emit = () => {
      throw new Error("instrumentation exploded");
    };
    await expect(dt.run("agent", {}, async () => boom())).rejects.toThrow("tool failed");
  });

  it("an argument with a throwing getter does not fail the call", async () => {
    const dt = newClient();
    const hostile = { get boom(): never { throw new Error("getter exploded"); } };
    const use = dt.tool(async (_x: unknown) => "fine", "use");
    await expect(dt.run("agent", {}, async () => use(hostile))).resolves.toBe("fine");
  });
});

// ── Finding 5: redaction and caps ─────────────────────────────────────────────

describe("tool args are redacted before they leave the process", () => {
  it("redacts an Authorization header nested in tool args", async () => {
    const dt = newClient();
    const fetchTool = dt.tool(async (_req: unknown) => "ok", "http_get");
    await dt.run("agent", {}, async () =>
      fetchTool({ url: "https://api.example.com", headers: { Authorization: "Bearer sk-live-LEAK" } }),
    );

    const args = String(ofType("tool.called")[0].payload["args"]);
    expect(args).not.toContain("sk-live-LEAK");
    expect(args).toContain(REDACTED);
    expect(args).toContain("https://api.example.com"); // non-secrets survive
  });

  it("redacts the camelCase spellings JS tool args actually use", async () => {
    const dt = newClient();
    const call = dt.tool(async (_c: unknown) => "ok", "call");
    await dt.run("agent", {}, async () =>
      call({ accessToken: "AT", clientSecret: "CS", refreshToken: "RT", userId: "u-1" }),
    );

    const args = JSON.parse(String(ofType("tool.called")[0].payload["args"])) as Record<
      string,
      Record<string, unknown>
    >;
    const inner = Object.values(args)[0];
    expect(inner["accessToken"]).toBe(REDACTED);
    expect(inner["clientSecret"]).toBe(REDACTED);
    expect(inner["refreshToken"]).toBe(REDACTED);
    expect(inner["userId"]).toBe("u-1");
  });

  it("honours extra redactKeys from the client", async () => {
    const dt = newClient({ redactKeys: ["patientId"] });
    const call = dt.tool(async (_c: unknown) => "ok", "call");
    await dt.run("agent", {}, async () => call({ patientId: "P-0001" }));
    expect(String(ofType("tool.called")[0].payload["args"])).not.toContain("P-0001");
  });

  it("honours a customer redact hook", async () => {
    const dt = newClient({
      redact: (args: Record<string, unknown>) => ({ ...args, scrubbed: true }),
    });
    const call = dt.tool(async (_c: unknown) => "ok", "call");
    await dt.run("agent", {}, async () => call({ anything: 1 }));
    expect(String(ofType("tool.called")[0].payload["args"])).toContain('"scrubbed":true');
  });
});

describe("content caps", () => {
  it("caps tool args and records the pre-cap length", async () => {
    const dt = newClient({ maxFieldChars: 64 });
    const call = dt.tool(async (_c: unknown) => "ok", "call");
    await dt.run("agent", {}, async () => call({ blob: "z".repeat(5000) }));

    const p = ofType("tool.called")[0].payload;
    expect(String(p["args"])).toHaveLength(64);
    expect(p["args_truncated"]).toBe(true);
    expect(p["args_original_length"]).toBeGreaterThan(5000);
    // Same number under the key the shared run builder reads on the OTLP path,
    // so server-side OVERSIZED_TOOL_ARGUMENTS still fires on a shortened payload.
    expect(p["args_length"]).toBe(p["args_original_length"]);
  });

  it("caps tool output while keeping output_length honest", async () => {
    const dt = newClient({ maxFieldChars: 32 });
    const call = dt.tool(async () => "q".repeat(900), "call");
    await dt.run("agent", {}, async () => call());

    const p = ofType("tool.responded")[0].payload;
    expect(String(p["output"])).toHaveLength(32);
    expect(p["output_truncated"]).toBe(true);
    expect(p["output_original_length"]).toBe(900);
    expect(p["output_length"]).toBe(900); // the REAL length, never the capped one
  });

  it("caps input_text and system_prompt on run.started without changing agent_version", async () => {
    const long = "p".repeat(20_000);
    const capped = newClient({ maxFieldChars: 100 });
    await capped.run("agent", { systemPrompt: long, userInput: long }, async () => {});
    const started = ofType("run.started")[0];

    expect(String(started.payload["system_prompt"])).toHaveLength(100);
    expect(started.payload["system_prompt_truncated"]).toBe(true);
    expect(started.payload["system_prompt_original_length"]).toBe(20_000);
    expect(started.payload["input_text_truncated"]).toBe(true);

    // Grouping must not move because a prompt got long: the hash is taken
    // before the cap.
    const cappedVersion = started.agent_version;
    const uncapped = newClient({ maxFieldChars: 0 });
    await uncapped.run("agent", { systemPrompt: long, userInput: long }, async () => {});
    expect(ofType("run.started")[0].agent_version).toBe(cappedVersion);
  });

  it("leaves a short field byte-identical — no markers on an ordinary run", async () => {
    const dt = newClient();
    const call = dt.tool(async () => "small", "call");
    await dt.run("agent", { systemPrompt: "be helpful", userInput: "hi" }, async () => call());

    expect(ofType("run.started")[0].payload).not.toHaveProperty("system_prompt_truncated");
    expect(ofType("tool.called")[0].payload).not.toHaveProperty("args_truncated");
    expect(ofType("tool.responded")[0].payload).not.toHaveProperty("output_truncated");
  });

  it("caps llm output, retrieval content and memory values as well", async () => {
    const dt = newClient({ maxFieldChars: 16 });
    await dt.run("agent", {}, async (run) => {
      run.llmCalled("gpt-4o");
      run.llmResponded({ outputText: "o".repeat(400) });
      run.retrievalCalled("docs", "q".repeat(400));
      run.retrievalResponded("docs", 1, 0.9, 0, "c".repeat(400));
      run.memoryWritten("k", "m".repeat(400));
    });

    expect(String(ofType("llm.responded")[0].payload["output"])).toHaveLength(16);
    expect(ofType("llm.responded")[0].payload["output_length"]).toBe(400);
    expect(String(ofType("retrieval.called")[0].payload["query"])).toHaveLength(16);
    expect(String(ofType("retrieval.responded")[0].payload["content"])).toHaveLength(16);
    expect(String(ofType("memory.written")[0].payload["value"])).toHaveLength(16);
  });
});
