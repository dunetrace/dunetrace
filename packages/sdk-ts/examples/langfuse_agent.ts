/**
 * TypeScript agent example: Dunetrace + Langfuse integration.
 *
 * Shows how to correlate a Dunetrace run with a Langfuse trace using a shared UUID,
 * then use the Dunetrace explain endpoint for Langfuse-backed root-cause analysis.
 *
 * Install deps:
 *   cd packages/sdk-ts && npm install
 *
 * Run (happy path):
 *   OPENAI_API_KEY=sk-...
 *   LANGFUSE_PUBLIC_KEY=pk-lf-...
 *   LANGFUSE_SECRET_KEY=sk-lf-...
 *   npx tsx examples/langfuse_agent.ts
 *
 * Run (tool loop — triggers TOOL_LOOP signal + explain):
 *   SCENARIO=tool_loop npx tsx examples/langfuse_agent.ts
 *
 * LANGFUSE_HOST defaults to https://cloud.langfuse.com.
 * Set LANGFUSE_HOST=http://localhost:3001 for self-hosted Langfuse.
 *
 * If LANGFUSE_PUBLIC_KEY is absent, Langfuse traces are skipped but
 * Dunetrace monitoring still runs.
 */

import * as path from "node:path";
import { randomUUID } from "node:crypto";
import { config as loadEnv } from "dotenv";
import OpenAI from "openai";
import { Langfuse } from "langfuse";
import { Dunetrace } from "../src/index.js";

// Load .env from the repo root (3 levels above packages/sdk-ts/examples/)
loadEnv({ path: path.join(__dirname, "..", "..", "..", ".env") });

// ── Config ─────────────────────────────────────────────────────────────────────

const AGENT_ID      = "langfuse-ts-example-agent";
const DUNETRACE_API = process.env.DUNETRACE_API      ?? "http://localhost:8002";
const DUNETRACE_KEY = process.env.DUNETRACE_KEY      ?? "dt_dev_test";

const SYSTEM_PROMPT =
  "You are a research assistant. " +
  "Use the search tool to find information before answering.";

const SCENARIOS: Record<string, string> = {
  normal:    "What is the capital of France?",
  tool_loop: (
    "Search for 'latest AI news' exactly 6 times using the exact query each time. " +
    "Compile all results."
  ),
};

const TOOL_DEF: OpenAI.ChatCompletionTool = {
  type: "function",
  function: {
    name:        "web_search",
    description: "Search the web for information on a topic.",
    parameters: {
      type:       "object",
      properties: { query: { type: "string", description: "The search query." } },
      required:   ["query"],
    },
  },
};

// ── Clients ────────────────────────────────────────────────────────────────────

if (!process.env.OPENAI_API_KEY) {
  console.error("OPENAI_API_KEY is not set. Export it and re-run.");
  process.exit(1);
}

const openai = new OpenAI();

const dt = new Dunetrace({
  endpoint: process.env.DUNETRACE_ENDPOINT ?? "http://localhost:8001",
});

const langfuse = process.env.LANGFUSE_PUBLIC_KEY
  ? new Langfuse({
      publicKey: process.env.LANGFUSE_PUBLIC_KEY,
      secretKey: process.env.LANGFUSE_SECRET_KEY!,
      baseUrl:   process.env.LANGFUSE_HOST ?? "https://cloud.langfuse.com",
    })
  : null;

if (!langfuse) {
  console.warn(
    "LANGFUSE_PUBLIC_KEY not set — Langfuse traces will be skipped.\n" +
    "  Dunetrace monitoring still runs.",
  );
}

// ── Simulated tool ─────────────────────────────────────────────────────────────

function webSearch(query: string): string {
  return `Search results for '${query}': simulated result.`;
}

// ── Main agent run ─────────────────────────────────────────────────────────────

async function runAgent(scenario: string): Promise<void> {
  const query = SCENARIOS[scenario] ?? SCENARIOS.normal;

  console.log("=".repeat(60));
  console.log(`Dunetrace + Langfuse TS example  [scenario=${scenario}]`);
  console.log("=".repeat(60));
  console.log(`Query: ${query}\n`);

  // Pre-generate a shared UUID.
  // Dunetrace run and Langfuse trace both use this ID — linking them together.
  const sharedId = randomUUID();

  // Start a Langfuse trace with the shared ID so observations land on it.
  const trace = langfuse?.trace({
    id:    sharedId,
    name:  AGENT_ID,
    input: { query },
    tags:  ["dunetrace-example"],
  });

  let answer = "";

  await dt.run(AGENT_ID, {
    userInput:    query,
    model:        "gpt-4o-mini",
    tools:        ["web_search"],
    runId:        sharedId, // ← same UUID links Dunetrace run to Langfuse trace
    systemPrompt: SYSTEM_PROMPT,
  }, async (run) => {

    const messages: OpenAI.ChatCompletionMessageParam[] = [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user",   content: query },
    ];

    // Agent loop — max 12 steps to guard against infinite tool calling
    for (let step = 0; step < 12; step++) {
      const promptTokens = estimateTokens(messages);
      run.llmCalled("gpt-4o-mini", promptTokens);

      // Create a Langfuse generation for this LLM call
      const generation = trace?.generation({
        name:  `llm-step-${step}`,
        model: "gpt-4o-mini",
        input: messages.map(m => ({ role: m.role, content: "[hashed — privacy]" })),
      });

      const t0       = Date.now();
      const response = await openai.chat.completions.create({
        model:       "gpt-4o-mini",
        messages,
        tools:       [TOOL_DEF],
        tool_choice: "auto",
        temperature: 0,
      });
      const latencyMs = Date.now() - t0;

      const msg         = response.choices[0].message;
      const finishReason = response.choices[0].finish_reason;

      // The OpenAI tool-call union may include custom tool calls; this agent
      // only defines a function tool, so narrow to the function variant.
      const firstToolCall = msg.tool_calls?.[0];
      const firstToolArgs =
        firstToolCall?.type === "function" ? firstToolCall.function.arguments : "";

      run.llmResponded({
        completionTokens: response.usage?.completion_tokens,
        latencyMs,
        finishReason,
        // outputText is hashed before transmission — never sent raw
        outputText: msg.content ?? firstToolArgs,
      });

      // End the Langfuse generation — output is structural metadata only
      generation?.end({
        output:             "[hashed — see Langfuse for full content]",
        usage:              response.usage
          ? {
              input:  response.usage.prompt_tokens,
              output: response.usage.completion_tokens,
              total:  response.usage.total_tokens,
              unit:   "TOKENS",
            }
          : undefined,
        completionStartTime: new Date(t0),
      });

      messages.push(msg as OpenAI.ChatCompletionMessageParam);

      // Final answer — no tool calls
      if (finishReason === "stop" || !msg.tool_calls?.length) {
        answer = msg.content ?? "";
        run.finalAnswer();
        break;
      }

      // Process tool calls
      for (const tc of msg.tool_calls ?? []) {
        if (tc.type !== "function") continue;
        const args = JSON.parse(tc.function.arguments) as { query?: string };

        // Dunetrace: args are SHA-256 hashed before transmission
        run.toolCalled(tc.function.name, args);

        // Langfuse: record the tool span (input hashed for privacy)
        const span = trace?.span({
          name:  tc.function.name,
          input: { query: "[hashed]" },
        });

        const t1     = Date.now();
        const result = tc.function.name === "web_search"
          ? webSearch(args.query ?? "")
          : `Unknown tool: ${tc.function.name}`;

        span?.end({ output: { result_length: result.length } });
        run.toolResponded(tc.function.name, true, result.length, Date.now() - t1);

        messages.push({
          role:         "tool",
          tool_call_id: tc.id,
          content:      result,
        });
      }
    }

    trace?.update({ output: { answer: "[hashed — privacy]" } });
  });

  console.log(`\nAnswer: ${answer}`);

  // Flush Langfuse events before inspecting traces
  if (langfuse) {
    await langfuse.flushAsync();
    const host = process.env.LANGFUSE_HOST ?? "https://cloud.langfuse.com";
    console.log(`\nLangfuse trace: ${host}/trace/${sharedId}`);
  }

  if (scenario === "tool_loop") {
    // sharedId is both the Dunetrace run_id and the Langfuse trace_id
    await fetchAndExplain(sharedId, sharedId.replace(/-/g, ""));
  }

  console.log("\nDone. Check:");
  console.log("  Dashboard: http://localhost:3000");
  if (langfuse) {
    console.log(`  Langfuse:  ${process.env.LANGFUSE_HOST ?? "https://cloud.langfuse.com"}`);
  }
}

// ── Post-run: fetch signal + explain ─────────────────────────────────────────

async function fetchAndExplain(dtRunId: string, lfTraceId: string | null): Promise<void> {
  const headers: Record<string, string> = {
    "Authorization": `Bearer ${DUNETRACE_KEY}`,
    "Content-Type":  "application/json",
  };

  // Poll until the detector runs (up to ~30s)
  let signal: Record<string, unknown> | null = null;
  for (let attempt = 0; attempt < 6; attempt++) {
    await sleep(5000);
    const resp  = await fetch(`${DUNETRACE_API}/v1/agents/${AGENT_ID}/signals`, { headers });
    const data  = await resp.json() as { signals: Array<Record<string, unknown>> };
    signal = data.signals?.find(s => s["run_id"] === dtRunId) ?? null;
    if (signal) break;
    console.log(`  waiting for signal (attempt ${attempt + 1}/6)…`);
  }

  if (!signal) {
    console.log("\nNo signal for this run found — check the dashboard.");
    return;
  }

  const signalId    = signal["id"];
  const failureType = signal["failure_type"];
  const confidence  = Math.round((signal["confidence"] as number) * 100);

  console.log("\n" + "─".repeat(60));
  console.log(`Signal detected: ${failureType}  (confidence ${confidence}%)`);
  console.log(`  run_id = ${dtRunId}  ← same as Langfuse trace_id`);
  console.log(`  signal id = ${signalId}`);
  console.log("─".repeat(60));

  if (!process.env.LANGFUSE_PUBLIC_KEY) {
    console.log("\nSkipping explain — LANGFUSE_PUBLIC_KEY not set.");
    return;
  }

  console.log("\nCalling POST /v1/signals/{id}/explain …");

  const body: Record<string, string> = {};
  if (lfTraceId) body["langfuse_trace_id"] = lfTraceId;

  const explainResp = await fetch(`${DUNETRACE_API}/v1/signals/${signalId}/explain`, {
    method:  "POST",
    headers,
    body:    JSON.stringify(body),
  });

  if (!explainResp.ok) {
    const err = await explainResp.json() as { detail?: string };
    console.log(`\nExplain failed (${explainResp.status}): ${err.detail ?? explainResp.statusText}`);
    return;
  }

  const data = await explainResp.json() as {
    root_cause:         string;
    fix_content:        string;
    fix_type:           string;
    apply_blocked:      boolean;
    langfuse_prompt_name?: string | null;
  };

  console.log("\nRoot cause:");
  console.log("─".repeat(60));
  console.log(data.root_cause);
  console.log(`\nFix (${data.fix_type}):`);
  console.log(`  ${data.fix_content}`);
  if (!data.apply_blocked && data.langfuse_prompt_name) {
    console.log(`\n  → Apply via Langfuse: POST /v1/signals/${signalId}/apply-fix`);
    console.log(`    prompt_name=${JSON.stringify(data.langfuse_prompt_name)}`);
  } else if (data.fix_type === "no_auto_apply") {
    console.log("\n  → Security signal: review manually before applying.");
  } else if (data.apply_blocked) {
    console.log("\n  → Code/infra fix: apply manually.");
  }
  console.log("─".repeat(60));
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function estimateTokens(messages: OpenAI.ChatCompletionMessageParam[]): number {
  return messages.reduce((sum, m) => {
    const text = typeof m.content === "string" ? m.content : JSON.stringify(m.content ?? "");
    return sum + Math.ceil(text.length / 4);
  }, 0);
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ── Entry point ────────────────────────────────────────────────────────────────

const scenario = process.env.SCENARIO ?? "normal";
runAgent(scenario)
  .catch(err => {
    console.error("Error:", err);
    process.exit(1);
  })
  .finally(() => dt.shutdown());
