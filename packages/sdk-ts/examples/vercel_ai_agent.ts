/**
 * TypeScript agent example: Dunetrace + Vercel AI SDK integration.
 *
 * Drives the Vercel AI SDK (`generateText`) against a local Ollama instead
 * of raw OpenAI — no API key required, since Ollama exposes an
 * OpenAI-compatible endpoint. The Dunetrace run is opened via the
 * `traceGenerateText` wrapper, which instruments the SDK call and emits the
 * same llm.* / tool.* events a manual instrumentation loop would produce.
 *
 * Install deps:
 *   cd packages/sdk-ts && npm install
 *
 * Run (happy path):
 *   ollama pull llama3.2
 *   npx tsx examples/vercel_ai_agent.ts
 *
 * Run (tool loop — triggers TOOL_LOOP signal + explain):
 *   SCENARIO=tool_loop npx tsx examples/vercel_ai_agent.ts
 *
 * Override the endpoint/model with OLLAMA_BASE_URL / OLLAMA_MODEL.
 * OLLAMA_BASE_URL defaults to http://localhost:11434/v1.
 */

import * as path from "node:path";
import { randomUUID } from "node:crypto";
import { config as loadEnv } from "dotenv";
import { generateText, tool, stepCountIs } from "ai";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { z } from "zod";
import { Dunetrace, traceGenerateText } from "../src/index.js";

// Load .env from the repo root (3 levels above packages/sdk-ts/examples/)
loadEnv({ path: path.join(__dirname, "..", "..", "..", ".env") });

// ── Config ─────────────────────────────────────────────────────────────────────

const AGENT_ID      = "vercel-ai-ts-example-agent";
const DUNETRACE_API = process.env.DUNETRACE_API ?? "http://localhost:8002";
const DUNETRACE_KEY = process.env.DUNETRACE_KEY ?? "dt_dev_test";

const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL ?? "http://localhost:11434/v1";
const OLLAMA_MODEL    = process.env.OLLAMA_MODEL    ?? "llama3.2";

const SYSTEM_PROMPT =
  "You are a research assistant. " +
  "Use the weather tool to find information before answering.";

const SCENARIOS: Record<string, string> = {
  normal:    "What is the weather in Paris? Use the weather tool, then answer in one sentence.",
  tool_loop: (
    "Check the weather for 'Paris' exactly 6 times using the exact city each time. " +
    "Compile all results."
  ),
};

// ── Clients ────────────────────────────────────────────────────────────────────

const ollama = createOpenAICompatible({ name: "ollama", baseURL: OLLAMA_BASE_URL });

const dt = new Dunetrace({
  endpoint: process.env.DUNETRACE_ENDPOINT ?? "http://localhost:8001",
});

// ── Simulated tool ─────────────────────────────────────────────────────────────

const weather = tool({
  description: "Get the current weather for a city.",
  inputSchema: z.object({ city: z.string() }),
  execute: async ({ city }) => ({ city, tempC: 18, conditions: "sunny" }),
});

// ── Main agent run ─────────────────────────────────────────────────────────────

async function runAgent(scenario: string): Promise<void> {
  const query = SCENARIOS[scenario] ?? SCENARIOS.normal;

  console.log("=".repeat(60));
  console.log(`Dunetrace + Vercel AI SDK TS example  [scenario=${scenario}]`);
  console.log("=".repeat(60));
  console.log(`Query: ${query}\n`);

  // Pre-generate a shared UUID so the Dunetrace run id is known up front
  // (lets you correlate it with an external tracing system if desired).
  const sharedId = randomUUID();

  // traceGenerateText opens the Dunetrace run, instruments the SDK call,
  // and emits the per-step llm.* / tool.* events inside the run boundary.
  const result = await traceGenerateText(
    dt,
    AGENT_ID,
    {
      runId:        sharedId,
      systemPrompt: SYSTEM_PROMPT,
      userInput:    query,
    },
    generateText,
    {
      model:    ollama(OLLAMA_MODEL),
      system:   SYSTEM_PROMPT,
      prompt:   query,
      tools:    { weather },
      stopWhen: stepCountIs(12),
    },
  );

  const answer = result.text;
  console.log(`\nAnswer: ${answer}`);
  console.log(`(steps: ${result.steps.length})`);

  if (scenario === "tool_loop") {
    await fetchAndExplain(sharedId);
  }

  console.log("\nDone. Check:");
  console.log("  Dashboard: http://localhost:3000");
}

// ── Post-run: fetch signal + explain ─────────────────────────────────────────

async function fetchAndExplain(dtRunId: string): Promise<void> {
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
  console.log(`  run_id = ${dtRunId}`);
  console.log(`  signal id = ${signalId}`);
  console.log("─".repeat(60));

  console.log("\nCalling POST /v1/signals/{id}/explain …");

  const explainResp = await fetch(`${DUNETRACE_API}/v1/signals/${signalId}/explain`, {
    method:  "POST",
    headers,
    body:    JSON.stringify({}),
  });

  if (!explainResp.ok) {
    const err = await explainResp.json() as { detail?: string };
    console.log(`\nExplain failed (${explainResp.status}): ${err.detail ?? explainResp.statusText}`);
    return;
  }

  const data = await explainResp.json() as {
    root_cause:    string;
    fix_content:   string;
    fix_type:      string;
    apply_blocked: boolean;
  };

  console.log("\nRoot cause:");
  console.log("─".repeat(60));
  console.log(data.root_cause);
  console.log(`\nFix (${data.fix_type}):`);
  console.log(`  ${data.fix_content}`);
  if (data.fix_type === "no_auto_apply") {
    console.log("\n  → Security signal: review manually before applying.");
  } else if (data.apply_blocked) {
    console.log("\n  → Code/infra fix: apply manually.");
  }
  console.log("─".repeat(60));
}

// ── Helpers ────────────────────────────────────────────────────────────────────

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
