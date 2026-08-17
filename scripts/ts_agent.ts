/**
 * Standalone TypeScript agent — no external dependencies.
 * Instruments a simulated research agent directly against the ingest HTTP API,
 * producing a run with 3 consecutive web_search failures that fires RETRY_STORM.
 *
 * Run: npx tsx scripts/ts_agent.ts
 */

import { createHash, randomUUID } from "node:crypto";

// ── Config ────────────────────────────────────────────────────────────────────

const INGEST_URL   = "http://localhost:8001/v1/ingest";
const API_URL      = "http://localhost:8002";
const API_KEY      = "dt_dev_ts";
const AGENT_ID     = "ts-research-agent";
const MODEL        = "gpt-4o";
const TOOLS        = ["web_search", "calculator", "read_file"];
const SYSTEM_PROMPT = "You are a research assistant that finds academic papers.";

// ── Agent version fingerprint (must match Python SDK exactly) ────────────────

function sha256hex(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

/** Replicates Python: f"{system_prompt}:{model}:{sorted(tools)}" where sorted(tools)
 *  stringifies as Python list repr, e.g. "['calculator', 'read_file', 'web_search']" */
function agentVersion(systemPrompt: string, model: string, tools: string[]): string {
  const sorted      = [...tools].sort();
  const pythonList  = "[" + sorted.map(t => `'${t}'`).join(", ") + "]";
  const fingerprint = `${systemPrompt}:${model}:${pythonList}`;
  return sha256hex(fingerprint).slice(0, 8);
}

// ── Event builder ─────────────────────────────────────────────────────────────

interface AgentEvent {
  event_type:    string;
  run_id:        string;
  agent_id:      string;
  agent_version: string;
  step_index:    number;
  timestamp:     number;
  payload:       Record<string, unknown>;
  parent_run_id?: string | null;
}

function makeEmitter(runId: string, version: string) {
  let step = 0;

  function emit(
    eventType: string,
    payload:   Record<string, unknown>,
    advance = true,
  ): AgentEvent {
    if (advance) step++;
    return {
      event_type:    eventType,
      run_id:        runId,
      agent_id:      AGENT_ID,
      agent_version: version,
      step_index:    step,
      timestamp:     Date.now() / 1000,
      payload,
    };
  }

  function currentStep(): number { return step; }

  return { emit, currentStep };
}

// ── Ingest call ───────────────────────────────────────────────────────────────

async function ingest(events: AgentEvent[]): Promise<{ accepted: number; batch_id: string }> {
  const res = await fetch(INGEST_URL, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ api_key: API_KEY, agent_id: AGENT_ID, events }),
  });
  if (!res.ok) {
    throw new Error(`POST /v1/ingest → ${res.status}: ${await res.text()}`);
  }
  return res.json() as Promise<{ accepted: number; batch_id: string }>;
}

// ── Agent simulation ──────────────────────────────────────────────────────────

async function runAgent(): Promise<string> {
  const runId   = randomUUID();
  const version = agentVersion(SYSTEM_PROMPT, MODEL, TOOLS);
  const { emit, currentStep } = makeEmitter(runId, version);

  const USER_INPUT  = "Find recent papers on LLM agent failure modes and hallucination";
  const SEARCH_ARGS = JSON.stringify({ query: "LLM agent failures 2024 arxiv" });
  const ERROR_TEXT  = "ConnectionError: upstream API timeout after 10s";

  const events: AgentEvent[] = [];

  // ── run.started (step 0, never advances) ─────────────────────────────────
  events.push({
    event_type:    "run.started",
    run_id:        runId,
    agent_id:      AGENT_ID,
    agent_version: version,
    step_index:    0,
    timestamp:     Date.now() / 1000,
    payload: {
      input_text: USER_INPUT,
      model:      MODEL,
      tools:      TOOLS,
    },
  });

  // ── Iteration 1: plan → search → timeout ─────────────────────────────────
  events.push(emit("llm.called",    { model: MODEL }));
  events.push(emit("llm.responded", {
    finish_reason: "tool_calls",
    output:        "I will search for recent papers on LLM failures",
    output_length: 68,
    latency_ms:    843,
  }, false));

  events.push(emit("tool.called",    { tool_name: "web_search", args: SEARCH_ARGS }));
  events.push(emit("tool.responded", { tool_name: "web_search", success: false, error: ERROR_TEXT }, false));

  // ── Iteration 2: LLM retries same search ─────────────────────────────────
  events.push(emit("llm.called",    { model: MODEL }));
  events.push(emit("llm.responded", {
    finish_reason: "tool_calls",
    output:        "The search timed out, let me retry",
    output_length: 52,
    latency_ms:    912,
  }, false));

  events.push(emit("tool.called",    { tool_name: "web_search", args: SEARCH_ARGS }));
  events.push(emit("tool.responded", { tool_name: "web_search", success: false, error: ERROR_TEXT }, false));

  // ── Iteration 3: third attempt — RETRY_STORM threshold (≥3) ──────────────
  events.push(emit("llm.called",    { model: MODEL }));
  events.push(emit("llm.responded", {
    finish_reason: "tool_calls",
    output:        "One more attempt at the search",
    output_length: 44,
    latency_ms:    876,
  }, false));

  events.push(emit("tool.called",    { tool_name: "web_search", args: SEARCH_ARGS }));
  events.push(emit("tool.responded", { tool_name: "web_search", success: false, error: ERROR_TEXT }, false));

  // ── Iteration 4: gives up ─────────────────────────────────────────────────
  events.push(emit("llm.called",    { model: MODEL }));
  events.push(emit("llm.responded", {
    finish_reason: "stop",
    output:        "I was unable to complete the search due to connectivity issues",
    output_length: 118,
    latency_ms:    761,
  }, false));

  // ── run.completed ─────────────────────────────────────────────────────────
  events.push({
    event_type:    "run.completed",
    run_id:        runId,
    agent_id:      AGENT_ID,
    agent_version: version,
    step_index:    currentStep(),
    timestamp:     Date.now() / 1000,
    payload: {
      total_steps:     currentStep(),
      exit_reason:     "final_answer",
      tool_call_count: 3,
    },
  });

  console.log(`\nAgent: ${AGENT_ID}`);
  console.log(`run_id:        ${runId}`);
  console.log(`agent_version: ${version}`);
  console.log(`events:        ${events.length}  (expect RETRY_STORM: 3× web_search failure)`);

  const result = await ingest(events);
  console.log(`accepted:      ${result.accepted}  batch_id=${result.batch_id}`);

  return runId;
}

// ── Poll for the run to appear in the customer API ────────────────────────────

async function poll(runId: string): Promise<void> {
  console.log("\nWaiting for detector to process run (polls every 5s)...");
  for (let i = 0; i < 15; i++) {
    await new Promise(r => setTimeout(r, 4000));
    process.stdout.write(".");

    const res = await fetch(`${API_URL}/v1/runs/${encodeURIComponent(runId)}`).catch(() => null);
    if (!res?.ok) continue;

    const data = await res.json() as {
      run_id:     string;
      agent_id:   string;
      step_count: number;
      signals:    Array<{ failure_type: string; severity: string; confidence: number }>;
    };

    process.stdout.write("\n");
    console.log("\n✓ Run visible in dashboard API");
    console.log(`  run_id:     ${data.run_id}`);
    console.log(`  agent_id:   ${data.agent_id}`);
    console.log(`  step_count: ${data.step_count}`);

    if (data.signals.length === 0) {
      console.log("  signals:    (none yet — detector may still be finishing)");
    } else {
      console.log(`  signals (${data.signals.length}):`);
      for (const s of data.signals) {
        const pct = Math.round(s.confidence * 100);
        console.log(`    • ${s.failure_type} [${s.severity}] confidence=${pct}%`);
      }
    }

    console.log(`\n  Dashboard: http://localhost:3000`);
    console.log(`  Runs API:  ${API_URL}/v1/agents/${AGENT_ID}/runs`);
    return;
  }

  process.stdout.write("\n");
  console.log("⚠  Run not visible after 60s — check detector container logs:");
  console.log("   docker compose logs detector --tail=30");
}

// ── Main ──────────────────────────────────────────────────────────────────────

(async () => {
  try {
    const runId = await runAgent();
    await poll(runId);
  } catch (err) {
    console.error("\n✗", err instanceof Error ? err.message : err);
    process.exit(1);
  }
})();
