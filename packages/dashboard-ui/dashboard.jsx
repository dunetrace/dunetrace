const { useState, useEffect } = React;

// ── Mock run data ──────────────────────────────────────────────────────────────
const NOW = Date.now() / 1000;

const MOCK_RUNS = [
  { run_id: "run-a9b8c7d6", label: "CONTEXT_BLOAT" },
  { run_id: "run-f4a9b2c1", label: "TOOL_LOOP" },
  { run_id: "run-71bce930", label: "TOOL_THRASHING" },
  { run_id: "run-c1d2e3f4", label: "SLOW_STEP" },
  { run_id: "run-8d3e1f77", label: "Clean run" },
];

const MOCK_DETAIL = {
  // ── CONTEXT_BLOAT: tokens grow 7.9x, each jump traced to a tool output ──────
  "run-a9b8c7d6": {
    run_id: "run-a9b8c7d6", agent_id: "research-agent-v2", agent_version: "a7f3d9b2",
    started_at: NOW - 290, completed_at: NOW - 180, exit_reason: "completed", step_count: 16,
    events: [
      { event_type: "run.started",    step_index: 0,  timestamp: NOW-290, payload: { input_hash: "3f8c2a1b", tools: ["web_search","database_lookup","code_runner"] } },
      // LLM call 1 — clean start
      { event_type: "llm.called",     step_index: 1,  timestamp: NOW-283, payload: { model: "gpt-4o-mini", prompt_tokens: 620 } },
      { event_type: "llm.responded",  step_index: 2,  timestamp: NOW-276, payload: { finish_reason: "tool_calls", latency_ms: 810 } },
      // tool.responded: web_search returns 2,800 chars → ~700 tokens added
      { event_type: "tool.called",    step_index: 3,  timestamp: NOW-274, payload: { tool_name: "web_search", args_hash: "aa11" } },
      { event_type: "tool.responded", step_index: 4,  timestamp: NOW-267, payload: { success: true, output_length: 2800 } },
      // LLM call 2 — +700 tok from web_search
      { event_type: "llm.called",     step_index: 5,  timestamp: NOW-265, payload: { model: "gpt-4o-mini", prompt_tokens: 1340 } },
      { event_type: "llm.responded",  step_index: 6,  timestamp: NOW-258, payload: { finish_reason: "tool_calls", latency_ms: 790 } },
      // tool.responded: database_lookup returns 8,400 chars → ~2,100 tokens added (the big jump)
      { event_type: "tool.called",    step_index: 7,  timestamp: NOW-256, payload: { tool_name: "database_lookup", args_hash: "bb22" } },
      { event_type: "tool.responded", step_index: 8,  timestamp: NOW-248, payload: { success: true, output_length: 8400 } },
      // LLM call 3 — +2,100 tok from database_lookup (largest jump, should be highlighted)
      { event_type: "llm.called",     step_index: 9,  timestamp: NOW-246, payload: { model: "gpt-4o-mini", prompt_tokens: 3460 } },
      { event_type: "llm.responded",  step_index: 10, timestamp: NOW-238, payload: { finish_reason: "tool_calls", latency_ms: 1240 } },
      // two tools: web_search 3,200 chars + code_runner 1,600 chars → ~1,200 tokens
      { event_type: "tool.called",    step_index: 11, timestamp: NOW-236, payload: { tool_name: "web_search", args_hash: "cc33" } },
      { event_type: "tool.responded", step_index: 12, timestamp: NOW-229, payload: { success: true, output_length: 3200 } },
      { event_type: "tool.called",    step_index: 13, timestamp: NOW-227, payload: { tool_name: "code_runner", args_hash: "dd44" } },
      { event_type: "tool.responded", step_index: 14, timestamp: NOW-221, payload: { success: true, output_length: 1600 } },
      // LLM call 4 — +1,200 tok from web_search+code_runner
      { event_type: "llm.called",     step_index: 15, timestamp: NOW-219, payload: { model: "gpt-4o-mini", prompt_tokens: 4680 } },
      { event_type: "run.completed",  step_index: 16, timestamp: NOW-180, payload: { exit_reason: "final_answer", total_steps: 16 } },
    ],
    signals: [
      { id: 1, failure_type: "CONTEXT_BLOAT", severity: "MEDIUM", step_index: 15, confidence: 0.80,
        evidence: { first_tokens: 620, last_tokens: 4680, growth_factor: 7.5, llm_call_count: 4, first_call_step: 1, last_call_step: 15 },
        title: "Context bloat: prompt grew 7.5x (620 \u2192 4,680 tokens) across 4 LLM calls",
        what: "The prompt grew from 620 to 4,680 tokens — a 7.5x increase across 4 LLM calls. Tool outputs (database_lookup: 8.4k chars, two web_search calls, code_runner) are being appended to context without summarisation. At this trajectory the model will hit its context window within 2-3 more calls.",
        why_it_matters: "Every token in the prompt is charged on every LLM call. A 7.5x bloat means 7.5x the cost per call vs the run start. When the context window fills, the API either throws an error or silently drops early context — the agent loses its earlier reasoning without any explicit failure signal.",
        evidence_summary: "prompt_tokens: 620 \u2192 1,340 \u2192 3,460 \u2192 4,680. Biggest jump: +2,120 tok at step 9 (database_lookup returned 8,400 chars). Growth factor 7.5x over 4 calls. Confidence: 80%.",
        suggested_fixes: [{ description: "Summarise tool outputs before appending to context", language: "python", code: "MAX_TOOL_OUTPUT_CHARS = 1500  # ~375 tokens\n\ndef format_tool_output(tool_name, output):\n    output_str = str(output)\n    if len(output_str) > MAX_TOOL_OUTPUT_CHARS:\n        output_str = (\n            output_str[:MAX_TOOL_OUTPUT_CHARS]\n            + f'\\n... [{len(output_str) - MAX_TOOL_OUTPUT_CHARS} chars truncated]'\n        )\n    return f'Result from {tool_name}:\\n{output_str}'" }],
      }
    ],
  },

  "run-f4a9b2c1": {
    run_id: "run-f4a9b2c1", agent_id: "research-agent-v2", agent_version: "a7f3d9b2",
    started_at: NOW - 182, completed_at: NOW - 54, exit_reason: "completed", step_count: 12,
    events: [
      { event_type: "run.started",    step_index: 0,  timestamp: NOW-182, payload: { input_hash: "e3b0c442", tools: ["web_search","calculator"] } },
      { event_type: "llm.called",     step_index: 1,  timestamp: NOW-175, payload: { model: "gpt-4o-mini", prompt_tokens: 480 } },
      { event_type: "llm.responded",  step_index: 2,  timestamp: NOW-168, payload: { finish_reason: "tool_calls", latency_ms: 820 } },
      { event_type: "tool.called",    step_index: 3,  timestamp: NOW-166, payload: { tool_name: "web_search", args_hash: "a1b2c3d4" } },
      { event_type: "tool.responded", step_index: 4,  timestamp: NOW-158, payload: { success: true, output_length: 1024 } },
      { event_type: "tool.called",    step_index: 5,  timestamp: NOW-155, payload: { tool_name: "web_search", args_hash: "a1b2c3d4" } },
      { event_type: "tool.responded", step_index: 6,  timestamp: NOW-147, payload: { success: true, output_length: 1019 } },
      { event_type: "tool.called",    step_index: 7,  timestamp: NOW-143, payload: { tool_name: "web_search", args_hash: "a1b2c3d4" } },
      { event_type: "tool.responded", step_index: 8,  timestamp: NOW-135, payload: { success: true, output_length: 1021 } },
      { event_type: "tool.called",    step_index: 9,  timestamp: NOW-132, payload: { tool_name: "web_search", args_hash: "a1b2c3d4" } },
      { event_type: "tool.responded", step_index: 10, timestamp: NOW-124, payload: { success: true, output_length: 1018 } },
      { event_type: "tool.called",    step_index: 11, timestamp: NOW-121, payload: { tool_name: "web_search", args_hash: "a1b2c3d4" } },
      { event_type: "run.completed",  step_index: 12, timestamp: NOW-54,  payload: { exit_reason: "final_answer", total_steps: 12 } },
    ],
    signals: [
      { id: 2, failure_type: "TOOL_LOOP", severity: "HIGH", step_index: 11, confidence: 0.95,
        evidence: { tool: "web_search", count: 5, window: 5 },
        title: "Tool loop: web_search called 5x in 5 steps",
        what: "The agent called web_search 5 consecutive times with identical arguments, making no progress between calls.",
        why_it_matters: "Looping agents burn tokens and cost money without producing value.",
        evidence_summary: "Tool web_search called 5x in steps 7-11 with identical args. Confidence: 95%.",
        suggested_fixes: [{ description: "Add per-tool call limit", language: "python", code: "MAX_CALLS_PER_TOOL = 3\nif tool_call_counts[tool] > MAX_CALLS_PER_TOOL:\n    raise RuntimeError(f'{tool} called too many times')" }],
      }
    ],
  },

  "run-71bce930": {
    run_id: "run-71bce930", agent_id: "research-agent-v2", agent_version: "a7f3d9b2",
    started_at: NOW-4200, completed_at: NOW-4100, exit_reason: "stalled", step_count: 17,
    events: [
      { event_type: "run.started",    step_index: 0,  timestamp: NOW-4200, payload: { input_hash: "f9e8d7c6", tools: ["web_search","database_lookup"] } },
      { event_type: "llm.called",     step_index: 1,  timestamp: NOW-4192, payload: { model: "gpt-4o-mini", prompt_tokens: 520 } },
      { event_type: "tool.called",    step_index: 2,  timestamp: NOW-4185, payload: { tool_name: "web_search", args_hash: "bb11" } },
      { event_type: "tool.responded", step_index: 3,  timestamp: NOW-4177, payload: { success: true, output_length: 512 } },
      { event_type: "tool.called",    step_index: 4,  timestamp: NOW-4172, payload: { tool_name: "database_lookup", args_hash: "cc22" } },
      { event_type: "tool.responded", step_index: 5,  timestamp: NOW-4165, payload: { success: true, output_length: 256 } },
      { event_type: "tool.called",    step_index: 6,  timestamp: NOW-4160, payload: { tool_name: "web_search", args_hash: "bb11" } },
      { event_type: "tool.responded", step_index: 7,  timestamp: NOW-4152, payload: { success: true, output_length: 508 } },
      { event_type: "tool.called",    step_index: 8,  timestamp: NOW-4147, payload: { tool_name: "database_lookup", args_hash: "cc22" } },
      { event_type: "tool.responded", step_index: 9,  timestamp: NOW-4140, payload: { success: true, output_length: 254 } },
      { event_type: "tool.called",    step_index: 10, timestamp: NOW-4135, payload: { tool_name: "web_search", args_hash: "bb11" } },
      { event_type: "tool.responded", step_index: 11, timestamp: NOW-4127, payload: { success: true, output_length: 510 } },
      { event_type: "tool.called",    step_index: 12, timestamp: NOW-4122, payload: { tool_name: "database_lookup", args_hash: "cc22" } },
      { event_type: "tool.responded", step_index: 13, timestamp: NOW-4115, payload: { success: true, output_length: 255 } },
      { event_type: "tool.called",    step_index: 14, timestamp: NOW-4110, payload: { tool_name: "web_search", args_hash: "bb11" } },
      { event_type: "tool.responded", step_index: 15, timestamp: NOW-4102, payload: { success: true, output_length: 511 } },
      { event_type: "llm.called",     step_index: 16, timestamp: NOW-4097, payload: { model: "gpt-4o-mini", prompt_tokens: 620 } },
      { event_type: "run.completed",  step_index: 17, timestamp: NOW-4100, payload: { exit_reason: "max_iterations" } },
    ],
    signals: [
      { id: 3, failure_type: "TOOL_THRASHING", severity: "HIGH", step_index: 14, confidence: 0.90,
        evidence: { tool_a: "web_search", tool_b: "database_lookup", oscillation_count: 4 },
        title: "Tool thrashing: oscillating between web_search and database_lookup",
        what: "The agent oscillated between web_search and database_lookup 4 times without converging.",
        why_it_matters: "Thrashing agents never produce an answer and exhaust token budgets.",
        evidence_summary: "4 oscillations across steps 2-15. Confidence: 90%.",
        suggested_fixes: [{ description: "Detect oscillation and break early", language: "python", code: "recent = deque(maxlen=6)\nif len(set(recent)) == 2 and len(recent) == 6:\n    raise RuntimeError('Oscillation detected')" }],
      }
    ],
  },

  "run-c1d2e3f4": {
    run_id: "run-c1d2e3f4", agent_id: "research-agent-v2", agent_version: "a7f3d9b2",
    started_at: NOW-320, completed_at: NOW-240, exit_reason: "completed", step_count: 6,
    events: [
      { event_type: "run.started",    step_index: 0, timestamp: NOW-320, payload: { input_hash: "9c3f2d1a", tools: ["web_search","database_lookup"] } },
      { event_type: "llm.called",     step_index: 1, timestamp: NOW-312, payload: { model: "gpt-4o-mini", prompt_tokens: 440 } },
      { event_type: "llm.responded",  step_index: 2, timestamp: NOW-305, payload: { finish_reason: "tool_calls", latency_ms: 740 } },
      { event_type: "tool.called",    step_index: 3, timestamp: NOW-303, payload: { tool_name: "database_lookup", args_hash: "fe9d8c7b" } },
      { event_type: "tool.responded", step_index: 4, timestamp: NOW-261, payload: { success: true, output_length: 3410 } },
      { event_type: "llm.called",     step_index: 5, timestamp: NOW-259, payload: { model: "gpt-4o-mini", prompt_tokens: 1380 } },
      { event_type: "run.completed",  step_index: 6, timestamp: NOW-240, payload: { exit_reason: "final_answer", total_steps: 6 } },
    ],
    signals: [
      { id: 4, failure_type: "SLOW_STEP", severity: "HIGH", step_index: 3, confidence: 0.92,
        evidence: { step_index: 3, duration_ms: 42000, threshold_ms: 15000, event_type: "tool.called", step_label: "tool execution", ratio: 2.8 },
        title: "Slow step: tool execution at step 3 took 42.0s (2.8x threshold)",
        what: "Step 3 (database_lookup) took 42.0s to complete, 2.8x the 15s threshold. The tool API either timed out or returned a very large payload. The agent was blocked waiting.",
        why_it_matters: "Slow tool calls add 42 seconds of latency to every user hitting this path. Under load, hung connections exhaust thread pools.",
        evidence_summary: "Step 3 (tool.called): 42.0s (threshold: 15s, ratio: 2.8x). Confidence: 92%.",
        suggested_fixes: [{ description: "Add a timeout to your tool call", language: "python", code: "result = await asyncio.wait_for(\n    run_tool('database_lookup', args),\n    timeout=10,\n)\n# Fails fast instead of hanging 42s" }],
      }
    ],
  },

  "run-8d3e1f77": {
    run_id: "run-8d3e1f77", agent_id: "research-agent-v2", agent_version: "a7f3d9b2",
    started_at: NOW-980, completed_at: NOW-910, exit_reason: "completed", step_count: 5,
    events: [
      { event_type: "run.started",    step_index: 0, timestamp: NOW-980, payload: { input_hash: "a1b2c3d4", tools: ["web_search"] } },
      { event_type: "llm.called",     step_index: 1, timestamp: NOW-972, payload: { model: "gpt-4o-mini", prompt_tokens: 390 } },
      { event_type: "llm.responded",  step_index: 2, timestamp: NOW-964, payload: { finish_reason: "tool_calls", latency_ms: 780 } },
      { event_type: "tool.called",    step_index: 3, timestamp: NOW-962, payload: { tool_name: "web_search", args_hash: "e5f6a7b8" } },
      { event_type: "tool.responded", step_index: 4, timestamp: NOW-950, payload: { success: true, output_length: 2048 } },
      { event_type: "run.completed",  step_index: 5, timestamp: NOW-910, payload: { exit_reason: "final_answer", total_steps: 5 } },
    ],
    signals: [],
  },
};

// ── Constants ──────────────────────────────────────────────────────────────────
const EVENT_META = {
  "run.started":    { color: "#22c55e", label: "Start" },
  "run.completed":  { color: "#22c55e", label: "Done" },
  "run.errored":    { color: "#ff3b3b", label: "Error" },
  "llm.called":     { color: "#818cf8", label: "LLM" },
  "llm.responded":  { color: "#6366f1", label: "LLM\u2193" },
  "tool.called":    { color: "#f97316", label: "Tool" },
  "tool.responded": { color: "#fb923c", label: "Tool\u2193" },
};

const SEVERITY_COLOR = {
  CRITICAL: "#ff3b3b", HIGH: "#ff7a00", MEDIUM: "#f5c518", LOW: "#22c55e",
};

// Token thresholds → color
function tokenColor(tok) {
  if (tok < 1500)  return "#22c55e";
  if (tok < 4000)  return "#f5c518";
  if (tok < 10000) return "#f97316";
  return "#ff3b3b";
}

function durationColor(ms) {
  if (ms < 2000)  return "#22c55e";
  if (ms < 10000) return "#f5c518";
  if (ms < 30000) return "#f97316";
  return "#ff3b3b";
}

function fmtDuration(ms) {
  if (ms < 1000) return ms + "ms";
  return (ms / 1000).toFixed(1) + "s";
}

function fmtK(n) {
  if (n < 1000) return String(n);
  return (n / 1000).toFixed(1) + "k";
}

function computeStepDurations(events) {
  const gaps = {};
  for (let i = 0; i < events.length - 1; i++) {
    const gap = Math.round((events[i + 1].timestamp - events[i].timestamp) * 1000);
    if (gap >= 0) gaps[events[i].step_index] = gap;
  }
  return gaps;
}

// ── Token series analysis ──────────────────────────────────────────────────────
// Returns array of { stepIndex, tokens, delta, cause } for each llm.called event.
// "cause" = the tool.responded with biggest output_length between this and the prior LLM call.
function computeTokenSeries(events) {
  const llmEvents = events.filter(e => e.event_type === "llm.called" && e.payload.prompt_tokens != null);
  if (llmEvents.length === 0) return [];

  const series = [];
  for (let i = 0; i < llmEvents.length; i++) {
    const ev = llmEvents[i];
    const tokens = ev.payload.prompt_tokens;
    const prevTokens = i > 0 ? llmEvents[i - 1].payload.prompt_tokens : null;
    const delta = prevTokens != null ? tokens - prevTokens : null;

    // Find tool.responded events between prior llm.called and this one
    const prevStep = i > 0 ? llmEvents[i - 1].step_index : -1;
    const thisStep = ev.step_index;
    const intervening = events.filter(e =>
      e.event_type === "tool.responded" &&
      e.step_index > prevStep &&
      e.step_index < thisStep &&
      e.payload.output_length != null
    );

    let cause = null;
    if (intervening.length > 0) {
      // Find the biggest contributor
      const biggest = intervening.reduce((a, b) =>
        (b.payload.output_length > a.payload.output_length) ? b : a
      );
      // Find its paired tool.called to get the tool name
      const toolCalledStep = biggest.step_index - 1;
      const toolCalled = events.find(e => e.event_type === "tool.called" && e.step_index === toolCalledStep);
      cause = {
        toolName: toolCalled ? (toolCalled.payload.tool_name || "tool") : "tool",
        outputLength: biggest.payload.output_length,
        toolCount: intervening.length,
      };
    }

    series.push({ stepIndex: ev.step_index, tokens, delta, cause });
  }
  return series;
}

function toolName(ev) { return ev.payload && ev.payload.tool_name ? ev.payload.tool_name : ""; }

// ── Signal popup ───────────────────────────────────────────────────────────────
function SignalPopup({ signal, onClose }) {
  const sc = SEVERITY_COLOR[signal.severity];
  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 100,
      display: "flex", alignItems: "center", justifyContent: "center",
      background: "rgba(0,0,0,0.75)", backdropFilter: "blur(4px)",
    }} onClick={onClose}>
      <div onClick={e => e.stopPropagation()} style={{
        width: 540, maxHeight: "88vh", overflowY: "auto",
        background: "#0f1117",
        border: `1px solid ${sc}44`, borderLeft: `4px solid ${sc}`,
        borderRadius: 8, padding: "24px 28px",
        boxShadow: `0 0 60px ${sc}22`,
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 20 }}>
          <div>
            <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 8 }}>
              <span style={{ fontSize: 10, fontWeight: 800, letterSpacing: "0.1em", color: sc, background: `${sc}18`, border: `1px solid ${sc}44`, padding: "2px 8px", borderRadius: 3 }}>{signal.severity}</span>
              <span style={{ fontSize: 10, color: "#4b5563", fontFamily: "monospace" }}>step {signal.step_index} · {Math.round(signal.confidence * 100)}% conf</span>
            </div>
            <div style={{ fontSize: 14, color: "#e8eaf0", lineHeight: 1.4, fontWeight: 500 }}>{signal.title}</div>
          </div>
          <button onClick={onClose} style={{ background: "none", border: "none", color: "#374151", cursor: "pointer", fontSize: 18, padding: 0, marginLeft: 16 }}>✕</button>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
          <div>
            <div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: "0.1em", color: "#374151", marginBottom: 6 }}>What happened</div>
            <div style={{ fontSize: 12, color: "#9ba3af", lineHeight: 1.6 }}>{signal.what}</div>
          </div>
          <div>
            <div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: "0.1em", color: "#374151", marginBottom: 6 }}>Why it matters</div>
            <div style={{ fontSize: 12, color: "#9ba3af", lineHeight: 1.6 }}>{signal.why_it_matters}</div>
          </div>
        </div>
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: "0.1em", color: "#374151", marginBottom: 6 }}>Evidence</div>
          <div style={{ fontFamily: "monospace", fontSize: 11, color: "#6b7280", background: "rgba(0,0,0,0.4)", padding: "10px 14px", borderRadius: 4, border: "1px solid rgba(255,255,255,0.05)" }}>{signal.evidence_summary}</div>
        </div>
        {signal.suggested_fixes[0] && (
          <div>
            <div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: "0.1em", color: "#374151", marginBottom: 6 }}>Fix — {signal.suggested_fixes[0].description}</div>
            <pre style={{ fontFamily: "monospace", fontSize: 11, color: "#86efac", background: "rgba(0,0,0,0.5)", padding: "12px 14px", borderRadius: 4, border: "1px solid rgba(34,197,94,0.12)", margin: 0, overflowX: "auto", lineHeight: 1.6 }}>{signal.suggested_fixes[0].code}</pre>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Duration strip ─────────────────────────────────────────────────────────────
function DurationStrip({ events, stepDurations, STEP_W, LEFT_PAD, signalByStep, onSignalClick }) {
  const MAX_BAR_H = 24;
  const allMs = Object.values(stepDurations);
  const maxMs = allMs.length ? Math.max(...allMs) : 1;
  const displayCap = Math.max(maxMs, 5000);

  return (
    <div>
      <div style={{ paddingLeft: LEFT_PAD, marginBottom: 4, display: "flex", alignItems: "center", gap: 10, fontSize: 9, color: "#1f2937", textTransform: "uppercase", letterSpacing: "0.1em" }}>
        <span>Step duration</span>
        <span style={{ display: "flex", gap: 8 }}>
          {[["#22c55e","<2s"],["#f5c518","2-10s"],["#f97316","10-30s"],["#ff3b3b",">30s"]].map(l => (
            <span key={l[1]} style={{ display: "inline-flex", alignItems: "center", gap: 3 }}>
              <span style={{ width: 6, height: 6, borderRadius: 1, background: l[0], display: "inline-block" }} />{l[1]}
            </span>
          ))}
        </span>
      </div>
      <svg width={LEFT_PAD + events.length * STEP_W + 40} height={MAX_BAR_H + 18} style={{ display: "block", overflow: "visible" }}>
        <line x1={LEFT_PAD - 8} y1={MAX_BAR_H} x2={LEFT_PAD + events.length * STEP_W + 20} y2={MAX_BAR_H} stroke="rgba(255,255,255,0.05)" strokeWidth={1} />
        {events.map((ev, i) => {
          const dms = stepDurations[ev.step_index];
          if (dms === undefined) return null;
          const sig = signalByStep[ev.step_index];
          const isSlowSig = sig && sig.failure_type === "SLOW_STEP";
          const x = LEFT_PAD + i * STEP_W;
          const barH = Math.max(2, (Math.min(dms, displayCap) / displayCap) * MAX_BAR_H);
          const color = isSlowSig ? SEVERITY_COLOR[sig.severity] : durationColor(dms);
          const barW = Math.min(STEP_W - 10, 28);
          return (
            <g key={i} style={{ cursor: isSlowSig ? "pointer" : "default" }} onClick={isSlowSig ? () => onSignalClick(sig) : undefined}>
              <rect x={x - barW/2} y={MAX_BAR_H - barH} width={barW} height={barH} rx={2} fill={color} opacity={isSlowSig ? 0.92 : 0.42} />
              {isSlowSig && <rect x={x - barW/2 - 1} y={MAX_BAR_H - barH - 1} width={barW + 2} height={barH + 2} rx={3} fill="none" stroke={color} strokeWidth={1.5} opacity={0.7} />}
              <text x={x} y={MAX_BAR_H + 12} textAnchor="middle" fill={dms >= 10000 ? color : "#1f2937"} fontSize={7.5} fontFamily="monospace" fontWeight={isSlowSig ? "700" : "400"}>{fmtDuration(dms)}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ── Token strip ────────────────────────────────────────────────────────────────
// Shows a bar per LLM call at its step x-position, with delta + cause annotation
// between consecutive bars.
function TokenStrip({ events, tokenSeries, STEP_W, LEFT_PAD, signalByStep, onSignalClick }) {
  if (tokenSeries.length === 0) {
    return (
      <div style={{ paddingLeft: LEFT_PAD, fontSize: 9, color: "#1f2937", fontFamily: "monospace", padding: "8px 0 4px " + LEFT_PAD + "px" }}>
        no prompt_tokens reported — pass prompt_tokens in llm.called payload to enable this view
      </div>
    );
  }

  const MAX_BAR_H = 48;
  const maxTokens = Math.max(...tokenSeries.map(s => s.tokens), 1);

  // Build lookup: stepIndex → series entry
  const byStep = {};
  tokenSeries.forEach(s => { byStep[s.stepIndex] = s; });

  // Step x positions
  const stepX = {};
  events.forEach((ev, i) => { stepX[ev.step_index] = LEFT_PAD + i * STEP_W; });

  const svgW = LEFT_PAD + events.length * STEP_W + 40;

  return (
    <div>
      <div style={{ paddingLeft: LEFT_PAD, marginBottom: 4, display: "flex", alignItems: "center", gap: 10, fontSize: 9, color: "#1f2937", textTransform: "uppercase", letterSpacing: "0.1em" }}>
        <span>Prompt tokens per LLM call</span>
        <span style={{ display: "flex", gap: 8 }}>
          {[["#22c55e","<1.5k"],["#f5c518","1.5-4k"],["#f97316","4-10k"],["#ff3b3b",">10k"]].map(l => (
            <span key={l[1]} style={{ display: "inline-flex", alignItems: "center", gap: 3 }}>
              <span style={{ width: 6, height: 6, borderRadius: 1, background: l[0], display: "inline-block" }} />{l[1]}
            </span>
          ))}
        </span>
      </div>

      <svg width={svgW} height={MAX_BAR_H + 52} style={{ display: "block", overflow: "visible" }}>
        {/* Baseline */}
        <line x1={LEFT_PAD - 8} y1={MAX_BAR_H} x2={LEFT_PAD + events.length * STEP_W + 20} y2={MAX_BAR_H}
          stroke="rgba(255,255,255,0.05)" strokeWidth={1} />

        {/* Trend line between LLM bars */}
        {tokenSeries.map((s, i) => {
          if (i === 0) return null;
          const prev = tokenSeries[i - 1];
          const x1 = stepX[prev.stepIndex];
          const x2 = stepX[s.stepIndex];
          if (x1 == null || x2 == null) return null;
          const y1 = MAX_BAR_H - (prev.tokens / maxTokens) * MAX_BAR_H;
          const y2 = MAX_BAR_H - (s.tokens / maxTokens) * MAX_BAR_H;
          return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="rgba(99,102,241,0.25)" strokeWidth={1} strokeDasharray="3,3" />;
        })}

        {/* Delta annotations between LLM calls */}
        {tokenSeries.map((s, i) => {
          if (i === 0 || s.delta == null) return null;
          const prev = tokenSeries[i - 1];
          const x1 = stepX[prev.stepIndex];
          const x2 = stepX[s.stepIndex];
          if (x1 == null || x2 == null) return null;
          const midX = (x1 + x2) / 2;
          const color = s.delta > 1500 ? "#ff7a00" : s.delta > 500 ? "#f5c518" : "#22c55e";
          const isLargest = tokenSeries.slice(1).every(other => other === s || (other.delta == null) || s.delta >= other.delta);

          return (
            <g key={"delta-" + i}>
              {/* Delta badge */}
              <rect x={midX - 26} y={MAX_BAR_H + 6} width={52} height={14} rx={3}
                fill={isLargest ? `${color}28` : "rgba(0,0,0,0.3)"}
                stroke={isLargest ? `${color}66` : "rgba(255,255,255,0.06)"} strokeWidth={0.5} />
              <text x={midX} y={MAX_BAR_H + 16} textAnchor="middle"
                fill={isLargest ? color : "#374151"} fontSize={8} fontFamily="monospace" fontWeight={isLargest ? "700" : "400"}>
                +{fmtK(s.delta)} tok
              </text>
              {/* Cause: biggest tool output responsible */}
              {s.cause && (
                <text x={midX} y={MAX_BAR_H + 30} textAnchor="middle"
                  fill={isLargest ? `${color}cc` : "#1f2937"} fontSize={7.5} fontFamily="monospace">
                  &#8592; {s.cause.toolName} {fmtK(s.cause.outputLength)}ch
                  {s.cause.toolCount > 1 ? ` +${s.cause.toolCount - 1}` : ""}
                </text>
              )}
            </g>
          );
        })}

        {/* LLM call bars */}
        {tokenSeries.map((s, i) => {
          const x = stepX[s.stepIndex];
          if (x == null) return null;
          const sig = signalByStep[s.stepIndex];
          const isBloatSig = sig && sig.failure_type === "CONTEXT_BLOAT";
          const barH = Math.max(3, (s.tokens / maxTokens) * MAX_BAR_H);
          const color = isBloatSig ? SEVERITY_COLOR[sig.severity] : tokenColor(s.tokens);
          const barW = 24;
          const barY = MAX_BAR_H - barH;

          return (
            <g key={i} style={{ cursor: isBloatSig ? "pointer" : "default" }}
              onClick={isBloatSig ? () => onSignalClick(sig) : undefined}>
              {/* Glow for signal */}
              {isBloatSig && <rect x={x - barW/2 - 3} y={barY - 3} width={barW + 6} height={barH + 6} rx={4} fill="none" stroke={color} strokeWidth={1.5} opacity={0.5} />}
              {/* Bar */}
              <rect x={x - barW/2} y={barY} width={barW} height={barH} rx={2}
                fill={color} opacity={isBloatSig ? 0.9 : 0.6} />
              {/* Token count above bar */}
              <text x={x} y={barY - 5} textAnchor="middle"
                fill={color} fontSize={8.5} fontFamily="monospace"
                fontWeight={isBloatSig ? "700" : "500"}>
                {fmtK(s.tokens)}
              </text>
              {/* Step label below */}
              <text x={x} y={MAX_BAR_H + 44} textAnchor="middle" fill="#1f2937" fontSize={7.5} fontFamily="monospace">
                s{s.stepIndex}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ── Event node ─────────────────────────────────────────────────────────────────
function EventNode({ event, signal, x, isHovered, onClick }) {
  const meta = EVENT_META[event.event_type] || { color: "#374151", label: "?" };
  const hasSig = !!signal;
  const sc = hasSig ? SEVERITY_COLOR[signal.severity] : null;
  const tn = event.event_type === "tool.called" ? toolName(event) : "";
  const icons = { "run.started": "▶", "run.completed": "■", "run.errored": "✕", "llm.called": "◆", "llm.responded": "◇", "tool.called": "⬡", "tool.responded": "⬡" };
  const icon = icons[event.event_type] || "•";

  return (
    <g>
      {hasSig && (
        <g>
          <line x1={x} y1={-6} x2={x} y2={-40} stroke={sc} strokeWidth={1.5} strokeDasharray="2,2" opacity={0.7} />
          <polygon points={`${x},${-6} ${x-5},${-16} ${x+5},${-16}`} fill={sc} opacity={0.9} />
          <rect x={x-42} y={-62} width={84} height={16} rx={2} fill={`${sc}20`} stroke={`${sc}55`} strokeWidth={0.5} />
          <text x={x} y={-50} textAnchor="middle" fill={sc} fontSize={7.5} fontWeight="700" fontFamily="monospace" letterSpacing="0.04em">
            {signal.failure_type.replace(/_/g, " ")}
          </text>
        </g>
      )}
      <circle cx={x} cy={0} r={hasSig ? 10 : 7}
        fill={hasSig ? `${sc}22` : `${meta.color}18`}
        stroke={hasSig ? sc : isHovered ? meta.color : `${meta.color}77`}
        strokeWidth={hasSig ? 2 : 1}
        style={{ cursor: hasSig ? "pointer" : "default" }}
        onClick={hasSig ? onClick : undefined}
      />
      {hasSig && <circle cx={x} cy={0} r={14} fill="none" stroke={sc} strokeWidth={0.5} opacity={0.3} strokeDasharray="3,3" />}
      <text x={x} y={4} textAnchor="middle" fill={hasSig ? sc : meta.color} fontSize={hasSig ? 9 : 7} fontFamily="monospace" style={{ pointerEvents: "none" }}>
        {hasSig ? "!" : icon}
      </text>
      <text x={x} y={22} textAnchor="middle" fill={isHovered ? "#6b7280" : "#1f2937"} fontSize={8} fontFamily="monospace">{event.step_index}</text>
      <text x={x} y={34} textAnchor="middle" fill={isHovered ? meta.color : "#1f2937"} fontSize={8} fontFamily="monospace">
        {meta.label}{tn ? " " + tn : ""}
      </text>
    </g>
  );
}

// ── Run selector ───────────────────────────────────────────────────────────────
function RunSelector({ selectedId, onSelect }) {
  const exitColor = { completed: "#22c55e", error: "#ff3b3b", stalled: "#f5c518" };
  return (
    <div style={{ display: "flex", gap: 6, marginBottom: 24, flexWrap: "wrap" }}>
      {MOCK_RUNS.map(r => {
        const detail = MOCK_DETAIL[r.run_id];
        const isSel = selectedId === r.run_id;
        const worstSig = detail.signals.length
          ? detail.signals.reduce((a, b) => ({ CRITICAL:3,HIGH:2,MEDIUM:1,LOW:0 }[a.severity] >= { CRITICAL:3,HIGH:2,MEDIUM:1,LOW:0 }[b.severity] ? a : b))
          : null;
        return (
          <button key={r.run_id} onClick={() => onSelect(r.run_id)} style={{
            padding: "8px 14px", borderRadius: 5, cursor: "pointer",
            background: isSel ? "rgba(249,115,22,0.1)" : "rgba(255,255,255,0.02)",
            border: `1px solid ${isSel ? "rgba(249,115,22,0.4)" : "rgba(255,255,255,0.07)"}`,
            textAlign: "left",
          }}>
            <div style={{ fontFamily: "monospace", fontSize: 10, color: isSel ? "#f97316" : "#4b5563", marginBottom: 3 }}>{r.run_id}</div>
            <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <span style={{ fontSize: 10, color: exitColor[detail.exit_reason] || "#6b7280", fontWeight: 600, textTransform: "uppercase" }}>{detail.exit_reason}</span>
              {worstSig
                ? <span style={{ fontSize: 9, color: SEVERITY_COLOR[worstSig.severity], fontWeight: 700 }}>▲ {r.label}</span>
                : <span style={{ fontSize: 9, color: "#22c55e" }}>✓ clean</span>}
            </div>
          </button>
        );
      })}
    </div>
  );
}

// ── Main timeline ──────────────────────────────────────────────────────────────
function Timeline({ run }) {
  const [activeSignal, setActiveSignal] = useState(null);
  const [hoveredStep, setHoveredStep] = useState(null);

  const events = run.events;
  const signals = run.signals;
  const signalByStep = Object.fromEntries(signals.map(s => [s.step_index, s]));
  const stepDurations = computeStepDurations(events);
  const tokenSeries = computeTokenSeries(events);

  const STEP_W = 72;
  const LEFT_PAD = 60;
  const svgWidth = LEFT_PAD + events.length * STEP_W + 40;
  const TRACK_Y = 72;
  const svgHeight = 130;

  const totalDuration = run.completed_at - run.started_at;
  let slowestMs = 0, slowestStep = null;
  Object.entries(stepDurations).forEach(([k, v]) => { if (v > slowestMs) { slowestMs = v; slowestStep = k; } });

  const lastTokens = tokenSeries.length ? tokenSeries[tokenSeries.length - 1].tokens : null;
  const firstTokens = tokenSeries.length ? tokenSeries[0].tokens : null;
  const tokenGrowth = (firstTokens && lastTokens && tokenSeries.length > 1) ? (lastTokens / firstTokens).toFixed(1) : null;

  return (
    <div>
      {/* Metadata row */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 0, marginBottom: 20, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 6, overflow: "hidden" }}>
        {[
          { label: "Run ID",       value: run.run_id },
          { label: "Agent",        value: run.agent_id },
          { label: "Version",      value: run.agent_version.slice(0, 8) },
          { label: "Duration",     value: `${totalDuration.toFixed(1)}s` },
          { label: "Token growth",
            value: tokenGrowth ? `${firstTokens} → ${lastTokens} (${tokenGrowth}×)` : "—",
            color: tokenGrowth >= 3 ? "#ff7a00" : tokenGrowth >= 1.5 ? "#f5c518" : "#6b7280" },
          { label: "Exit",
            value: run.exit_reason,
            color: { completed: "#22c55e", error: "#ff3b3b", stalled: "#f5c518" }[run.exit_reason] },
        ].map((m, i) => (
          <div key={i} style={{ padding: "12px 16px", borderRight: i < 5 ? "1px solid rgba(255,255,255,0.05)" : "none" }}>
            <div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: "0.1em", color: "#374151", marginBottom: 4 }}>{m.label}</div>
            <div style={{ fontSize: 12, fontWeight: 600, color: m.color || "#9ba3af", fontFamily: "monospace" }}>{m.value}</div>
          </div>
        ))}
      </div>

      {/* Signal badges */}
      {signals.length > 0 && (
        <div style={{ display: "flex", gap: 6, marginBottom: 16, flexWrap: "wrap" }}>
          {signals.map(s => {
            const sc = SEVERITY_COLOR[s.severity];
            return (
              <button key={s.id} onClick={() => setActiveSignal(s)} style={{ display: "flex", alignItems: "center", gap: 6, padding: "5px 12px", borderRadius: 4, cursor: "pointer", background: `${sc}12`, border: `1px solid ${sc}44` }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: sc }} />
                <span style={{ fontSize: 11, fontWeight: 700, color: sc, letterSpacing: "0.06em" }}>{s.severity}</span>
                <span style={{ fontSize: 11, color: "#6b7280" }}>step {s.step_index} — {s.failure_type.replace(/_/g, " ")}</span>
              </button>
            );
          })}
        </div>
      )}

      {/* Timeline + both strips */}
      <div style={{ background: "rgba(0,0,0,0.25)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 8, padding: "20px 0 14px", overflowX: "auto" }}>
        {/* Event track */}
        <svg width={svgWidth} height={svgHeight} style={{ display: "block", overflow: "visible" }}>
          <line x1={LEFT_PAD} y1={TRACK_Y} x2={LEFT_PAD + (events.length-1) * STEP_W} y2={TRACK_Y} stroke="rgba(255,255,255,0.07)" strokeWidth={1.5} />
          {events.map((ev, i) => {
            if (i === 0) return null;
            const x1 = LEFT_PAD + (i-1) * STEP_W, x2 = LEFT_PAD + i * STEP_W;
            const gap = (ev.timestamp - events[i-1].timestamp) * 1000;
            const op = Math.min(1, gap / 20000);
            return <line key={i} x1={x1} y1={TRACK_Y} x2={x2} y2={TRACK_Y} stroke={`rgba(249,115,22,${0.08 + op * 0.55})`} strokeWidth={1 + op * 2.5} />;
          })}
          <text x={LEFT_PAD - 8} y={TRACK_Y + 4} textAnchor="end" fill="#1f2937" fontSize={9} fontFamily="monospace">EVT</text>
          {events.map((ev, i) => (
            <g key={i} transform={`translate(0,${TRACK_Y})`}
              onMouseEnter={() => setHoveredStep(i)} onMouseLeave={() => setHoveredStep(null)}>
              <EventNode event={ev} signal={signalByStep[ev.step_index]} x={LEFT_PAD + i * STEP_W}
                isHovered={hoveredStep === i} onClick={() => signalByStep[ev.step_index] && setActiveSignal(signalByStep[ev.step_index])} />
            </g>
          ))}
        </svg>

        {/* Token strip */}
        <div style={{ borderTop: "1px solid rgba(255,255,255,0.04)", marginTop: 4, paddingTop: 10 }}>
          <TokenStrip events={events} tokenSeries={tokenSeries} STEP_W={STEP_W} LEFT_PAD={LEFT_PAD} signalByStep={signalByStep} onSignalClick={setActiveSignal} />
        </div>

        {/* Duration strip */}
        <div style={{ borderTop: "1px solid rgba(255,255,255,0.04)", marginTop: 8, paddingTop: 10 }}>
          <DurationStrip events={events} stepDurations={stepDurations} STEP_W={STEP_W} LEFT_PAD={LEFT_PAD} signalByStep={signalByStep} onSignalClick={setActiveSignal} />
        </div>
      </div>

      {/* Hover detail panel */}
      {hoveredStep !== null && (() => {
        const ev = events[hoveredStep];
        const meta = EVENT_META[ev.event_type] || {};
        const dur = stepDurations[ev.step_index];
        const tokEntry = tokenSeries.find(s => s.stepIndex === ev.step_index);
        return (
          <div style={{ marginTop: 8, padding: "10px 16px", background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 5, display: "flex", gap: 24, alignItems: "flex-start", flexWrap: "wrap" }}>
            <div>
              <div style={{ fontSize: 9, color: "#374151", textTransform: "uppercase", marginBottom: 3 }}>Event</div>
              <div style={{ fontFamily: "monospace", fontSize: 12, color: meta.color || "#9ba3af" }}>{ev.event_type}</div>
            </div>
            <div>
              <div style={{ fontSize: 9, color: "#374151", textTransform: "uppercase", marginBottom: 3 }}>Step</div>
              <div style={{ fontFamily: "monospace", fontSize: 12, color: "#6b7280" }}>{ev.step_index}</div>
            </div>
            {dur !== undefined && (
              <div>
                <div style={{ fontSize: 9, color: "#374151", textTransform: "uppercase", marginBottom: 3 }}>Duration</div>
                <div style={{ fontFamily: "monospace", fontSize: 12, color: durationColor(dur), fontWeight: dur > 10000 ? 700 : 400 }}>{fmtDuration(dur)}</div>
              </div>
            )}
            {tokEntry && (
              <div>
                <div style={{ fontSize: 9, color: "#374151", textTransform: "uppercase", marginBottom: 3 }}>Prompt tokens</div>
                <div style={{ fontFamily: "monospace", fontSize: 12, color: tokenColor(tokEntry.tokens) }}>{tokEntry.tokens.toLocaleString()}{tokEntry.delta != null ? ` (+${tokEntry.delta.toLocaleString()})` : ""}</div>
              </div>
            )}
            {tokEntry && tokEntry.cause && (
              <div>
                <div style={{ fontSize: 9, color: "#374151", textTransform: "uppercase", marginBottom: 3 }}>Biggest source</div>
                <div style={{ fontFamily: "monospace", fontSize: 12, color: "#f97316" }}>{tokEntry.cause.toolName} · {tokEntry.cause.outputLength.toLocaleString()} chars</div>
              </div>
            )}
            {Object.entries(ev.payload).slice(0, 3).map(([k, v]) => (
              <div key={k}>
                <div style={{ fontSize: 9, color: "#374151", textTransform: "uppercase", marginBottom: 3 }}>{k.replace(/_/g, " ")}</div>
                <div style={{ fontFamily: "monospace", fontSize: 12, color: "#6b7280" }}>{Array.isArray(v) ? v.join(", ") : String(v)}</div>
              </div>
            ))}
          </div>
        );
      })()}

      {activeSignal && <SignalPopup signal={activeSignal} onClose={() => setActiveSignal(null)} />}
    </div>
  );
}

// ── App ────────────────────────────────────────────────────────────────────────
export default function App() {
  const [selectedRun, setSelectedRun] = useState("run-a9b8c7d6");
  const run = MOCK_DETAIL[selectedRun];

  return (
    <div style={{ minHeight: "100vh", background: "#0a0b0d", fontFamily: "'DM Mono','Fira Code',monospace", color: "#e8eaf0", padding: "32px" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&display=swap');
        * { box-sizing: border-box; }
        button { font-family: inherit; }
        ::-webkit-scrollbar { height: 4px; width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 2px; }
      `}</style>

      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 32 }}>
        <div style={{ width: 28, height: 28, borderRadius: 5, background: "linear-gradient(135deg,#f97316,#dc2626)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13, fontWeight: 900, color: "#fff" }}>D</div>
        <div>
          <div style={{ fontSize: 15, fontWeight: 500, color: "#e8eaf0" }}>Dunetrace — Run Timeline</div>
          <div style={{ fontSize: 10, color: "#374151", letterSpacing: "0.08em" }}>research-agent-v2 · token strip shows prompt size per LLM call + what fed the growth</div>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: 16, alignItems: "center" }}>
          {[["#818cf8","LLM calls"],["#f97316","Tool calls"],["#22c55e","< 1.5k tok"],["#f5c518","1.5-4k tok"],["#f97316","4-10k tok"],["#ff7a00","Signal"]].map(l => (
            <div key={l[1]} style={{ display: "flex", alignItems: "center", gap: 5 }}>
              <div style={{ width: 6, height: 6, borderRadius: "50%", background: l[0] }} />
              <span style={{ fontSize: 9, color: "#374151" }}>{l[1]}</span>
            </div>
          ))}
        </div>
      </div>

      <RunSelector selectedId={selectedRun} onSelect={setSelectedRun} />
      <Timeline run={run} key={selectedRun} />

      <div style={{ marginTop: 14, fontSize: 10, color: "#1f2937" }}>
        Token strip: bar height = prompt_tokens at each LLM call · +N tok = delta since prior call · ← tool Xch = biggest tool output that caused the growth · hover step for details
      </div>
    </div>
  );
}