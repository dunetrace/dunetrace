const { useState } = React;

// ── Mock run data ──────────────────────────────────────────────────────────────
const NOW = Date.now() / 1000;

const MOCK_RUNS = [
  { run_id: "run-f4a9b2c1", label: "TOOL_LOOP" },
  { run_id: "run-2c9a5b4e", label: "PROMPT_INJECTION" },
  { run_id: "run-71bce930", label: "TOOL_THRASHING" },
  { run_id: "run-c1d2e3f4", label: "SLOW_STEP" },
  { run_id: "run-8d3e1f77", label: "Clean run" },
];

const MOCK_DETAIL = {
  "run-f4a9b2c1": {
    run_id: "run-f4a9b2c1", agent_id: "research-agent-v2", agent_version: "a7f3d9b2",
    started_at: NOW - 182, completed_at: NOW - 54, exit_reason: "completed", step_count: 12,
    events: [
      { event_type: "run.started",    step_index: 0,  timestamp: NOW - 182, payload: { input_hash: "e3b0c442", tools: ["web_search", "calculator"] } },
      { event_type: "llm.called",     step_index: 1,  timestamp: NOW - 175, payload: { model: "gpt-4o-mini", prompt_tokens: 480 } },
      { event_type: "llm.responded",  step_index: 2,  timestamp: NOW - 168, payload: { finish_reason: "tool_calls", latency_ms: 820, output_length: 312 } },
      { event_type: "tool.called",    step_index: 3,  timestamp: NOW - 166, payload: { tool_name: "web_search", args_hash: "a1b2c3d4" } },
      { event_type: "tool.responded", step_index: 4,  timestamp: NOW - 158, payload: { success: true, output_length: 1024 } },
      { event_type: "tool.called",    step_index: 5,  timestamp: NOW - 155, payload: { tool_name: "web_search", args_hash: "a1b2c3d4" } },
      { event_type: "tool.responded", step_index: 6,  timestamp: NOW - 147, payload: { success: true, output_length: 1019 } },
      { event_type: "tool.called",    step_index: 7,  timestamp: NOW - 143, payload: { tool_name: "web_search", args_hash: "a1b2c3d4" } },
      { event_type: "tool.responded", step_index: 8,  timestamp: NOW - 135, payload: { success: true, output_length: 1021 } },
      { event_type: "tool.called",    step_index: 9,  timestamp: NOW - 132, payload: { tool_name: "web_search", args_hash: "a1b2c3d4" } },
      { event_type: "tool.responded", step_index: 10, timestamp: NOW - 124, payload: { success: true, output_length: 1018 } },
      { event_type: "tool.called",    step_index: 11, timestamp: NOW - 121, payload: { tool_name: "web_search", args_hash: "a1b2c3d4" } },
      { event_type: "run.completed",  step_index: 12, timestamp: NOW - 54,  payload: { exit_reason: "final_answer", total_steps: 12 } },
    ],
    signals: [
      { id: 1, failure_type: "TOOL_LOOP", severity: "HIGH", step_index: 11, confidence: 0.95,
        evidence: { tool: "web_search", count: 5, window: 5 },
        title: "Tool loop: `web_search` called 5x in 5 steps",
        what: "The agent called `web_search` 5 consecutive times with identical arguments, making no progress between calls.",
        why_it_matters: "Looping agents burn tokens and cost money without producing value.",
        evidence_summary: "Tool `web_search` called 5x in steps 7-11 with identical args. Confidence: 95%.",
        suggested_fixes: [{ description: "Add per-tool call limit", language: "python", code: "MAX_CALLS_PER_TOOL = 3\nif tool_call_counts[tool] > MAX_CALLS_PER_TOOL:\n    raise RuntimeError(f'{tool} called too many times')" }],
      }
    ],
  },
  "run-2c9a5b4e": {
    run_id: "run-2c9a5b4e", agent_id: "research-agent-v2", agent_version: "a7f3d9b2",
    started_at: NOW - 1850, completed_at: NOW - 1780, exit_reason: "error", step_count: 3,
    events: [
      { event_type: "run.started",  step_index: 0, timestamp: NOW - 1850, payload: { input_hash: "d4e5f6a7", tools: ["web_search"] } },
      { event_type: "llm.called",   step_index: 1, timestamp: NOW - 1845, payload: { model: "gpt-4o-mini", prompt_tokens: 310 } },
      { event_type: "run.errored",  step_index: 2, timestamp: NOW - 1780, payload: { error: "InputRejected: injection detected" } },
    ],
    signals: [
      { id: 2, failure_type: "PROMPT_INJECTION_SIGNAL", severity: "CRITICAL", step_index: 1, confidence: 0.85,
        evidence: { matched_patterns: ["ignore_instructions", "you_are_now"], pattern_count: 2 },
        title: "Prompt injection attempt (2 patterns matched)",
        what: "The user input matched 2 known injection patterns at step 1.",
        why_it_matters: "A successful injection can cause the agent to ignore safety instructions.",
        evidence_summary: "Matched: ignore_instructions, you_are_now. Confidence: 85%.",
        suggested_fixes: [{ description: "Reject and log injection attempts", language: "python", code: "if injection_detected:\n    log_security_event(user_id, input)\n    return {'error': 'INPUT_REJECTED'}" }],
      }
    ],
  },
  "run-71bce930": {
    run_id: "run-71bce930", agent_id: "research-agent-v2", agent_version: "a7f3d9b2",
    started_at: NOW - 4200, completed_at: NOW - 4100, exit_reason: "stalled", step_count: 18,
    events: [
      { event_type: "run.started",    step_index: 0,  timestamp: NOW - 4200, payload: { input_hash: "f9e8d7c6", tools: ["web_search", "database_lookup"] } },
      { event_type: "llm.called",     step_index: 1,  timestamp: NOW - 4192, payload: { model: "gpt-4o-mini", prompt_tokens: 520 } },
      { event_type: "tool.called",    step_index: 2,  timestamp: NOW - 4185, payload: { tool_name: "web_search", args_hash: "bb11" } },
      { event_type: "tool.responded", step_index: 3,  timestamp: NOW - 4177, payload: { success: true, output_length: 512 } },
      { event_type: "tool.called",    step_index: 4,  timestamp: NOW - 4172, payload: { tool_name: "database_lookup", args_hash: "cc22" } },
      { event_type: "tool.responded", step_index: 5,  timestamp: NOW - 4165, payload: { success: true, output_length: 256 } },
      { event_type: "tool.called",    step_index: 6,  timestamp: NOW - 4160, payload: { tool_name: "web_search", args_hash: "bb11" } },
      { event_type: "tool.responded", step_index: 7,  timestamp: NOW - 4152, payload: { success: true, output_length: 508 } },
      { event_type: "tool.called",    step_index: 8,  timestamp: NOW - 4147, payload: { tool_name: "database_lookup", args_hash: "cc22" } },
      { event_type: "tool.responded", step_index: 9,  timestamp: NOW - 4140, payload: { success: true, output_length: 254 } },
      { event_type: "tool.called",    step_index: 10, timestamp: NOW - 4135, payload: { tool_name: "web_search", args_hash: "bb11" } },
      { event_type: "tool.responded", step_index: 11, timestamp: NOW - 4127, payload: { success: true, output_length: 510 } },
      { event_type: "tool.called",    step_index: 12, timestamp: NOW - 4122, payload: { tool_name: "database_lookup", args_hash: "cc22" } },
      { event_type: "tool.responded", step_index: 13, timestamp: NOW - 4115, payload: { success: true, output_length: 255 } },
      { event_type: "tool.called",    step_index: 14, timestamp: NOW - 4110, payload: { tool_name: "web_search", args_hash: "bb11" } },
      { event_type: "tool.responded", step_index: 15, timestamp: NOW - 4102, payload: { success: true, output_length: 511 } },
      { event_type: "llm.called",     step_index: 16, timestamp: NOW - 4097, payload: { model: "gpt-4o-mini", prompt_tokens: 620 } },
      { event_type: "run.completed",  step_index: 17, timestamp: NOW - 4100, payload: { exit_reason: "max_iterations" } },
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
    started_at: NOW - 320, completed_at: NOW - 240, exit_reason: "completed", step_count: 6,
    events: [
      { event_type: "run.started",    step_index: 0, timestamp: NOW - 320, payload: { input_hash: "9c3f2d1a", tools: ["web_search", "database_lookup"] } },
      { event_type: "llm.called",     step_index: 1, timestamp: NOW - 312, payload: { model: "gpt-4o-mini", prompt_tokens: 440 } },
      { event_type: "llm.responded",  step_index: 2, timestamp: NOW - 305, payload: { finish_reason: "tool_calls", latency_ms: 740, output_length: 280 } },
      { event_type: "tool.called",    step_index: 3, timestamp: NOW - 303, payload: { tool_name: "database_lookup", args_hash: "fe9d8c7b" } },
      { event_type: "tool.responded", step_index: 4, timestamp: NOW - 261, payload: { success: true, output_length: 3410 } },
      { event_type: "llm.called",     step_index: 5, timestamp: NOW - 259, payload: { model: "gpt-4o-mini", prompt_tokens: 1380 } },
      { event_type: "run.completed",  step_index: 6, timestamp: NOW - 240, payload: { exit_reason: "final_answer", total_steps: 6 } },
    ],
    signals: [
      { id: 4, failure_type: "SLOW_STEP", severity: "HIGH", step_index: 3, confidence: 0.92,
        evidence: { step_index: 3, duration_ms: 42000, threshold_ms: 15000, event_type: "tool.called", step_label: "tool execution", ratio: 2.8 },
        title: "Slow step: tool execution at step 3 took 42.0s (2.8x threshold)",
        what: "Step 3 (tool.called: database_lookup) took 42.0s to complete, 2.8x the 15s threshold. The tool API either timed out or returned a very large payload. The agent was blocked waiting the entire time.",
        why_it_matters: "Slow tool calls add 42 seconds of latency to every user hitting this path. Under load, hung connections exhaust thread pools and cause cascading timeouts.",
        evidence_summary: "Step 3 (tool.called): 42.0s (threshold: 15s, ratio: 2.8x). Confidence: 92%.",
        suggested_fixes: [{ description: "Add a timeout to your tool call", language: "python", code: "result = await asyncio.wait_for(\n    run_tool('database_lookup', args),\n    timeout=10,  # fail fast\n)\n# Returns error cleanly instead of hanging 42s" }],
      }
    ],
  },
  "run-8d3e1f77": {
    run_id: "run-8d3e1f77", agent_id: "research-agent-v2", agent_version: "a7f3d9b2",
    started_at: NOW - 980, completed_at: NOW - 910, exit_reason: "completed", step_count: 5,
    events: [
      { event_type: "run.started",    step_index: 0, timestamp: NOW - 980, payload: { input_hash: "a1b2c3d4", tools: ["web_search"] } },
      { event_type: "llm.called",     step_index: 1, timestamp: NOW - 972, payload: { model: "gpt-4o-mini", prompt_tokens: 390 } },
      { event_type: "llm.responded",  step_index: 2, timestamp: NOW - 964, payload: { finish_reason: "tool_calls", latency_ms: 780, output_length: 290 } },
      { event_type: "tool.called",    step_index: 3, timestamp: NOW - 962, payload: { tool_name: "web_search", args_hash: "e5f6a7b8" } },
      { event_type: "tool.responded", step_index: 4, timestamp: NOW - 950, payload: { success: true, output_length: 2048 } },
      { event_type: "run.completed",  step_index: 5, timestamp: NOW - 910, payload: { exit_reason: "final_answer", total_steps: 5 } },
    ],
    signals: [],
  },
};

// ── Constants ──────────────────────────────────────────────────────────────────
const EVENT_META = {
  "run.started":    { icon: "PLAY", color: "#22c55e",  label: "Start" },
  "run.completed":  { icon: "END",  color: "#22c55e",  label: "Done" },
  "run.errored":    { icon: "ERR",  color: "#ff3b3b",  label: "Error" },
  "llm.called":     { icon: "LLM",  color: "#818cf8",  label: "LLM" },
  "llm.responded":  { icon: "LLM",  color: "#6366f1",  label: "LLM dn" },
  "tool.called":    { icon: "TOOL", color: "#f97316",  label: "Tool" },
  "tool.responded": { icon: "TOOL", color: "#fb923c",  label: "Tool dn" },
};

const SEVERITY_COLOR = {
  CRITICAL: "#ff3b3b", HIGH: "#ff7a00", MEDIUM: "#f5c518", LOW: "#22c55e",
};

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

function computeStepDurations(events) {
  const gaps = {};
  for (let i = 0; i < events.length - 1; i++) {
    const gap = Math.round((events[i + 1].timestamp - events[i].timestamp) * 1000);
    if (gap >= 0) gaps[events[i].step_index] = gap;
  }
  return gaps;
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
      <div onClick={function(e){ e.stopPropagation(); }} style={{
        width: 540, maxHeight: "88vh", overflowY: "auto",
        background: "#0f1117",
        border: "1px solid " + sc + "44",
        borderLeft: "4px solid " + sc,
        borderRadius: 8, padding: "24px 28px",
        boxShadow: "0 0 60px " + sc + "22",
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 20 }}>
          <div>
            <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 8 }}>
              <span style={{
                fontSize: 10, fontWeight: 800, letterSpacing: "0.1em", color: sc,
                background: sc + "18", border: "1px solid " + sc + "44",
                padding: "2px 8px", borderRadius: 3,
              }}>{signal.severity}</span>
              <span style={{ fontSize: 10, color: "#4b5563", fontFamily: "monospace" }}>
                step {signal.step_index} &middot; {Math.round(signal.confidence * 100)}% conf
              </span>
            </div>
            <div style={{ fontSize: 14, color: "#e8eaf0", lineHeight: 1.4, fontWeight: 500 }}>{signal.title}</div>
          </div>
          <button onClick={onClose} style={{ background: "none", border: "none", color: "#374151", cursor: "pointer", fontSize: 18, padding: 0, marginLeft: 16 }}>x</button>
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
          <div style={{ fontFamily: "monospace", fontSize: 11, color: "#6b7280", background: "rgba(0,0,0,0.4)", padding: "10px 14px", borderRadius: 4, border: "1px solid rgba(255,255,255,0.05)" }}>
            {signal.evidence_summary}
          </div>
        </div>
        {signal.suggested_fixes[0] && (
          <div>
            <div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: "0.1em", color: "#374151", marginBottom: 6 }}>
              Fix &mdash; {signal.suggested_fixes[0].description}
            </div>
            <pre style={{ fontFamily: "monospace", fontSize: 11, color: "#86efac", background: "rgba(0,0,0,0.5)", padding: "12px 14px", borderRadius: 4, border: "1px solid rgba(34,197,94,0.12)", margin: 0, overflowX: "auto", lineHeight: 1.6 }}>
              {signal.suggested_fixes[0].code}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Duration strip ─────────────────────────────────────────────────────────────
function DurationStrip({ events, stepDurations, STEP_W, LEFT_PAD, signalByStep, onSignalClick }) {
  const MAX_BAR_H = 30;
  const allMs = Object.values(stepDurations);
  const maxMs = allMs.length ? Math.max.apply(null, allMs) : 1;
  const displayCap = Math.max(maxMs, 5000);

  return (
    <div style={{ marginTop: 4 }}>
      <div style={{ marginBottom: 4, paddingLeft: LEFT_PAD, display: "flex", alignItems: "center", gap: 4, fontSize: 9, color: "#1f2937", textTransform: "uppercase", letterSpacing: "0.1em" }}>
        <span>Step duration</span>
        <span style={{ marginLeft: 10, display: "flex", gap: 8 }}>
          {[["#22c55e","<2s"],["#f5c518","2-10s"],["#f97316","10-30s"],["#ff3b3b",">30s"]].map(function(l) {
            return (
              <span key={l[1]} style={{ display: "inline-flex", alignItems: "center", gap: 3 }}>
                <span style={{ display: "inline-block", width: 6, height: 6, borderRadius: 1, background: l[0] }} />
                <span>{l[1]}</span>
              </span>
            );
          })}
        </span>
      </div>
      <div style={{ overflowX: "auto" }}>
        <svg width={LEFT_PAD + events.length * STEP_W + 40} height={MAX_BAR_H + 20} style={{ display: "block", overflow: "visible" }}>
          <line x1={LEFT_PAD - 8} y1={MAX_BAR_H} x2={LEFT_PAD + events.length * STEP_W + 20} y2={MAX_BAR_H}
            stroke="rgba(255,255,255,0.05)" strokeWidth={1} />
          {events.map(function(ev, i) {
            var dms = stepDurations[ev.step_index];
            if (dms === undefined) return null;
            var sig = signalByStep[ev.step_index];
            var isSlowSig = sig && sig.failure_type === "SLOW_STEP";
            var x = LEFT_PAD + i * STEP_W;
            var barH = Math.max(2, (Math.min(dms, displayCap) / displayCap) * MAX_BAR_H);
            var color = isSlowSig ? SEVERITY_COLOR[sig.severity] : durationColor(dms);
            var barY = MAX_BAR_H - barH;
            var barW = Math.min(STEP_W - 10, 30);
            var barX = x - barW / 2;
            return (
              <g key={i} style={{ cursor: isSlowSig ? "pointer" : "default" }}
                onClick={isSlowSig ? function() { onSignalClick(sig); } : undefined}>
                <rect x={barX} y={barY} width={barW} height={barH} rx={2}
                  fill={color} opacity={isSlowSig ? 0.92 : 0.45} />
                {isSlowSig && (
                  <rect x={barX - 1} y={barY - 1} width={barW + 2} height={barH + 2} rx={3}
                    fill="none" stroke={color} strokeWidth={1.5} opacity={0.7} />
                )}
                <text x={x} y={MAX_BAR_H + 13} textAnchor="middle"
                  fill={dms >= 10000 ? color : "#1f2937"}
                  fontSize={8} fontFamily="monospace"
                  fontWeight={isSlowSig ? "700" : "400"}>
                  {fmtDuration(dms)}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}

// ── Event node ─────────────────────────────────────────────────────────────────
function EventNode({ event, signal, x, isHovered, onClick }) {
  var meta = EVENT_META[event.event_type] || { icon: "?", color: "#374151", label: "?" };
  var hasSig = !!signal;
  var sc = hasSig ? SEVERITY_COLOR[signal.severity] : null;
  var tn = event.event_type === "tool.called" ? toolName(event) : "";

  // Encode icon as short text
  var iconChars = { "PLAY": "▶", "END": "■", "ERR": "✕", "LLM": "◆", "TOOL": "⬡" };
  var icon = iconChars[meta.icon] || meta.icon;

  return (
    <g>
      {hasSig && (
        <g>
          <line x1={x} y1={-6} x2={x} y2={-40} stroke={sc} strokeWidth={1.5} strokeDasharray="2,2" opacity={0.7} />
          <polygon points={x + "," + (-6) + " " + (x-5) + "," + (-16) + " " + (x+5) + "," + (-16)} fill={sc} opacity={0.9} />
          <rect x={x-42} y={-62} width={84} height={16} rx={2} fill={sc + "20"} stroke={sc + "55"} strokeWidth={0.5} />
          <text x={x} y={-50} textAnchor="middle" fill={sc} fontSize={8} fontWeight="700" fontFamily="monospace" letterSpacing="0.04em">
            {signal.failure_type.replace(/_/g, " ")}
          </text>
        </g>
      )}
      <circle cx={x} cy={0} r={hasSig ? 10 : 7}
        fill={hasSig ? sc + "22" : meta.color + "18"}
        stroke={hasSig ? sc : isHovered ? meta.color : meta.color + "77"}
        strokeWidth={hasSig ? 2 : 1}
        style={{ cursor: hasSig ? "pointer" : "default" }}
        onClick={hasSig ? onClick : undefined}
      />
      {hasSig && <circle cx={x} cy={0} r={14} fill="none" stroke={sc} strokeWidth={0.5} opacity={0.3} strokeDasharray="3,3" />}
      <text x={x} y={4} textAnchor="middle" fill={hasSig ? sc : meta.color}
        fontSize={hasSig ? 9 : 7} fontFamily="monospace" style={{ pointerEvents: "none" }}>
        {hasSig ? "!" : icon}
      </text>
      <text x={x} y={22} textAnchor="middle" fill={isHovered ? "#6b7280" : "#1f2937"} fontSize={8} fontFamily="monospace">
        {event.step_index}
      </text>
      <text x={x} y={34} textAnchor="middle" fill={isHovered ? meta.color : "#1f2937"} fontSize={8} fontFamily="monospace">
        {meta.label}{tn ? " " + tn : ""}
      </text>
    </g>
  );
}

// ── Run selector ───────────────────────────────────────────────────────────────
function RunSelector({ selectedId, onSelect }) {
  var exitColor = { completed: "#22c55e", error: "#ff3b3b", stalled: "#f5c518" };
  return (
    <div style={{ display: "flex", gap: 6, marginBottom: 24, flexWrap: "wrap" }}>
      {MOCK_RUNS.map(function(r) {
        var detail = MOCK_DETAIL[r.run_id];
        var isSel = selectedId === r.run_id;
        var worstSig = null;
        if (detail.signals.length) {
          var order = { CRITICAL: 3, HIGH: 2, MEDIUM: 1, LOW: 0 };
          worstSig = detail.signals.reduce(function(a, b) { return order[a.severity] >= order[b.severity] ? a : b; });
        }
        return (
          <button key={r.run_id} onClick={function() { onSelect(r.run_id); }} style={{
            padding: "8px 14px", borderRadius: 5, cursor: "pointer",
            background: isSel ? "rgba(249,115,22,0.1)" : "rgba(255,255,255,0.02)",
            border: "1px solid " + (isSel ? "rgba(249,115,22,0.4)" : "rgba(255,255,255,0.07)"),
            textAlign: "left",
          }}>
            <div style={{ fontFamily: "monospace", fontSize: 10, color: isSel ? "#f97316" : "#4b5563", marginBottom: 3 }}>
              {r.run_id}
            </div>
            <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <span style={{ fontSize: 10, color: exitColor[detail.exit_reason] || "#6b7280", fontWeight: 600, textTransform: "uppercase" }}>
                {detail.exit_reason}
              </span>
              {worstSig
                ? <span style={{ fontSize: 9, color: SEVERITY_COLOR[worstSig.severity], fontWeight: 700 }}>&#9650; {r.label}</span>
                : <span style={{ fontSize: 9, color: "#22c55e" }}>&#10003; clean</span>
              }
            </div>
          </button>
        );
      })}
    </div>
  );
}

// ── Main timeline ──────────────────────────────────────────────────────────────
function Timeline({ run }) {
  var [activeSignal, setActiveSignal] = useState(null);
  var [hoveredStep, setHoveredStep] = useState(null);

  var events = run.events;
  var signals = run.signals;
  var signalByStep = {};
  signals.forEach(function(s) { signalByStep[s.step_index] = s; });
  var stepDurations = computeStepDurations(events);

  var STEP_W = 72;
  var LEFT_PAD = 60;
  var svgWidth = LEFT_PAD + events.length * STEP_W + 40;
  var TRACK_Y = 72;
  var svgHeight = 130;

  var totalDuration = run.completed_at - run.started_at;
  var slowestMs = 0, slowestStep = null;
  Object.keys(stepDurations).forEach(function(k) {
    if (stepDurations[k] > slowestMs) { slowestMs = stepDurations[k]; slowestStep = k; }
  });

  return (
    <div>
      {/* Metadata row */}
      <div style={{
        display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 0, marginBottom: 20,
        background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
        borderRadius: 6, overflow: "hidden",
      }}>
        {[
          { label: "Run ID",         value: run.run_id },
          { label: "Agent",          value: run.agent_id },
          { label: "Version",        value: run.agent_version.slice(0, 8) },
          { label: "Total duration", value: totalDuration.toFixed(1) + "s" },
          { label: "Slowest step",
            value: slowestStep !== null ? "step " + slowestStep + " \u00b7 " + fmtDuration(slowestMs) : "\u2014",
            color: slowestMs > 15000 ? "#ff7a00" : slowestMs > 2000 ? "#f5c518" : "#6b7280" },
          { label: "Exit",
            value: run.exit_reason,
            color: ({ completed: "#22c55e", error: "#ff3b3b", stalled: "#f5c518" })[run.exit_reason] },
        ].map(function(m, i) {
          return (
            <div key={i} style={{ padding: "12px 16px", borderRight: i < 5 ? "1px solid rgba(255,255,255,0.05)" : "none" }}>
              <div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: "0.1em", color: "#374151", marginBottom: 4 }}>{m.label}</div>
              <div style={{ fontSize: 12, fontWeight: 600, color: m.color || "#9ba3af", fontFamily: "monospace" }}>{m.value}</div>
            </div>
          );
        })}
      </div>

      {/* Signal badges */}
      {signals.length > 0 && (
        <div style={{ display: "flex", gap: 6, marginBottom: 16, flexWrap: "wrap" }}>
          {signals.map(function(s) {
            var sc = SEVERITY_COLOR[s.severity];
            return (
              <button key={s.id} onClick={function() { setActiveSignal(s); }} style={{
                display: "flex", alignItems: "center", gap: 6, padding: "5px 12px",
                borderRadius: 4, cursor: "pointer",
                background: sc + "12", border: "1px solid " + sc + "44",
              }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: sc }} />
                <span style={{ fontSize: 11, fontWeight: 700, color: sc, letterSpacing: "0.06em" }}>{s.severity}</span>
                <span style={{ fontSize: 11, color: "#6b7280" }}>step {s.step_index} &mdash; {s.failure_type.replace(/_/g, " ")}</span>
              </button>
            );
          })}
        </div>
      )}

      {/* Timeline + duration strip */}
      <div style={{ background: "rgba(0,0,0,0.25)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 8, padding: "20px 0 14px", overflowX: "auto" }}>
        <svg width={svgWidth} height={svgHeight} style={{ display: "block", overflow: "visible" }}>
          {/* Track */}
          <line x1={LEFT_PAD} y1={TRACK_Y} x2={LEFT_PAD + (events.length - 1) * STEP_W} y2={TRACK_Y}
            stroke="rgba(255,255,255,0.07)" strokeWidth={1.5} />
          {/* Connectors colored by gap */}
          {events.map(function(ev, i) {
            if (i === 0) return null;
            var x1 = LEFT_PAD + (i - 1) * STEP_W;
            var x2 = LEFT_PAD + i * STEP_W;
            var gap = (ev.timestamp - events[i - 1].timestamp) * 1000;
            var op = Math.min(1, gap / 20000);
            return <line key={i} x1={x1} y1={TRACK_Y} x2={x2} y2={TRACK_Y}
              stroke={"rgba(249,115,22," + (0.08 + op * 0.55) + ")"} strokeWidth={1 + op * 2.5} />;
          })}
          {/* Lane label */}
          <text x={LEFT_PAD - 8} y={TRACK_Y + 4} textAnchor="end" fill="#1f2937" fontSize={9} fontFamily="monospace">EVT</text>
          {/* Nodes */}
          {events.map(function(ev, i) {
            var x = LEFT_PAD + i * STEP_W;
            var sig = signalByStep[ev.step_index];
            return (
              <g key={i} transform={"translate(0," + TRACK_Y + ")"}
                onMouseEnter={function() { setHoveredStep(i); }}
                onMouseLeave={function() { setHoveredStep(null); }}>
                <EventNode event={ev} signal={sig} x={x} isHovered={hoveredStep === i}
                  onClick={function() { if (sig) setActiveSignal(sig); }} />
              </g>
            );
          })}
        </svg>

        {/* Duration strip */}
        <div style={{ borderTop: "1px solid rgba(255,255,255,0.04)", marginTop: 6, paddingTop: 10 }}>
          <DurationStrip
            events={events}
            stepDurations={stepDurations}
            STEP_W={STEP_W}
            LEFT_PAD={LEFT_PAD}
            signalByStep={signalByStep}
            onSignalClick={setActiveSignal}
          />
        </div>
      </div>

      {/* Hover detail */}
      {hoveredStep !== null && (function() {
        var ev = events[hoveredStep];
        var meta = EVENT_META[ev.event_type] || {};
        var dur = stepDurations[ev.step_index];
        return (
          <div style={{
            marginTop: 8, padding: "10px 16px",
            background: "rgba(255,255,255,0.02)",
            border: "1px solid rgba(255,255,255,0.06)", borderRadius: 5,
            display: "flex", gap: 24, alignItems: "flex-start", flexWrap: "wrap",
          }}>
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
                <div style={{ fontSize: 9, color: "#374151", textTransform: "uppercase", marginBottom: 3 }}>Duration to next</div>
                <div style={{ fontFamily: "monospace", fontSize: 12, color: durationColor(dur), fontWeight: dur > 10000 ? 700 : 400 }}>
                  {fmtDuration(dur)}
                </div>
              </div>
            )}
            {Object.entries(ev.payload).slice(0, 4).map(function([k, v]) {
              return (
                <div key={k}>
                  <div style={{ fontSize: 9, color: "#374151", textTransform: "uppercase", marginBottom: 3 }}>{k.replace(/_/g, " ")}</div>
                  <div style={{ fontFamily: "monospace", fontSize: 12, color: "#6b7280" }}>
                    {Array.isArray(v) ? v.join(", ") : String(v)}
                  </div>
                </div>
              );
            })}
          </div>
        );
      })()}

      {activeSignal && <SignalPopup signal={activeSignal} onClose={function() { setActiveSignal(null); }} />}
    </div>
  );
}

// ── App ────────────────────────────────────────────────────────────────────────
function TimelineApp() {
  var [selectedRun, setSelectedRun] = useState("run-c1d2e3f4");
  var run = MOCK_DETAIL[selectedRun];

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

      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 32 }}>
        <div style={{ width: 28, height: 28, borderRadius: 5, background: "linear-gradient(135deg,#f97316,#dc2626)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13, fontWeight: 900, color: "#fff" }}>D</div>
        <div>
          <div style={{ fontSize: 15, fontWeight: 500, color: "#e8eaf0" }}>Dunetrace &mdash; Run Timeline</div>
          <div style={{ fontSize: 10, color: "#374151", letterSpacing: "0.08em" }}>research-agent-v2 &middot; click signal badges or nodes for explanation + fix</div>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: 16, alignItems: "center" }}>
          {[["#22c55e","Run lifecycle"],["#6366f1","LLM calls"],["#f97316","Tool calls"],["#ff7a00","Signal (click)"]].map(function(l) {
            return (
              <div key={l[1]} style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: l[0] }} />
                <span style={{ fontSize: 10, color: "#374151" }}>{l[1]}</span>
              </div>
            );
          })}
        </div>
      </div>

      <RunSelector selectedId={selectedRun} onSelect={setSelectedRun} />
      <Timeline run={run} key={selectedRun} />

      <div style={{ marginTop: 14, fontSize: 10, color: "#1f2937" }}>
        Hover any node to inspect payload &middot; Duration bars show time to next event (green &lt;2s &middot; yellow 2-10s &middot; orange 10-30s &middot; red &gt;30s) &middot; Click colored bars or badges to see full explanation
      </div>
    </div>
  );
}
