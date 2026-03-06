const { useState, useEffect, useRef } = React;

// ── API ────────────────────────────────────────────────────────────────────────
const API_URL = "http://localhost:8002";
const AUTH    = { "Authorization": "Bearer dt_dev_test" };

async function apiFetch(path) {
  const r = await fetch(`${API_URL}${path}`, { headers: AUTH });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

// ── Constants ──────────────────────────────────────────────────────────────────
const EVENT_META = {
  "run.started":    { color: "#22c55e", label: "Start"   },
  "run.completed":  { color: "#22c55e", label: "Done"    },
  "run.errored":    { color: "#ff3b3b", label: "Error"   },
  "llm.called":     { color: "#818cf8", label: "LLM"     },
  "llm.responded":  { color: "#6366f1", label: "LLM\u2193" },
  "tool.called":    { color: "#f97316", label: "Tool"    },
  "tool.responded": { color: "#fb923c", label: "Tool\u2193" },
};

const SEVERITY_COLOR = {
  CRITICAL: "#ff3b3b", HIGH: "#ff7a00", MEDIUM: "#f5c518", LOW: "#22c55e",
};

const SEV_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"];

// ── Helpers ────────────────────────────────────────────────────────────────────
function tokenColor(t) {
  return t < 1500 ? "#22c55e" : t < 4000 ? "#f5c518" : t < 10000 ? "#f97316" : "#ff3b3b";
}
function durationColor(ms) {
  return ms < 2000 ? "#22c55e" : ms < 10000 ? "#f5c518" : ms < 30000 ? "#f97316" : "#ff3b3b";
}
function fmtDuration(ms) {
  if (ms == null || ms < 0) return "—";
  return ms < 1000 ? ms + "ms" : (ms / 1000).toFixed(1) + "s";
}
function fmtK(n) {
  return n < 1000 ? String(n) : (n / 1000).toFixed(1) + "k";
}
function fmtAge(ts) {
  if (!ts) return "—";
  const s = Math.round(Date.now() / 1000 - ts);
  if (s < 60)   return s + "s ago";
  if (s < 3600) return Math.floor(s / 60) + "m ago";
  return Math.floor(s / 3600) + "h ago";
}
function fmtExit(reason) {
  if (!reason) return "?";
  // "run.completed" → "COMPLETED", "run.errored" → "ERRORED"
  return reason.replace(/^run\./, "").toUpperCase();
}
function exitColor(reason) {
  const r = (reason || "").toLowerCase();
  if (r.includes("complet") || r === "final_answer") return "#22c55e";
  if (r.includes("error"))  return "#ff3b3b";
  if (r === "stalled")      return "#f5c518";
  return "#6b7280";
}

function computeStepDurations(events) {
  const gaps = {};
  for (let i = 0; i < events.length - 1; i++) {
    const gap = Math.round((events[i + 1].timestamp - events[i].timestamp) * 1000);
    // Keep the FIRST gap per step_index. When llm.called and llm.responded share
    // the same step_index, the first gap (llm.called→llm.responded) is the LLM
    // inference time — the real cost. Later gaps at the same step (e.g.
    // llm.responded→run.completed ≈ 0ms) would overwrite it with near-zero.
    if (gap >= 0 && !(events[i].step_index in gaps)) gaps[events[i].step_index] = gap;
  }
  return gaps;
}

// Looks at llm.responded (accurate, has prompt_tokens from API) then llm.called as fallback.
function computeTokenSeries(events) {
  const byStep = {};
  events.forEach(e => {
    if ((e.event_type === "llm.called" || e.event_type === "llm.responded")
        && e.payload.prompt_tokens != null) {
      // llm.responded wins (tokens confirmed by API); llm.called is fallback
      if (!byStep[e.step_index] || e.event_type === "llm.responded") {
        byStep[e.step_index] = e;
      }
    }
  });
  const sorted = Object.values(byStep).sort((a, b) => a.step_index - b.step_index);
  if (!sorted.length) return [];

  return sorted.map((ev, i) => {
    const tokens   = ev.payload.prompt_tokens;
    const prevTok  = i > 0 ? sorted[i - 1].payload.prompt_tokens : null;
    const delta    = prevTok != null ? tokens - prevTok : null;
    const prevStep = i > 0 ? sorted[i - 1].step_index : -1;
    const between  = events.filter(e =>
      e.event_type === "tool.responded" &&
      e.step_index > prevStep && e.step_index < ev.step_index &&
      e.payload.output_length != null
    );
    let cause = null;
    if (between.length) {
      const big = between.reduce((a, b) => b.payload.output_length > a.payload.output_length ? b : a);
      const tc  = events.find(e => e.event_type === "tool.called" && e.step_index === big.step_index - 1);
      cause = { toolName: tc ? (tc.payload.tool_name || "tool") : "tool",
                outputLength: big.payload.output_length, toolCount: between.length };
    }
    return { stepIndex: ev.step_index, tokens, delta, cause };
  });
}

// ── Section wrapper ────────────────────────────────────────────────────────────
function Section({ title, badge, children, id }) {
  return (
    <div id={id} style={{ marginBottom: 32 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
        <div style={{ width: 3, height: 14, background: "#f97316", borderRadius: 2 }} />
        <span style={{ fontSize: 11, fontWeight: 700, color: "#9ba3af", textTransform: "uppercase", letterSpacing: "0.1em" }}>
          {title}
        </span>
        {badge != null && (
          <span style={{ fontSize: 10, color: "#6b7280", background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 10, padding: "1px 8px" }}>
            {badge}
          </span>
        )}
      </div>
      {children}
    </div>
  );
}

// ── Signal popup ───────────────────────────────────────────────────────────────
function SignalPopup({ signal, onClose }) {
  const sc = SEVERITY_COLOR[signal.severity];
  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.78)", backdropFilter: "blur(4px)" }} onClick={onClose}>
      <div onClick={e => e.stopPropagation()} style={{ width: 600, maxHeight: "90vh", overflowY: "auto", background: "#0f1117", border: `1px solid ${sc}44`, borderLeft: `4px solid ${sc}`, borderRadius: 8, padding: "24px 28px", boxShadow: `0 0 60px ${sc}22` }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 20 }}>
          <div>
            <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 8 }}>
              <span style={{ fontSize: 10, fontWeight: 800, letterSpacing: "0.1em", color: sc, background: `${sc}18`, border: `1px solid ${sc}44`, padding: "2px 8px", borderRadius: 3 }}>{signal.severity}</span>
              <span style={{ fontSize: 10, color: "#6b7280", fontFamily: "monospace" }}>step {signal.step_index} · {Math.round(signal.confidence * 100)}% conf</span>
            </div>
            <div style={{ fontSize: 14, color: "#e8eaf0", lineHeight: 1.4, fontWeight: 500 }}>{signal.title}</div>
          </div>
          <button onClick={onClose} style={{ background: "none", border: "none", color: "#6b7280", cursor: "pointer", fontSize: 18, padding: 0, marginLeft: 16 }}>✕</button>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
          {[["What happened", signal.what], ["Why it matters", signal.why_it_matters]].map(([lbl, txt]) => (
            <div key={lbl}>
              <div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: "0.1em", color: "#4b5563", marginBottom: 6 }}>{lbl}</div>
              <div style={{ fontSize: 12, color: "#9ba3af", lineHeight: 1.7 }}>{txt}</div>
            </div>
          ))}
        </div>
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: "0.1em", color: "#4b5563", marginBottom: 6 }}>Evidence</div>
          <div style={{ fontFamily: "monospace", fontSize: 11, color: "#9ba3af", background: "rgba(0,0,0,0.4)", padding: "10px 14px", borderRadius: 4, border: "1px solid rgba(255,255,255,0.08)" }}>{signal.evidence_summary}</div>
        </div>
        {signal.suggested_fixes && signal.suggested_fixes[0] && (
          <div>
            <div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: "0.1em", color: "#4b5563", marginBottom: 6 }}>Fix — {signal.suggested_fixes[0].description}</div>
            <pre style={{ fontFamily: "monospace", fontSize: 11, color: "#86efac", background: "rgba(0,0,0,0.5)", padding: "12px 14px", borderRadius: 4, border: "1px solid rgba(34,197,94,0.15)", margin: 0, overflowX: "auto", lineHeight: 1.6 }}>{signal.suggested_fixes[0].code}</pre>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Sparkline ──────────────────────────────────────────────────────────────────
function Sparkline({ data, color }) {
  // data: 7 daily signal counts oldest→newest. Empty or all-zero → flat baseline.
  const W = 88, H = 28, PAD_X = 3, PAD_Y = 4;
  const max = Math.max(...data, 1);
  const pts = data.map((v, i) => {
    const x = PAD_X + (i / (data.length - 1)) * (W - 2 * PAD_X);
    const y = H - PAD_Y - (v / max) * (H - 2 * PAD_Y);
    return [x, y];
  });
  const polyline = pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const allZero  = data.every(v => v === 0);
  const [lx, ly] = pts[pts.length - 1];

  return (
    <svg width={W} height={H} style={{ display: "block", overflow: "visible" }}>
      {/* Baseline */}
      <line x1={PAD_X} y1={H - PAD_Y} x2={W - PAD_X} y2={H - PAD_Y}
        stroke="rgba(255,255,255,0.07)" strokeWidth={1} />
      {/* Day tick marks */}
      {pts.map(([x], i) => (
        <line key={i} x1={x} y1={H - PAD_Y} x2={x} y2={H - PAD_Y + 2}
          stroke="rgba(255,255,255,0.12)" strokeWidth={1} />
      ))}
      {allZero ? (
        /* Flat green line when no signals in the window */
        <line x1={PAD_X} y1={H - PAD_Y} x2={W - PAD_X} y2={H - PAD_Y}
          stroke="#22c55e" strokeWidth={1.5} opacity={0.4} />
      ) : (
        <>
          {/* Filled area under the line */}
          <polygon
            points={`${PAD_X.toFixed(1)},${(H - PAD_Y).toFixed(1)} ${polyline} ${(W - PAD_X).toFixed(1)},${(H - PAD_Y).toFixed(1)}`}
            fill={`${color}18`}
          />
          {/* Line */}
          <polyline points={polyline} fill="none" stroke={color} strokeWidth={1.5} opacity={0.85} strokeLinejoin="round" />
          {/* Today's dot */}
          <circle cx={lx} cy={ly} r={2.5} fill={color} />
          {/* Today's value label if > 0 */}
          {data[data.length - 1] > 0 && (
            <text x={lx + 4} y={ly + 3.5} fill={color} fontSize={8} fontFamily="monospace" fontWeight="700">
              {data[data.length - 1]}
            </text>
          )}
        </>
      )}
    </svg>
  );
}

// ── Agent cards ────────────────────────────────────────────────────────────────
function AgentCards({ agents, selectedAgent, onSelect }) {
  if (!agents.length) return (
    <div style={{ color: "#6b7280", fontSize: 11 }}>
      No agents yet — run <code style={{ color: "#f97316" }}>python scripts/smoke_test.py</code>
    </div>
  );
  return (
    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
      {agents.map(a => {
        const isSel  = a.agent_id === selectedAgent;
        const worst  = a.critical_count > 0 ? "CRITICAL" : a.high_count > 0 ? "HIGH" : a.signal_count > 0 ? "MEDIUM" : null;
        const wc     = worst ? SEVERITY_COLOR[worst] : "#22c55e";
        const spark  = (a.sparkline && a.sparkline.length === 7) ? a.sparkline : [0,0,0,0,0,0,0];
        return (
          <button key={a.agent_id} onClick={() => onSelect(a.agent_id)} style={{
            padding: "12px 16px", borderRadius: 6, cursor: "pointer", textAlign: "left", minWidth: 210,
            background: isSel ? "rgba(249,115,22,0.06)" : "rgba(255,255,255,0.02)",
            border: `1px solid ${isSel ? "rgba(249,115,22,0.35)" : "rgba(255,255,255,0.07)"}`,
          }}>
            {/* Agent ID */}
            <div style={{ fontFamily: "monospace", fontSize: 11, color: isSel ? "#f97316" : "#9ba3af", fontWeight: 600, marginBottom: 8, wordBreak: "break-all" }}>
              {a.agent_id}
            </div>
            {/* Sparkline + stats side by side */}
            <div style={{ display: "flex", alignItems: "flex-end", gap: 12, marginBottom: 6 }}>
              <div>
                <Sparkline data={spark} color={wc} />
                <div style={{ fontSize: 8, color: "#4b5563", fontFamily: "monospace", marginTop: 2, textAlign: "center" }}>
                  7d signal rate
                </div>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span style={{ fontSize: 9, color: "#6b7280" }}>{a.run_count} run{a.run_count !== 1 ? "s" : ""}</span>
                {a.signal_count > 0 ? (
                  <span style={{ fontSize: 9, color: wc, fontWeight: 700 }}>▲ {a.signal_count} signal{a.signal_count !== 1 ? "s" : ""}</span>
                ) : (
                  <span style={{ fontSize: 9, color: "#22c55e" }}>✓ clean</span>
                )}
                {a.critical_count > 0 && <span style={{ fontSize: 9, color: "#ff3b3b", fontWeight: 700 }}>CRIT {a.critical_count}</span>}
                <span style={{ fontSize: 9, color: "#4b5563" }}>{fmtAge(a.last_seen)}</span>
              </div>
            </div>
            {/* Failure type dots — one per distinct type, count as label */}
            {a.failure_types && Object.keys(a.failure_types).length > 0 && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 2 }}>
                {Object.entries(a.failure_types).map(([ft, cnt]) => {
                  const ftColor = {
                    TOOL_LOOP: "#f97316", TOOL_THRASHING: "#fb923c",
                    TOOL_AVOIDANCE: "#f5c518", GOAL_ABANDONMENT: "#f5c518",
                    CONTEXT_BLOAT: "#818cf8", LLM_TRUNCATION_LOOP: "#a78bfa",
                    RAG_EMPTY_RETRIEVAL: "#22d3ee", SLOW_STEP: "#94a3b8",
                    PROMPT_INJECTION_SIGNAL: "#ff3b3b",
                  }[ft] || "#6b7280";
                  const short = ft.replace(/_/g, " ").replace(/TOOL /,"").replace(/LLM /,"").replace(/ SIGNAL/,"");
                  return (
                    <span key={ft} style={{
                      display: "inline-flex", alignItems: "center", gap: 3,
                      fontSize: 8, fontFamily: "monospace",
                      color: ftColor, background: `${ftColor}14`,
                      border: `1px solid ${ftColor}33`,
                      borderRadius: 3, padding: "1px 5px",
                    }}>
                      <span style={{ width: 5, height: 5, borderRadius: "50%", background: ftColor, display: "inline-block", flexShrink: 0 }} />
                      {short}{cnt > 1 ? ` ×${cnt}` : ""}
                    </span>
                  );
                })}
              </div>
            )}
          </button>
        );
      })}
    </div>
  );
}

// ── Agent sidebar (compact vertical list) ──────────────────────────────────────
function AgentSidebar({ agents, selectedAgent, onSelect }) {
  if (!agents.length) return (
    <div style={{ color: "#6b7280", fontSize: 10, lineHeight: 1.7 }}>
      No agents yet —<br />
      <code style={{ color: "#f97316", fontSize: 9 }}>python scripts/smoke_test.py</code>
    </div>
  );
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
      {agents.map(a => {
        const isSel = a.agent_id === selectedAgent;
        const worst = a.critical_count > 0 ? "CRITICAL" : a.high_count > 0 ? "HIGH" : a.signal_count > 0 ? "MEDIUM" : null;
        const wc    = worst ? SEVERITY_COLOR[worst] : "#22c55e";
        return (
          <button key={a.agent_id} onClick={() => onSelect(a.agent_id)} style={{
            padding: "9px 11px", borderRadius: 5, cursor: "pointer", textAlign: "left", width: "100%",
            background: isSel ? "rgba(249,115,22,0.06)" : "rgba(255,255,255,0.02)",
            border: `1px solid ${isSel ? "rgba(249,115,22,0.35)" : "rgba(255,255,255,0.07)"}`,
          }}>
            <div style={{ fontFamily: "monospace", fontSize: 9, color: isSel ? "#f97316" : "#9ba3af", fontWeight: 600, marginBottom: 4, wordBreak: "break-all", lineHeight: 1.4 }}>
              {a.agent_id}
            </div>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <span style={{ fontSize: 9, color: "#4b5563" }}>{a.run_count}r</span>
              {a.signal_count > 0
                ? <span style={{ fontSize: 9, color: wc, fontWeight: 700 }}>▲{a.signal_count}</span>
                : <span style={{ fontSize: 9, color: "#22c55e" }}>✓</span>}
              <span style={{ fontSize: 8, color: "#4b5563" }}>{fmtAge(a.last_seen)}</span>
            </div>
          </button>
        );
      })}
    </div>
  );
}

// ── Tabs ────────────────────────────────────────────────────────────────────────
function Tabs({ tabs, active, onChange }) {
  return (
    <div style={{ display: "flex", borderBottom: "1px solid rgba(255,255,255,0.08)", marginBottom: 22 }}>
      {tabs.map(t => {
        const on = t.id === active;
        return (
          <button key={t.id} onClick={() => onChange(t.id)} style={{
            padding: "9px 20px", border: "none", cursor: "pointer",
            fontFamily: "inherit", fontSize: 11, fontWeight: on ? 700 : 400,
            color: on ? "#e8eaf0" : "#6b7280",
            background: "none",
            borderBottom: `2px solid ${on ? "#f97316" : "transparent"}`,
            marginBottom: -1,
          }}>
            {t.label}
            {t.badge != null && (
              <span style={{
                marginLeft: 7, fontSize: 9,
                background: on ? "rgba(249,115,22,0.18)" : "rgba(255,255,255,0.06)",
                border: `1px solid ${on ? "rgba(249,115,22,0.4)" : "rgba(255,255,255,0.1)"}`,
                borderRadius: 8, padding: "1px 6px",
                color: on ? "#f97316" : "#6b7280",
              }}>{t.badge}</span>
            )}
          </button>
        );
      })}
    </div>
  );
}

// ── Signals list ───────────────────────────────────────────────────────────────
function SignalsList({ signals, onRunSelect, onInspect }) {
  if (!signals.length) return (
    <div style={{ padding: "14px 16px", background: "rgba(34,197,94,0.04)", border: "1px solid rgba(34,197,94,0.15)", borderRadius: 6, display: "flex", alignItems: "center", gap: 10 }}>
      <span style={{ color: "#22c55e", fontSize: 14 }}>✓</span>
      <span style={{ fontSize: 12, color: "#6b7280" }}>No failure signals detected for this agent.</span>
    </div>
  );

  // Group signals by run_id, preserving order of first appearance
  const groups = [];
  const seen = {};
  signals.forEach(s => {
    const key = s.run_id || "unknown";
    if (!seen[key]) { seen[key] = []; groups.push({ run_id: key, rows: seen[key] }); }
    seen[key].push(s);
  });

  const COLS = "90px 200px 60px 70px 80px 100px";

  return (
    <div style={{ background: "rgba(0,0,0,0.2)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 6, overflow: "hidden" }}>
      {/* Header */}
      <div style={{ display: "grid", gridTemplateColumns: COLS, gap: 0, padding: "8px 16px", borderBottom: "1px solid rgba(255,255,255,0.05)", background: "rgba(255,255,255,0.02)" }}>
        {["Severity", "Failure type", "Step", "Conf", "Detected", ""].map((h, i) => (
          <div key={i} style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: "0.08em", color: "#4b5563" }}>{h}</div>
        ))}
      </div>
      {/* Groups */}
      {groups.map((g, gi) => (
        <div key={g.run_id}>
          {/* Run ID separator — makes it obvious multiple signals share one run */}
          <div style={{
            display: "flex", alignItems: "center", gap: 10, padding: "6px 16px",
            borderTop: gi > 0 ? "1px solid rgba(255,255,255,0.07)" : "none",
            background: "rgba(255,255,255,0.015)",
          }}>
            <span style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: "0.08em", color: "#4b5563" }}>Run</span>
            <span style={{ fontFamily: "monospace", fontSize: 10, color: "#6b7280" }}>{g.run_id.slice(0, 18)}…</span>
            {g.rows.length > 1 && (
              <span style={{ fontSize: 9, color: "#6b7280" }}>· {g.rows.length} signals on this run</span>
            )}
            <button onClick={() => onRunSelect(g.run_id)} style={{
              marginLeft: "auto", fontSize: 9, color: "#9ba3af",
              background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 3, padding: "2px 7px", cursor: "pointer",
            }}>Timeline →</button>
          </div>
          {/* Signal rows within this run */}
          {g.rows.map((s, i) => {
            const sc = SEVERITY_COLOR[s.severity];
            return (
              <div key={s.id} style={{
                display: "grid", gridTemplateColumns: COLS,
                gap: 0, padding: "9px 16px", alignItems: "center",
                borderTop: "1px solid rgba(255,255,255,0.03)",
                background: "transparent", transition: "background 0.1s",
              }}
                onMouseEnter={e => e.currentTarget.style.background = "rgba(255,255,255,0.02)"}
                onMouseLeave={e => e.currentTarget.style.background = "transparent"}
              >
                <div>
                  <span style={{ fontSize: 10, fontWeight: 700, color: sc, background: `${sc}14`, border: `1px solid ${sc}44`, padding: "2px 7px", borderRadius: 3, letterSpacing: "0.06em" }}>
                    {s.severity}
                  </span>
                </div>
                <div style={{ fontFamily: "monospace", fontSize: 11, color: "#9ba3af" }}>
                  {s.failure_type.replace(/_/g, " ")}
                </div>
                <div style={{ fontFamily: "monospace", fontSize: 11, color: "#6b7280" }}>s{s.step_index}</div>
                <div style={{ fontSize: 11, color: sc }}>{Math.round(s.confidence * 100)}%</div>
                <div style={{ fontSize: 10, color: "#4b5563" }}>{fmtAge(s.detected_at)}</div>
                <div>
                  <button onClick={() => onInspect(s)} style={{
                    fontSize: 9, color: sc, background: `${sc}10`, border: `1px solid ${sc}33`,
                    borderRadius: 3, padding: "3px 8px", cursor: "pointer",
                  }}>Inspect</button>
                </div>
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

// ── Run selector ───────────────────────────────────────────────────────────────
function RunSelector({ runs, selectedId, onSelect }) {
  if (!runs.length) return (
    <div style={{ color: "#6b7280", fontSize: 11 }}>No runs found for this agent.</div>
  );
  return (
    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
      {runs.map(r => {
        const isSel = selectedId === r.run_id;
        const durMs = r.started_at && r.completed_at ? (r.completed_at - r.started_at) * 1000 : null;
        return (
          <button key={r.run_id} onClick={() => onSelect(r.run_id)} style={{
            padding: "8px 14px", borderRadius: 5, cursor: "pointer", minWidth: 160, textAlign: "left",
            background: isSel ? "rgba(249,115,22,0.07)" : "rgba(255,255,255,0.02)",
            border: `1px solid ${isSel ? "rgba(249,115,22,0.4)" : "rgba(255,255,255,0.07)"}`,
          }}>
            <div style={{ fontFamily: "monospace", fontSize: 10, color: isSel ? "#f97316" : "#6b7280", marginBottom: 5 }}>
              {r.run_id.slice(0, 16)}…
            </div>
            <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
              <span style={{ fontSize: 10, color: exitColor(r.exit_reason), fontWeight: 700 }}>{fmtExit(r.exit_reason)}</span>
              {durMs != null && <span style={{ fontSize: 9, color: durationColor(durMs) }}>{fmtDuration(durMs)}</span>}
              {r.signal_count > 0
                ? <span style={{ fontSize: 9, color: "#ff7a00", fontWeight: 700 }}>▲ {r.signal_count}</span>
                : <span style={{ fontSize: 9, color: "#22c55e" }}>✓</span>}
              <span style={{ fontSize: 9, color: "#4b5563" }}>{r.step_count} steps</span>
            </div>
          </button>
        );
      })}
    </div>
  );
}

// ── Duration strip ─────────────────────────────────────────────────────────────
function DurationStrip({ events, stepDurations, STEP_W, LEFT_PAD, signalByStep, onSignalClick }) {
  const MAX_BAR_H  = 30;
  const allMs      = Object.values(stepDurations);
  const maxMs      = allMs.length ? Math.max(...allMs) : 1;
  const displayCap = Math.max(maxMs, 5000);

  return (
    <div>
      <div style={{ paddingLeft: LEFT_PAD, marginBottom: 6, display: "flex", alignItems: "center", gap: 12, fontSize: 10, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.08em" }}>
        <span style={{ fontWeight: 600 }}>Step duration</span>
        <span style={{ display: "flex", gap: 10 }}>
          {[["#22c55e","< 2s"],["#f5c518","2–10s"],["#f97316","10–30s"],["#ff3b3b","> 30s"]].map(([c, l]) => (
            <span key={l} style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 9 }}>
              <span style={{ width: 7, height: 7, borderRadius: 1, background: c, display: "inline-block" }} />{l}
            </span>
          ))}
        </span>
      </div>
      <svg width={LEFT_PAD + events.length * STEP_W + 40} height={MAX_BAR_H + 22} style={{ display: "block", overflow: "visible" }}>
        <line x1={LEFT_PAD - 8} y1={MAX_BAR_H} x2={LEFT_PAD + events.length * STEP_W + 20} y2={MAX_BAR_H} stroke="rgba(255,255,255,0.06)" strokeWidth={1} />
        {events.map((ev, i) => {
          const dms    = stepDurations[ev.step_index];
          if (dms === undefined) return null;
          const sig    = signalByStep[ev.step_index];
          const isSlow = sig && sig.failure_type === "SLOW_STEP";
          const x      = LEFT_PAD + i * STEP_W;
          const barH   = Math.max(2, (Math.min(dms, displayCap) / displayCap) * MAX_BAR_H);
          const color  = isSlow ? SEVERITY_COLOR[sig.severity] : durationColor(dms);
          const barW   = Math.min(STEP_W - 12, 28);
          return (
            <g key={i} style={{ cursor: isSlow ? "pointer" : "default" }} onClick={isSlow ? () => onSignalClick(sig) : undefined}>
              <rect x={x - barW/2} y={MAX_BAR_H - barH} width={barW} height={barH} rx={2} fill={color} opacity={isSlow ? 0.9 : 0.5} />
              {isSlow && <rect x={x-barW/2-1} y={MAX_BAR_H-barH-1} width={barW+2} height={barH+2} rx={3} fill="none" stroke={color} strokeWidth={1.5} opacity={0.7} />}
              <text x={x} y={MAX_BAR_H + 16} textAnchor="middle" fill={dms >= 2000 ? color : "#6b7280"} fontSize={8.5} fontFamily="monospace" fontWeight={dms >= 10000 ? "700" : "400"}>
                {fmtDuration(dms)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ── Token strip ────────────────────────────────────────────────────────────────
function TokenStrip({ events, tokenSeries, STEP_W, LEFT_PAD, signalByStep, onSignalClick }) {
  if (!tokenSeries.length) return (
    <div style={{ paddingLeft: LEFT_PAD, padding: "10px 0 6px", fontSize: 10, color: "#6b7280", fontFamily: "monospace" }}>
      No <code style={{ color: "#f97316" }}>prompt_tokens</code> in events.
      {" "}Requires <code style={{ color: "#f97316" }}>on_chat_model_start</code> in callback (SDK ≥ latest).
    </div>
  );

  const MAX_BAR_H = 54;
  const maxTok    = Math.max(...tokenSeries.map(s => s.tokens), 1);
  const stepX     = {};
  events.forEach((ev, i) => { stepX[ev.step_index] = LEFT_PAD + i * STEP_W; });
  const svgW = LEFT_PAD + events.length * STEP_W + 40;

  return (
    <div>
      <div style={{ paddingLeft: LEFT_PAD, marginBottom: 6, display: "flex", alignItems: "center", gap: 12, fontSize: 10, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.08em" }}>
        <span style={{ fontWeight: 600 }}>Prompt tokens / LLM call</span>
        {tokenSeries.length === 1 && (
          <span style={{ fontSize: 9, color: "#4b5563", textTransform: "none", letterSpacing: 0, fontFamily: "monospace" }}>
            1 LLM call · no growth to show
          </span>
        )}
        <span style={{ display: "flex", gap: 10 }}>
          {[["#22c55e","< 1.5k"],["#f5c518","1.5–4k"],["#f97316","4–10k"],["#ff3b3b","> 10k"]].map(([c, l]) => (
            <span key={l} style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 9 }}>
              <span style={{ width: 7, height: 7, borderRadius: 1, background: c, display: "inline-block" }} />{l}
            </span>
          ))}
        </span>
      </div>
      <svg width={svgW} height={MAX_BAR_H + 52} style={{ display: "block", overflow: "visible" }}>
        <line x1={LEFT_PAD - 8} y1={MAX_BAR_H} x2={LEFT_PAD + events.length * STEP_W + 20} y2={MAX_BAR_H} stroke="rgba(255,255,255,0.06)" strokeWidth={1} />
        {/* Trend line */}
        {tokenSeries.map((s, i) => {
          if (!i) return null;
          const prev = tokenSeries[i-1];
          const x1 = stepX[prev.stepIndex], x2 = stepX[s.stepIndex];
          if (x1 == null || x2 == null) return null;
          const y1 = MAX_BAR_H - (prev.tokens/maxTok)*MAX_BAR_H;
          const y2 = MAX_BAR_H - (s.tokens/maxTok)*MAX_BAR_H;
          return <line key={"tl"+i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="rgba(99,102,241,0.35)" strokeWidth={1.5} strokeDasharray="3,3" />;
        })}
        {/* Delta badges */}
        {tokenSeries.map((s, i) => {
          if (!i || s.delta == null) return null;
          const prev = tokenSeries[i-1];
          const x1 = stepX[prev.stepIndex], x2 = stepX[s.stepIndex];
          if (x1 == null || x2 == null) return null;
          const midX  = (x1+x2)/2;
          const color = s.delta > 1500 ? "#ff7a00" : s.delta > 500 ? "#f5c518" : "#22c55e";
          return (
            <g key={"dt"+i}>
              <rect x={midX-30} y={MAX_BAR_H+4} width={60} height={16} rx={3} fill={`${color}1a`} stroke={`${color}55`} strokeWidth={0.8} />
              <text x={midX} y={MAX_BAR_H+16} textAnchor="middle" fill={color} fontSize={9} fontFamily="monospace" fontWeight="600">+{fmtK(s.delta)} tok</text>
              {s.cause && (
                <text x={midX} y={MAX_BAR_H+32} textAnchor="middle" fill="#6b7280" fontSize={8} fontFamily="monospace">
                  ← {s.cause.toolName} {fmtK(s.cause.outputLength)}ch{s.cause.toolCount>1?` +${s.cause.toolCount-1}`:""}
                </text>
              )}
            </g>
          );
        })}
        {/* Bars */}
        {tokenSeries.map((s, i) => {
          const x = stepX[s.stepIndex];
          if (x == null) return null;
          const sig  = signalByStep[s.stepIndex];
          const isBloat = sig && sig.failure_type === "CONTEXT_BLOAT";
          const barH = Math.max(4, (s.tokens/maxTok)*MAX_BAR_H);
          const color = isBloat ? SEVERITY_COLOR[sig.severity] : tokenColor(s.tokens);
          const barW = 26, barY = MAX_BAR_H - barH;
          return (
            <g key={"bar"+i} style={{ cursor: isBloat?"pointer":"default" }} onClick={isBloat?()=>onSignalClick(sig):undefined}>
              {isBloat && <rect x={x-barW/2-3} y={barY-3} width={barW+6} height={barH+6} rx={4} fill="none" stroke={color} strokeWidth={1.5} opacity={0.5} />}
              <rect x={x-barW/2} y={barY} width={barW} height={barH} rx={2} fill={color} opacity={isBloat?0.9:0.65} />
              <text x={x} y={barY-5} textAnchor="middle" fill={color} fontSize={9} fontFamily="monospace" fontWeight="600">{fmtK(s.tokens)}</text>
              <text x={x} y={MAX_BAR_H+46} textAnchor="middle" fill="#6b7280" fontSize={8.5} fontFamily="monospace">s{s.stepIndex}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ── Event node ─────────────────────────────────────────────────────────────────
function EventNode({ event, signal, x, isHovered, onClick }) {
  const meta   = EVENT_META[event.event_type] || { color: "#6b7280", label: "?" };
  const hasSig = !!signal;
  const sc     = hasSig ? SEVERITY_COLOR[signal.severity] : null;
  const tn     = event.event_type === "tool.called" && event.payload?.tool_name ? event.payload.tool_name : "";
  const icons  = { "run.started":"\u25B6","run.completed":"\u25A0","run.errored":"\u2715","llm.called":"\u25C6","llm.responded":"\u25C7","tool.called":"\u2B21","tool.responded":"\u2B21" };
  const icon   = icons[event.event_type] || "\u2022";

  return (
    <g>
      {hasSig && (
        <g style={{ cursor:"pointer" }} onClick={onClick}>
          <line x1={x} y1={-8} x2={x} y2={-44} stroke={sc} strokeWidth={1.5} strokeDasharray="2,2" opacity={0.75} />
          <polygon points={`${x},${-8} ${x-5},${-20} ${x+5},${-20}`} fill={sc} opacity={0.9} />
          <rect x={x-46} y={-68} width={92} height={18} rx={3} fill={`${sc}20`} stroke={`${sc}55`} strokeWidth={0.6} />
          <text x={x} y={-54} textAnchor="middle" fill={sc} fontSize={8} fontWeight="700" fontFamily="monospace" letterSpacing="0.04em">
            {signal.failure_type.replace(/_/g," ")}
          </text>
        </g>
      )}
      <circle cx={x} cy={0} r={hasSig?11:7}
        fill={hasSig?`${sc}22`:`${meta.color}18`}
        stroke={hasSig?sc:isHovered?meta.color:`${meta.color}99`}
        strokeWidth={hasSig?2:1}
        style={{ cursor: hasSig?"pointer":"default" }}
        onClick={hasSig?onClick:undefined}
      />
      {hasSig && <circle cx={x} cy={0} r={15} fill="none" stroke={sc} strokeWidth={0.5} opacity={0.3} strokeDasharray="3,3" />}
      <text x={x} y={4} textAnchor="middle" fill={hasSig?sc:meta.color} fontSize={hasSig?9:8} fontFamily="monospace" style={{ pointerEvents:"none" }}>
        {hasSig?"!":icon}
      </text>
      <text x={x} y={23} textAnchor="middle" fill={isHovered?"#c0c6d0":"#9ba3af"} fontSize={9} fontFamily="monospace" fontWeight="500">{event.step_index}</text>
      <text x={x} y={35} textAnchor="middle" fill={hasSig?sc:isHovered?meta.color:"#9ba3af"} fontSize={8} fontFamily="monospace">
        {meta.label}{tn?" "+tn:""}
      </text>
    </g>
  );
}

// ── Timeline ───────────────────────────────────────────────────────────────────
function Timeline({ run }) {
  const [activeSignal, setActiveSignal] = useState(null);
  const [hoveredStep,  setHoveredStep]  = useState(null);

  const events        = run.events;
  const signals       = run.signals;
  const signalByStep  = Object.fromEntries(signals.map(s => [s.step_index, s]));
  const stepDurations = computeStepDurations(events);
  const tokenSeries   = computeTokenSeries(events);

  const STEP_W  = 76;
  const LEFT_PAD = 64;
  const TRACK_Y  = 84;
  const svgH     = 148;
  const svgW     = LEFT_PAD + events.length * STEP_W + 40;

  const durMs      = run.started_at && run.completed_at ? (run.completed_at - run.started_at) * 1000 : null;
  const lastTokens  = tokenSeries.length ? tokenSeries[tokenSeries.length-1].tokens : null;
  const firstTokens = tokenSeries.length ? tokenSeries[0].tokens : null;
  const tokenGrowth = firstTokens && lastTokens && tokenSeries.length > 1 ? (lastTokens/firstTokens).toFixed(1) : null;
  const llmEvent    = events.find(e => e.event_type === "llm.called");
  const model       = llmEvent?.payload?.model || null;
  const llmCount    = events.filter(e => e.event_type === "llm.called").length;
  const toolCount   = events.filter(e => e.event_type === "tool.called").length;

  const metaRow = [
    { label:"Run ID",    value: run.run_id.slice(0,18)+"…",         mono:true },
    model && { label:"Model", value: model,                          mono:true },
    { label:"Duration",  value: durMs!=null ? fmtDuration(durMs):"—", color: durMs!=null?durationColor(durMs):"#6b7280" },
    { label:"Steps",     value: String(run.step_count) },
    { label:"LLM calls", value: String(llmCount),  color:"#818cf8" },
    { label:"Tool calls",value: String(toolCount), color:"#f97316" },
    { label:"Tokens",    value: lastTokens!=null ? fmtK(lastTokens)+(tokenGrowth?` (${tokenGrowth}\u00d7)`:"") : "—",
      color: lastTokens!=null ? tokenColor(lastTokens) : "#4b5563" },
    { label:"Signals",   value: signals.length>0 ? `${signals.length} signal${signals.length!==1?"s":""}` : "clean",
      color: signals.length>0 ? SEVERITY_COLOR[signals.reduce((a,b)=>SEV_ORDER.indexOf(a.severity)<=SEV_ORDER.indexOf(b.severity)?a:b).severity] : "#22c55e" },
    { label:"Exit",      value: fmtExit(run.exit_reason), color: exitColor(run.exit_reason) },
  ].filter(Boolean);

  return (
    <div>
      {/* Metadata bar */}
      <div style={{ display:"flex", flexWrap:"wrap", marginBottom:14, background:"rgba(255,255,255,0.02)", border:"1px solid rgba(255,255,255,0.07)", borderRadius:6, overflow:"hidden" }}>
        {metaRow.map((m,i) => (
          <div key={i} style={{ padding:"10px 16px", borderRight: i<metaRow.length-1?"1px solid rgba(255,255,255,0.05)":"none", flexShrink:0 }}>
            <div style={{ fontSize:9, textTransform:"uppercase", letterSpacing:"0.08em", color:"#4b5563", marginBottom:4 }}>{m.label}</div>
            <div style={{ fontSize:12, fontWeight:600, color:m.color||"#9ba3af", fontFamily:m.mono?"monospace":"inherit" }}>{m.value}</div>
          </div>
        ))}
      </div>

      {/* Signal badges */}
      {signals.length>0 && (
        <div style={{ display:"flex", gap:6, marginBottom:14, flexWrap:"wrap" }}>
          {signals.map(s => {
            const sc = SEVERITY_COLOR[s.severity];
            return (
              <button key={s.id} onClick={() => setActiveSignal(s)} style={{ display:"flex", alignItems:"center", gap:8, padding:"6px 14px", borderRadius:4, cursor:"pointer", background:`${sc}0e`, border:`1px solid ${sc}44` }}>
                <span style={{ width:6, height:6, borderRadius:"50%", background:sc }} />
                <span style={{ fontSize:11, fontWeight:700, color:sc, letterSpacing:"0.05em" }}>{s.severity}</span>
                <span style={{ fontSize:11, color:"#9ba3af" }}>step {s.step_index} — {s.failure_type.replace(/_/g," ")}</span>
                <span style={{ fontSize:10, color:"#6b7280" }}>{Math.round(s.confidence*100)}% conf</span>
              </button>
            );
          })}
        </div>
      )}

      {/* Track + strips */}
      <div style={{ background:"rgba(0,0,0,0.28)", border:"1px solid rgba(255,255,255,0.07)", borderRadius:8, padding:"16px 0 10px", overflowX:"auto" }}>
        <svg width={svgW} height={svgH} style={{ display:"block", overflow:"visible" }}>
          <line x1={LEFT_PAD} y1={TRACK_Y} x2={LEFT_PAD+(events.length-1)*STEP_W} y2={TRACK_Y} stroke="rgba(255,255,255,0.08)" strokeWidth={1.5} />
          {events.map((ev,i) => {
            if (!i) return null;
            const x1=LEFT_PAD+(i-1)*STEP_W, x2=LEFT_PAD+i*STEP_W;
            const gap=(ev.timestamp-events[i-1].timestamp)*1000;
            const op=Math.min(1,gap/20000);
            return <line key={i} x1={x1} y1={TRACK_Y} x2={x2} y2={TRACK_Y} stroke={`rgba(249,115,22,${0.07+op*0.55})`} strokeWidth={1+op*2.5} />;
          })}
          <text x={LEFT_PAD-10} y={TRACK_Y+4} textAnchor="end" fill="#4b5563" fontSize={9} fontFamily="monospace">EVT</text>
          {(() => {
            const _spikeDone = new Set();
            return events.map((ev, i) => {
              const sig = signalByStep[ev.step_index];
              // Only the first event at a given step_index renders the spike.
              // Without this guard, run.completed and llm.responded sharing
              // step_index=N both render the spike, overlapping the label.
              const showSig = sig && !_spikeDone.has(ev.step_index);
              if (showSig) _spikeDone.add(ev.step_index);
              return (
                <g key={i} transform={`translate(0,${TRACK_Y})`}
                  onMouseEnter={()=>setHoveredStep(i)} onMouseLeave={()=>setHoveredStep(null)}>
                  <EventNode event={ev} signal={showSig ? sig : null} x={LEFT_PAD+i*STEP_W}
                    isHovered={hoveredStep===i}
                    onClick={()=>sig&&setActiveSignal(sig)} />
                </g>
              );
            });
          })()}
        </svg>

        {/* Token strip */}
        <div style={{ borderTop:"1px solid rgba(255,255,255,0.05)", marginTop:8, paddingTop:12, paddingLeft:16 }}>
          <TokenStrip events={events} tokenSeries={tokenSeries} STEP_W={STEP_W} LEFT_PAD={LEFT_PAD-16} signalByStep={signalByStep} onSignalClick={setActiveSignal} />
        </div>

        {/* Duration strip */}
        <div style={{ borderTop:"1px solid rgba(255,255,255,0.05)", marginTop:12, paddingTop:12, paddingLeft:16 }}>
          <DurationStrip events={events} stepDurations={stepDurations} STEP_W={STEP_W} LEFT_PAD={LEFT_PAD-16} signalByStep={signalByStep} onSignalClick={setActiveSignal} />
        </div>
      </div>

      {/* Hover detail */}
      {hoveredStep !== null && (() => {
        const ev    = events[hoveredStep];
        const meta2 = EVENT_META[ev.event_type] || {};
        const dur   = stepDurations[ev.step_index];
        const tok   = tokenSeries.find(s => s.stepIndex === ev.step_index);
        const cols  = [
          { label:"Event",    value:ev.event_type, color:meta2.color },
          { label:"Step",     value:String(ev.step_index), color:"#9ba3af" },
          dur!==undefined && { label:"Duration", value:fmtDuration(dur), color:durationColor(dur) },
          tok && { label:"Prompt tokens", value:tok.tokens.toLocaleString()+(tok.delta!=null?` (+${tok.delta.toLocaleString()})`:""), color:tokenColor(tok.tokens) },
          tok?.cause && { label:"Token source", value:`${tok.cause.toolName} · ${tok.cause.outputLength.toLocaleString()} chars`, color:"#f97316" },
          ...Object.entries(ev.payload||{}).slice(0,4).map(([k,v])=>({ label:k.replace(/_/g," "), value:Array.isArray(v)?v.join(", "):String(v), color:"#9ba3af" })),
        ].filter(Boolean);
        return (
          <div style={{ marginTop:8, padding:"12px 18px", background:"rgba(255,255,255,0.02)", border:"1px solid rgba(255,255,255,0.07)", borderRadius:5, display:"flex", gap:20, flexWrap:"wrap" }}>
            {cols.map((c,i) => (
              <div key={i}>
                <div style={{ fontSize:9, color:"#4b5563", textTransform:"uppercase", letterSpacing:"0.06em", marginBottom:3 }}>{c.label}</div>
                <div style={{ fontFamily:"monospace", fontSize:12, color:c.color||"#9ba3af", fontWeight:500 }}>{c.value}</div>
              </div>
            ))}
          </div>
        );
      })()}

      {activeSignal && <SignalPopup signal={activeSignal} onClose={() => setActiveSignal(null)} />}
    </div>
  );
}

// ── InsightsPanel ──────────────────────────────────────────────────────────────
function InsightsPanel({ insights }) {
  if (!insights) return null;

  const card = (children) => (
    <div style={{ background:"rgba(255,255,255,0.02)", border:"1px solid rgba(255,255,255,0.06)",
                  borderRadius:6, padding:"14px 16px", flex:"1 1 260px", minWidth:220 }}>
      {children}
    </div>
  );
  const label = (txt) => (
    <div style={{ fontSize:9, textTransform:"uppercase", letterSpacing:"0.1em",
                  color:"#4b5563", marginBottom:8 }}>{txt}</div>
  );

  // ── 1. Version comparison ──────────────────────────────────────────────────
  const VersionTable = () => {
    if (!insights.versions.length) return <div style={{ fontSize:11, color:"#4b5563" }}>No version data yet.</div>;
    return (
      <table style={{ width:"100%", borderCollapse:"collapse", fontSize:11 }}>
        <thead>
          <tr>
            {["Version","Runs","Signal rate","Signals"].map(h=>(
              <th key={h} style={{ textAlign:"left", color:"#4b5563", fontWeight:500,
                                   paddingBottom:6, fontSize:9, letterSpacing:"0.06em" }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {insights.versions.map((v, i) => {
            const pct = Math.round(v.signal_rate * 100);
            const bar = pct > 0 ? (
              <div style={{ display:"inline-block", width:Math.max(pct * 0.5, 2), height:5,
                            background: pct > 30 ? "#ff7a00" : pct > 15 ? "#f5c518" : "#22c55e",
                            borderRadius:2, marginRight:5, verticalAlign:"middle" }} />
            ) : null;
            return (
              <tr key={v.agent_version} style={{ borderTop:"1px solid rgba(255,255,255,0.04)" }}>
                <td style={{ padding:"5px 0", color:"#9ba3af", fontFamily:"monospace" }}>
                  {i === 0 && <span style={{ fontSize:8, color:"#22c55e", marginRight:4 }}>●</span>}
                  {v.agent_version.slice(0, 8)}
                </td>
                <td style={{ padding:"5px 0", color:"#6b7280" }}>{v.run_count}</td>
                <td style={{ padding:"5px 0", color: pct > 30?"#ff7a00": pct>15?"#f5c518":"#22c55e" }}>
                  {bar}{pct}%
                </td>
                <td style={{ padding:"5px 0", color:"#6b7280" }}>{v.signal_count}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    );
  };

  // ── 2. Signal recurrence ───────────────────────────────────────────────────
  const RecurrenceTable = () => {
    if (!insights.signal_trends.length)
      return <div style={{ fontSize:11, color:"#4b5563" }}>No recurring signals yet.</div>;

    // Group by failure_type, summing counts across all days/versions
    const totals = {};
    const byVersion = {};
    insights.signal_trends.forEach(pt => {
      totals[pt.failure_type] = (totals[pt.failure_type] || 0) + pt.count;
      if (!byVersion[pt.failure_type]) byVersion[pt.failure_type] = new Set();
      byVersion[pt.failure_type].add(pt.agent_version);
    });
    const sorted = Object.entries(totals).sort((a,b) => b[1]-a[1]).slice(0,6);
    const maxCount = Math.max(...sorted.map(([,c])=>c), 1);

    return (
      <div>
        {sorted.map(([ft, count]) => {
          const versions = [...byVersion[ft]];
          const w = Math.round((count / maxCount) * 80);
          return (
            <div key={ft} style={{ marginBottom:8 }}>
              <div style={{ display:"flex", justifyContent:"space-between", marginBottom:3 }}>
                <span style={{ fontSize:10, color:"#9ba3af" }}>{ft.replace(/_/g," ")}</span>
                <span style={{ fontSize:10, color:"#6b7280" }}>×{count} · {versions.length}v</span>
              </div>
              <div style={{ height:4, background:"rgba(255,255,255,0.06)", borderRadius:2 }}>
                <div style={{ width:w+"%", height:"100%",
                              background: count>10?"#ff7a00": count>4?"#f5c518":"#818cf8",
                              borderRadius:2, transition:"width 0.4s" }} />
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  // ── 3. Input hash patterns ─────────────────────────────────────────────────
  const InputPatterns = () => {
    if (!insights.input_patterns.length)
      return <div style={{ fontSize:11, color:"#4b5563" }}>No input patterns yet (need ≥2 runs with the same input hash).</div>;

    const top = insights.input_patterns.slice(0, 5);
    return (
      <div>
        {top.map((p, i) => {
          const pct = Math.round(p.rate * 100);
          const bar = Math.round(pct * 0.8);
          return (
            <div key={i} style={{ marginBottom:10 }}>
              <div style={{ display:"flex", justifyContent:"space-between", marginBottom:3 }}>
                <span style={{ fontSize:10, color:"#9ba3af", fontFamily:"monospace" }}>
                  {p.input_hash}
                </span>
                <span style={{ fontSize:9, color: pct>=70?"#ff7a00":pct>=40?"#f5c518":"#6b7280" }}>
                  {p.triggered_count}/{p.total_runs} = {pct}%
                </span>
              </div>
              <div style={{ display:"flex", alignItems:"center", gap:6 }}>
                <div style={{ height:3, flex:1, background:"rgba(255,255,255,0.06)", borderRadius:2 }}>
                  <div style={{ width:bar+"%", height:"100%",
                                background: pct>=70?"#ff7a00":pct>=40?"#f5c518":"#818cf8",
                                borderRadius:2 }} />
                </div>
                <span style={{ fontSize:9, color:"#6b7280", whiteSpace:"nowrap" }}>
                  → {p.failure_type.replace(/_/g," ")}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  // ── 4. Time-to-first-tool ──────────────────────────────────────────────────
  const TimeToTool = () => {
    const { p25, p50, p75, runs_with_tool, total_runs, daily_trend } = insights.time_to_tool;
    const hasTool = runs_with_tool > 0;

    // Mini trend sparkline
    const trendPts = daily_trend.filter(d => d.avg_first_tool_step != null);
    let sparkSvg = null;
    if (trendPts.length >= 2) {
      const vals = trendPts.map(d => d.avg_first_tool_step);
      const minV = Math.min(...vals), maxV = Math.max(...vals, minV + 0.1);
      const W = 120, H = 24, PAD = 2;
      const pts = vals.map((v, i) => {
        const x = PAD + (i / (vals.length - 1)) * (W - 2 * PAD);
        const y = H - PAD - ((v - minV) / (maxV - minV)) * (H - 2 * PAD);
        return [x.toFixed(1), y.toFixed(1)];
      });
      sparkSvg = (
        <svg width={W} height={H} style={{ display:"block", marginTop:8 }}>
          <polyline points={pts.map(p=>p.join(",")).join(" ")}
            fill="none" stroke="#818cf8" strokeWidth={1.5} strokeLinejoin="round" />
          <circle cx={pts[pts.length-1][0]} cy={pts[pts.length-1][1]} r={2.5} fill="#818cf8" />
        </svg>
      );
    }

    return (
      <div>
        {hasTool ? (
          <>
            <div style={{ display:"flex", gap:16, marginBottom:8 }}>
              {[["P25", p25], ["P50", p50], ["P75", p75]].map(([lbl, v]) => (
                <div key={lbl} style={{ textAlign:"center" }}>
                  <div style={{ fontSize:16, fontWeight:600, color:"#e8eaf0" }}>
                    {v != null ? Math.round(v) : "—"}
                  </div>
                  <div style={{ fontSize:9, color:"#4b5563" }}>{lbl}</div>
                </div>
              ))}
              <div style={{ textAlign:"center" }}>
                <div style={{ fontSize:13, fontWeight:500, color:"#6b7280" }}>
                  {runs_with_tool}/{total_runs}
                </div>
                <div style={{ fontSize:9, color:"#4b5563" }}>used tools</div>
              </div>
            </div>
            {sparkSvg && (
              <div>
                <div style={{ fontSize:9, color:"#4b5563" }}>avg steps · 14d trend</div>
                {sparkSvg}
              </div>
            )}
          </>
        ) : (
          <div style={{ fontSize:11, color:"#4b5563" }}>No tool calls recorded yet.</div>
        )}
      </div>
    );
  };

  // ── 5. Signal rate by hour ─────────────────────────────────────────────────
  const HourlyChart = () => {
    const data = insights.hourly_pattern;
    if (!data.length)
      return <div style={{ fontSize:11, color:"#4b5563" }}>Need 30+ days of data.</div>;

    // Build a full 24-slot array
    const byHour = {};
    data.forEach(d => { byHour[d.hour_of_day] = d; });
    const maxRate = Math.max(...data.map(d => d.signal_rate), 0.01);
    const BAR_W = 9, GAP = 2, H = 40, PAD_Y = 4;

    return (
      <div>
        <svg width={24 * (BAR_W + GAP)} height={H + 16} style={{ display:"block", overflow:"visible" }}>
          {Array.from({length:24}, (_, h) => {
            const d = byHour[h];
            const rate = d ? d.signal_rate : 0;
            const barH = d ? Math.max(2, (rate / maxRate) * (H - PAD_Y)) : 1;
            const col  = rate > 0.5 ? "#ff7a00" : rate > 0.2 ? "#f5c518" : rate > 0 ? "#818cf8" : "rgba(255,255,255,0.08)";
            const x = h * (BAR_W + GAP);
            return (
              <g key={h}>
                <rect x={x} y={H - barH} width={BAR_W} height={barH}
                  fill={col} rx={1} opacity={d ? 1 : 0.4} />
                {h % 6 === 0 && (
                  <text x={x + BAR_W / 2} y={H + 12} textAnchor="middle"
                    style={{ fontSize:8, fill:"#4b5563" }}>{h}h</text>
                )}
              </g>
            );
          })}
        </svg>
        <div style={{ fontSize:9, color:"#4b5563", marginTop:2 }}>
          UTC hour · last 30d · height = signal rate · hover coming soon
        </div>
      </div>
    );
  };

  return (
    <div style={{ marginBottom:32 }}>
      <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:14 }}>
        <div style={{ width:3, height:14, background:"#818cf8", borderRadius:2 }} />
        <span style={{ fontSize:11, fontWeight:700, color:"#9ba3af", textTransform:"uppercase", letterSpacing:"0.1em" }}>
          Agent Insights
        </span>
      </div>
      {/* Row 1: version comparison + signal recurrence */}
      <div style={{ display:"flex", gap:12, flexWrap:"wrap", marginBottom:12 }}>
        {card(<><div>{label("Version comparison")}</div><VersionTable /></>)}
        {card(<><div>{label("Signal recurrence (last 30d)")}</div><RecurrenceTable /></>)}
      </div>
      {/* Row 2: input patterns + time-to-first-tool */}
      <div style={{ display:"flex", gap:12, flexWrap:"wrap", marginBottom:12 }}>
        {card(<><div>{label("Failure rate by input hash")}</div><InputPatterns /></>)}
        {card(<><div>{label("Steps to first tool call")}</div><TimeToTool /></>)}
      </div>
      {/* Row 3: hourly pattern (full width) */}
      <div style={{ background:"rgba(255,255,255,0.02)", border:"1px solid rgba(255,255,255,0.06)",
                    borderRadius:6, padding:"14px 16px" }}>
        {label("Signal rate by hour of day (UTC)")}
        <HourlyChart />
      </div>
    </div>
  );
}

// ── App ────────────────────────────────────────────────────────────────────────
function App() {
  const [agents,        setAgents]        = useState([]);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [signals,       setSignals]       = useState([]);
  const [runs,          setRuns]          = useState([]);
  const [runDetail,     setRunDetail]     = useState({});
  const [selectedId,    setSelectedId]    = useState(null);
  const [loading,       setLoading]       = useState(true);
  const [error,         setError]         = useState(null);
  const [lastRefresh,   setLastRefresh]   = useState(null);
  const [activeSignal,  setActiveSignal]  = useState(null);
  const [insights,      setInsights]      = useState(null);
  const [activeTab,     setActiveTab]     = useState("signals");

  // Poll agent list every 10 s
  useEffect(() => {
    let cancelled = false;
    async function refresh() {
      try {
        const data = await apiFetch("/v1/agents");
        if (cancelled) return;
        const list = (data.agents||[]).sort((a,b)=>(b.last_seen||0)-(a.last_seen||0));
        setAgents(list);
        setSelectedAgent(prev => prev || (list[0]?.agent_id));
        setError(null);
      } catch(e) {
        if (!cancelled) setError(String(e));
      } finally {
        if (!cancelled) { setLoading(false); setLastRefresh(new Date()); }
      }
    }
    refresh();
    const id = setInterval(refresh, 10_000);
    return () => { cancelled=true; clearInterval(id); };
  }, []);

  // When selected agent changes: fetch runs + signals + insights
  useEffect(() => {
    if (!selectedAgent) return;
    let cancelled = false;
    async function fetchAgent() {
      try {
        const [runsData, sigsData, insightsData] = await Promise.all([
          apiFetch(`/v1/agents/${selectedAgent}/runs`),
          apiFetch(`/v1/agents/${selectedAgent}/signals`),
          apiFetch(`/v1/agents/${selectedAgent}/insights`),
        ]);
        if (cancelled) return;
        const list = runsData.runs||[];
        setRuns(list);
        setSignals(sigsData.signals||[]);
        setInsights(insightsData);
        setSelectedId(prev => (prev && list.some(r=>r.run_id===prev)) ? prev : (list[0]?.run_id||null));
      } catch(e) {
        if (!cancelled) setError(String(e));
      }
    }
    fetchAgent();
    const id = setInterval(fetchAgent, 10_000);
    return () => { cancelled=true; clearInterval(id); };
  }, [selectedAgent]);

  // Fetch run detail on demand
  useEffect(() => {
    if (!selectedId || runDetail[selectedId]) return;
    apiFetch(`/v1/runs/${selectedId}`)
      .then(d => setRunDetail(prev=>({...prev,[selectedId]:d})))
      .catch(e => setError(String(e)));
  }, [selectedId]);

  const run = selectedId ? runDetail[selectedId] : null;

  function selectRun(id) { setSelectedId(id); }

  function selectAgent(id) {
    setSelectedAgent(id);
    setSelectedId(null);
    setRuns([]);
    setSignals([]);
    setInsights(null);
    setActiveTab("signals");
  }

  const tabs = [
    { id: "signals",  label: "Signals",  badge: signals.length > 0 ? signals.length : "✓" },
    { id: "insights", label: "Insights" },
    { id: "timeline", label: "Timeline", badge: runs.length || null },
  ];

  return (
    <div style={{ minHeight:"100vh", background:"#0a0b0d", fontFamily:"'DM Mono','Fira Code',monospace", color:"#e8eaf0", padding:"20px 28px" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&display=swap');
        * { box-sizing:border-box; }
        button { font-family:inherit; }
        ::-webkit-scrollbar { height:4px; width:4px; }
        ::-webkit-scrollbar-track { background:transparent; }
        ::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.1); border-radius:2px; }
      `}</style>

      {/* Header */}
      <div style={{ display:"flex", alignItems:"center", gap:12, marginBottom:20, paddingBottom:16, borderBottom:"1px solid rgba(255,255,255,0.06)" }}>
        <div style={{ width:26, height:26, borderRadius:5, background:"linear-gradient(135deg,#f97316,#dc2626)", display:"flex", alignItems:"center", justifyContent:"center", fontSize:12, fontWeight:900, color:"#fff", flexShrink:0 }}>D</div>
        <div>
          <div style={{ fontSize:14, fontWeight:600, color:"#e8eaf0" }}>DuneTrace</div>
          <div style={{ fontSize:9, color:"#4b5563", letterSpacing:"0.06em" }}>
            {lastRefresh ? `updated ${lastRefresh.toLocaleTimeString()} · auto-refresh 10s` : "Connecting…"}
          </div>
        </div>
        <div style={{ marginLeft:"auto", display:"flex", gap:12, alignItems:"center", flexWrap:"wrap" }}>
          {[["#22c55e","Lifecycle"],["#818cf8","LLM"],["#f97316","Tool"],["#ff7a00","Signal"]].map(([c,l]) => (
            <div key={l} style={{ display:"flex", alignItems:"center", gap:4 }}>
              <div style={{ width:6, height:6, borderRadius:"50%", background:c }} />
              <span style={{ fontSize:9, color:"#4b5563" }}>{l}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div style={{ marginBottom:14, padding:"8px 14px", background:"rgba(255,59,59,0.07)", border:"1px solid rgba(255,59,59,0.3)", borderRadius:5, fontSize:11, color:"#ff8080" }}>
          API error: {error} — is the stack running? <code style={{ color:"#f97316" }}>docker compose up -d --build</code>
        </div>
      )}

      {loading ? (
        <div style={{ color:"#6b7280", fontSize:12 }}>Connecting to {API_URL}…</div>
      ) : (
        <div style={{ display:"flex", alignItems:"flex-start" }}>

          {/* ── LEFT SIDEBAR: agent list ── */}
          <div style={{ width:200, flexShrink:0, borderRight:"1px solid rgba(255,255,255,0.06)", paddingRight:14, marginRight:22 }}>
            <div style={{ fontSize:9, textTransform:"uppercase", letterSpacing:"0.1em", color:"#4b5563", marginBottom:10, fontWeight:700 }}>
              Agents
              <span style={{ marginLeft:6, background:"rgba(255,255,255,0.06)", border:"1px solid rgba(255,255,255,0.1)", borderRadius:8, padding:"1px 6px", fontSize:8, color:"#6b7280" }}>
                {agents.length}
              </span>
            </div>
            <AgentSidebar agents={agents} selectedAgent={selectedAgent} onSelect={selectAgent} />
          </div>

          {/* ── RIGHT: tabbed content ── */}
          <div style={{ flex:1, minWidth:0 }}>
            {!selectedAgent ? (
              <div style={{ color:"#6b7280", fontSize:12, paddingTop:32 }}>
                No agents found — run <code style={{ color:"#f97316" }}>python scripts/smoke_test.py</code>
              </div>
            ) : (
              <>
                <Tabs tabs={tabs} active={activeTab} onChange={setActiveTab} />

                {/* ── Signals tab ── */}
                {activeTab === "signals" && (
                  <SignalsList
                    signals={signals}
                    onRunSelect={id => { setActiveTab("timeline"); selectRun(id); }}
                    onInspect={s => setActiveSignal(s)}
                  />
                )}

                {/* ── Insights tab ── */}
                {activeTab === "insights" && (
                  insights
                    ? <InsightsPanel insights={insights} />
                    : <div style={{ color:"#6b7280", fontSize:12 }}>Loading insights…</div>
                )}

                {/* ── Timeline tab ── */}
                {activeTab === "timeline" && (
                  <div>
                    <div style={{ marginBottom:18 }}>
                      <RunSelector runs={runs} selectedId={selectedId} onSelect={selectRun} />
                    </div>
                    {run ? (
                      <Timeline run={run} key={selectedId} />
                    ) : selectedId ? (
                      <div style={{ color:"#6b7280", fontSize:12, marginTop:16 }}>Loading run detail…</div>
                    ) : (
                      <div style={{ color:"#4b5563", fontSize:11, marginTop:16 }}>Select a run above to inspect the timeline.</div>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      )}

      {/* Footer */}
      {!loading && (
        <div style={{ marginTop:32, padding:"10px 14px", background:"rgba(255,255,255,0.01)", border:"1px solid rgba(255,255,255,0.04)", borderRadius:4, fontSize:10, color:"#4b5563", lineHeight:1.9 }}>
          <strong style={{ color:"#6b7280" }}>How to read:</strong>{" "}
          Track connector brightness = time gap between steps ·
          Signal <strong style={{ color:"#ff7a00" }}>!</strong> node = click for explanation + fix ·
          Token bars = prompt_tokens per LLM call · +Ntok = growth · Duration bars = step latency
        </div>
      )}

      {activeSignal && <SignalPopup signal={activeSignal} onClose={() => setActiveSignal(null)} />}
    </div>
  );
}
