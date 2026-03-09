const { useState } = React;

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

function fmtDurationMs(ms) {
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
                  {fmtDurationMs(dms)}
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
            value: slowestStep !== null ? "step " + slowestStep + " \u00b7 " + fmtDurationMs(slowestMs) : "\u2014",
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
                  {fmtDurationMs(dur)}
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

