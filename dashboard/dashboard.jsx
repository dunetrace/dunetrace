const { useState, useEffect, useMemo, useCallback } = React;
const API = "http://localhost:8002";

// ── Colour tokens ─────────────────────────────────────────────────────────────
const C = {
  bg:       "#080B11",
  surface:  "#0E1320",
  surfaceB: "#131B2A",
  border:   "#1E2D45",
  orange:   "#E86A2B",
  blue:     "#4A9EFF",
  green:    "#22C55E",
  red:      "#EF4444",
  yellow:   "#F59E0B",
  purple:   "#C026D3",
  text:     "#E2E8F0",
  textM:    "#94A3B8",
  textD:    "#4B6A8A",
  navy:     "#1A2942",
};

const SEV_COLOR = { CRITICAL: C.purple, HIGH: C.red, MEDIUM: C.yellow, LOW: C.blue };

// ── Helpers ───────────────────────────────────────────────────────────────────
function fmtDate(tsMs) {
  return new Date(tsMs).toLocaleDateString("en-GB", { day: "2-digit", month: "short" });
}
function fmtTime(tsMs) {
  return new Date(tsMs).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
}
function fmtDuration(s) {
  if (s == null || s < 0) return "—";
  return s < 60 ? `${s.toFixed(1)}s` : `${(s / 60).toFixed(1)}m`;
}
function toMs(unixSec) { return unixSec * 1000; }

function exitStatus(exitReason) {
  if (!exitReason) return { label: "—", ok: null };
  const r = exitReason.toLowerCase();
  if (r.includes("error") || r.includes("fail")) return { label: "ERR", ok: false };
  if (r.includes("complet"))                      return { label: "OK",  ok: true  };
  return { label: exitReason.replace("run.", "").toUpperCase().slice(0, 8), ok: null };
}

function bucketByDay(runs) {
  const map = {};
  runs.forEach(r => {
    const d   = new Date(toMs(r.started_at));
    const key = `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,"0")}-${String(d.getDate()).padStart(2,"0")}`;
    if (!map[key]) {
      const [y, mo, dy] = key.split("-").map(Number);
      map[key] = { key, ts: new Date(y, mo - 1, dy).getTime(), runs: [] };
    }
    map[key].runs.push(r);
  });
  return Object.values(map).sort((a, b) => a.ts - b.ts);
}

function dayStats(runs) {
  const total    = runs.length;
  const errored  = runs.filter(r => exitStatus(r.exit_reason).ok === false).length;
  const withSigs = runs.filter(r => r.has_signals).length;
  const durs     = runs.filter(r => r.completed_at && r.started_at)
                       .map(r => r.completed_at - r.started_at)
                       .sort((a, b) => a - b);
  const p95dur   = durs[Math.floor(durs.length * 0.95)] ?? 0;
  return {
    total, errored, withSigs,
    errorRate:  total ? Math.round(errored  / total * 100) : 0,
    signalRate: total ? Math.round(withSigs / total * 100) : 0,
    p95dur,
  };
}

// ── Sub-components ────────────────────────────────────────────────────────────
function MiniBar({ value, max, color, width = 40, height = 6 }) {
  const pct = Math.min(100, (value / (max || 1)) * 100);
  return (
    <div style={{ width, height, background: C.border, borderRadius: 2, overflow: "hidden" }}>
      <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 2 }} />
    </div>
  );
}

function Sparkline({ values, color, w = 80, h = 24 }) {
  if (!values || values.length < 2) return null;
  const max = Math.max(...values, 1);
  const min = Math.min(...values);
  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * w;
    const y = h - ((v - min) / (max - min + 0.001)) * (h - 4) - 2;
    return `${x},${y}`;
  }).join(" ");
  return (
    <svg width={w} height={h} style={{ display: "block" }}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  );
}

function HeatCell({ runs, date, selected, onClick }) {
  const s         = dayStats(runs);
  const intensity = Math.min(1, runs.length / 30);
  const hasSigs   = runs.some(r => r.has_signals);
  const bg = selected
    ? C.orange
    : s.errorRate > 40 ? `rgba(239,68,68,${0.15 + intensity * 0.35})`
    : hasSigs           ? `rgba(232,106,43,${0.1  + intensity * 0.3})`
    :                     `rgba(74,158,255,${0.08 + intensity * 0.25})`;

  return (
    <div onClick={onClick} style={{
      flex: 1, minWidth: 56, maxWidth: 80, cursor: "pointer", padding: "8px 6px",
      background: bg, borderRadius: 4,
      border: selected ? `1px solid ${C.orange}` : "1px solid transparent",
      transition: "all 0.15s",
    }}>
      <div style={{ fontSize: 10, color: selected ? "#fff" : C.textM, fontFamily: "monospace", marginBottom: 3 }}>
        {fmtDate(date)}
      </div>
      <div style={{ fontSize: 14, fontWeight: 700, color: selected ? "#fff" : C.text, marginBottom: 1 }}>
        {runs.length}
      </div>
      <div style={{ fontSize: 9, color: selected ? "rgba(255,255,255,0.7)" : C.textD, fontFamily: "monospace" }}>
        {s.errorRate}% err
      </div>
      {s.withSigs > 0 && (
        <div style={{ fontSize: 9, color: selected ? "rgba(255,255,255,0.7)" : C.orange, fontFamily: "monospace" }}>
          ▲{s.withSigs} sig
        </div>
      )}
    </div>
  );
}

// ── API helpers ───────────────────────────────────────────────────────────────
async function apiFetch(path) {
  const r = await fetch(`${API}${path}`);
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json();
}

// ── Main Dashboard ────────────────────────────────────────────────────────────
function Dashboard() {
  const [agents,        setAgents]        = useState([]);
  const [selectedAgent, setSelectedAgent] = useState(null); // null = all-agents view
  const [runs,          setRuns]          = useState([]);
  const [signals,       setSignals]       = useState([]);
  const [loading,       setLoading]       = useState(false);
  const [selectedDay,   setSelectedDay]   = useState(null);
  const [selectedRun,   setSelectedRun]   = useState(null);
  const [sortBy,           setSortBy]           = useState("started_at");
  const [sortDir,          setSortDir]          = useState("desc");
  const [runsPage,         setRunsPage]         = useState(0);
  const [runDetail,        setRunDetail]        = useState(null);
  const [runDetailLoading, setRunDetailLoading] = useState(false);
  const [timelineExpanded, setTimelineExpanded] = useState(false);
  const PAGE_SIZE = 15;

  // Load agents list
  const loadAgents = useCallback(async () => {
    try {
      const d = await apiFetch("/v1/agents?limit=50");
      setAgents(d.agents || []);
    } catch (e) { console.error(e); }
  }, []);

  useEffect(() => {
    loadAgents();
    const t = setInterval(loadAgents, 10000);
    return () => clearInterval(t);
  }, [loadAgents]);

  // Load runs + signals when agent selected
  const loadAgentData = useCallback(async (agentId) => {
    if (!agentId) { setRuns([]); setSignals([]); return; }
    setLoading(true);
    try {
      const [rd, sd] = await Promise.all([
        apiFetch(`/v1/agents/${encodeURIComponent(agentId)}/runs?limit=200`),
        apiFetch(`/v1/agents/${encodeURIComponent(agentId)}/signals?limit=500`),
      ]);
      setRuns(rd.runs    || []);
      setSignals(sd.signals || []);
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => {
    setSelectedRun(null);
    setSelectedDay(null);
    setRunsPage(0);
    setRunDetail(null);
    loadAgentData(selectedAgent);
  }, [selectedAgent, loadAgentData]);

  // Fetch full run detail (events + signals) when a run is selected
  useEffect(() => {
    if (!selectedRun) { setRunDetail(null); setTimelineExpanded(false); return; }
    setRunDetailLoading(true);
    setRunDetail(null);
    setTimelineExpanded(false);
    apiFetch(`/v1/runs/${encodeURIComponent(selectedRun.run_id)}`)
      .then(d => setRunDetail(d))
      .catch(e => console.error(e))
      .finally(() => setRunDetailLoading(false));
  }, [selectedRun]);

  // Auto-refresh agent data
  useEffect(() => {
    if (!selectedAgent) return;
    const t = setInterval(() => loadAgentData(selectedAgent), 10000);
    return () => clearInterval(t);
  }, [selectedAgent, loadAgentData]);

  // Current agent record (for sidebar)
  const agentRecord = useMemo(
    () => agents.find(a => a.agent_id === selectedAgent) || null,
    [agents, selectedAgent]
  );

  // KPI stats
  const globalStats = useMemo(() => {
    if (selectedAgent) {
      const total   = runs.length;
      const errored = runs.filter(r => exitStatus(r.exit_reason).ok === false).length;
      return {
        total,
        errored,
        errorRate:  total ? Math.round(errored   / total * 100) : 0,
        signals:    signals.length,
        signalRate: total ? Math.round(signals.length / total * 100) : 0,
      };
    }
    const total = agents.reduce((n, a) => n + a.run_count,    0);
    const sigs  = agents.reduce((n, a) => n + a.signal_count, 0);
    return {
      total, errored: 0, errorRate: 0,
      signals: sigs,
      signalRate: total ? Math.round(sigs / total * 100) : 0,
    };
  }, [selectedAgent, agents, runs, signals]);

  // Calendar buckets
  const days = useMemo(() => selectedAgent ? bucketByDay(runs) : [], [runs, selectedAgent]);
  const sparkErrRates = useMemo(() => days.map(d => dayStats(d.runs).errorRate),  [days]);
  const sparkSigRates = useMemo(() => days.map(d => dayStats(d.runs).signalRate), [days]);

  // Day-filtered runs
  const dayRuns = useMemo(() => {
    if (!selectedDay) return runs;
    return runs.filter(r => {
      const d   = new Date(toMs(r.started_at));
      const key = `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,"0")}-${String(d.getDate()).padStart(2,"0")}`;
      return key === selectedDay;
    });
  }, [selectedDay, runs]);

  const dsStats = useMemo(() => selectedDay ? dayStats(dayRuns) : null, [selectedDay, dayRuns]);

  // Sorted + paginated runs
  const sorted = useMemo(() => [...dayRuns].sort((a, b) => {
    let va = a[sortBy] ?? 0, vb = b[sortBy] ?? 0;
    return sortDir === "asc" ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
  }), [dayRuns, sortBy, sortDir]);

  const page       = sorted.slice(runsPage * PAGE_SIZE, (runsPage + 1) * PAGE_SIZE);
  const totalPages = Math.ceil(sorted.length / PAGE_SIZE);

  function toggleSort(col) {
    if (sortBy === col) setSortDir(d => d === "asc" ? "desc" : "asc");
    else { setSortBy(col); setSortDir("desc"); }
    setRunsPage(0);
  }

  // Top failure types
  const topFailures = useMemo(() => {
    const map = {};
    const src = selectedAgent
      ? Object.entries(agentRecord?.failure_types || {})
      : agents.flatMap(a => Object.entries(a.failure_types || {}));
    src.forEach(([k, v]) => { map[k] = (map[k] || 0) + v; });
    return Object.entries(map).sort((a, b) => b[1] - a[1]);
  }, [selectedAgent, agents, agentRecord]);

  function selectAgent(id) {
    setSelectedAgent(id);
    setSelectedRun(null);
    setSelectedDay(null);
    setRunsPage(0);
  }

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <div style={{
      background: C.bg, minHeight: "100vh", color: C.text,
      fontFamily: "'JetBrains Mono','Fira Code','SF Mono',monospace",
      fontSize: 12,
    }}>

      {/* ── Top bar ── */}
      <div style={{
        background: C.surface, borderBottom: `1px solid ${C.border}`,
        padding: "0 24px", display: "flex", alignItems: "center",
        height: 48, gap: 20,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{
            width: 28, height: 28, borderRadius: 6,
            background: "linear-gradient(135deg,#1A2942,#0B0D13)",
            display: "flex", alignItems: "center", justifyContent: "center",
            border: `1px solid ${C.border}`,
          }}>
            <span style={{ color: C.orange, fontWeight: 700, fontSize: 13 }}>D</span>
          </div>
          <span style={{ fontWeight: 700, color: C.text, fontSize: 13, letterSpacing: "0.05em" }}>
            DUNETRACE
          </span>
        </div>

        <div style={{ width: 1, height: 20, background: C.border }} />

        {/* Agent tabs */}
        <div style={{ display: "flex", gap: 6, overflowX: "auto" }}>
          <button onClick={() => selectAgent(null)} style={{
            padding: "3px 10px", borderRadius: 4, border: "none", cursor: "pointer",
            fontSize: 10, fontFamily: "inherit", whiteSpace: "nowrap",
            background: selectedAgent === null ? C.orange : C.surfaceB,
            color:      selectedAgent === null ? "#fff"   : C.textM,
            letterSpacing: "0.04em",
          }}>ALL</button>
          {agents.map(a => (
            <button key={a.agent_id} onClick={() => selectAgent(a.agent_id)} style={{
              padding: "3px 10px", borderRadius: 4, border: "none", cursor: "pointer",
              fontSize: 10, fontFamily: "inherit", whiteSpace: "nowrap",
              background: selectedAgent === a.agent_id ? C.orange : C.surfaceB,
              color:      selectedAgent === a.agent_id ? "#fff"   : C.textM,
              letterSpacing: "0.04em",
            }}>
              {a.agent_id.split("-").slice(0, 2).join("-").toUpperCase()}
            </button>
          ))}
        </div>

        <div style={{ marginLeft: "auto", display: "flex", gap: 20, fontSize: 11, color: C.textD }}>
          <span>auto-refresh 10s</span>
          <span style={{ color: C.green }}>● LIVE</span>
        </div>
      </div>

      <div style={{ display: "flex", height: "calc(100vh - 48px)", overflow: "hidden" }}>

        {/* ── Left panel ── */}
        <div style={{
          width: 280, flexShrink: 0,
          background: C.surface, borderRight: `1px solid ${C.border}`,
          overflowY: "auto", padding: 16,
          display: "flex", flexDirection: "column", gap: 16,
        }}>
          {/* Label */}
          <div style={{ fontSize: 9, color: C.textD, letterSpacing: "0.15em" }}>
            {selectedAgent ? selectedAgent.toUpperCase() : "ALL AGENTS"}
          </div>

          {/* KPIs */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
            {(selectedAgent ? [
              { label: "TOTAL RUNS", value: globalStats.total, color: C.blue },
              { label: "ERROR RATE", value: `${globalStats.errorRate}%`,
                color: globalStats.errorRate > 30 ? C.red : C.green },
              { label: "SIGNALS",    value: globalStats.signals, color: C.orange },
              { label: "SIG RATE",   value: `${globalStats.signalRate}%`,
                color: globalStats.signalRate > 20 ? C.yellow : C.textM },
            ] : [
              { label: "TOTAL RUNS", value: globalStats.total,   color: C.blue },
              { label: "AGENTS",     value: agents.length,        color: C.textM },
              { label: "SIGNALS",    value: globalStats.signals,  color: C.orange },
              { label: "HIGH SIGS",  value: agents.reduce((n, a) => n + a.high_count, 0),
                color: C.red },
            ]).map(k => (
              <div key={k.label} style={{
                background: C.surfaceB, borderRadius: 6, padding: 10,
                border: `1px solid ${C.border}`,
              }}>
                <div style={{ fontSize: 9, color: C.textD, letterSpacing: "0.1em", marginBottom: 4 }}>{k.label}</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: k.color }}>{k.value}</div>
              </div>
            ))}
          </div>

          {/* Sparklines — agent view only */}
          {selectedAgent && days.length > 1 && (
            <div style={{ background: C.surfaceB, borderRadius: 6, padding: 12, border: `1px solid ${C.border}` }}>
              <div style={{ fontSize: 9, color: C.textD, letterSpacing: "0.1em", marginBottom: 8 }}>ERROR RATE TREND</div>
              <Sparkline values={sparkErrRates} color={C.red}    w={240} h={32} />
              <div style={{ fontSize: 9, color: C.textD, letterSpacing: "0.1em", margin: "10px 0 8px" }}>SIGNAL RATE TREND</div>
              <Sparkline values={sparkSigRates} color={C.orange} w={240} h={32} />
            </div>
          )}

          {/* Signal breakdown */}
          <div>
            <div style={{ fontSize: 9, color: C.textD, letterSpacing: "0.15em", marginBottom: 8 }}>
              {selectedAgent ? "SIGNAL BREAKDOWN" : "TOP SIGNALS"}
            </div>
            {topFailures.length === 0
              ? <div style={{ fontSize: 10, color: C.textD }}>No signals yet</div>
              : (() => {
                  const maxV = Math.max(...topFailures.map(e => e[1]), 1);
                  return topFailures.map(([sig, cnt]) => (
                    <div key={sig} style={{ marginBottom: 8 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                        <span style={{ fontSize: 10, color: C.textM }}>{sig.replace(/_/g, " ")}</span>
                        <span style={{ fontSize: 10, color: C.orange, fontWeight: 700 }}>{cnt}</span>
                      </div>
                      <MiniBar value={cnt} max={maxV} color={C.orange} width={248} height={4} />
                    </div>
                  ));
                })()
            }
          </div>
        </div>

        {/* ── Main content ── */}
        <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column" }}>

          {selectedAgent ? (
            /* ── Agent view: heatmap + runs table ── */
            <>
              {/* Calendar heatmap */}
              {days.length > 0 && (
                <div style={{
                  padding: "16px 20px", borderBottom: `1px solid ${C.border}`,
                  background: C.surface,
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                    <span style={{ fontSize: 9, color: C.textD, letterSpacing: "0.15em" }}>
                      RUN VOLUME — CLICK A DAY TO DRILL DOWN
                    </span>
                    {selectedDay && (
                      <button onClick={() => { setSelectedDay(null); setSelectedRun(null); setRunsPage(0); }}
                        style={{
                          background: "none", border: `1px solid ${C.border}`, color: C.textM,
                          borderRadius: 4, padding: "2px 10px", cursor: "pointer",
                          fontSize: 10, fontFamily: "inherit",
                        }}>✕ clear filter</button>
                    )}
                  </div>
                  <div style={{ display: "flex", gap: 4, overflowX: "auto" }}>
                    {days.map(d => (
                      <HeatCell key={d.key} runs={d.runs} date={d.ts}
                        selected={selectedDay === d.key}
                        onClick={() => {
                          setSelectedDay(selectedDay === d.key ? null : d.key);
                          setSelectedRun(null);
                          setRunsPage(0);
                        }}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* Day summary banner */}
              {selectedDay && dsStats && (
                <div style={{
                  background: `${C.orange}11`, borderBottom: `1px solid ${C.orange}33`,
                  padding: "10px 20px", display: "flex", gap: 24, alignItems: "center",
                }}>
                  <span style={{ fontSize: 11, color: C.orange, fontWeight: 700 }}>
                    {fmtDate(days.find(d => d.key === selectedDay)?.ts)} — {dsStats.total} runs
                  </span>
                  {[
                    { label: "ERROR RATE", value: `${dsStats.errorRate}%`, color: dsStats.errorRate > 30 ? C.red : C.green },
                    { label: "SIGNALS",    value: dsStats.withSigs,         color: C.orange },
                    { label: "P95 DUR",    value: fmtDuration(dsStats.p95dur), color: C.textM },
                  ].map(k => (
                    <div key={k.label} style={{ display: "flex", gap: 6, alignItems: "baseline" }}>
                      <span style={{ fontSize: 9, color: C.textD, letterSpacing: "0.1em" }}>{k.label}</span>
                      <span style={{ fontSize: 12, fontWeight: 700, color: k.color }}>{k.value}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Runs table */}
              {loading ? (
                <div style={{ padding: 40, textAlign: "center", color: C.textD }}>Loading…</div>
              ) : (
                <div style={{ flex: 1, padding: "0 0 16px" }}>
                  {/* Header */}
                  <div style={{
                    display: "grid",
                    gridTemplateColumns: "200px 70px 70px 60px 80px 1fr 130px",
                    padding: "8px 20px",
                    borderBottom: `1px solid ${C.border}`,
                    background: C.surface,
                    position: "sticky", top: 0, zIndex: 1,
                  }}>
                    {[
                      ["RUN ID",   "run_id"],
                      ["STATUS",   "exit_reason"],
                      ["DUR",      "duration"],
                      ["STEPS",    "step_count"],
                      ["SIGNALS",  "signal_count"],
                      ["VERSION",  "agent_version"],
                      ["TIME",     "started_at"],
                    ].map(([label, col]) => (
                      <div key={col} onClick={() => toggleSort(col)} style={{
                        fontSize: 9, color: sortBy === col ? C.orange : C.textD,
                        letterSpacing: "0.12em", cursor: "pointer", userSelect: "none",
                        display: "flex", alignItems: "center", gap: 4,
                      }}>
                        {label}
                        {sortBy === col && <span style={{ fontSize: 8 }}>{sortDir === "asc" ? "▲" : "▼"}</span>}
                      </div>
                    ))}
                  </div>

                  {/* Rows */}
                  {page.map(r => {
                    const st  = exitStatus(r.exit_reason);
                    const dur = r.completed_at && r.started_at ? r.completed_at - r.started_at : null;
                    const isSel = selectedRun?.run_id === r.run_id;
                    return (
                      <div key={r.run_id}
                        onClick={() => setSelectedRun(isSel ? null : r)}
                        style={{
                          display: "grid",
                          gridTemplateColumns: "200px 70px 70px 60px 80px 1fr 130px",
                          padding: "8px 20px",
                          borderBottom: `1px solid ${C.border}22`,
                          background: isSel      ? `${C.orange}12`
                                    : r.has_signals ? `${C.surfaceB}99`
                                    : "transparent",
                          cursor: "pointer", transition: "background 0.1s", alignItems: "center",
                        }}>
                        <span style={{ fontSize: 10, color: C.textM, fontFamily: "monospace" }}>
                          {r.run_id.slice(0, 16)}…
                        </span>
                        <span style={{
                          fontSize: 10, fontWeight: 700,
                          color: st.ok === true ? C.green : st.ok === false ? C.red : C.textM,
                        }}>
                          {st.ok === true ? "✓" : "✕"} {st.label}
                        </span>
                        <span style={{ fontSize: 11, color: dur > 15 ? C.yellow : C.textM }}>
                          {fmtDuration(dur)}
                        </span>
                        <span style={{ fontSize: 11, color: C.textM }}>{r.step_count}</span>
                        <span style={{
                          fontSize: 11, fontWeight: 700,
                          color: r.signal_count > 0 ? C.orange : C.textD,
                        }}>
                          {r.signal_count > 0 ? `▲ ${r.signal_count}` : "—"}
                        </span>
                        <span style={{ fontSize: 10, color: C.textD, fontFamily: "monospace" }}>
                          {r.agent_version || "—"}
                        </span>
                        <span style={{ fontSize: 10, color: C.textD, textAlign: "right" }}>
                          {fmtDate(toMs(r.started_at))} {fmtTime(toMs(r.started_at))}
                        </span>
                      </div>
                    );
                  })}

                  {page.length === 0 && !loading && (
                    <div style={{ padding: 40, textAlign: "center", color: C.textD }}>No runs</div>
                  )}

                  {/* Pagination */}
                  {totalPages > 1 && (
                    <div style={{
                      padding: "12px 20px", display: "flex", alignItems: "center",
                      gap: 12, borderTop: `1px solid ${C.border}`,
                    }}>
                      <span style={{ fontSize: 10, color: C.textD }}>
                        {sorted.length} runs · page {runsPage + 1} of {totalPages}
                      </span>
                      <div style={{ display: "flex", gap: 6 }}>
                        {[...Array(Math.min(totalPages, 8))].map((_, i) => (
                          <button key={i} onClick={() => setRunsPage(i)} style={{
                            width: 24, height: 24, borderRadius: 4, border: "none", cursor: "pointer",
                            fontSize: 10, fontFamily: "inherit",
                            background: runsPage === i ? C.orange : C.surfaceB,
                            color:      runsPage === i ? "#fff"   : C.textM,
                          }}>{i + 1}</button>
                        ))}
                        {totalPages > 8 && (
                          <span style={{ fontSize: 10, color: C.textD, alignSelf: "center" }}>…{totalPages}</span>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          ) : (
            /* ── All-agents overview table ── */
            <div style={{ padding: 20 }}>
              <div style={{ fontSize: 9, color: C.textD, letterSpacing: "0.15em", marginBottom: 12 }}>
                ALL AGENTS — {agents.length} agents
              </div>
              <div style={{
                display: "grid",
                gridTemplateColumns: "220px 70px 80px 70px 90px 1fr",
                padding: "8px 12px",
                borderBottom: `1px solid ${C.border}`,
                background: C.surface, borderRadius: "6px 6px 0 0",
              }}>
                {["AGENT", "RUNS", "SIGNALS", "HIGH", "LAST SEEN", "FAILURE TYPES"].map(h => (
                  <div key={h} style={{ fontSize: 9, color: C.textD, letterSpacing: "0.12em" }}>{h}</div>
                ))}
              </div>
              {agents.map(a => (
                <div key={a.agent_id} onClick={() => selectAgent(a.agent_id)}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "220px 70px 80px 70px 90px 1fr",
                    padding: "10px 12px",
                    borderBottom: `1px solid ${C.border}22`,
                    background: C.surfaceB, cursor: "pointer", alignItems: "center",
                    transition: "background 0.1s",
                  }}
                  onMouseEnter={e => e.currentTarget.style.background = C.navy}
                  onMouseLeave={e => e.currentTarget.style.background = C.surfaceB}
                >
                  <span style={{ fontSize: 11, color: C.blue, fontFamily: "monospace" }}>{a.agent_id}</span>
                  <span style={{ fontSize: 12, fontWeight: 600, color: C.text }}>{a.run_count}</span>
                  <span style={{ fontSize: 12, fontWeight: 600, color: a.signal_count > 0 ? C.orange : C.textD }}>
                    {a.signal_count || "—"}
                  </span>
                  <span style={{ fontSize: 12, fontWeight: 600, color: a.high_count > 0 ? C.red : C.textD }}>
                    {a.high_count || "—"}
                  </span>
                  <span style={{ fontSize: 10, color: C.textD }}>{fmtDate(toMs(a.last_seen))}</span>
                  <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                    {Object.entries(a.failure_types || {}).slice(0, 3).map(([ft, cnt]) => (
                      <span key={ft} style={{
                        fontSize: 9, padding: "1px 5px", borderRadius: 3,
                        background: `${C.orange}22`, color: C.orange,
                        border: `1px solid ${C.orange}44`, fontFamily: "monospace",
                      }}>{ft.replace(/_/g, " ")} ×{cnt}</span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* ── Right panel: run detail ── */}
        {selectedRun && (
          <div style={{
            width: 400, flexShrink: 0,
            background: C.surface, borderLeft: `1px solid ${C.border}`,
            overflowY: "auto", padding: 16,
            display: "flex", flexDirection: "column", gap: 14,
          }}>
            {/* Header */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: C.orange }}>RUN DETAIL</span>
              <button onClick={() => { setSelectedRun(null); setTimelineExpanded(false); }}
                style={{ background: "none", border: "none", color: C.textD, cursor: "pointer", fontSize: 16 }}>✕</button>
            </div>

            {runDetailLoading && (
              <div style={{ padding: 40, textAlign: "center", color: C.textD }}>Loading…</div>
            )}

            {runDetail && (() => {
              const st  = exitStatus(runDetail.exit_reason);
              const dur = runDetail.completed_at && runDetail.started_at
                ? runDetail.completed_at - runDetail.started_at : null;
              const llmEvents  = (runDetail.events || []).filter(e => e.event_type === "llm.called");
              const tokenSeries = llmEvents.map(e => e.payload?.prompt_tokens || 0).filter(t => t > 0);
              const lastTokens  = tokenSeries[tokenSeries.length - 1] || 0;
              const tokenGrowth = tokenSeries.length > 1
                ? Math.round(((tokenSeries[tokenSeries.length - 1] - tokenSeries[0]) / (tokenSeries[0] || 1)) * 100)
                : 0;

              return (
                <>
                  {/* ── Section 1: Run ID / Status / Duration ── */}
                  <div style={{ background: C.surfaceB, borderRadius: 6, padding: 12, border: `1px solid ${C.border}` }}>
                    <div style={{ fontSize: 9, color: C.textD, marginBottom: 4, letterSpacing: "0.1em" }}>RUN ID</div>
                    <div style={{ fontSize: 10, color: C.textM, wordBreak: "break-all", fontFamily: "monospace", marginBottom: 10 }}>
                      {runDetail.run_id}
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                      <div>
                        <div style={{ fontSize: 9, color: C.textD, marginBottom: 3, letterSpacing: "0.1em" }}>STATUS</div>
                        <div style={{ fontSize: 12, fontWeight: 700,
                          color: st.ok === true ? C.green : st.ok === false ? C.red : C.textM }}>
                          {st.ok === true ? "✓" : "✕"} {runDetail.exit_reason}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: 9, color: C.textD, marginBottom: 3, letterSpacing: "0.1em" }}>DURATION</div>
                        <div style={{ fontSize: 13, fontWeight: 700, color: dur > 15 ? C.yellow : C.text }}>
                          {fmtDuration(dur)}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* ── Section 2: Metrics ── */}
                  <div style={{ background: C.surfaceB, borderRadius: 6, padding: 12, border: `1px solid ${C.border}` }}>
                    <div style={{ fontSize: 9, color: C.textD, marginBottom: 10, letterSpacing: "0.1em" }}>METRICS</div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, marginBottom: tokenSeries.length > 0 ? 12 : 0 }}>
                      {[
                        { label: "STEPS",    value: runDetail.step_count,           color: C.text },
                        { label: "SIGNALS",  value: runDetail.signals?.length || 0,
                          color: (runDetail.signals?.length || 0) > 0 ? C.orange : C.textD },
                        { label: "LLM CALLS", value: llmEvents.length,              color: C.blue },
                      ].map(m => (
                        <div key={m.label} style={{ textAlign: "center" }}>
                          <div style={{ fontSize: 9, color: C.textD, letterSpacing: "0.08em", marginBottom: 4 }}>{m.label}</div>
                          <div style={{ fontSize: 16, fontWeight: 700, color: m.color }}>{m.value}</div>
                        </div>
                      ))}
                    </div>
                    {tokenSeries.length > 0 && (
                      <div style={{ paddingTop: 10, borderTop: `1px solid ${C.border}` }}>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                          <span style={{ fontSize: 9, color: C.textD, letterSpacing: "0.1em" }}>TOKEN GROWTH</span>
                          <span style={{ fontSize: 10, fontFamily: "monospace",
                            color: tokenGrowth > 100 ? C.yellow : C.textM }}>
                            {lastTokens.toLocaleString()} tok{tokenGrowth > 0 ? ` +${tokenGrowth}%` : ""}
                          </span>
                        </div>
                        <Sparkline values={tokenSeries} color={tokenGrowth > 100 ? C.yellow : C.blue} w={344} h={28} />
                      </div>
                    )}
                  </div>

                  {/* ── Section 3: Signals detected ── */}
                  <div>
                    <div style={{ fontSize: 9, color: C.textD, letterSpacing: "0.1em", marginBottom: 8 }}>SIGNALS DETECTED</div>
                    {(!runDetail.signals || runDetail.signals.length === 0) ? (
                      <div style={{ fontSize: 11, color: C.green }}>✓ No signals — clean run</div>
                    ) : (
                      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                        {runDetail.signals.map(s => {
                          const col = SEV_COLOR[s.severity] || C.textM;
                          return (
                            <div key={s.id} style={{
                              background: `${col}11`, border: `1px solid ${col}33`,
                              borderRadius: 6, padding: "10px 12px",
                            }}>
                              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                                <span style={{ fontSize: 10, fontWeight: 700, color: col }}>
                                  {s.failure_type.replace(/_/g, " ")}
                                </span>
                                <span style={{ fontSize: 9, color: col, opacity: 0.7 }}>{s.severity}</span>
                              </div>
                              {s.evidence_summary && (
                                <div style={{ fontSize: 10, color: C.textM, lineHeight: 1.5, marginBottom: 4 }}>
                                  {s.evidence_summary}
                                </div>
                              )}
                              {s.what && s.what !== s.evidence_summary && (
                                <div style={{ fontSize: 10, color: C.textD, lineHeight: 1.5, marginBottom: 4 }}>
                                  {s.what}
                                </div>
                              )}
                              {s.suggested_fixes?.[0] && (
                                <details style={{ marginTop: 6 }}>
                                  <summary style={{ fontSize: 9, color: col, cursor: "pointer", opacity: 0.8, userSelect: "none" }}>
                                    Fix: {s.suggested_fixes[0].description}
                                  </summary>
                                  <pre style={{
                                    fontSize: 9, color: "#86efac", margin: "6px 0 0",
                                    background: "rgba(0,0,0,0.4)", padding: "8px 10px",
                                    borderRadius: 4, overflowX: "auto", lineHeight: 1.6,
                                  }}>
                                    {s.suggested_fixes[0].code}
                                  </pre>
                                </details>
                              )}
                              <div style={{ fontSize: 9, color: C.textD, marginTop: 4 }}>
                                step {s.step_index} · confidence {Math.round(s.confidence * 100)}%
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>

                  {/* ── Section 4: Timeline (collapsed) ── */}
                  {(runDetail.events?.length > 0) && (
                    <div style={{ borderTop: `1px solid ${C.border}`, paddingTop: 12 }}>
                      <button
                        onClick={() => setTimelineExpanded(x => !x)}
                        style={{
                          width: "100%", textAlign: "left", display: "flex", alignItems: "center", gap: 6,
                          background: "none", border: `1px solid ${C.border}`, color: C.textM,
                          borderRadius: 4, padding: "6px 12px", cursor: "pointer",
                          fontSize: 10, fontFamily: "inherit",
                        }}>
                        <span style={{ fontSize: 9 }}>{timelineExpanded ? "▼" : "▶"}</span>
                        <span>{timelineExpanded ? "Hide timeline" : "Show timeline"}</span>
                      </button>
                      {timelineExpanded && (
                        <div style={{ marginTop: 12 }}>
                          <Timeline run={runDetail} />
                        </div>
                      )}
                    </div>
                  )}
                </>
              );
            })()}
          </div>
        )}
      </div>
    </div>
  );
}
