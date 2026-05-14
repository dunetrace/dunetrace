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
function fmtCost(usd) {
  if (usd == null) return "—";
  if (usd === 0) return "$0.00";
  if (usd < 0.001) return `$${usd.toFixed(5)}`;
  if (usd < 0.01)  return `$${usd.toFixed(4)}`;
  if (usd < 1)     return `$${usd.toFixed(3)}`;
  if (usd < 1000)  return `$${usd.toFixed(2)}`;
  return `$${Math.round(usd).toLocaleString()}`;
}

function verdictColor(verdict, C) {
  if (verdict === "verified")        return C.green;
  if (verdict === "likely_fixed")    return C.yellow;
  if (verdict === "still_occurring") return C.red;
  return C.textD;
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

const DETECTOR_COLORS = [
  C.orange, C.blue, C.red, C.green, C.yellow, C.purple,
  "#06B6D4", "#F97316", "#8B5CF6", "#EC4899",
];

function DeployTimeline({ failureRates, deployEvents }) {
  if (!failureRates || failureRates.length === 0) return null;

  const W = 620, H = 160, PAD_L = 36, PAD_R = 12, PAD_T = 12, PAD_B = 28;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  // Build sorted day list (last 30 days)
  const daySet = new Set(failureRates.map(r => r.day));
  const days   = Array.from(daySet).sort();
  if (days.length < 2) return null;

  const dayIdx = Object.fromEntries(days.map((d, i) => [d, i]));
  const xOf    = i => PAD_L + (i / (days.length - 1)) * plotW;

  // Group by failure_type
  const byType = {};
  failureRates.forEach(r => {
    if (!byType[r.failure_type]) byType[r.failure_type] = {};
    byType[r.failure_type][r.day] = r.rate;
  });
  const types = Object.keys(byType);

  // y scale: 0..maxRate, min ceiling 0.2
  const maxRate = Math.max(0.2, ...failureRates.map(r => r.rate));
  const yOf     = rate => PAD_T + plotH - (rate / maxRate) * plotH;

  // Deploy event x positions (clamp to day range)
  const dayTs  = days.map(d => new Date(d).getTime());
  const minTs  = dayTs[0], maxTs = dayTs[dayTs.length - 1];
  const deploys = (deployEvents || []).filter(d => {
    const ts = d.deployed_at * 1000;
    return ts >= minTs - 86400000 && ts <= maxTs + 86400000;
  });

  return (
    <div style={{ marginTop: 16 }}>
      <div style={{ fontSize: 9, color: C.textD, letterSpacing: "0.15em", marginBottom: 8 }}>
        DETECTOR TIMELINE — failure rate per day
        {deploys.length > 0 && (
          <span style={{ marginLeft: 10, color: "#60A5FA" }}>
            ┊ deploy markers
          </span>
        )}
      </div>
      <svg width={W} height={H} style={{ display: "block", overflow: "visible" }}>
        {/* y-axis grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map(f => {
          const rate = maxRate * f;
          const y    = yOf(rate);
          return (
            <g key={f}>
              <line x1={PAD_L} y1={y} x2={W - PAD_R} y2={y}
                stroke={C.border} strokeWidth="0.5" strokeDasharray="3,3" />
              <text x={PAD_L - 4} y={y + 3} textAnchor="end"
                fontSize="7" fill={C.textD} fontFamily="monospace">
                {Math.round(rate * 100)}%
              </text>
            </g>
          );
        })}

        {/* detector rate lines */}
        {types.map((ft, ti) => {
          const color = DETECTOR_COLORS[ti % DETECTOR_COLORS.length];
          const pts   = days
            .map((d, i) => {
              const r = byType[ft][d];
              return r != null ? `${xOf(i)},${yOf(r)}` : null;
            })
            .filter(Boolean);
          if (pts.length < 2) return null;
          return (
            <polyline key={ft} points={pts.join(" ")}
              fill="none" stroke={color} strokeWidth="1.5"
              strokeLinejoin="round" strokeLinecap="round" opacity="0.85" />
          );
        })}

        {/* deploy marker lines */}
        {deploys.map((dep, di) => {
          const ts  = dep.deployed_at * 1000;
          const frac = Math.max(0, Math.min(1, (ts - minTs) / (maxTs - minTs)));
          const x    = PAD_L + frac * plotW;
          return (
            <g key={dep.id}>
              <line x1={x} y1={PAD_T} x2={x} y2={H - PAD_B}
                stroke="#60A5FA" strokeWidth="1.5" strokeDasharray="4,3" opacity="0.8" />
              <text x={x + 3} y={PAD_T + 9} fontSize="7" fill="#60A5FA"
                fontFamily="monospace" style={{ pointerEvents: "none" }}>
                {dep.version}
              </text>
            </g>
          );
        })}

        {/* x-axis day labels — show ~5 evenly spaced */}
        {days.filter((_, i) => {
          const step = Math.max(1, Math.floor(days.length / 5));
          return i % step === 0 || i === days.length - 1;
        }).map((d, _, arr) => {
          const i = dayIdx[d];
          return (
            <text key={d} x={xOf(i)} y={H - 4} textAnchor="middle"
              fontSize="7" fill={C.textD} fontFamily="monospace">
              {d.slice(5)}
            </text>
          );
        })}
      </svg>

      {/* legend */}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 6 }}>
        {types.map((ft, ti) => (
          <div key={ft} style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <div style={{ width: 16, height: 2, background: DETECTOR_COLORS[ti % DETECTOR_COLORS.length], borderRadius: 1 }} />
            <span style={{ fontSize: 8, color: C.textD, fontFamily: "monospace" }}>
              {ft.replace(/_/g, " ")}
            </span>
          </div>
        ))}
        {deploys.length > 0 && (
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <div style={{ width: 16, height: 2, background: "#60A5FA", borderRadius: 1, borderTop: "1px dashed #60A5FA" }} />
            <span style={{ fontSize: 8, color: "#60A5FA", fontFamily: "monospace" }}>deploy</span>
          </div>
        )}
      </div>
    </div>
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

async function apiPost(path, body) {
  const r = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || `${r.status} ${path}`);
  }
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
  const [graphRunId,       setGraphRunId]       = useState(null);
  const [insights,              setInsights]              = useState(null);
  const [selectedFailureType,   setSelectedFailureType]   = useState(null);
  const [failurePattern,        setFailurePattern]        = useState(null);
  const [failurePatternLoading, setFailurePatternLoading] = useState(false);
  const PAGE_SIZE = 15;

  const [activeView,    setActiveView]    = useState("runs");   // "runs" | "patterns" | "fixes"
  const [patterns,      setPatterns]      = useState(null);
  const [agentFixes,    setAgentFixes]    = useState([]);
  const [fixesLoading,  setFixesLoading]  = useState(false);

  // ── Autofix state ──────────────────────────────────────────────────────────
  // explainStates[signalId] = { loading, error, rootCause, fixContent,
  //                              langfusePromptName, langfusePromptVersion }
  const [explainStates, setExplainStates] = useState({});
  // copyFeedback[signalId] = true while "Copied ✓" is showing
  const [copyFeedback,  setCopyFeedback]  = useState({});
  // applyConfirm = null | { signalId, promptName, fixContent }
  const [applyConfirm,  setApplyConfirm]  = useState(null);
  // applyResults[signalId] = { loading, version, url, error }
  const [applyResults,     setApplyResults]     = useState({});
  // null = unknown (loading), true/false once /v1/config responds
  const [langfuseEnabled,  setLangfuseEnabled]  = useState(null);

  // Fetch server config once on mount to know whether Langfuse is wired up
  useEffect(() => {
    apiFetch("/v1/config")
      .then(d => setLangfuseEnabled(!!d.langfuse_configured))
      .catch(() => setLangfuseEnabled(false));
  }, []);

  // Deep-link: /runs/<run_id> from Slack alert → auto-open that run
  useEffect(() => {
    const match = window.location.pathname.match(/^\/runs\/([^/]+)$/);
    if (!match) return;
    const runId = match[1];
    // Don't setSelectedAgent — its useEffect calls setSelectedRun(null) and would clear our selection.
    // Just set selectedRun directly; the detail panel fetches everything it needs from run_id alone.
    setSelectedRun({ run_id: runId });
    window.history.replaceState(null, "", "/");
  }, []);

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

  // Load runs + signals + insights + fixes when agent selected
  const loadAgentData = useCallback(async (agentId) => {
    if (!agentId) { setRuns([]); setSignals([]); setInsights(null); setAgentFixes([]); return; }
    setLoading(true);
    try {
      const [rd, sd, id, fx] = await Promise.all([
        apiFetch(`/v1/agents/${encodeURIComponent(agentId)}/runs?limit=200`),
        apiFetch(`/v1/agents/${encodeURIComponent(agentId)}/signals?limit=500`),
        apiFetch(`/v1/agents/${encodeURIComponent(agentId)}/insights`).catch(() => null),
        apiFetch(`/v1/agents/${encodeURIComponent(agentId)}/fixes`).catch(() => null),
      ]);
      setRuns(rd.runs       || []);
      setSignals(sd.signals || []);
      setInsights(id        || null);
      setAgentFixes((fx?.fixes) || []);
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => {
    setSelectedRun(null);
    setSelectedDay(null);
    setRunsPage(0);
    setRunDetail(null);
    setInsights(null);
    setSelectedFailureType(null);
    setFailurePattern(null);
    setAgentFixes([]);
    setActiveView("runs");
    loadAgentData(selectedAgent);
  }, [selectedAgent, loadAgentData]);

  // Fetch failure pattern when a failure type is clicked
  const loadFailurePattern = useCallback(async (agentId, failureType) => {
    if (!agentId || !failureType) { setFailurePattern(null); return; }
    setFailurePatternLoading(true);
    try {
      const d = await apiFetch(
        `/v1/agents/${encodeURIComponent(agentId)}/failure-patterns/${encodeURIComponent(failureType)}`
      );
      setFailurePattern(d);
    } catch (e) { console.error(e); setFailurePattern(null); }
    finally { setFailurePatternLoading(false); }
  }, []);

  const handleFailureTypeClick = useCallback((failureType) => {
    if (selectedFailureType === failureType) {
      setSelectedFailureType(null);
      setFailurePattern(null);
    } else {
      setSelectedFailureType(failureType);
      loadFailurePattern(selectedAgent, failureType);
    }
  }, [selectedAgent, selectedFailureType, loadFailurePattern]);

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

  // Load cross-run patterns
  const loadPatterns = useCallback(async () => {
    try {
      const d = await apiFetch("/v1/patterns");
      setPatterns(d);
    } catch (e) { console.error(e); }
  }, []);

  useEffect(() => {
    loadPatterns();
    const t = setInterval(loadPatterns, 30000);
    return () => clearInterval(t);
  }, [loadPatterns]);

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

  // ── Autofix handlers ─────────────────────────────────────────────────────────
  async function handleExplain(signalId) {
    setExplainStates(s => ({ ...s, [signalId]: { loading: true } }));
    try {
      const res = await apiPost(`/v1/signals/${signalId}/explain`, {});
      setExplainStates(s => ({
        ...s,
        [signalId]: {
          loading: false,
          rootCause:             res.root_cause,
          fixContent:            res.fix_content,
          fixPatch:              res.fix_patch,
          fixType:               res.fix_type,
          applyBlocked:          res.apply_blocked,
          source:                res.source,
          langfusePromptName:    res.langfuse_prompt_name,
          langfusePromptVersion: res.langfuse_prompt_version,
        },
      }));
    } catch (e) {
      setExplainStates(s => ({ ...s, [signalId]: { loading: false, error: e.message } }));
    }
  }

  function handleCopyFix(signalId, text) {
    navigator.clipboard.writeText(text).catch(() => {});
    setCopyFeedback(s => ({ ...s, [signalId]: true }));
    setTimeout(() => setCopyFeedback(s => ({ ...s, [signalId]: false })), 2000);
    // Record clipboard path (fire-and-forget)
    const st = explainStates[signalId];
    if (st?.fixContent) {
      apiPost(`/v1/signals/${signalId}/record-copy`, {
        fix_content: st.fixContent,
        langfuse_prompt_name: st.langfusePromptName || "",
        applied_via: "clipboard",
      }).catch(() => {});
    }
  }

  async function handleApplyConfirm() {
    if (!applyConfirm) return;
    const { signalId, promptName, fixContent } = applyConfirm;
    setApplyResults(s => ({ ...s, [signalId]: { loading: true } }));
    setApplyConfirm(null);
    try {
      const res = await apiPost(`/v1/signals/${signalId}/apply-fix`, {
        fix_content: fixContent,
        langfuse_prompt_name: promptName,
        applied_via: "langfuse",
      });
      setApplyResults(s => ({
        ...s,
        [signalId]: { loading: false, version: res.new_version, url: res.prompt_url },
      }));
    } catch (e) {
      setApplyResults(s => ({ ...s, [signalId]: { loading: false, error: e.message } }));
    }
  }

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <div style={{
      background: C.bg, minHeight: "100vh", color: C.text,
      fontFamily: "'JetBrains Mono','Fira Code','SF Mono',monospace",
      fontSize: 12,
      position: "relative",
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

        <div style={{ width: 1, height: 20, background: C.border }} />
        <button onClick={() => setActiveView(v => v === "patterns" ? "runs" : "patterns")} style={{
          padding: "3px 10px", borderRadius: 4, border: "none", cursor: "pointer",
          fontSize: 10, fontFamily: "inherit", whiteSpace: "nowrap", letterSpacing: "0.04em",
          background: activeView === "patterns" ? C.blue : C.surfaceB,
          color:      activeView === "patterns" ? "#fff"  : C.textM,
        }}>
          PATTERNS
          {patterns?.agents?.length > 0 && (
            <span style={{
              marginLeft: 5, fontSize: 9, background: C.orange, color: "#fff",
              borderRadius: 8, padding: "0 5px",
            }}>
              {patterns.agents.reduce((n, a) => n + a.rows.length, 0)}
            </span>
          )}
        </button>

        {selectedAgent && (
          <button onClick={() => setActiveView(v => v === "fixes" ? "runs" : "fixes")} style={{
            padding: "3px 10px", borderRadius: 4, border: "none", cursor: "pointer",
            fontSize: 10, fontFamily: "inherit", whiteSpace: "nowrap", letterSpacing: "0.04em",
            background: activeView === "fixes" ? C.green : C.surfaceB,
            color:      activeView === "fixes" ? "#fff"  : C.textM,
          }}>
            FIXES
            {agentFixes.length > 0 && (
              <span style={{
                marginLeft: 5, fontSize: 9, borderRadius: 8, padding: "0 5px",
                background: agentFixes.some(f => f.verdict === "still_occurring") ? C.red
                          : agentFixes.some(f => f.verdict === "verified")       ? C.green
                          : C.textD,
                color: "#fff",
              }}>
                {agentFixes.length}
              </span>
            )}
          </button>
        )}

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

          {/* Cost card — agent view only, when cost data available */}
          {selectedAgent && insights?.cost_stats && insights.cost_stats.total_cost_usd > 0 && (() => {
            const cs = insights.cost_stats;
            return (
              <div style={{ background: C.surfaceB, borderRadius: 6, padding: 12, border: `1px solid ${C.border}` }}>
                <div style={{ fontSize: 9, color: C.textD, letterSpacing: "0.1em", marginBottom: 8 }}>
                  EST. API COST — 30 DAYS
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 8 }}>
                  <div>
                    <div style={{ fontSize: 9, color: C.textD, marginBottom: 3 }}>TOTAL</div>
                    <div style={{ fontSize: 16, fontWeight: 700, color: C.text }}>
                      {fmtCost(cs.total_cost_usd)}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 9, color: C.textD, marginBottom: 3 }}>WASTED</div>
                    <div style={{ fontSize: 16, fontWeight: 700, color: cs.wasted_pct > 0.3 ? C.red : C.yellow }}>
                      {fmtCost(cs.wasted_cost_usd)}
                    </div>
                    <div style={{ fontSize: 9, color: C.textD }}>
                      {Math.round(cs.wasted_pct * 100)}% of total
                    </div>
                  </div>
                </div>
                {cs.cost_by_failure_type?.slice(0, 3).map(ft => (
                  <div key={ft.failure_type} style={{ marginBottom: 4 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
                      <span style={{ fontSize: 9, color: C.textM }}>{ft.failure_type.replace(/_/g, " ")}</span>
                      <span style={{ fontSize: 9, color: C.orange, fontWeight: 600 }}>{fmtCost(ft.wasted_usd)}</span>
                    </div>
                    <MiniBar value={ft.wasted_usd} max={cs.cost_by_failure_type[0].wasted_usd} color={C.orange} width={248} height={3} />
                  </div>
                ))}
              </div>
            );
          })()}

          {/* Sparklines — agent view only */}
          {selectedAgent && days.length > 1 && (
            <div style={{ background: C.surfaceB, borderRadius: 6, padding: 12, border: `1px solid ${C.border}` }}>
              <div style={{ fontSize: 9, color: C.textD, letterSpacing: "0.1em", marginBottom: 8 }}>ERROR RATE TREND</div>
              <Sparkline values={sparkErrRates} color={C.red}    w={240} h={32} />
              <div style={{ fontSize: 9, color: C.textD, letterSpacing: "0.1em", margin: "10px 0 8px" }}>SIGNAL RATE TREND</div>
              <Sparkline values={sparkSigRates} color={C.orange} w={240} h={32} />
            </div>
          )}

          {/* Systemic patterns — agent view only */}
          {selectedAgent && insights?.systemic_patterns?.filter(p => p.is_systemic).length > 0 && (
            <div style={{ background: `${C.red}11`, border: `1px solid ${C.red}33`, borderRadius: 6, padding: 12 }}>
              <div style={{ fontSize: 9, color: C.red, letterSpacing: "0.15em", marginBottom: 8, fontWeight: 700 }}>
                SYSTEMIC PATTERNS
              </div>
              {insights.systemic_patterns.filter(p => p.is_systemic).map(p => (
                <div key={p.failure_type} style={{ marginBottom: 8, cursor: "pointer" }}
                  onClick={() => handleFailureTypeClick(p.failure_type)}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 3 }}>
                    <span style={{ fontSize: 10, color: selectedFailureType === p.failure_type ? C.red : C.text, fontWeight: selectedFailureType === p.failure_type ? 700 : 400 }}>
                      {p.failure_type.replace(/_/g, " ")}
                    </span>
                    <span style={{ fontSize: 10, color: C.red, fontWeight: 700 }}>
                      {Math.round(p.rate * 100)}% of runs
                    </span>
                  </div>
                  <div style={{ fontSize: 9, color: C.textD }}>
                    {p.affected_runs}/{p.total_runs} runs · 7 days
                  </div>
                  <MiniBar value={p.rate * 100} max={100} color={C.red} width={248} height={4} />
                </div>
              ))}
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
                  const userImpactMap = {};
                  (insights?.user_impact || []).forEach(u => { userImpactMap[u.failure_type] = u; });
                  return topFailures.map(([sig, cnt]) => {
                    const ui = userImpactMap[sig];
                    return (
                      <div key={sig} style={{ marginBottom: 8, cursor: selectedAgent ? "pointer" : "default" }}
                        onClick={() => selectedAgent && handleFailureTypeClick(sig)}>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                          <span style={{ fontSize: 10, color: selectedFailureType === sig ? C.orange : C.textM, fontWeight: selectedFailureType === sig ? 700 : 400 }}>
                            {sig.replace(/_/g, " ")}
                          </span>
                          <span style={{ fontSize: 10, color: C.orange, fontWeight: 700 }}>{cnt}</span>
                        </div>
                        <MiniBar value={cnt} max={maxV} color={C.orange} width={248} height={4} />
                        {ui && (
                          <div style={{ fontSize: 9, color: C.textD, marginTop: 2 }}>
                            <span style={{ color: ui.user_impact_rate > 0.5 ? C.red : C.textM }}>
                              {ui.affected_users} user{ui.affected_users !== 1 ? "s" : ""} affected
                            </span>
                            {" "}· {Math.round(ui.user_impact_rate * 100)}% of total
                          </div>
                        )}
                      </div>
                    );
                  });
                })()
            }
          </div>
        </div>

        {/* ── Main content ── */}
        <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column" }}>

          {/* ── Patterns view ── */}
          {activeView === "patterns" && (() => {
            const DAY_LABELS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];
            const agents     = patterns?.agents || [];

            function heatBg(count) {
              if (!count) return C.surfaceB;
              const alpha = Math.min(0.85, 0.2 + count * 0.12);
              return `rgba(232,106,43,${alpha})`;
            }

            function handlePatternRowClick(agentId, failureType) {
              setActiveView("runs");
              selectAgent(agentId);
              // handleFailureTypeClick won't have insights yet — set pending and let
              // the agent data load trigger it via a flag stored in state.
              setSelectedFailureType(failureType);
              setFailurePattern(null);
              loadFailurePattern(agentId, failureType);
            }

            if (agents.length === 0) {
              return (
                <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 12, color: C.textD }}>
                  <span style={{ fontSize: 32 }}>✓</span>
                  <div style={{ fontSize: 13, color: C.green }}>No patterns detected in the last 7 days</div>
                  <div style={{ fontSize: 10 }}>Signals will appear here when a detector fires on more than one run</div>
                </div>
              );
            }

            // Day header labels from API response (use first row's days array)
            const dayDates = agents[0]?.rows[0]?.days?.map(d => {
              const dt  = new Date(d.date + "T00:00:00Z");
              return DAY_LABELS[dt.getUTCDay() === 0 ? 6 : dt.getUTCDay() - 1];
            }) || DAY_LABELS;

            return (
              <div style={{ padding: "20px 24px", display: "flex", flexDirection: "column", gap: 24 }}>
                <div style={{ fontSize: 9, color: C.textD, letterSpacing: "0.15em" }}>
                  CROSS-RUN PATTERNS — LAST 7 DAYS · {agents.reduce((n, a) => n + a.rows.length, 0)} active detector{agents.reduce((n, a) => n + a.rows.length, 0) !== 1 ? "s" : ""}
                </div>

                {agents.map(agent => (
                  <div key={agent.agent_id} style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, overflow: "hidden" }}>

                    {/* Agent header */}
                    <div style={{ padding: "10px 16px", borderBottom: `1px solid ${C.border}`, background: C.surfaceB, display: "flex", alignItems: "center", gap: 10 }}>
                      <span style={{ fontSize: 11, fontWeight: 700, color: C.text, letterSpacing: "0.05em" }}>
                        {agent.agent_id}
                      </span>
                      <span style={{ fontSize: 9, color: C.textD }}>
                        {agent.rows.length} detector{agent.rows.length !== 1 ? "s" : ""} firing
                      </span>
                    </div>

                    {/* Heat table */}
                    <div style={{ overflowX: "auto" }}>
                      <table style={{ width: "100%", borderCollapse: "collapse" }}>
                        <thead>
                          <tr>
                            <th style={{ padding: "8px 16px", textAlign: "left", fontSize: 9, color: C.textD, fontWeight: 400, letterSpacing: "0.1em", borderBottom: `1px solid ${C.border}`, minWidth: 220 }}>
                              DETECTOR
                            </th>
                            {dayDates.map((label, i) => (
                              <th key={i} style={{ padding: "8px 12px", fontSize: 9, color: C.textD, fontWeight: 400, letterSpacing: "0.08em", textAlign: "center", borderBottom: `1px solid ${C.border}`, minWidth: 52 }}>
                                {label}
                              </th>
                            ))}
                            <th style={{ padding: "8px 16px", fontSize: 9, color: C.textD, fontWeight: 400, letterSpacing: "0.08em", textAlign: "right", borderBottom: `1px solid ${C.border}`, minWidth: 160 }}>
                              7-DAY SUMMARY
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {agent.rows.map((row, ri) => (
                            <tr key={row.failure_type}
                              onClick={() => handlePatternRowClick(agent.agent_id, row.failure_type)}
                              style={{ cursor: "pointer", borderBottom: ri < agent.rows.length - 1 ? `1px solid ${C.border}` : "none" }}
                              onMouseEnter={e => e.currentTarget.style.background = C.navy}
                              onMouseLeave={e => e.currentTarget.style.background = ""}
                            >
                              <td style={{ padding: "10px 16px" }}>
                                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                  <span style={{ fontSize: 11, color: C.text }}>
                                    {row.failure_type.replace(/_/g, " ")}
                                  </span>
                                  {row.trending_up && (
                                    <span style={{ fontSize: 9, color: C.red, background: `${C.red}22`, padding: "1px 5px", borderRadius: 3, fontWeight: 700 }}>
                                      ↑ TRENDING
                                    </span>
                                  )}
                                </div>
                              </td>
                              {row.days.map((day, di) => (
                                <td key={di} style={{
                                  padding: "10px 8px", textAlign: "center",
                                  background: heatBg(day.signal_count),
                                }}>
                                  <span style={{ fontSize: 12, fontWeight: day.signal_count ? 700 : 400, color: day.signal_count ? C.text : C.textD, fontFamily: "monospace" }}>
                                    {day.signal_count || "—"}
                                  </span>
                                </td>
                              ))}
                              <td style={{ padding: "10px 16px", textAlign: "right" }}>
                                <span style={{ fontSize: 11, color: C.orange, fontWeight: 700 }}>
                                  {row.total_occurrences}
                                </span>
                                <span style={{ fontSize: 9, color: C.textD }}> signals · </span>
                                <span style={{ fontSize: 11, color: row.pct_of_runs > 0.1 ? C.red : C.textM, fontWeight: 700 }}>
                                  {Math.round(row.pct_of_runs * 100)}%
                                </span>
                                <span style={{ fontSize: 9, color: C.textD }}> of runs</span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ))}
              </div>
            );
          })()}

          {/* ── Fixes view ── */}
          {activeView === "fixes" && selectedAgent && (
            <div style={{ padding: 20 }}>
              <div style={{ fontSize: 9, color: C.textD, letterSpacing: "0.15em", marginBottom: 16 }}>
                FIX HISTORY — {agentFixes.length} fix{agentFixes.length !== 1 ? "es" : ""} applied
              </div>
              {agentFixes.length === 0 ? (
                <div style={{ fontSize: 11, color: C.textD, padding: "40px 0", textAlign: "center" }}>
                  No fixes applied yet. Use "Explain with Langfuse" on a signal to get a fix recommendation.
                </div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {agentFixes.map(fix => {
                    const vc = verdictColor(fix.verdict, C);
                    const verdictLabel = {
                      verified:          "✓ VERIFIED",
                      likely_fixed:      "~ LIKELY FIXED",
                      still_occurring:   "✕ STILL OCCURRING",
                      insufficient_data: "? INSUFFICIENT DATA",
                    }[fix.verdict] || fix.verdict;
                    return (
                      <div key={fix.id} style={{
                        background: C.surface, border: `1px solid ${C.border}`,
                        borderRadius: 6, padding: "12px 16px",
                        borderLeft: `3px solid ${vc}`,
                      }}>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
                          <div>
                            <span style={{ fontSize: 11, fontWeight: 700, color: C.text }}>
                              {fix.failure_type.replace(/_/g, " ")}
                            </span>
                            <span style={{ fontSize: 9, color: C.textD, marginLeft: 8 }}>
                              {fix.severity}
                            </span>
                          </div>
                          <span style={{
                            fontSize: 9, padding: "2px 8px", borderRadius: 10, fontWeight: 700,
                            background: `${vc}22`, color: vc, border: `1px solid ${vc}44`,
                            letterSpacing: "0.05em", whiteSpace: "nowrap",
                          }}>
                            {verdictLabel}
                          </span>
                        </div>

                        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 8 }}>
                          <div style={{ fontSize: 9, color: C.textD }}>
                            Applied via{" "}
                            <span style={{ color: fix.applied_via === "langfuse" ? C.blue : C.textM }}>
                              {fix.applied_via}
                            </span>
                            {fix.langfuse_prompt_name && (
                              <> · prompt <span style={{ color: C.text }}>{fix.langfuse_prompt_name}</span>
                                {fix.langfuse_version && <> v{fix.langfuse_version}</>}
                              </>
                            )}
                          </div>
                          <div style={{ fontSize: 9, color: C.textD }}>
                            {new Date(fix.applied_at * 1000).toLocaleString()}
                          </div>
                        </div>

                        <div style={{ display: "flex", gap: 20, flexWrap: "wrap", marginBottom: 6 }}>
                          <div>
                            <span style={{ fontSize: 9, color: C.textD }}>Runs after fix: </span>
                            <span style={{ fontSize: 10, color: C.text, fontWeight: 600 }}>{fix.runs_after}</span>
                          </div>
                          <div>
                            <span style={{ fontSize: 9, color: C.textD }}>Recurrences: </span>
                            <span style={{
                              fontSize: 10, fontWeight: 600,
                              color: fix.recurrences_after > 0 ? C.red : C.green,
                            }}>
                              {fix.recurrences_after}
                            </span>
                          </div>
                          {fix.runs_after >= 5 && (
                            <div>
                              <span style={{ fontSize: 9, color: C.textD }}>Recurrence rate: </span>
                              <span style={{
                                fontSize: 10, fontWeight: 600,
                                color: fix.recurrences_after / fix.runs_after > 0.1 ? C.red : C.green,
                              }}>
                                {fix.runs_after > 0 ? Math.round(fix.recurrences_after / fix.runs_after * 100) : 0}%
                              </span>
                            </div>
                          )}
                        </div>

                        <div style={{ fontSize: 9, color: C.textD }}>
                          signal #{fix.signal_id} · run {fix.run_id.slice(0, 8)}…
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {activeView === "runs" && (selectedAgent ? (
            /* ── Agent view: health record + heatmap + runs table ── */
            <>
              {/* ── Deploy regression banners ── */}
              {insights?.deploy_regressions?.filter(r => r.is_regression).map(reg => (
                <div key={reg.deploy_id} style={{
                  padding: "10px 20px",
                  background: `${C.red}15`,
                  borderBottom: `1px solid ${C.red}44`,
                  display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap",
                }}>
                  <span style={{ fontSize: 11, color: C.red, fontWeight: 700 }}>⚠ REGRESSION DETECTED</span>
                  <span style={{ fontSize: 10, color: C.textM }}>
                    Deploy <span style={{ color: C.text, fontWeight: 600 }}>{reg.version}</span>
                    {" "}increased signal rate{" "}
                    <span style={{ color: C.textM }}>{Math.round(reg.before_rate * 100)}%</span>
                    {" → "}
                    <span style={{ color: C.red, fontWeight: 700 }}>{Math.round(reg.after_rate * 100)}%</span>
                    {" "}
                    <span style={{ color: C.red }}>(+{Math.round(reg.delta_rate * 100)}pp)</span>
                  </span>
                  <span style={{ fontSize: 9, color: C.textD }}>
                    {reg.before_runs} runs before · {reg.after_runs} runs after
                    {" · "}{new Date(reg.deployed_at * 1000).toLocaleString()}
                  </span>
                </div>
              ))}

              {/* ── Health record: failure rate trends ── */}
              {insights && (insights.systemic_patterns?.length > 0 || insights.failure_rates?.length > 0) && (
                <div style={{
                  padding: "14px 20px",
                  borderBottom: `1px solid ${C.border}`,
                  background: C.surface,
                }}>
                  <div style={{ fontSize: 9, color: C.textD, letterSpacing: "0.15em", marginBottom: 10 }}>
                    STRUCTURAL HEALTH RECORD — 30 DAYS
                  </div>
                  <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                    {(insights.systemic_patterns || []).map(p => {
                      const isSys  = p.is_systemic;
                      const color  = isSys ? C.red : p.rate > 0.05 ? C.yellow : C.textM;
                      // Build a 30-day sparkline from failure_rates for this type
                      const pts    = (insights.failure_rates || [])
                        .filter(r => r.failure_type === p.failure_type)
                        .sort((a, b) => a.day < b.day ? -1 : 1)
                        .map(r => Math.round(r.rate * 100));
                      return (
                        <div key={p.failure_type} style={{
                          background: isSys ? `${C.red}11` : C.surfaceB,
                          border: `1px solid ${isSys ? C.red + "44" : C.border}`,
                          borderRadius: 6, padding: "10px 12px", minWidth: 160,
                        }}>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                            <span style={{ fontSize: 9, color: isSys ? C.red : C.textM, fontWeight: isSys ? 700 : 400 }}>
                              {p.failure_type.replace(/_/g, " ")}
                            </span>
                            {isSys && (
                              <span style={{ fontSize: 8, color: C.red, background: `${C.red}22`, padding: "1px 5px", borderRadius: 3 }}>
                                SYSTEMIC
                              </span>
                            )}
                          </div>
                          <div style={{ fontSize: 18, fontWeight: 700, color, marginBottom: 4 }}>
                            {Math.round(p.rate * 100)}%
                          </div>
                          <div style={{ fontSize: 9, color: C.textD, marginBottom: pts.length > 1 ? 6 : 0 }}>
                            {p.affected_runs}/{p.total_runs} runs · 7d
                          </div>
                          {pts.length > 1 && (
                            <Sparkline values={pts} color={color} w={136} h={20} />
                          )}
                        </div>
                      );
                    })}
                  </div>
                  <DeployTimeline
                    failureRates={insights.failure_rates || []}
                    deployEvents={insights.deploy_events || []}
                  />
                </div>
              )}

              {/* ── Why is this happening? — failure pattern deep-dive ── */}
              {selectedFailureType && (
                <div style={{ padding: "14px 20px", borderBottom: `1px solid ${C.border}`, background: C.surface }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                    <div>
                      <span style={{ fontSize: 9, color: C.textD, letterSpacing: "0.15em" }}>WHY IS THIS HAPPENING? — </span>
                      <span style={{ fontSize: 9, color: C.orange, letterSpacing: "0.15em", fontWeight: 700 }}>
                        {selectedFailureType.replace(/_/g, " ")}
                      </span>
                    </div>
                    <button onClick={() => { setSelectedFailureType(null); setFailurePattern(null); }}
                      style={{ background: "none", border: `1px solid ${C.border}`, color: C.textM, borderRadius: 4, padding: "2px 10px", cursor: "pointer", fontSize: 10, fontFamily: "inherit" }}>
                      ✕ close
                    </button>
                  </div>

                  {failurePatternLoading && (
                    <div style={{ fontSize: 10, color: C.textD }}>Loading pattern data…</div>
                  )}

                  {!failurePatternLoading && failurePattern && (() => {
                    const fp = failurePattern;
                    const ov = fp.overview;
                    const sd = fp.step_distribution;
                    return (
                      <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>

                        {/* Overview card */}
                        <div style={{ background: C.surfaceB, border: `1px solid ${C.border}`, borderRadius: 6, padding: "10px 14px", minWidth: 160 }}>
                          <div style={{ fontSize: 9, color: C.textD, marginBottom: 6 }}>OVERVIEW</div>
                          <div style={{ fontSize: 22, fontWeight: 700, color: C.orange, marginBottom: 2 }}>
                            {Math.round(ov.rate * 100)}%
                          </div>
                          <div style={{ fontSize: 9, color: C.textD, marginBottom: 6 }}>
                            {ov.affected_runs}/{ov.total_runs} runs affected
                          </div>
                          <div style={{ fontSize: 9, color: C.textD, marginBottom: 2 }}>
                            avg confidence: <span style={{ color: C.text }}>{Math.round(ov.avg_confidence * 100)}%</span>
                          </div>
                          <div style={{ display: "flex", gap: 4, marginTop: 6, flexWrap: "wrap" }}>
                            {Object.entries(ov.severity_breakdown).filter(([,v]) => v > 0).map(([sev, cnt]) => (
                              <span key={sev} style={{ fontSize: 8, padding: "1px 5px", borderRadius: 3, background: sev === "CRITICAL" ? `${C.red}22` : sev === "HIGH" ? `${C.orange}22` : `${C.textD}22`, color: sev === "CRITICAL" ? C.red : sev === "HIGH" ? C.orange : C.textD }}>
                                {sev} {cnt}
                              </span>
                            ))}
                          </div>
                        </div>

                        {/* Step distribution */}
                        {sd.p50 != null && (
                          <div style={{ background: C.surfaceB, border: `1px solid ${C.border}`, borderRadius: 6, padding: "10px 14px", minWidth: 140 }}>
                            <div style={{ fontSize: 9, color: C.textD, marginBottom: 6 }}>FIRES AT STEP</div>
                            <div style={{ fontSize: 22, fontWeight: 700, color: C.text, marginBottom: 2 }}>~{Math.round(sd.p50)}</div>
                            <div style={{ fontSize: 9, color: C.textD }}>median (P50)</div>
                            <div style={{ fontSize: 9, color: C.textD, marginTop: 6 }}>
                              P25 <span style={{ color: C.text }}>{Math.round(sd.p25 ?? 0)}</span>
                              {"  ·  "}
                              P75 <span style={{ color: C.text }}>{Math.round(sd.p75 ?? 0)}</span>
                            </div>
                            <div style={{ fontSize: 9, color: C.textD, marginTop: 2 }}>
                              avg <span style={{ color: C.text }}>{sd.avg_step?.toFixed(1)}</span>
                            </div>
                          </div>
                        )}

                        {/* Evidence aggregates */}
                        {fp.evidence_aggregates.length > 0 && (
                          <div style={{ background: C.surfaceB, border: `1px solid ${C.border}`, borderRadius: 6, padding: "10px 14px", minWidth: 180 }}>
                            <div style={{ fontSize: 9, color: C.textD, marginBottom: 6 }}>EVIDENCE PATTERNS</div>
                            {fp.evidence_aggregates.slice(0, 3).map((ev, i) => (
                              <div key={i} style={{ marginBottom: 6, paddingBottom: 6, borderBottom: i < Math.min(fp.evidence_aggregates.length, 3) - 1 ? `1px solid ${C.border}` : "none" }}>
                                {ev.tool && <div style={{ fontSize: 10, color: C.text, fontWeight: 600, marginBottom: 2 }}>{ev.tool}</div>}
                                {ev.index_name && <div style={{ fontSize: 10, color: C.text, fontWeight: 600, marginBottom: 2 }}>{ev.index_name}</div>}
                                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                                  {ev.avg_count != null && <span style={{ fontSize: 9, color: C.textD }}>avg calls: <span style={{ color: C.orange }}>{ev.avg_count}</span></span>}
                                  {ev.avg_consecutive_fails != null && <span style={{ fontSize: 9, color: C.textD }}>consec fails: <span style={{ color: C.red }}>{ev.avg_consecutive_fails}</span></span>}
                                  {ev.args_identical_rate != null && <span style={{ fontSize: 9, color: C.textD }}>same args: <span style={{ color: C.text }}>{Math.round(ev.args_identical_rate * 100)}%</span></span>}
                                  {ev.avg_growth_factor != null && <span style={{ fontSize: 9, color: C.textD }}>token growth: <span style={{ color: C.orange }}>{ev.avg_growth_factor}×</span></span>}
                                  {ev.avg_duration_ms != null && <span style={{ fontSize: 9, color: C.textD }}>avg duration: <span style={{ color: C.text }}>{(ev.avg_duration_ms / 1000).toFixed(1)}s</span></span>}
                                  {ev.avg_ratio != null && <span style={{ fontSize: 9, color: C.textD }}>ratio: <span style={{ color: C.orange }}>{ev.avg_ratio}×</span></span>}
                                  {ev.avg_top_score != null && <span style={{ fontSize: 9, color: C.textD }}>top score: <span style={{ color: C.text }}>{ev.avg_top_score}</span></span>}
                                  {ev.avg_stall_steps != null && <span style={{ fontSize: 9, color: C.textD }}>stall steps: <span style={{ color: C.text }}>{ev.avg_stall_steps}</span></span>}
                                  {ev.avg_inflation_ratio != null && <span style={{ fontSize: 9, color: C.textD }}>inflation: <span style={{ color: C.orange }}>{ev.avg_inflation_ratio}×</span></span>}
                                </div>
                                <div style={{ fontSize: 9, color: C.textD, marginTop: 2 }}>
                                  {ev.sample_count} signal{ev.sample_count !== 1 ? "s" : ""}
                                </div>
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Co-occurring failures */}
                        {fp.co_occurring.length > 0 && (
                          <div style={{ background: C.surfaceB, border: `1px solid ${C.border}`, borderRadius: 6, padding: "10px 14px", minWidth: 160 }}>
                            <div style={{ fontSize: 9, color: C.textD, marginBottom: 6 }}>CO-OCCURS WITH</div>
                            {fp.co_occurring.slice(0, 5).map(c => (
                              <div key={c.failure_type} style={{ marginBottom: 5 }}>
                                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
                                  <span style={{ fontSize: 9, color: C.textM }}>{c.failure_type.replace(/_/g, " ")}</span>
                                  <span style={{ fontSize: 9, color: C.orange, fontWeight: 700 }}>{Math.round(c.co_rate * 100)}%</span>
                                </div>
                                <MiniBar value={c.co_rate * 100} max={100} color={C.orange} width={140} height={3} />
                              </div>
                            ))}
                          </div>
                        )}

                        {/* 14-day trend */}
                        {fp.daily_trend.length > 0 && (() => {
                          const trendVals = fp.daily_trend.map(d => Math.round(d.rate * 100));
                          return (
                            <div style={{ background: C.surfaceB, border: `1px solid ${C.border}`, borderRadius: 6, padding: "10px 14px", minWidth: 160 }}>
                              <div style={{ fontSize: 9, color: C.textD, marginBottom: 6 }}>14-DAY TREND</div>
                              <Sparkline values={trendVals} color={C.orange} w={140} h={32} />
                              <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
                                <span style={{ fontSize: 8, color: C.textD }}>{fp.daily_trend[0]?.day?.slice(5)}</span>
                                <span style={{ fontSize: 8, color: C.textD }}>{fp.daily_trend[fp.daily_trend.length - 1]?.day?.slice(5)}</span>
                              </div>
                            </div>
                          );
                        })()}

                        {/* Top runs */}
                        {fp.top_runs.length > 0 && (
                          <div style={{ background: C.surfaceB, border: `1px solid ${C.border}`, borderRadius: 6, padding: "10px 14px", minWidth: 200 }}>
                            <div style={{ fontSize: 9, color: C.textD, marginBottom: 6 }}>HIGHEST CONFIDENCE RUNS</div>
                            {fp.top_runs.map(r => (
                              <div key={r.run_id} style={{ marginBottom: 6, cursor: "pointer" }}
                                onClick={() => setSelectedRun({ run_id: r.run_id })}>
                                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 1 }}>
                                  <span style={{ fontSize: 9, color: C.blue, textDecoration: "underline" }}>
                                    {r.run_id.slice(0, 8)}…
                                  </span>
                                  <span style={{ fontSize: 9, color: r.severity === "CRITICAL" ? C.red : C.orange, fontWeight: 700 }}>
                                    {Math.round(r.confidence * 100)}%
                                  </span>
                                </div>
                                <div style={{ fontSize: 8, color: C.textD }}>step {r.step_index} · {r.severity}</div>
                              </div>
                            ))}
                          </div>
                        )}

                      </div>
                    );
                  })()}
                </div>
              )}

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
                    gridTemplateColumns: "200px 70px 70px 60px 80px 130px 1fr",
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
                      ["TIME",     "started_at"],
                      ["VERSION",  "agent_version"],
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
                          gridTemplateColumns: "200px 70px 70px 60px 80px 130px 1fr",
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
                        <span style={{ fontSize: 10, color: C.textD }}>
                          {fmtDate(toMs(r.started_at))} {fmtTime(toMs(r.started_at))}
                        </span>
                        <span style={{ fontSize: 10, color: C.textD, fontFamily: "monospace" }}>
                          {r.agent_version || "—"}
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
                {["AGENT", "RUNS", "SIGNALS", "HIGH", "LAST SEEN", "FAILURE TYPES", ""].map(h => (
                  <div key={h} style={{ fontSize: 9, color: C.textD, letterSpacing: "0.12em" }}>{h}</div>
                ))}
              </div>
              {agents.map(a => (
                <div key={a.agent_id} onClick={() => selectAgent(a.agent_id)}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "220px 70px 80px 70px 90px 1fr 80px",
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
                  <div>
                    {a.signal_count > 0 && (
                      <span style={{
                        fontSize: 9, padding: "2px 6px", borderRadius: 3,
                        background: `${C.textD}22`, color: C.textD,
                        fontFamily: "monospace",
                      }}>health →</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ))}
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
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <button
                  onClick={() => setGraphRunId(selectedRun.run_id)}
                  style={{
                    background: C.surfaceB, border: `1px solid ${C.border}`,
                    color: C.textM, borderRadius: 4, padding: "3px 10px",
                    cursor: "pointer", fontSize: 10, fontFamily: "inherit",
                    display: "flex", alignItems: "center", gap: 5,
                  }}
                >⬡ View Graph</button>
                <button onClick={() => { setSelectedRun(null); setTimelineExpanded(false); }}
                  style={{ background: "none", border: "none", color: C.textD, cursor: "pointer", fontSize: 16 }}>✕</button>
              </div>
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
                          const col  = SEV_COLOR[s.severity] || C.textM;
                          const exSt = explainStates[s.id] || {};
                          const apSt = applyResults[s.id]  || {};
                          const staticFix = s.suggested_fixes?.[0]?.code || "";
                          const llmFix    = exSt.fixContent || "";
                          const bestFix   = llmFix || staticFix;
                          return (
                            <div key={s.id} style={{
                              background: `${col}11`, border: `1px solid ${col}33`,
                              borderRadius: 6, padding: "10px 12px",
                            }}>
                              {/* Header */}
                              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                                <span style={{ fontSize: 10, fontWeight: 700, color: col }}>
                                  {s.failure_type.replace(/_/g, " ")}
                                </span>
                                <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                                  {s.co_signal_count >= 2 && (
                                    <span style={{
                                      fontSize: 8, padding: "1px 5px", borderRadius: 3,
                                      background: `${C.orange}22`, border: `1px solid ${C.orange}55`,
                                      color: C.orange, letterSpacing: "0.04em",
                                    }}>
                                      confidence boosted · {s.co_signal_count} co-occurring signals
                                    </span>
                                  )}
                                  <span style={{ fontSize: 9, color: col, opacity: 0.7 }}>{s.severity}</span>
                                </div>
                              </div>

                              {/* Static evidence */}
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

                              {/* Static fix (collapsed, only before explain) */}
                              {!exSt.rootCause && s.suggested_fixes?.[0] && (
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

                              {/* Action bar — always visible, anchors the card bottom */}
                              <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap", alignItems: "center" }}>
                                {!exSt.rootCause && !exSt.loading && (
                                  langfuseEnabled
                                    ? (
                                      <button
                                        onClick={() => handleExplain(s.id)}
                                        style={{
                                          fontSize: 9, padding: "3px 8px", cursor: "pointer",
                                          background: "none", border: `1px solid ${col}55`,
                                          color: col, borderRadius: 3, fontFamily: "inherit",
                                        }}
                                      >Explain with Langfuse</button>
                                    ) : langfuseEnabled === false ? (
                                      <span style={{ fontSize: 9, color: C.textD, fontStyle: "italic" }}>
                                        Connect Langfuse to enable deep analysis
                                      </span>
                                    ) : null
                                )}
                                {exSt.loading && (
                                  <span style={{ fontSize: 9, color: C.textD }}>
                                    <span style={{
                                      display: "inline-block", width: 8, height: 8,
                                      border: `1px solid ${C.textD}`, borderTopColor: "transparent",
                                      borderRadius: "50%", marginRight: 4,
                                      animation: "spin 0.7s linear infinite",
                                    }} />
                                    analysing…
                                  </span>
                                )}
                                {exSt.error && (
                                  <span style={{ fontSize: 9, color: C.red }}>{exSt.error}</span>
                                )}
                                {bestFix && (
                                  <button
                                    onClick={() => handleCopyFix(s.id, bestFix)}
                                    style={{
                                      fontSize: 9, padding: "3px 8px", cursor: "pointer",
                                      background: copyFeedback[s.id] ? `${C.green}22` : "none",
                                      border: `1px solid ${copyFeedback[s.id] ? C.green : C.border}`,
                                      color: copyFeedback[s.id] ? C.green : C.textM,
                                      borderRadius: 3, fontFamily: "inherit",
                                      transition: "all 0.15s",
                                    }}
                                  >{copyFeedback[s.id] ? "Copied ✓" : "Copy fix"}</button>
                                )}
                                {!exSt.applyBlocked && exSt.langfusePromptName && llmFix && !apSt.version && !apSt.loading && (
                                  <button
                                    onClick={() => setApplyConfirm({
                                      signalId: s.id,
                                      promptName: exSt.langfusePromptName,
                                      fixContent: llmFix,
                                    })}
                                    style={{
                                      fontSize: 9, padding: "3px 8px", cursor: "pointer",
                                      background: `${C.blue}15`,
                                      border: `1px solid ${C.blue}44`,
                                      color: C.blue, borderRadius: 3, fontFamily: "inherit",
                                    }}
                                  >Apply via Langfuse</button>
                                )}
                                {apSt.loading && <span style={{ fontSize: 9, color: C.textD }}>applying…</span>}
                                {apSt.version && (
                                  <a href={apSt.url} target="_blank" rel="noopener noreferrer"
                                    style={{ fontSize: 9, color: C.green, textDecoration: "none", borderBottom: `1px solid ${C.green}44` }}>
                                    Applied as v{apSt.version} in Langfuse ↗
                                  </a>
                                )}
                                {apSt.error && <span style={{ fontSize: 9, color: C.red }}>{apSt.error}</span>}
                              </div>

                              {/* Results — expand downward below the action bar */}
                              {exSt.rootCause && (
                                <div style={{ marginTop: 8 }}>
                                  <div style={{
                                    fontSize: 10, color: C.text, lineHeight: 1.6,
                                    background: "rgba(0,0,0,0.3)", borderRadius: 4,
                                    padding: "8px 10px", marginBottom: 4,
                                  }}>
                                    <span style={{ color: C.orange, fontSize: 9, letterSpacing: "0.08em" }}>ROOT CAUSE  </span>
                                    {exSt.rootCause}
                                  </div>
                                  {exSt.source === "signal_only" && (
                                    <div style={{ fontSize: 9, color: C.textD, marginBottom: 4, fontStyle: "italic" }}>
                                      No Langfuse trace found — analysis based on signal evidence only.
                                    </div>
                                  )}
                                  {llmFix && (() => {
                                    const isCode     = exSt.fixType === "code_change";
                                    const isSecurity = exSt.fixType === "no_auto_apply";
                                    const label       = isCode ? "CODE CHANGE NEEDED" : isSecurity ? "SECURITY FIX (review manually)" : "SUGGESTED FIX";
                                    const border      = isCode ? C.yellow : isSecurity ? C.red : C.green;
                                    const textCol     = isCode ? "#fde68a" : isSecurity ? "#fca5a5" : "#86efac";
                                    return (
                                      <div style={{
                                        fontSize: 10, color: textCol,
                                        background: "rgba(0,0,0,0.4)", borderRadius: 4,
                                        padding: "8px 10px", borderLeft: `2px solid ${border}`,
                                        marginBottom: 4,
                                      }}>
                                        <span style={{ color: border, fontSize: 9, letterSpacing: "0.08em" }}>{label}  </span>
                                        {llmFix}
                                      </div>
                                    );
                                  })()}
                                  {!apSt.version && !apSt.loading && (() => {
                                    if (exSt.fixType === "no_auto_apply")
                                      return <span style={{ fontSize: 9, color: C.red, fontStyle: "italic" }}>Security signal — review and apply manually.</span>;
                                    if (exSt.fixType === "code_change")
                                      return <span style={{ fontSize: 9, color: C.textD, fontStyle: "italic" }}>Code/infra fix — apply manually.</span>;
                                    if (!exSt.langfusePromptName)
                                      return <span style={{ fontSize: 9, color: C.textD, fontStyle: "italic" }}>To auto-apply, manage prompts through Langfuse.</span>;
                                    return null;
                                  })()}
                                </div>
                              )}

                              <div style={{ fontSize: 9, color: C.textD, marginTop: 6 }}>
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

      {/* ── Apply-fix confirm dialog ── */}
      {applyConfirm && (
        <div style={{
          position: "fixed", inset: 0, zIndex: 200,
          background: "rgba(8,11,17,0.85)",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <div style={{
            background: C.surface, border: `1px solid ${C.border}`,
            borderRadius: 8, padding: "20px 24px", maxWidth: 480, width: "90%",
          }}>
            <div style={{ fontSize: 10, color: C.orange, letterSpacing: "0.1em", marginBottom: 12 }}>
              CONFIRM FIX
            </div>
            <div style={{ fontSize: 11, color: C.textM, lineHeight: 1.6, marginBottom: 12 }}>
              This will create a new version of{" "}
              <span style={{ color: C.text, fontWeight: 600 }}>{applyConfirm.promptName}</span>{" "}
              in Langfuse. Your next agent run will use the updated prompt.
            </div>
            <div style={{
              fontSize: 10, color: "#86efac",
              background: "rgba(0,0,0,0.4)", borderRadius: 4,
              padding: "8px 10px", marginBottom: 16,
              borderLeft: `2px solid ${C.green}`,
            }}>
              + {applyConfirm.fixContent}
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button
                onClick={handleApplyConfirm}
                style={{
                  padding: "6px 16px", fontSize: 10, cursor: "pointer",
                  background: `${C.green}22`, border: `1px solid ${C.green}66`,
                  color: C.green, borderRadius: 4, fontFamily: "inherit",
                }}
              >Confirm</button>
              <button
                onClick={() => setApplyConfirm(null)}
                style={{
                  padding: "6px 16px", fontSize: 10, cursor: "pointer",
                  background: "none", border: `1px solid ${C.border}`,
                  color: C.textM, borderRadius: 4, fontFamily: "inherit",
                }}
              >Cancel</button>
            </div>
          </div>
        </div>
      )}

      {/* ── Graph overlay ── */}
      {graphRunId && (
        <div style={{
          position: "fixed", inset: 0, zIndex: 100,
          background: "rgba(8,11,17,0.92)",
          display: "flex", flexDirection: "column",
        }}>
          {/* Close bar */}
          <div style={{
            height: 36, background: "#0F1824", borderBottom: "1px solid #1C2D45",
            display: "flex", alignItems: "center", padding: "0 16px", gap: 12, flexShrink: 0,
          }}>
            <span style={{ fontSize: 10, color: "#475B75", fontFamily: "monospace" }}>
              graph · {graphRunId}
            </span>
            <div style={{ flex: 1 }} />
            <button
              onClick={() => setGraphRunId(null)}
              style={{
                background: "none", border: "1px solid #1C2D45", color: "#94A3B8",
                borderRadius: 4, padding: "2px 12px", cursor: "pointer",
                fontSize: 10, fontFamily: "monospace",
              }}
            >✕ close</button>
          </div>
          <div style={{ flex: 1, overflow: "hidden" }}>
            <DunetraceGraph runId={graphRunId} apiBase="http://localhost:8002" token="dev" />
          </div>
        </div>
      )}
    </div>
  );
}
