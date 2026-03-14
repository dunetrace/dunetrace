const { useState, useRef, useEffect, useCallback } = React;

// ── Brand tokens ──────────────────────────────────────────────────────────────
const C = {
  bg:      "#080B11",
  surface: "#0F1824",
  surf2:   "#131D2E",
  border:  "#1C2D45",
  orange:  "#E86A2B",
  orangeD: "#7A2E0A",
  white:   "#FFFFFF",
  light:   "#E2E8F0",
  mid:     "#94A3B8",
  dim:     "#475B75",
  dim2:    "#2A3A50",
  red:     "#EF4444",
  redD:    "#7F1D1D",
  green:   "#22C55E",
  greenD:  "#14532D",
  blue:    "#60A5FA",
  blueD:   "#1E3A5F",
  yellow:  "#F59E0B",
  navy:    "#1A2942",
};

// ── Layout constants ───────────────────────────────────────────────────────────
const NW = 180, NH = 52, Y_STEP = 110;
const MAIN_X  = 400;  // x for normal events
const LOOP_X  = 680;  // x for loop-lane events
const MAX_CONSECUTIVE_COMPRESS = 3;  // compress consecutive runs longer than this

// ── Graph builder from API response ──────────────────────────────────────────
//
// API shape (GET /v1/runs/{run_id}):
//   run.events[]  { event_type, step_index, timestamp, payload }
//   run.signals[] { id, failure_type, severity, step_index, confidence,
//                   evidence_summary, evidence, suggested_fixes[] }
//   run.agent_id, run.exit_reason, run.started_at, run.completed_at, run.step_count
//
// Payload fields per event_type:
//   llm.called:         model, prompt_tokens
//   llm.responded:      finish_reason, completion_tokens, latency_ms, output_length
//   tool.called:        tool_name, args_hash
//   tool.responded:     success, output_length, latency_ms, error_hash
//   retrieval.called:   index_name, query_hash
//   retrieval.responded:result_count, top_score, latency_ms
//   external.signal:    signal_name, source, meta
//   run.started:        input_hash, model, tools
//   run.completed:      total_steps, exit_reason, tool_call_count
//   run.errored:        error_type, error_hash, step_index

function buildGraph(run) {
  const events = [...run.events].sort((a, b) =>
    a.step_index !== b.step_index
      ? a.step_index - b.step_index
      : a.timestamp - b.timestamp
  );

  // Index responded payloads
  const responded = {};
  for (const e of events) {
    if (e.event_type.endsWith(".responded"))
      responded[`${e.step_index}:${e.event_type}`] = e.payload || {};
  }

  // Detect looping tools (called ≥ 3 times)
  const toolCounts = {};
  for (const e of events) {
    if (e.event_type === "tool.called") {
      const name = e.payload?.tool_name || "tool";
      toolCounts[name] = (toolCounts[name] || 0) + 1;
    }
  }
  const loopingTools = new Set(Object.keys(toolCounts).filter(k => toolCounts[k] >= 3));

  // Index signals by step_index
  const signalsByStep = {};
  for (const sig of run.signals || []) {
    if (!signalsByStep[sig.step_index]) signalsByStep[sig.step_index] = [];
    signalsByStep[sig.step_index].push(sig);
  }

  const primaryTypes = new Set([
    "run.started", "run.completed", "run.errored",
    "llm.called", "tool.called", "retrieval.called", "external.signal",
  ]);

  // Deduplicate primary events (keep temporal order)
  const seenKeys = new Set();
  const primaryEvts = [];
  for (const e of events) {
    if (!primaryTypes.has(e.event_type)) continue;
    const key = `${e.step_index}:${e.event_type}`;
    if (seenKeys.has(key)) continue;
    seenKeys.add(key);
    primaryEvts.push(e);
  }

  // Build segments: group consecutive same-looping-tool calls so they can be compressed.
  // Non-consecutive loop calls (interleaved with LLMs) are individual "loopSingle" segments.
  const segments = [];
  let si = 0;
  while (si < primaryEvts.length) {
    const e = primaryEvts[si];
    if (e.event_type === "tool.called") {
      const name = e.payload?.tool_name || "tool";
      if (loopingTools.has(name)) {
        // Collect consecutive calls to the same tool
        const run_evts = [e];
        let j = si + 1;
        while (
          j < primaryEvts.length &&
          primaryEvts[j].event_type === "tool.called" &&
          (primaryEvts[j].payload?.tool_name || "tool") === name
        ) {
          run_evts.push(primaryEvts[j]);
          j++;
        }
        if (run_evts.length >= 2) {
          segments.push({ type: "run", toolName: name, events: run_evts });
          si = j;
          continue;
        }
        // Single (non-consecutive) occurrence of a looping tool
        segments.push({ type: "loopSingle", event: e, toolName: name });
        si++;
        continue;
      }
    }
    segments.push({ type: "single", event: e });
    si++;
  }

  const nodes = [];
  const edges = [];
  let y = 40;
  let prevId = null;

  function addSignals(stepIndex) {
    const sigs = signalsByStep[stepIndex];
    if (!sigs) return;
    for (const sig of sigs) {
      const sigId = `signal-${sig.id}`;
      const fixText = sig.suggested_fixes?.[0]?.description || "";
      nodes.push({
        id: sigId, type: "signal",
        label: sig.failure_type,
        sub: `${sig.severity} · ${sig.evidence_summary || ""}`,
        x: MAIN_X, y,
        meta: {
          detector:   sig.failure_type,
          severity:   sig.severity,
          confidence: `${Math.round((sig.confidence || 0) * 100)}%`,
          evidence:   sig.evidence_summary || JSON.stringify(sig.evidence),
          ...(fixText && { fix: fixText }),
        },
      });
      edges.push({ from: prevId, to: sigId, type: "signal" });
      prevId = sigId;
      y += Y_STEP;
    }
    delete signalsByStep[stepIndex];
  }

  function pushLoopNode(e, groupCount) {
    const resp = responded[`${e.step_index}:tool.responded`] || {};
    const name = e.payload?.tool_name || "tool";
    const nodeId = groupCount
      ? `tool-run-${name}-${e.step_index}`
      : `tool-called-${e.step_index}`;
    nodes.push({
      id: nodeId, type: "tool", label: name,
      sub: groupCount ? `step ${e.step_index}+` : `step ${e.step_index}`,
      x: LOOP_X, y,
      meta: {
        tool_name:  name,
        success:    resp.success,
        latency_ms: resp.latency_ms,
        error_hash: resp.error_hash,
        ...(groupCount && { calls: groupCount, note: `${groupCount} consecutive calls` }),
      },
      loop: true, ok: false, groupCount: groupCount || null,
    });
    if (prevId) edges.push({ from: prevId, to: nodeId, type: "loop" });
    prevId = nodeId;
    y += Y_STEP;
    return nodeId;
  }

  for (const seg of segments) {
    if (seg.type === "loopSingle") {
      pushLoopNode(seg.event, null);
      addSignals(seg.event.step_index);
      continue;
    }

    if (seg.type === "run") {
      const { events: runEvts } = seg;
      const N = runEvts.length;
      if (N <= MAX_CONSECUTIVE_COMPRESS) {
        // Show each individually in the loop lane
        for (const e of runEvts) {
          pushLoopNode(e, null);
          addSignals(e.step_index);
        }
      } else {
        // Compress: single node with ×N badge
        pushLoopNode(runEvts[0], N);
        for (const e of runEvts) addSignals(e.step_index);
      }
      continue;
    }

    // Normal event
    const e = seg.event;
    const p = e.payload || {};
    let type, label, sub, ok = false;
    let meta = { ...p };

    switch (e.event_type) {
      case "run.started":
        type = "start"; label = "RUN START"; sub = run.agent_id;
        break;
      case "run.completed":
      case "run.errored": {
        const dur = run.started_at && run.completed_at
          ? `${(run.completed_at - run.started_at).toFixed(1)}s` : "—";
        type = "end"; label = "RUN END";
        sub = `${run.exit_reason || (e.event_type === "run.errored" ? "errored" : "completed")} · ${dur}`;
        break;
      }
      case "llm.called": {
        const resp = responded[`${e.step_index}:llm.responded`] || {};
        Object.assign(meta, {
          latency_ms:        resp.latency_ms,
          finish_reason:     resp.finish_reason,
          completion_tokens: resp.completion_tokens,
          output_length:     resp.output_length,
        });
        type = "llm"; label = "LLM CALL";
        sub = `${p.model || "unknown"} · step ${e.step_index}`;
        break;
      }
      case "tool.called": {
        const resp = responded[`${e.step_index}:tool.responded`] || {};
        ok = resp.success !== false;
        Object.assign(meta, {
          success:    resp.success,
          latency_ms: resp.latency_ms,
          error_hash: resp.error_hash,
        });
        type = "tool"; label = p.tool_name || "tool"; sub = `step ${e.step_index}`;
        break;
      }
      case "retrieval.called": {
        const resp = responded[`${e.step_index}:retrieval.responded`] || {};
        Object.assign(meta, {
          result_count: resp.result_count,
          top_score:    resp.top_score,
          latency_ms:   resp.latency_ms,
        });
        type = "retrieval"; label = "RETRIEVAL";
        sub = p.index_name || `step ${e.step_index}`;
        break;
      }
      case "external.signal":
        type = "external"; label = p.signal_name || "EXTERNAL";
        sub = p.source || `step ${e.step_index}`;
        break;
      default: continue;
    }

    const nodeId = `${e.event_type.replace(".", "-")}-${e.step_index}`;
    nodes.push({ id: nodeId, type, label, sub, x: MAIN_X, y, meta, loop: false, ok });
    if (prevId) {
      const prevNode = nodes.find(n => n.id === prevId);
      edges.push({ from: prevId, to: nodeId, type: prevNode?.type === "signal" ? "signal" : "normal" });
    }
    prevId = nodeId;
    y += Y_STEP;
    addSignals(e.step_index);
  }

  // Loop zone: vertical band on the right side spanning all loop nodes
  const loopNodes = nodes.filter(n => n.loop);
  const loopToolNames = [...new Set(loopNodes.map(n => n.label))].join(", ");
  const loopZone = loopNodes.length >= 1 ? {
    x:        Math.min(...loopNodes.map(n => n.x)) - NW / 2 - 20,
    y:        Math.min(...loopNodes.map(n => n.y)) - NH / 2 - 6,
    w:        NW + 40,
    h:        Math.max(...loopNodes.map(n => n.y)) - Math.min(...loopNodes.map(n => n.y)) + NH + 12,
    toolName: loopToolNames,
    count:    loopNodes.length,
  } : null;

  return { nodes, edges, loopZone, graphH: y + 40 };
}

// ── Run summary helpers ───────────────────────────────────────────────────────
function runSummary(run) {
  const dur = run.started_at && run.completed_at
    ? `${(run.completed_at - run.started_at).toFixed(1)}s` : "—";

  const totalTokens = (run.events || []).reduce(
    (s, e) => s + (e.payload?.prompt_tokens || 0) + (e.payload?.completion_tokens || 0), 0
  );

  const severityOrder = { CRITICAL: 3, HIGH: 2, MEDIUM: 1, LOW: 0 };
  const worstSignal = [...(run.signals || [])]
    .sort((a, b) => (severityOrder[b.severity] || 0) - (severityOrder[a.severity] || 0))[0] || null;

  return { dur, totalTokens, worstSignal, stepCount: run.step_count || 0 };
}

// ── Node visuals ──────────────────────────────────────────────────────────────
function nodeColor(node) {
  if (node.type === "start")     return { fill: "#0A1F38", border: C.blue,    text: C.white  };
  if (node.type === "end")       return { fill: "#111C2A", border: "#5A7A9A", text: C.light  };
  if (node.type === "signal")    return { fill: "#4A1010", border: "#FF5555", text: C.white  };
  if (node.type === "external")  return { fill: "#1C1500", border: C.yellow,  text: C.yellow };
  if (node.type === "retrieval") return { fill: "#091C30", border: "#5BB5E0", text: C.light  };
  if (node.type === "llm")       return { fill: "#0C1D32", border: "#4A9FC8", text: C.light  };
  if (node.type === "tool") {
    if (node.loop) return { fill: "#2A0808", border: "#FF5555", text: "#FF9999" };
    if (node.ok)   return { fill: "#0A1E14", border: "#4ADE80", text: "#86EFAC" };
    return                { fill: "#0C1C2A", border: "#4A7090", text: C.light   };
  }
  return { fill: C.surf2, border: C.border, text: C.mid };
}

function nodeIcon(type) {
  if (type === "start")     return "▶";
  if (type === "end")       return "■";
  if (type === "llm")       return "◆";
  if (type === "tool")      return "⚙";
  if (type === "retrieval") return "⊞";
  if (type === "external")  return "⚑";
  if (type === "signal")    return "⚡";
  return "○";
}

// ── Animated edge dot ─────────────────────────────────────────────────────────
// Uses animateMotion `path` attribute (same coordinate space as the element),
// which avoids the mpath-in-defs coordinate mismatch that caused off-screen dots.
function AnimatedDot({ pathD, duration, color, delay, radius }) {
  return (
    <circle r={radius || 2.5} fill={color} opacity={0.85}>
      <animateMotion
        dur={`${duration}s`}
        repeatCount="indefinite"
        begin={`${delay}s`}
        path={pathD}
      />
    </circle>
  );
}

// ── Edge path ─────────────────────────────────────────────────────────────────
function edgePath(from, to) {
  const fx = from.x, fy = from.y + NH / 2;
  const tx = to.x,   ty = to.y - NH / 2;
  const my = (fy + ty) / 2;
  if (Math.abs(fx - tx) < 5) return `M ${fx} ${fy} L ${tx} ${ty}`;
  return `M ${fx} ${fy} C ${fx} ${my} ${tx} ${my} ${tx} ${ty}`;
}

// ── Detail panel ──────────────────────────────────────────────────────────────
function DetailPanel({ node, onClose }) {
  if (!node) return null;
  const col = nodeColor(node);
  const rows = Object.entries(node.meta).filter(([, v]) => v !== null && v !== undefined && v !== "");

  return (
    <div style={{
      position: "absolute", right: 0, top: 0, bottom: 0, width: 280,
      background: C.surface, borderLeft: `1px solid ${C.border}`,
      display: "flex", flexDirection: "column", zIndex: 20,
      fontFamily: "'JetBrains Mono', 'Fira Code', 'Courier New', monospace",
    }}>
      <div style={{ padding: "14px 16px 12px", borderBottom: `1px solid ${C.border}`, background: C.surf2 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 16, color: col.text }}>{nodeIcon(node.type)}</span>
            <span style={{ fontSize: 12, fontWeight: 700, color: col.text, letterSpacing: 1 }}>{node.label}</span>
          </div>
          <button onClick={onClose} style={{
            background: "none", border: "none", color: C.dim, cursor: "pointer",
            fontSize: 16, lineHeight: 1, padding: "2px 4px",
          }}>✕</button>
        </div>
        <div style={{ fontSize: 10, color: C.mid, marginTop: 4 }}>{node.sub}</div>
        {node.type === "signal" && (
          <div style={{
            marginTop: 8, display: "inline-block",
            background: C.redD, border: `1px solid ${C.red}`,
            borderRadius: 3, padding: "2px 8px",
            fontSize: 10, color: C.red, fontWeight: 700, letterSpacing: 1,
          }}>{node.meta.severity} SEVERITY</div>
        )}
      </div>

      <div style={{ height: 3, background: col.border, flexShrink: 0 }} />

      <div style={{ flex: 1, overflowY: "auto", padding: "12px 16px" }}>
        {rows.map(([k, v]) => (
          <div key={k} style={{ marginBottom: 10 }}>
            <div style={{ fontSize: 9, color: C.dim, letterSpacing: 1, textTransform: "uppercase", marginBottom: 2 }}>
              {k.replace(/_/g, " ")}
            </div>
            <div style={{
              fontSize: 11, lineHeight: 1.5,
              color:      k === "fix" ? C.orange : k === "evidence" ? C.red : C.light,
              background: k === "fix" ? "#1A0D00" : k === "evidence" ? "#2A0808" : "transparent",
              padding:    (k === "fix" || k === "evidence") ? "6px 8px" : 0,
              borderRadius: 3,
              borderLeft: k === "fix" ? `2px solid ${C.orange}` : k === "evidence" ? `2px solid ${C.red}` : "none",
            }}>{String(v)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
// Props:
//   runId   — run UUID to load.
//   apiBase — base URL of the Customer API (default: http://localhost:8002)
//   token   — bearer token (AUTH_MODE=dev accepts any non-empty string)
function DunetraceGraph({
  runId   = "REPLACE_WITH_REAL_RUN_ID",
  apiBase = "http://localhost:8002",
  token   = "dev",
}) {
  const [runData,  setRunData]  = useState(null);
  const [loading,  setLoading]  = useState(true);
  const [error,    setError]    = useState(null);
  const [selected, setSelected] = useState(null);
  const [pan,      setPan]      = useState({ x: 0, y: 0 });
  const [zoom,     setZoom]     = useState(0.85);
  const [dragging, setDragging] = useState(false);
  const [dragStart,setDragStart]= useState(null);
  const [panStart, setPanStart] = useState(null);
  const [tick,     setTick]     = useState(0);
  const svgRef       = useRef(null);
  const containerRef = useRef(null);

  // Fetch run from API
  useEffect(() => {
    if (!runId || runId === "REPLACE_WITH_REAL_RUN_ID") {
      setError("Set the runId prop to a real run UUID.");
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    fetch(`${apiBase}/v1/runs/${runId}`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(r => {
        if (!r.ok) throw new Error(`API ${r.status}: ${r.statusText}`);
        return r.json();
      })
      .then(data => { setRunData(data); setLoading(false); })
      .catch(e  => { setError(e.message); setLoading(false); });
  }, [runId, apiBase, token]);

  // Pulse animation
  useEffect(() => {
    const id = setInterval(() => setTick(t => t + 1), 60);
    return () => clearInterval(id);
  }, []);
  const pulse = Math.abs(Math.sin(tick * 0.05 * Math.PI));

  // Wheel zoom (touchpad pinch + two-finger scroll)
  const onWheel = useCallback((e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? (e.ctrlKey ? 0.95 : 0.92) : (e.ctrlKey ? 1.05 : 1.08);
    setZoom(z => Math.max(0.2, Math.min(2.5, z * delta)));
  }, []);

  const onMouseDown = useCallback((e) => {
    if (e.target.closest(".node-hit")) return;
    setDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
    setPanStart({ ...pan });
  }, [pan]);

  const onMouseMove = useCallback((e) => {
    if (!dragging || !dragStart || !panStart) return;
    setPan({ x: panStart.x + (e.clientX - dragStart.x), y: panStart.y + (e.clientY - dragStart.y) });
  }, [dragging, dragStart, panStart]);

  const onMouseUp = useCallback(() => setDragging(false), []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [onWheel]);

  // Derived graph data
  const graph   = runData ? buildGraph(runData) : null;
  const summary = runData ? runSummary(runData)  : null;
  const NODES   = graph?.nodes || [];
  const EDGES   = graph?.edges || [];

  const canvasW = selected ? "calc(100% - 280px)" : "100%";

  if (loading) return (
    <div style={{ width: "100%", height: "100vh", background: C.bg, display: "flex", alignItems: "center", justifyContent: "center" }}>
      <span style={{ color: C.dim, fontFamily: "monospace", fontSize: 13 }}>loading run {runId}…</span>
    </div>
  );
  if (error) return (
    <div style={{ width: "100%", height: "100vh", background: C.bg, display: "flex", alignItems: "center", justifyContent: "center" }}>
      <span style={{ color: C.red, fontFamily: "monospace", fontSize: 13 }}>⚠ {error}</span>
    </div>
  );

  return (
    <div ref={containerRef} style={{
      width: "100%", height: "100vh", background: C.bg,
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      display: "flex", flexDirection: "column",
      overflow: "hidden", position: "relative",
    }}>

      {/* ── Top bar ── */}
      <div style={{
        height: 48, flexShrink: 0, background: C.surface,
        borderBottom: `1px solid ${C.border}`,
        display: "flex", alignItems: "center",
        padding: "0 20px", gap: 12, zIndex: 10,
      }}>
        <div style={{
          width: 26, height: 26, background: C.navy, borderRadius: 4,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 14, fontWeight: 700, color: C.orange,
        }}>D</div>
        <span style={{ fontSize: 12, fontWeight: 700, color: C.white, letterSpacing: 2 }}>DUNETRACE</span>
        <div style={{ width: 1, height: 20, background: C.border, margin: "0 4px" }} />

        <span style={{ fontSize: 10, color: C.dim }}>run</span>
        <span style={{ fontSize: 10, color: C.light, fontWeight: 600 }}>{runData.run_id}</span>
        <span style={{ fontSize: 10, color: C.dim }}>·</span>
        <span style={{ fontSize: 10, color: C.mid }}>{runData.agent_id}</span>

        <div style={{ flex: 1 }} />

        {summary.worstSignal && (
          <div style={{
            display: "flex", alignItems: "center", gap: 6,
            background: C.redD, border: `1px solid ${C.red}`,
            borderRadius: 4, padding: "3px 10px",
          }}>
            <span style={{ fontSize: 10, color: C.red }}>⚡</span>
            <span style={{ fontSize: 10, fontWeight: 700, color: C.red, letterSpacing: 1 }}>
              {summary.worstSignal.failure_type}
            </span>
            <span style={{ fontSize: 9, background: C.red, color: C.white, borderRadius: 2, padding: "1px 5px", fontWeight: 700 }}>
              {summary.worstSignal.severity}
            </span>
          </div>
        )}

        {[
          ["steps",    String(summary.stepCount)],
          ["tokens",   summary.totalTokens > 0 ? summary.totalTokens.toLocaleString() : "—"],
          ["duration", summary.dur],
        ].map(([k, v]) => (
          <div key={k} style={{ textAlign: "center" }}>
            <div style={{ fontSize: 9, color: C.dim, textTransform: "uppercase", letterSpacing: 1 }}>{k}</div>
            <div style={{ fontSize: 12, color: C.light, fontWeight: 600 }}>{v}</div>
          </div>
        ))}

        <div style={{ display: "flex", gap: 4, marginLeft: 8 }}>
          {[
            ["−", () => setZoom(z => Math.max(0.2, z * 0.85))],
            ["+", () => setZoom(z => Math.min(2.5, z * 1.15))],
            ["⊡", () => { setZoom(0.85); setPan({ x: 0, y: 0 }); }],
          ].map(([label, fn]) => (
            <button key={label} onClick={fn} style={{
              width: 26, height: 26, background: C.surf2,
              border: `1px solid ${C.border}`, borderRadius: 4,
              color: C.mid, cursor: "pointer", fontSize: 13,
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>{label}</button>
          ))}
        </div>
      </div>

      {/* ── Main area ── */}
      <div style={{ flex: 1, position: "relative", overflow: "hidden", display: "flex" }}>

        <svg
          ref={svgRef}
          style={{ width: canvasW, height: "100%", cursor: dragging ? "grabbing" : "grab", transition: "width 0.2s" }}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={onMouseUp}
        >
          <defs>
            <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke={C.dim2} strokeWidth="0.5" opacity="0.4" />
            </pattern>
            <marker id="arrow-normal" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M 0 0 L 6 3 L 0 6 Z" fill="#3A5070" />
            </marker>
            <marker id="arrow-loop" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M 0 0 L 6 3 L 0 6 Z" fill={C.red} opacity="0.7" />
            </marker>
            <marker id="arrow-signal" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M 0 0 L 6 3 L 0 6 Z" fill={C.orange} opacity="0.8" />
            </marker>
            <filter id="glow-red">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
            <filter id="glow-orange">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </defs>

          <rect width="100%" height="100%" fill="url(#grid)" />

          <g transform={`translate(${pan.x + 400}, ${pan.y + 30}) scale(${zoom})`}>
            <g transform={`translate(${-400}, 0)`}>

              {/* ── Loop zone highlight ── */}
              {graph.loopZone && (() => {
                const lz = graph.loopZone;
                const labelW = Math.min(lz.w - 4, 120);
                return (
                  <g>
                    {/* zone rect */}
                    <rect
                      x={lz.x} y={lz.y} width={lz.w} height={lz.h} rx={6}
                      fill={C.red} fillOpacity={0.07 + pulse * 0.04}
                      stroke="#FF5555" strokeOpacity={0.55} strokeWidth={1.5} strokeDasharray="6 4"
                    />
                    {/* label pill pinned to the top of the rect */}
                    <rect
                      x={lz.x + (lz.w - labelW) / 2} y={lz.y - 1}
                      width={labelW} height={18} rx={4}
                      fill="#3A0808" stroke="#FF5555" strokeWidth={1} strokeOpacity={0.8}
                    />
                    <text
                      x={lz.x + lz.w / 2} y={lz.y + 12}
                      textAnchor="middle" fontSize={9} fontWeight={700}
                      fill="#FF8888" fontFamily="monospace" letterSpacing={1.5}
                    >
                      LOOP · {lz.toolName}
                    </text>
                  </g>
                );
              })()}

              {/* ── Edges ── */}
              {EDGES.map((e, i) => {
                const from = NODES.find(n => n.id === e.from);
                const to   = NODES.find(n => n.id === e.to);
                if (!from || !to) return null;
                const d       = edgePath(from, to);
                const isLoop  = e.type === "loop";
                const isSig   = e.type === "signal";
                const stroke  = isLoop ? "#FF5555" : isSig ? C.orange : "#3A5070";
                const opacity = isLoop ? 0.75 : isSig ? 0.8 : 0.7;
                const dashArr = isLoop ? "5 4" : isSig ? "4 3" : "none";
                const marker  = isLoop ? "url(#arrow-loop)" : isSig ? "url(#arrow-signal)" : "url(#arrow-normal)";
                const dotCol  = isLoop ? "#FF5555" : isSig ? C.orange : "#4A7090";
                const dotSpd  = isLoop ? 1.2  : isSig ? 0.9 : 1.8;

                return (
                  <g key={i}>
                    <path
                      d={d} fill="none"
                      stroke={stroke} strokeWidth={isLoop || isSig ? 1.5 : 1}
                      strokeDasharray={dashArr} opacity={opacity}
                      markerEnd={marker}
                    />
                    <AnimatedDot
                      pathD={d}
                      duration={dotSpd}
                      color={dotCol}
                      delay={(i * 0.3) % dotSpd}
                      radius={isLoop ? 3.5 : 2.5}
                    />
                  </g>
                );
              })}

              {/* ── Nodes ── */}
              {NODES.map(node => {
                const col        = nodeColor(node);
                const isSelected = selected?.id === node.id;
                const isSignal   = node.type === "signal";
                const nx = node.x - NW / 2;
                const ny = node.y - NH / 2;
                const filterAttr = isSignal ? "url(#glow-red)" : isSelected ? "url(#glow-orange)" : "none";

                return (
                  <g
                    key={node.id}
                    className="node-hit"
                    style={{ cursor: "pointer" }}
                    onClick={(ev) => { ev.stopPropagation(); setSelected(selected?.id === node.id ? null : node); }}
                    filter={filterAttr}
                  >
                    {isSelected && (
                      <rect x={nx - 3} y={ny - 3} width={NW + 6} height={NH + 6}
                        rx={9} fill="none" stroke={C.orange} strokeWidth={1.5} opacity={0.8} />
                    )}
                    <rect x={nx} y={ny} width={NW} height={NH} rx={6}
                      fill={col.fill} stroke={col.border} strokeWidth={isSelected ? 1.5 : 1} />
                    <rect x={nx} y={ny} width={4} height={NH} rx={2} fill={col.border} />

                    {isSignal && (
                      <rect
                        x={nx - 6} y={ny - 6} width={NW + 12} height={NH + 12} rx={10}
                        fill="none" stroke={C.red} strokeWidth={1} opacity={pulse * 0.5}
                      />
                    )}

                    <text
                      x={nx + 16} y={ny + NH / 2 + 1}
                      textAnchor="middle" dominantBaseline="middle"
                      fontSize={isSignal ? 14 : 12} fill={col.text} fontFamily="monospace"
                    >{nodeIcon(node.type)}</text>

                    <text x={nx + 26} y={ny + 17} fontSize={10} fontWeight={700}
                      fill={col.text} fontFamily="monospace" letterSpacing={0.5}
                    >{node.label}</text>

                    <text x={nx + 26} y={ny + 32} fontSize={9}
                      fill={C.mid} fontFamily="monospace" opacity={0.9}
                    >{node.sub.length > 22 ? node.sub.slice(0, 22) + "…" : node.sub}</text>

                    {node.loop && node.type === "tool" && (
                      <g>
                        <rect x={nx + NW - 40} y={ny + 4} width={34} height={14} rx={3} fill={C.redD} />
                        <text x={nx + NW - 23} y={ny + 14} textAnchor="middle"
                          fontSize={8} fontWeight={700} fill={C.red} fontFamily="monospace">
                          {node.groupCount ? `\u00d7${node.groupCount}` : "LOOP"}
                        </text>
                      </g>
                    )}

                    {!isSelected && (
                      <text x={nx + NW - 8} y={ny + NH - 6} textAnchor="end"
                        fontSize={8} fill={C.dim} fontFamily="monospace" opacity={0.5}>inspect →</text>
                    )}
                  </g>
                );
              })}
            </g>
          </g>
        </svg>

        {selected && <DetailPanel node={selected} onClose={() => setSelected(null)} />}

        {/* ── Legend ── */}
        <div style={{
          position: "absolute", left: 16, bottom: 16,
          background: C.surface, border: `1px solid ${C.border}`,
          borderRadius: 6, padding: "10px 14px",
          display: "flex", flexDirection: "column", gap: 6,
          fontSize: 10, color: C.dim, fontFamily: "monospace",
        }}>
          {[
            { icon: "◆", col: C.mid,    label: "LLM call"        },
            { icon: "⚙", col: C.green,  label: "Tool call"       },
            { icon: "⚙", col: C.red,    label: "Tool (looping)"  },
            { icon: "⊞", col: C.blue,   label: "Retrieval"       },
            { icon: "⚑", col: C.yellow, label: "External signal" },
            { icon: "⚡", col: C.red,   label: "Failure signal"  },
          ].map(({ icon, col, label }) => (
            <div key={label} style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ color: col, fontSize: 11 }}>{icon}</span>
              <span>{label}</span>
            </div>
          ))}
          <div style={{ borderTop: `1px solid ${C.border}`, paddingTop: 6, marginTop: 2, color: C.dim2, fontSize: 9 }}>
            scroll/pinch to zoom · drag to pan · click to inspect
          </div>
        </div>

        {!selected && (
          <div style={{
            position: "absolute", right: 16, top: 16,
            background: C.surface, border: `1px solid ${C.border}`,
            borderRadius: 6, padding: "8px 12px",
            fontSize: 10, color: C.dim, fontFamily: "monospace",
          }}>
            click any node to inspect
          </div>
        )}
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: ${C.surface}; }
        ::-webkit-scrollbar-thumb { background: ${C.border}; border-radius: 2px; }
      `}</style>
    </div>
  );
}
