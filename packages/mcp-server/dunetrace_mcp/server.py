"""
Dunetrace MCP server.

Exposes Dunetrace agent signals, run details, and health scores as MCP tools
so Claude Code, Cursor, Codex, and any MCP-compatible client can query them.

Usage:
    dunetrace-mcp                           # stdio (Claude Code / Cursor)
    dunetrace-mcp --sse                     # HTTP/SSE on :8000
    dunetrace-mcp --sse --port 9000         # HTTP/SSE on :9000
    dunetrace-mcp --help                    # show usage
    dunetrace-mcp --version                 # show version

Environment:
    DUNETRACE_API_URL   Customer API base URL  (default: http://localhost:8002)
    DUNETRACE_API_KEY   Bearer token           (default: dt_dev_test)
"""
from __future__ import annotations

import pathlib
import textwrap
from datetime import datetime, timezone
from typing import Optional

from mcp.server.fastmcp import FastMCP

from . import client

# Docs live at <repo-root>/docs/ — four levels up from this file.
_DOCS = pathlib.Path(__file__).resolve().parents[3] / "docs"

mcp = FastMCP(
    "dunetrace",
    instructions=textwrap.dedent("""
        Dunetrace monitors AI agents in production and detects structural failures
        (tool loops, cost spikes, context bloat, goal abandonment, and 14 more).

        Use these tools to answer questions like:
        - "Is my agent healthy?"
        - "What failures happened in the last 24 hours?"
        - "Show me the details of run <id>"
        - "Which failure type is most common across all agents?"
        - "Did a recent deploy cause a spike in errors?"

        All timestamps returned are UTC ISO-8601.
        Confidence values are 0–1 (higher = more certain the failure occurred).
    """).strip(),
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _ts(epoch: float | None) -> str:
    if epoch is None:
        return "—"
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _ago(epoch: float | None) -> str:
    if epoch is None:
        return "—"
    secs = datetime.now(tz=timezone.utc).timestamp() - epoch
    if secs < 60:
        return f"{int(secs)}s ago"
    if secs < 3600:
        return f"{int(secs/60)}m ago"
    if secs < 86400:
        return f"{int(secs/3600)}h ago"
    return f"{int(secs/86400)}d ago"


def _sev_icon(sev: str) -> str:
    return {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(sev, "⚪")


def _score_icon(score: int | None) -> str:
    if score is None:
        return "❓"
    if score >= 80:
        return "✅"
    if score >= 60:
        return "🟡"
    return "🔴"


# ── tools ────────────────────────────────────────────────────────────────────

@mcp.tool()
def list_agents() -> str:
    """
    List all monitored AI agents with their health summary: run count, signal
    count, failure type breakdown, and last seen time.
    """
    data = client.get("/v1/agents", limit=100)
    agents = data.get("agents", [])
    if not agents:
        return "No agents found. Instrument your agent with the Dunetrace SDK to start monitoring."

    lines = [f"{'AGENT':<35} {'RUNS':>5} {'SIGS':>5} {'CRIT':>4} {'HIGH':>4}  LAST SEEN"]
    lines.append("─" * 75)
    for a in agents:
        ft_str = ", ".join(f"{k}×{v}" for k, v in (a.get("failure_types") or {}).items())
        lines.append(
            f"{a['agent_id']:<35} {a['run_count']:>5} {a['signal_count']:>5}"
            f" {a['critical_count']:>4} {a['high_count']:>4}  {_ago(a.get('last_seen'))}"
        )
        if ft_str:
            lines.append(f"  {'':35} {ft_str}")

    total = data.get("page", {}).get("total", len(agents))
    lines.append(f"\n{total} agent(s) total.")
    return "\n".join(lines)


@mcp.tool()
def get_agent_signals(
    agent_id: str,
    limit: int = 20,
    severity: Optional[str] = None,
) -> str:
    """
    Get recent failure signals detected for a specific agent.

    Args:
        agent_id:  The agent ID to query (use list_agents to find IDs).
        limit:     Max signals to return (default 20, max 100).
        severity:  Filter to one severity: CRITICAL, HIGH, MEDIUM, or LOW.
    """
    params: dict = {"limit": min(limit, 100), "include_shadow": "false"}
    data = client.get(f"/v1/agents/{agent_id}/signals", **params)
    signals = data.get("signals", [])

    if severity:
        signals = [s for s in signals if s["severity"] == severity.upper()]

    if not signals:
        return f"No signals found for agent '{agent_id}'" + (f" with severity {severity}" if severity else "") + "."

    lines = [f"Signals for agent: {agent_id}\n"]
    for s in signals:
        icon = _sev_icon(s["severity"])
        lines.append(
            f"{icon} [{s['severity']}] {s['failure_type']}  conf={s['confidence']:.0%}"
            f"  step={s['step_index']}  {_ago(s.get('detected_at'))}"
        )
        lines.append(f"   {s['title']}")
        lines.append(f"   What: {s['what']}")
        if s.get("suggested_fixes"):
            fix = s["suggested_fixes"][0]
            lines.append(f"   Fix:  {fix.get('description', '')}")
        lines.append("")

    page = data.get("page", {})
    lines.append(f"Showing {len(signals)} of {page.get('total', len(signals))} signals.")
    return "\n".join(lines)


@mcp.tool()
def get_agent_health(agent_id: str) -> str:
    """
    Get the health score and component breakdown for an agent.

    The score is 0–100. Components: failure_rate (40pts), loop_avoidance (25pts),
    token_efficiency (20pts), latency (15pts). Requires ≥3 runs; needs ≥30 runs
    for token/latency baselines.

    Args:
        agent_id: The agent ID to query.
    """
    h = client.get(f"/v1/agents/{agent_id}/health-score")

    score = h.get("score")
    icon  = _score_icon(score)
    score_str = str(score) if score is not None else "N/A (need ≥3 runs)"

    lines = [
        f"{icon} Health score for {agent_id}: {score_str}/100",
        f"   Sample runs:     {h['sample_runs']}",
        f"   Baseline ready:  {'yes' if h['baseline_ready'] else 'no (need ≥30 runs for token/latency)'}",
        "",
        "Component breakdown:",
    ]
    for name, comp in (h.get("components") or {}).items():
        pct = f"{comp['score']}/{comp['max']}"
        val = comp.get("value")
        val_str = f"  (current: {val:.1f} {comp['label']})" if val is not None else ""
        lines.append(f"  {name:<20} {pct:>7}{val_str}")

    return "\n".join(lines)


@mcp.tool()
def get_run_detail(run_id: str, agent_id: str = "") -> str:
    """
    Get full detail for a run: metadata, detected signals, and event timeline.

    Args:
        run_id:   The run UUID to inspect.
        agent_id: Agent ID (optional; speeds up the signal lookup).
    """
    run = client.get(f"/v1/runs/{run_id}")

    r       = run.get("run", run)          # handle both flat and nested shapes
    signals = run.get("signals", [])
    events  = run.get("events", [])

    dur = "—"
    if r.get("completed_at") and r.get("started_at"):
        secs = r["completed_at"] - r["started_at"]
        dur = f"{secs:.1f}s"

    lines = [
        f"Run: {run_id}",
        f"Agent:    {r.get('agent_id', '—')}  v{r.get('agent_version', '?')}",
        f"Started:  {_ts(r.get('started_at'))}  ({_ago(r.get('started_at'))})",
        f"Duration: {dur}",
        f"Steps:    {r.get('step_count', '—')}",
        f"Exit:     {r.get('exit_reason', '—')}",
    ]
    if r.get("total_tokens"):
        lines.append(f"Tokens:   {r['total_tokens']:,}")

    # Signals
    if signals:
        lines.append(f"\nSignals ({len(signals)}):")
        for s in signals:
            icon = _sev_icon(s["severity"])
            lines.append(
                f"  {icon} {s['failure_type']}  [{s['severity']}]  conf={s['confidence']:.0%}"
                f"  step={s['step_index']}"
            )
            lines.append(f"     {s['title']}")
            lines.append(f"     {s['what']}")
            if s.get("suggested_fixes"):
                lines.append(f"     Fix: {s['suggested_fixes'][0].get('description', '')}")
    else:
        lines.append("\nNo signals — clean run.")

    # Event timeline (compact)
    if events:
        lines.append(f"\nEvent timeline ({len(events)} events):")
        for e in events[:40]:         # cap at 40 for readability
            ts_rel = ""
            if e.get("timestamp") and r.get("started_at"):
                ts_rel = f"+{e['timestamp'] - r['started_at']:.1f}s"
            p = e.get("payload", {})
            detail = ""
            if e["event_type"] == "tool.called":
                detail = f"tool={p.get('tool_name', '?')}  ok={p.get('success', '?')}"
                if p.get("latency_ms"):
                    detail += f"  {p['latency_ms']}ms"
            elif e["event_type"] == "llm.called":
                # prompt_tokens may be absent here for LangChain (they're in llm.responded)
                pt = p.get('prompt_tokens', '?')
                detail = f"model={p.get('model', '?')}  p={pt}"
            elif e["event_type"] == "llm.responded":
                # completion_tokens and latency always land here; prompt_tokens here for LangChain
                pt = p.get('prompt_tokens')
                ct = p.get('completion_tokens', '?')
                detail = f"c={ct}"
                if pt:
                    detail = f"p={pt}  {detail}"
                if p.get("latency_ms"):
                    detail += f"  {p['latency_ms']}ms"
            elif e["event_type"] in ("run.started", "run.completed"):
                detail = p.get("exit_reason", "")
            lines.append(f"  [{e['step_index']:>3}] {ts_rel:>8}  {e['event_type']:<20} {detail}")
        if len(events) > 40:
            lines.append(f"  … {len(events)-40} more events")

    return "\n".join(lines)


@mcp.tool()
def search_signals(
    severity: Optional[str] = None,
    failure_type: Optional[str] = None,
    since_hours: Optional[int] = None,
    agent_id: Optional[str] = None,
    limit: int = 30,
) -> str:
    """
    Search signals across all agents with optional filters.

    Args:
        severity:     Filter to one severity: CRITICAL, HIGH, MEDIUM, or LOW.
        failure_type: Filter by detector type e.g. TOOL_LOOP, COST_SPIKE, CONTEXT_BLOAT.
        since_hours:  Only return signals from the last N hours (e.g. 24 for last day).
        agent_id:     Restrict to one agent (optional; searches all agents if omitted).
        limit:        Max signals to return (default 30, max 200).
    """
    import time as _time

    cutoff = (_time.time() - since_hours * 3600) if since_hours else None

    # Collect agents to search
    if agent_id:
        agents = [{"agent_id": agent_id}]
    else:
        agents = client.get("/v1/agents", limit=100).get("agents", [])

    params: dict = {"limit": 200, "include_shadow": "false"}
    if severity:
        params["severity"] = severity.upper()
    if failure_type:
        params["failure_type"] = failure_type.upper()

    all_signals: list = []
    for a in agents:
        aid = a["agent_id"]
        sigs = client.get(f"/v1/agents/{aid}/signals", **params).get("signals", [])
        all_signals.extend(sigs)

    # Client-side filters — API also filters severity/failure_type but we apply
    # them here too so results are always correct regardless of API version.
    if cutoff:
        all_signals = [s for s in all_signals if (s.get("detected_at") or 0) >= cutoff]
    if severity:
        all_signals = [s for s in all_signals if s.get("severity") == severity.upper()]
    if failure_type:
        all_signals = [s for s in all_signals if s.get("failure_type") == failure_type.upper()]

    all_signals.sort(key=lambda s: s.get("detected_at") or 0, reverse=True)
    shown = all_signals[:min(limit, 200)]

    if not shown:
        parts = []
        if severity:
            parts.append(severity)
        if failure_type:
            parts.append(failure_type)
        if since_hours:
            parts.append(f"last {since_hours}h")
        qualifier = " / ".join(parts)
        return f"No signals found{(' matching ' + qualifier) if qualifier else ''}."

    lines = [f"Signals ({len(shown)} shown, {len(all_signals)} matched):\n"]
    for s in shown:
        icon = _sev_icon(s["severity"])
        lines.append(
            f"{icon} {_ago(s.get('detected_at')):>10}  [{s['severity']:<8}]"
            f"  {s['failure_type']:<30}  agent={s['agent_id']}"
        )
        lines.append(f"   id={s['id']}  run={s['run_id'][:12]}…  conf={s['confidence']:.0%}")
        lines.append(f"   {s['title']}")
        lines.append("")

    return "\n".join(lines)


# keep backward-compatible alias
list_all_signals = search_signals


@mcp.tool()
def get_signal_detail(signal_id: int, agent_id: str = "") -> str:
    """
    Get the full detail for a specific signal: evidence, impact, and all
    suggested fixes.

    Args:
        signal_id: The integer signal ID (visible in list_agents / search_signals output).
        agent_id:  Agent ID (optional but speeds up the lookup significantly).
    """
    # If agent_id provided, search only that agent; otherwise scan all agents.
    if agent_id:
        agents = [{"agent_id": agent_id}]
    else:
        agents = client.get("/v1/agents", limit=100).get("agents", [])

    signal = None
    for a in agents:
        aid = a["agent_id"]
        sigs = client.get(f"/v1/agents/{aid}/signals", limit=500, include_shadow="false").get("signals", [])
        signal = next((s for s in sigs if s["id"] == signal_id), None)
        if signal:
            break

    if not signal:
        return (
            f"Signal {signal_id} not found."
            + (" Try omitting agent_id to search all agents." if agent_id else "")
        )

    icon = _sev_icon(signal["severity"])
    lines = [
        f"{icon} Signal #{signal['id']}",
        f"Type:      {signal['failure_type']}",
        f"Severity:  {signal['severity']}  confidence={signal['confidence']:.0%}",
        f"Agent:     {signal['agent_id']}  v{signal['agent_version']}",
        f"Run:       {signal['run_id']}",
        f"Step:      {signal['step_index']}",
        f"Detected:  {_ts(signal.get('detected_at'))}  ({_ago(signal.get('detected_at'))})",
        "",
        f"Title:     {signal['title']}",
        "",
        f"What happened:",
        f"  {signal['what']}",
        "",
        f"Why it matters:",
        f"  {signal['why_it_matters']}",
        "",
        f"Evidence summary:",
        f"  {signal['evidence_summary']}",
        "",
    ]

    # Evidence dict (structured data, hashed values only — no raw content)
    ev = signal.get("evidence") or {}
    if ev:
        lines.append("Evidence (hashed/structural data):")
        for k, v in ev.items():
            if isinstance(v, list) and len(v) > 6:
                v = v[:6] + [f"…+{len(v)-6} more"]
            lines.append(f"  {k}: {v}")
        lines.append("")

    # Suggested fixes
    fixes = signal.get("suggested_fixes") or []
    if fixes:
        lines.append(f"Suggested fixes ({len(fixes)}):")
        for i, fix in enumerate(fixes, 1):
            lines.append(f"  {i}. {fix.get('description', '')}")
            if fix.get("code"):
                code = fix["code"]
                lang = fix.get("language", "")
                # Show first 20 lines of code
                code_lines = code.splitlines()[:20]
                lines.append(f"     ```{lang}")
                lines.extend(f"     {l}" for l in code_lines)
                if len(code.splitlines()) > 20:
                    lines.append(f"     … ({len(code.splitlines())-20} more lines)")
                lines.append("     ```")

    return "\n".join(lines)


@mcp.tool()
def get_agent_patterns(agent_id: str) -> str:
    """
    Analyze failure patterns for an agent: which failures are systemic (recurring
    across many runs) vs. one-off, trend over the past 14 days, and input hashes
    that consistently trigger failures.

    Args:
        agent_id: The agent ID to analyze.
    """
    insights = client.get(f"/v1/agents/{agent_id}/insights")

    lines = [f"Failure patterns for: {agent_id}\n"]

    # Systemic patterns
    systemic = insights.get("systemic_patterns") or []
    if systemic:
        lines.append("Systemic patterns (recurring across many runs):")
        for p in systemic:
            badge = "🚨 SYSTEMIC" if p["is_systemic"] else "⚠ Occasional"
            lines.append(
                f"  {badge}  {p['failure_type']}"
                f"  {p['affected_runs']}/{p['total_runs']} runs ({p['rate']*100:.0f}%)"
            )
            lines.append(
                f"            first seen {_ago(p.get('first_seen'))}  "
                f"last seen {_ago(p.get('last_seen'))}"
            )
        lines.append("")

    # Signal trends (daily counts per failure type over last 14 days)
    trends = insights.get("signal_trends") or []
    if trends:
        # Aggregate by failure_type
        from collections import defaultdict
        by_ft: dict = defaultdict(dict)
        for t in trends:
            by_ft[t["failure_type"]][t["day"]] = (by_ft[t["failure_type"]].get(t["day"], 0) + t["count"])
        all_days = sorted({t["day"] for t in trends})
        recent_days = all_days[-7:]  # last 7 days for display

        lines.append(f"Daily signal counts (last {len(recent_days)} days):")
        header = f"  {'FAILURE TYPE':<30}" + "".join(f"  {d[5:]}" for d in recent_days)
        lines.append(header)
        lines.append("  " + "─" * (28 + len(recent_days) * 8))
        for ft, day_map in sorted(by_ft.items()):
            row = f"  {ft:<30}"
            for day in recent_days:
                cnt = day_map.get(day, 0)
                row += f"  {'   ' + str(cnt) if cnt else '    —':>5}"
            lines.append(row)
        lines.append("")

    # Failure rates (by detector over rolling window)
    rates = insights.get("failure_rates") or []
    if rates:
        # Show worst current rate per failure type
        from collections import defaultdict
        rate_map: dict = {}
        for r in rates:
            ft = r["failure_type"]
            if ft not in rate_map or r["rate"] > rate_map[ft]["rate"]:
                rate_map[ft] = r
        if rate_map:
            lines.append("Failure rate by type (worst single-day rate):")
            for ft, r in sorted(rate_map.items(), key=lambda x: -x[1]["rate"]):
                bar = "█" * int(r["rate"] * 20)
                lines.append(
                    f"  {ft:<30}  {bar:<20}  {r['rate']*100:.0f}%"
                    f"  ({r['affected_runs']}/{r['total_runs']} runs on {r['day']})"
                )
            lines.append("")

    # Input hashes that consistently trigger failures (only strong patterns)
    input_pats = [p for p in (insights.get("input_patterns") or []) if p["rate"] >= 0.5]
    if input_pats:
        lines.append("Input patterns that reliably trigger failures (rate ≥ 50%):")
        for p in sorted(input_pats, key=lambda x: -x["rate"])[:8]:
            lines.append(
                f"  hash={p['input_hash']}  {p['failure_type']}"
                f"  {p['triggered_count']}/{p['total_runs']} runs ({p['rate']*100:.0f}%)"
            )
            lines.append("    → This input hash consistently causes this failure.")
        lines.append("")

    if not (systemic or trends or rates):
        lines.append("No pattern data yet — run more traffic through this agent.")

    return "\n".join(lines)


@mcp.tool()
def summarize_agent(agent_id: str) -> str:
    """
    One-shot summary of an agent: health score, recent failure pattern,
    most common signal types, and top suggested fixes.

    Use this when you want a quick diagnosis before diving deeper.

    Args:
        agent_id: The agent ID to summarize.
    """
    # Fetch in parallel would be ideal; sequential is fine for an MCP tool
    agents_data = client.get("/v1/agents", limit=100)
    agent_meta  = next(
        (a for a in agents_data.get("agents", []) if a["agent_id"] == agent_id), None
    )
    if not agent_meta:
        return f"Agent '{agent_id}' not found. Use list_agents to see available agents."

    signals_data = client.get(f"/v1/agents/{agent_id}/signals", limit=50, include_shadow="false")
    signals      = signals_data.get("signals", [])

    try:
        health = client.get(f"/v1/agents/{agent_id}/health-score")
    except Exception:
        health = None

    # Build summary
    score     = health.get("score") if health else None
    score_icon = _score_icon(score)
    score_str  = f"{score}/100" if score is not None else "N/A"

    lines = [
        f"═══ Agent summary: {agent_id} ═══",
        "",
        f"Health score:  {score_icon} {score_str}",
        f"Total runs:    {agent_meta['run_count']}",
        f"Total signals: {agent_meta['signal_count']}",
        f"Last seen:     {_ago(agent_meta.get('last_seen'))}",
        "",
    ]

    # Failure type breakdown
    ft = agent_meta.get("failure_types") or {}
    if ft:
        lines.append("Failure breakdown:")
        for k, v in sorted(ft.items(), key=lambda x: -x[1]):
            run_pct = f"{v/agent_meta['run_count']*100:.0f}% of runs" if agent_meta["run_count"] else ""
            lines.append(f"  {k:<35} {v:>4} signals  ({run_pct})")
        lines.append("")

    # Most recent signals with their fixes
    recent = signals[:5]
    if recent:
        lines.append("Most recent signals:")
        for s in recent:
            icon = _sev_icon(s["severity"])
            lines.append(
                f"  {icon} {s['failure_type']}  conf={s['confidence']:.0%}"
                f"  {_ago(s.get('detected_at'))}  run={s['run_id'][:8]}…"
            )
            lines.append(f"     {s['what']}")
            if s.get("why_it_matters"):
                lines.append(f"     Impact: {s['why_it_matters']}")
            if s.get("suggested_fixes"):
                lines.append(f"     Fix: {s['suggested_fixes'][0].get('description', '')}")
            lines.append("")

    # Health component detail
    if health and health.get("components"):
        lines.append("Health components:")
        for name, comp in health["components"].items():
            bar_len = int(comp["score"] / comp["max"] * 20) if comp["max"] else 0
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(
                f"  {name:<20} {bar}  {comp['score']}/{comp['max']}"
            )

    return "\n".join(lines)


@mcp.tool()
def get_agent_runs(agent_id: str, limit: int = 20) -> str:
    """
    List recent runs for a specific agent with their signal counts and durations.

    Args:
        agent_id: The agent ID to query.
        limit:    Max runs to return (default 20, max 100).
    """
    data = client.get(f"/v1/agents/{agent_id}/runs", limit=min(limit, 100))
    runs = data.get("runs", [])

    if not runs:
        return f"No runs found for agent '{agent_id}'."

    lines = [
        f"Recent runs for: {agent_id}\n",
        f"{'RUN ID':<12} {'STARTED':<22} {'DUR':>6} {'STEPS':>5} {'SIGS':>4}  STATUS",
        "─" * 70,
    ]
    for r in runs:
        dur = "—"
        if r.get("completed_at") and r.get("started_at"):
            secs = r["completed_at"] - r["started_at"]
            dur = f"{secs:.1f}s"
        sigs    = r.get("signal_count", 0)
        sig_str = f"🔴 {sigs}" if sigs > 0 else "✅  0"
        lines.append(
            f"{r['run_id'][:12]:<12} {_ago(r.get('started_at')):<22} "
            f"{dur:>6} {r.get('step_count',0):>5}  {sig_str}"
        )

    page = data.get("page", {})
    lines.append(f"\n{len(runs)} of {page.get('total', len(runs))} runs shown.")
    return "\n".join(lines)


# ── documentation resources ───────────────────────────────────────────────────
# Exposed as MCP resources so clients can browse them and LLMs can answer
# instrumentation / how-to questions without calling any live-data tools.

def _read_doc(name: str) -> str:
    p = _DOCS / name
    return p.read_text(encoding="utf-8") if p.exists() else f"(doc not found: {p})"


@mcp.resource("dunetrace://docs/integrate-python")
def doc_integrate_python() -> str:
    """Full guide: instrument a custom Python agent with Dunetrace (decorator
    style, context manager, auto-instrumentation, FastAPI, Flask, policies)."""
    return _read_doc("integrate-custom-python-agent.md")


@mcp.resource("dunetrace://docs/integrate-langchain")
def doc_integrate_langchain() -> str:
    """Full guide: add Dunetrace monitoring to a LangChain or LangGraph agent
    using DunetraceCallbackHandler."""
    return _read_doc("integrate-langchain-agent.md")


@mcp.resource("dunetrace://docs/integrate-typescript")
def doc_integrate_typescript() -> str:
    """Full guide: instrument a TypeScript / Node.js agent by posting events
    directly to the Dunetrace ingest HTTP endpoint."""
    return _read_doc("integrate-typescript-agent.md")


@mcp.resource("dunetrace://docs/policies")
def doc_policies() -> str:
    """Reference: runtime policies — conditions, actions, remote fetch,
    dashboard CRUD, stop/switch_model/inject_prompt/log action details."""
    return _read_doc("policies.md")


@mcp.resource("dunetrace://docs/integrate-langdock")
def doc_integrate_langdock() -> str:
    """Full guide: integrate Langdock (or any OTLP-emitting platform) with
    Dunetrace via the zero-code OTel receiver. Covers ngrok setup, Langdock
    workspace configuration, verification steps, and the MCP server path."""
    return _read_doc("integrate-langdock.md")


@mcp.resource("dunetrace://docs/detectors")
def doc_detectors() -> str:
    """Reference: all 17 detectors, thresholds, evidence fields, and tuning."""
    return _read_doc("detectors.md")


@mcp.resource("dunetrace://docs/mcp-server")
def doc_mcp_server() -> str:
    """Reference: MCP server tools, setup for Claude Code / Cursor / Codex,
    example workflows."""
    return _read_doc("mcp-server.md")


# ── instrumentation guide tool ────────────────────────────────────────────────

# Inline quick-start snippets for each framework so the tool works even if the
# docs directory isn't mounted (e.g. installed via pip without the repo).
_GUIDES: dict[str, str] = {
    "langchain": textwrap.dedent("""
        # Instrument a LangChain / LangGraph agent with Dunetrace

        ## Install
        ```bash
        pip install 'dunetrace[langchain]'
        ```

        ## Add the callback (one line)
        ```python
        from dunetrace import Dunetrace
        from dunetrace.integrations.langchain import DunetraceCallbackHandler

        dt = Dunetrace(endpoint="http://localhost:8001")  # or set DUNETRACE_ENDPOINT

        callback = DunetraceCallbackHandler(
            dt,
            agent_id="my-agent",        # identifies this agent in the dashboard
            system_prompt=SYSTEM_PROMPT, # optional — helps pattern analysis
            model="gpt-4o-mini",         # optional
            tools=[t.name for t in tools],
        )

        # Pass it to every invoke() call
        result = agent.invoke(
            {"messages": [("human", query)]},
            config={"callbacks": [callback]},
        )
        dt.shutdown(timeout=5)  # flush before process exits
        ```

        ## What gets tracked automatically
        - Every LLM call: model, token counts, latency, finish reason
        - Every tool call: name, success/failure, latency
        - Every retrieval: result count, top score
        - Run start / completion / error

        ## Deploy marker (optional — correlate failures with releases)
        ```python
        dt.mark_deploy("my-agent", version="v1.4.2", commit="abc1234", env="production")
        ```

        Full guide: docs/integrate-langchain-agent.md
    """).strip(),

    "python": textwrap.dedent("""
        # Instrument a custom Python agent with Dunetrace

        ## Install
        ```bash
        pip install dunetrace
        ```

        ## Option A — decorator style (simplest)
        ```python
        from dunetrace import Dunetrace

        dt = Dunetrace(endpoint="http://localhost:8001")

        @dt.tool
        def web_search(query: str) -> list:
            ...  # your tool implementation — args are hashed, never sent raw

        @dt.trace("my-agent")
        def run_agent(question: str) -> str:
            return web_search(question)[0]

        run_agent("What is the capital of France?")
        dt.shutdown()
        ```

        ## Option B — context manager (explicit control)
        ```python
        with dt.run("my-agent", user_input=query, tools=["web_search"]) as run:
            for step in agent_loop():
                run.tool_called("web_search", args)
                result = web_search(args)
                run.tool_responded("web_search", success=True, latency_ms=120)

                run.llm_called(model="gpt-4o", prompt_tokens=500)
                output = llm(...)
                run.llm_responded(completion_tokens=80, latency_ms=900)
        ```

        ## Option C — auto-instrumentation (patches openai / anthropic / httpx)
        ```python
        dt.init(agent_id="my-agent")   # patches clients globally

        @dt.agent(model="gpt-4o")
        def run_agent(query: str) -> str:
            return openai_client.chat.completions.create(...).choices[0].message.content
        ```

        ## Track tool calls explicitly
        ```python
        run.tool_called("my_tool", {"arg": value})   # before calling
        result = my_tool(value)
        run.tool_responded("my_tool", success=True, result=result, latency_ms=50)
        ```

        Full guide: docs/integrate-custom-python-agent.md
    """).strip(),

    "typescript": textwrap.dedent("""
        # Instrument a TypeScript / Node.js agent with Dunetrace

        No SDK required — post events directly to the ingest HTTP endpoint.

        ## Minimal client
        ```typescript
        import { randomUUID } from "crypto";
        import * as crypto from "crypto";

        const ENDPOINT = process.env.DUNETRACE_ENDPOINT ?? "http://localhost:8001";

        function sha256(value: unknown): string {
          return crypto.createHash("sha256")
            .update(JSON.stringify(value))
            .digest("hex")
            .slice(0, 16);
        }

        async function sendEvent(event: object) {
          await fetch(`${ENDPOINT}/v1/ingest`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ events: [event] }),
          });
        }
        ```

        ## Instrument a run
        ```typescript
        const runId = randomUUID();
        const agentId = "my-ts-agent";
        let step = 0;
        const t0 = Date.now() / 1000;

        // 1. Start the run
        await sendEvent({
          event_type: "run.started",
          run_id: runId, agent_id: agentId, agent_version: "1.0.0",
          step_index: step++, timestamp: t0,
          payload: { input_hash: sha256(userQuery), available_tools: ["web_search"] },
        });

        // 2. LLM call
        const llmStart = Date.now();
        const response = await openai.chat.completions.create({ ... });
        await sendEvent({
          event_type: "llm.called",
          run_id: runId, agent_id: agentId, agent_version: "1.0.0",
          step_index: step++, timestamp: llmStart / 1000,
          payload: {
            model: "gpt-4o-mini",
            prompt_tokens: response.usage.prompt_tokens,
            completion_tokens: response.usage.completion_tokens,
            latency_ms: Date.now() - llmStart,
            finish_reason: response.choices[0].finish_reason,
          },
        });

        // 3. Tool call
        await sendEvent({
          event_type: "tool.called",
          run_id: runId, agent_id: agentId, agent_version: "1.0.0",
          step_index: step, timestamp: Date.now() / 1000,
          payload: { tool_name: "web_search", args_hash: sha256(args) },
        });
        const toolResult = await webSearch(args);
        await sendEvent({
          event_type: "tool.responded",
          run_id: runId, agent_id: agentId, agent_version: "1.0.0",
          step_index: step++, timestamp: Date.now() / 1000,
          payload: { tool_name: "web_search", success: true, latency_ms: 200 },
        });

        // 4. Complete the run
        await sendEvent({
          event_type: "run.completed",
          run_id: runId, agent_id: agentId, agent_version: "1.0.0",
          step_index: step, timestamp: Date.now() / 1000,
          payload: { exit_reason: "final_answer" },
        });
        ```

        Full guide: docs/integrate-typescript-agent.md
    """).strip(),

    "tools": textwrap.dedent("""
        # How to track tool calls with Dunetrace

        ## Python — decorator style (automatic)
        ```python
        from dunetrace import Dunetrace

        dt = Dunetrace()

        @dt.tool                          # wraps the function; emits tool.called + tool.responded
        def web_search(query: str) -> list:
            ...

        @dt.trace("my-agent")
        def run(question):
            results = web_search(question)   # automatically tracked inside a run
            return results[0]
        ```
        Args are SHA-256 hashed before transmission — raw arguments never leave your process.

        ## Python — context manager (explicit)
        ```python
        with dt.run("my-agent") as run:
            run.tool_called("web_search", {"query": q})
            result = web_search(q)
            run.tool_responded(
                "web_search",
                success=True,
                result=result,      # hashed before sending
                latency_ms=120,
            )
        ```

        ## LangChain — automatic via callback
        ```python
        from dunetrace.integrations.langchain import DunetraceCallbackHandler

        callback = DunetraceCallbackHandler(dt, agent_id="my-agent", tools=["web_search"])
        agent.invoke(input, config={"callbacks": [callback]})
        # on_tool_start / on_tool_end are captured automatically
        ```

        ## TypeScript — manual events
        ```typescript
        // Before calling
        await sendEvent({ event_type: "tool.called", payload: {
          tool_name: "web_search", args_hash: sha256(args)
        }, ... });

        const result = await webSearch(args);

        // After calling
        await sendEvent({ event_type: "tool.responded", payload: {
          tool_name: "web_search", success: true, latency_ms: 150
        }, ... });
        ```

        ## What the detectors watch for
        - **TOOL_LOOP** — same tool called 3+ times in a 5-call window with identical args
        - **RETRY_STORM** — same tool fails 3+ times in a row
        - **CASCADING_TOOL_FAILURE** — 3+ consecutive failures across 2+ distinct tools
        - **TOOL_THRASHING** — agent oscillates between exactly two tools
        - **TOOL_AVOIDANCE** — agent gives a final answer without using any tools
        - **SLOW_STEP** — any tool call takes longer than 15 seconds
    """).strip(),

    "otel": textwrap.dedent("""
        # Zero-code monitoring via OpenTelemetry (OTLP receiver)

        If your agent or platform already emits OpenTelemetry traces, point its
        exporter at Dunetrace — no SDK, no code change required.

        ## Endpoint

        ```
        POST http://<dunetrace-host>:8001/v1/otlp/traces
        ```

        Auth: `Authorization: Bearer <api_key>` header (optional in dev mode).

        ## Python (OTLP/HTTP exporter)

        ```python
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": "my-agent"})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint="http://localhost:8001/v1/otlp/traces",
                headers={"Authorization": "Bearer YOUR_API_KEY"},
            )
        ))
        ```

        ## Node.js / TypeScript

        ```typescript
        import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
        const exporter = new OTLPTraceExporter({
            url: "http://localhost:8001/v1/otlp/traces",
            headers: { Authorization: "Bearer YOUR_API_KEY" },
        });
        ```

        ## No-code platforms (Langdock, Dify, etc.)

        Set the tracing/OTLP endpoint in your platform's workspace settings:
        ```
        https://<your-ngrok-or-deployed-host>/v1/otlp/traces
        ```
        Note: cloud platforms cannot reach localhost — use ngrok for local testing.

        ## Supported span types

        | Span | Events emitted |
        |---|---|
        | Root span (no parent) | run.started + run.completed / run.errored |
        | gen_ai.* attributes or name contains openai/anthropic/gpt/chat | llm.called + llm.responded |
        | tool.name attribute or name contains tool/function_call | tool.called + tool.responded |
        | vector_db.* / retrieval.* attributes or name contains pinecone/rag/search | retrieval.called + retrieval.responded |

        All 16 detectors activate automatically on each completed trace.

        ## Agent identity

        - `service.name` resource attribute → agent_id
        - `service.version` resource attribute → agent_version
        - Override per-request with `X-Dunetrace-Agent-Id` header
    """).strip(),
}

# Aliases map natural-language variants to canonical guide keys
_ALIASES: dict[str, str] = {
    "langchain": "langchain",
    "langgraph": "langchain",
    "lc": "langchain",
    "lc-graph": "langchain",
    "lc_graph": "langchain",
    "python": "python",
    "custom-python": "python",
    "custom python": "python",
    "py": "python",
    "typescript": "typescript",
    "ts": "typescript",
    "javascript": "typescript",
    "js": "typescript",
    "node": "typescript",
    "nodejs": "typescript",
    "tool": "tools",
    "tools": "tools",
    "tool-calls": "tools",
    "tool_calls": "tools",
    "tracking": "tools",
    "otel": "otel",
    "otlp": "otel",
    "opentelemetry": "otel",
    "open-telemetry": "otel",
    "langdock": "otel",
    "dify": "otel",
    "no-code": "otel",
    "zero-code": "otel",
}


@mcp.tool()
def get_instrumentation_guide(framework: str) -> str:
    """
    Return a step-by-step guide for instrumenting an AI agent with Dunetrace.

    Args:
        framework: One of:
            - "langchain"   — LangChain / LangGraph agents
            - "python"      — Custom Python agents (decorator, context manager,
                              auto-instrumentation, FastAPI, Flask)
            - "typescript"  — TypeScript / Node.js agents (raw HTTP ingest)
            - "tools"       — How to track tool calls specifically (all languages)
            - "otel"        — Zero-code OTel/OTLP receiver (Langdock, Dify, OpenLLMetry,
                              any platform that emits OTLP traces)

    Aliases accepted: langgraph, lc, custom-python, ts, javascript, js, node,
    tool-calls, tracking, otlp, opentelemetry, langdock, dify, no-code.
    """
    key = _ALIASES.get(framework.lower().strip())
    if key is None:
        supported = "langchain, python, typescript, tools"
        return (
            f"Unknown framework '{framework}'. Supported values: {supported}.\n\n"
            "Use list_agents to check what agents are already instrumented."
        )

    guide = _GUIDES[key]

    # Append the full doc if it's available on disk (richer content)
    doc_map = {
        "langchain": "integrate-langchain-agent.md",
        "python": "integrate-custom-python-agent.md",
        "typescript": "integrate-typescript-agent.md",
        "tools": "integrate-custom-python-agent.md",
        "otel": "integrate-langdock.md",
    }
    doc_path = _DOCS / doc_map[key]
    if doc_path.exists():
        full = doc_path.read_text(encoding="utf-8")
        return guide + "\n\n---\n\n" + full

    return guide


@mcp.tool()
def get_agent_token_stats(agent_id: str) -> str:
    """
    Per-window token usage and waste breakdown for an agent (1d / 7d / 30d).

    Shows how many tokens were spent, how many were wasted (on runs with detected
    failures), and the estimated API cost for each time window.  The 30-day view
    also breaks waste down by failure type so you can prioritise which failures
    to fix first.

    Use this when you want to understand the financial impact of agent failures or
    answer questions like "how much money is my agent wasting?".

    Args:
        agent_id: The agent ID to query.
    """
    try:
        data = client.get(f"/v1/agents/{agent_id}/token-stats")
    except Exception as exc:
        return f"Could not fetch token stats for '{agent_id}': {exc}"

    windows    = data.get("windows", {})
    waste_by_ft = data.get("waste_by_failure_type", [])

    def _fmt_tok(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}k"
        return str(n)

    def _fmt_cost(usd: float) -> str:
        if usd == 0:
            return "$0.00"
        if usd < 0.001:
            return f"${usd * 100:.4f}¢"
        if usd < 1:
            return f"${usd:.4f}"
        return f"${usd:.2f}"

    lines = [f"═══ Token stats: {agent_id} ═══", ""]

    for win in ("1d", "7d", "30d"):
        w = windows.get(win, {})
        if not w:
            continue
        label    = {"1d": "Last 24 h", "7d": "Last 7 days", "30d": "Last 30 days"}[win]
        wst_pct  = int(round((w.get("wasted_pct") or 0) * 100))
        lines += [
            f"── {label} ──",
            f"  Runs:            {w.get('run_count', 0):>6}  ({w.get('wasted_run_count', 0)} with failures)",
            f"  Total tokens:    {_fmt_tok(w.get('total_tokens', 0)):>8}",
            f"  Wasted tokens:   {_fmt_tok(w.get('wasted_tokens', 0)):>8}  ({wst_pct}% of total)",
            f"  Total cost:      {_fmt_cost(w.get('total_cost_usd', 0)):>10}",
            f"  Wasted cost:     {_fmt_cost(w.get('wasted_cost_usd', 0)):>10}  ({wst_pct}% of total)",
            "",
        ]

    if waste_by_ft:
        lines.append("Waste by failure type (30 days):")
        for ft in waste_by_ft:
            lines.append(
                f"  {ft['failure_type']:<35}  "
                f"{_fmt_tok(ft['wasted_tokens']):>7} tok  "
                f"{_fmt_cost(ft['wasted_cost_usd']):>9}  "
                f"({ft['affected_runs']} run{'s' if ft['affected_runs'] != 1 else ''})"
            )

    return "\n".join(lines)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import argparse, os

    parser = argparse.ArgumentParser(
        prog="dunetrace-mcp",
        description=(
            "Dunetrace MCP server — expose agent signals to Claude Code, Cursor, and Codex.\n\n"
            "Normally invoked automatically by your MCP client (stdio transport).\n"
            "Run with --sse to start an HTTP/SSE server instead."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sse",
        action="store_true",
        help="Run in SSE mode (HTTP server on --port) instead of stdio",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for SSE mode (default: 8000)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="dunetrace-mcp 0.1.0",
    )

    api_url = os.environ.get("DUNETRACE_API_URL", "http://localhost:8002")
    api_key = os.environ.get("DUNETRACE_API_KEY", "dt_dev_test")

    parser.epilog = (
        f"Environment:\n"
        f"  DUNETRACE_API_URL  Customer API base URL  (current: {api_url})\n"
        f"  DUNETRACE_API_KEY  Bearer token           (current: {'set' if api_key != 'dt_dev_test' else 'dt_dev_test (dev default)'})\n\n"
        f"Client config (Claude Code — add to ~/.claude.json):\n"
        f'  {{"mcpServers": {{"dunetrace": {{"command": "dunetrace-mcp", '
        f'"env": {{"DUNETRACE_API_URL": "{api_url}", "DUNETRACE_API_KEY": "..."}}}}}}}}'
    )

    args = parser.parse_args()

    if args.sse:
        mcp.run(transport="sse", port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
