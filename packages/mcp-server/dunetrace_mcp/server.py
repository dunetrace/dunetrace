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

import json
import pathlib
import re
import textwrap
from datetime import datetime, timezone
from importlib import resources
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
        - "How are my voice agents doing? Show me dropped calls."

        All timestamps returned are UTC ISO-8601.
        Confidence values are 0–1 (higher = more certain the failure occurred).
    """).strip(),
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _ts(epoch: float | None) -> str:
    if epoch is None:
        return "—"
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _ago(ts: float | str | None) -> str:
    """Format a timestamp (epoch float or ISO-8601 string) as a relative time."""
    if ts is None:
        return "—"
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            epoch: float = dt.timestamp()
        except (ValueError, AttributeError):
            return ts
    else:
        epoch = float(ts)
    secs = datetime.now(tz=timezone.utc).timestamp() - epoch
    if secs < 60:
        return f"{int(secs)}s ago"
    if secs < 3600:
        return f"{int(secs / 60)}m ago"
    if secs < 86400:
        return f"{int(secs / 3600)}h ago"
    return f"{int(secs / 86400)}d ago"


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
        return (
            f"No signals found for agent '{agent_id}'"
            + (f" with severity {severity}" if severity else "")
            + "."
        )

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
    icon = _score_icon(score)
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

    r = run.get("run", run)  # handle both flat and nested shapes
    signals = run.get("signals", [])
    events = run.get("events", [])

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
        for e in events[:40]:  # cap at 40 for readability
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
                pt = p.get("prompt_tokens", "?")
                detail = f"model={p.get('model', '?')}  p={pt}"
            elif e["event_type"] == "llm.responded":
                # completion_tokens and latency always land here; prompt_tokens here for LangChain
                pt = p.get("prompt_tokens")
                ct = p.get("completion_tokens", "?")
                detail = f"c={ct}"
                if pt:
                    detail = f"p={pt}  {detail}"
                if p.get("latency_ms"):
                    detail += f"  {p['latency_ms']}ms"
            elif e["event_type"] in ("run.started", "run.completed"):
                detail = p.get("exit_reason", "")
            lines.append(f"  [{e['step_index']:>3}] {ts_rel:>8}  {e['event_type']:<20} {detail}")
        if len(events) > 40:
            lines.append(f"  [Timeline truncated — showing first 40 of {len(events)} events]")

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

    params: dict = {"include_shadow": "false"}
    if severity:
        params["severity"] = severity.upper()
    if failure_type:
        params["failure_type"] = failure_type.upper()

    # Page through each agent rather than taking a single fixed slice.
    #
    # A flat per-agent cap silently truncated any agent with more signals than
    # the cap, and the truncated row count was then reported as the match total.
    # That made the numbers self-contradictory: an unfiltered query truncated one
    # busy agent to the cap, while each severity-filtered query got its own fresh
    # cap, so the severity breakdown summed to MORE than the unfiltered "total"
    # — while both were under the true figure.
    #
    # The API has no time-window parameter, so `since_hours` is applied here. It
    # returns rows detected_at DESC though, so once a page's oldest row predates
    # the cutoff every later page does too and we can stop — exact counts without
    # reading an agent's whole history.
    PAGE = 200
    MAX_PER_AGENT = 5000  # guard against unbounded paging on a huge agent

    all_signals: list = []
    truncated_agents: list[str] = []
    for a in agents:
        aid = a["agent_id"]
        offset = 0
        while offset < MAX_PER_AGENT:
            page = client.get(f"/v1/agents/{aid}/signals", limit=PAGE, offset=offset, **params)
            sigs = page.get("signals", [])
            if not sigs:
                break
            all_signals.extend(sigs)
            # Ordered newest-first: nothing older can qualify once we pass it.
            if cutoff and (sigs[-1].get("detected_at") or 0) < cutoff:
                break
            if not (page.get("page") or {}).get("has_more"):
                break
            offset += PAGE
        else:
            truncated_agents.append(aid)

    # Client-side filters — API also filters severity/failure_type but we apply
    # them here too so results are always correct regardless of API version.
    if cutoff:
        all_signals = [s for s in all_signals if (s.get("detected_at") or 0) >= cutoff]
    if severity:
        all_signals = [s for s in all_signals if s.get("severity") == severity.upper()]
    if failure_type:
        all_signals = [s for s in all_signals if s.get("failure_type") == failure_type.upper()]

    all_signals.sort(key=lambda s: s.get("detected_at") or 0, reverse=True)
    shown = all_signals[: min(limit, 200)]

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

    header = f"Signals ({len(shown)} shown, {len(all_signals)} matched)"
    if truncated_agents:
        header += (
            f" — count is a LOWER BOUND: hit the {MAX_PER_AGENT}-signal read cap on "
            f"{', '.join(truncated_agents)}"
        )
    lines = [header + ":\n"]
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
        sigs = client.get(f"/v1/agents/{aid}/signals", limit=500, include_shadow="false").get(
            "signals", []
        )
        signal = next((s for s in sigs if s["id"] == signal_id), None)
        if signal:
            break

    if not signal:
        return f"Signal {signal_id} not found." + (
            " Try omitting agent_id to search all agents." if agent_id else ""
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

    # Evidence dict — includes raw content fields (args, output, errors) where the detector used them
    ev = signal.get("evidence") or {}
    if ev:
        lines.append("Evidence:")
        for k, v in ev.items():
            if isinstance(v, list) and len(v) > 6:
                v = v[:6] + [f"…+{len(v) - 6} more"]
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
                    lines.append(f"     … ({len(code.splitlines()) - 20} more lines)")
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
                f"  {p['affected_runs']}/{p['total_runs']} runs ({p['rate'] * 100:.0f}%)"
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
            by_ft[t["failure_type"]][t["day"]] = (
                by_ft[t["failure_type"]].get(t["day"], 0) + t["count"]
            )
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
                    f"  {ft:<30}  {bar:<20}  {r['rate'] * 100:.0f}%"
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
                f"  {p['triggered_count']}/{p['total_runs']} runs ({p['rate'] * 100:.0f}%)"
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
    agent_meta = next((a for a in agents_data.get("agents", []) if a["agent_id"] == agent_id), None)
    if not agent_meta:
        return f"Agent '{agent_id}' not found. Use list_agents to see available agents."

    signals_data = client.get(f"/v1/agents/{agent_id}/signals", limit=50, include_shadow="false")
    signals = signals_data.get("signals", [])

    try:
        health = client.get(f"/v1/agents/{agent_id}/health-score")
    except Exception:
        health = None

    # Build summary
    score = health.get("score") if health else None
    score_icon = _score_icon(score)
    score_str = f"{score}/100" if score is not None else "N/A"

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
            run_pct = (
                f"{v / agent_meta['run_count'] * 100:.0f}% of runs"
                if agent_meta["run_count"]
                else ""
            )
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
            lines.append(f"  {name:<20} {bar}  {comp['score']}/{comp['max']}")

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
        sigs = r.get("signal_count", 0)
        sig_str = f"🔴 {sigs}" if sigs > 0 else "✅  0"
        # A run whose events have aged out of retention keeps its processed_runs
        # row but loses every derived field, because started_at / step_count are
        # computed from `events`. Rendering that as "— / 0 steps" is
        # indistinguishable from a real zero-step run, so say what happened.
        expired = r.get("started_at") is None and not r.get("step_count")
        started = "events expired" if expired else _ago(r.get("started_at"))
        steps = "—" if expired else str(r.get("step_count") or 0)
        lines.append(f"{r['run_id'][:12]:<12} {started:<22} {dur:>6} {steps:>5}  {sig_str}")

    page = data.get("page", {})
    lines.append(f"\n{len(runs)} of {page.get('total', len(runs))} runs shown.")
    expired_count = sum(1 for r in runs if r.get("started_at") is None and not r.get("step_count"))
    if expired_count:
        lines.append(
            f"{expired_count} run(s) show 'events expired': the run was analysed and its "
            f"signals are retained, but its raw events have aged out of "
            f"EVENT_RETENTION_DAYS, so duration and step count can no longer be derived."
        )
    return "\n".join(lines)


# ── voice / call tools ───────────────────────────────────────────────────────
#
# A "call" is a conversation for a voice agent, read through a voice lens:
# duration, how the call ended, silence percentage, agent-vs-caller talk ratio,
# and per-stage cost (STT / LLM / TTS / telephony). These wrap the /v1/calls
# endpoints so an operator can ask "how are my voice agents doing" without
# leaving the client. Voice failure signals are reported through the normal
# signal tools too; these add the voice-specific call metrics on top.


_COMPLETION_ICONS = {"natural": "✅", "dropped": "🔴", "escalated": "🟠"}


@mcp.tool()
def list_voice_calls(
    agent_id: str = "",
    completion_status: str = "",
    cost_bucket: str = "",
    limit: int = 20,
) -> str:
    """
    List recent voice calls with call-level metrics: duration, how the call
    ended, silence percentage, voice signal count, and cost. A "call" is a voice
    agent's conversation (the runs that share one conversation id).

    Use this for questions like "how are my voice agents doing", "show me dropped
    calls", or "which calls escalated to a human".

    Args:
        agent_id:          Filter to one voice agent (optional).
        completion_status: Filter by how the call ended:
                           natural | dropped | escalated (optional).
        cost_bucket:       Filter by cost: low (<$0.10) | medium ($0.10-$1) |
                           high (>$1) (optional).
        limit:             Max calls to return (default 20, max 100).
    """
    data = client.get(
        "/v1/calls",
        agent_id=agent_id or None,
        completion_status=completion_status or None,
        cost_bucket=cost_bucket or None,
        limit=min(limit, 100),
    )
    calls = data.get("calls", [])
    if not calls:
        return "No voice calls found for the given filters."

    lines = [
        "Voice calls\n",
        f"{'CALL':>5}  {'AGENT':<18} {'WHEN':<10} {'DUR':>6} {'SILENCE':>7} "
        f"{'SIGS':>5} {'COST':>9}  STATUS",
        "─" * 84,
    ]
    for c in calls:
        icon = _COMPLETION_ICONS.get(c.get("completion_status", ""), "⚪")
        dur = f"{c.get('duration_seconds', 0):.0f}s"
        silence = f"{c.get('silence_pct', 0):.0f}%"
        sigs = c.get("voice_signal_count", 0)
        sig_str = f"🔴 {sigs}" if sigs else "0"
        lines.append(
            f"{c.get('id', '?'):>5}  {str(c.get('agent_id', ''))[:18]:<18} "
            f"{_ago(c.get('last_run_at')):<10} {dur:>6} {silence:>7} "
            f"{sig_str:>5} ${c.get('cost_usd', 0):>8.4f}  {icon} {c.get('completion_status', '?')}"
        )

    page = data.get("page", {})
    lines.append(f"\n{len(calls)} of {page.get('total', len(calls))} calls shown.")
    return "\n".join(lines)


@mcp.tool()
def get_call_detail(conversation_id: int) -> str:
    """
    One voice call's full picture: call-level metrics, per-stage cost breakdown,
    the voice failure signals detected, and (when recorded) links to the call
    audio.

    Args:
        conversation_id: The integer call id (from list_voice_calls).
    """
    try:
        c = client.get(f"/v1/calls/{conversation_id}")
    except Exception as exc:
        return f"Error: {exc}"

    icon = _COMPLETION_ICONS.get(c.get("completion_status", ""), "⚪")
    lines = [
        f"Call #{c.get('id')}  ({c.get('agent_id', '')})",
        f"  Ended:      {icon} {c.get('completion_status', '?')}",
        f"  Duration:   {c.get('duration_seconds', 0):.0f}s over {c.get('run_count', 0)} run(s)",
        f"  Silence:    {c.get('silence_pct', 0):.0f}%",
    ]
    ratio = c.get("agent_talk_ratio")
    if ratio is not None:
        lines.append(f"  Talk ratio: agent {ratio * 100:.0f}% / caller {(1 - ratio) * 100:.0f}%")
    lines.append(f"  Cost:       ${c.get('cost_usd', 0):.4f}")
    breakdown = c.get("cost_breakdown") or {}
    parts = "  ".join(f"{k}=${v:.4f}" for k, v in breakdown.items() if v)
    if parts:
        lines.append(f"              {parts}")

    signals = c.get("voice_signals") or []
    lines.append("")
    if signals:
        lines.append(f"Voice signals ({len(signals)}):")
        lines.extend(f"  🔴 {s}" for s in signals)
    else:
        lines.append("Voice signals: ✅ none")

    recordings = c.get("recordings") or []
    if recordings:
        lines.append("")
        lines.append(f"Recordings ({len(recordings)}):")
        for rec in recordings:
            dur = rec.get("duration_seconds")
            suffix = f"  ({dur:.0f}s)" if isinstance(dur, (int, float)) else ""
            lines.append(f"  🎧 {rec.get('url', '')}{suffix}")

    return "\n".join(lines)


# ── fix tracking tools ───────────────────────────────────────────────────────


@mcp.tool()
def get_fix_status(signal_id: int, agent_id: str = "") -> str:
    """
    Check whether a fix applied for a signal reduced recurrence.

    Verdict meanings:
      verified         — signal has not recurred since the fix was applied
      likely_fixed     — recurrence dropped significantly but not to zero
      still_occurring  — signal is still firing at a similar rate
      insufficient_data — not enough post-fix runs to make a determination

    Args:
        signal_id: The integer signal ID to check.
        agent_id:  Agent ID (optional; not required for lookup).
    """
    try:
        data = client.get(f"/v1/signals/{signal_id}/fix-status")
    except Exception as exc:
        return f"Error: {exc}"

    verdict = data.get("verdict", "unknown")
    verdict_icons = {
        "verified": "✅",
        "likely_fixed": "🟡",
        "still_occurring": "🔴",
        "insufficient_data": "❓",
    }
    icon = verdict_icons.get(verdict, "⚪")

    lines = [
        f"Fix status for signal #{signal_id}",
        f"Verdict:  {icon} {verdict.upper().replace('_', ' ')}",
        "",
    ]

    runs_before = data.get("runs_before") or data.get("total_runs_before")
    runs_after = data.get("runs_after") or data.get("total_runs_after")
    recur_before = data.get("recurrence_before") or data.get("signal_recurrence_before")
    recur_after = data.get("recurrence_after") or data.get("signal_recurrence_after")
    runs_evaluated = data.get("runs_evaluated") or data.get("total_runs_evaluated")

    if runs_before is not None:
        lines.append(
            f"  Runs before fix:   {runs_before}"
            + (f"  (recurrence: {recur_before})" if recur_before is not None else "")
        )
    if runs_after is not None:
        lines.append(
            f"  Runs after fix:    {runs_after}"
            + (f"  (recurrence: {recur_after})" if recur_after is not None else "")
        )
    if runs_evaluated is not None:
        lines.append(f"  Runs evaluated:    {runs_evaluated}")

    fix_applied_at = data.get("fix_applied_at") or data.get("applied_at")
    if fix_applied_at:
        lines.append(f"  Fix applied:       {_ago(fix_applied_at)}")

    return "\n".join(lines)


@mcp.tool()
def list_agent_fixes(agent_id: str) -> str:
    """
    List all fixes that have been applied for an agent's signals.

    Shows which signals were fixed, how they were applied (clipboard copy or
    GitHub PR), the fix type, and whether the fix was verified effective.

    Args:
        agent_id: The agent ID to query.
    """
    try:
        data = client.get(f"/v1/agents/{agent_id}/fixes")
    except Exception as exc:
        return f"Error: {exc}"

    fixes = data if isinstance(data, list) else data.get("fixes", [])

    if not fixes:
        return f"No fixes recorded for agent '{agent_id}'."

    lines = [
        f"Fixes for: {agent_id}  ({len(fixes)} total)\n",
        f"{'SIGNAL':>7}  {'TYPE':<18}  {'VIA':<10}  {'VERDICT':<20}  WHEN",
        "─" * 75,
    ]
    for f in fixes:
        sig_id = str(f.get("signal_id", "—"))
        fix_type = (f.get("fix_type") or "—")[:18]
        via = (f.get("applied_via") or "—")[:10]
        verdict = (f.get("verdict") or "—")[:20]
        when = _ago(f.get("applied_at"))
        lines.append(f"{sig_id:>7}  {fix_type:<18}  {via:<10}  {verdict:<20}  {when}")

    return "\n".join(lines)


# ── explain / root cause tools ────────────────────────────────────────────────


@mcp.tool()
def trigger_explain(signal_id: int, agent_id: str = "") -> str:
    """
    Trigger LLM-powered root cause analysis for a signal and return a fix
    suggestion.

    Runs natively against Dunetrace's own stored events for the signal's run
    — no external tracing system involved — and returns a structured
    explanation with either a suggested runtime policy (for failure types
    Dunetrace can guard against directly) or a prompt/code diff.

    Note: this makes an LLM call and may take 5–15 seconds.

    Args:
        signal_id: The integer signal ID to analyze.
        agent_id:  Agent ID (optional; informational only).
    """
    try:
        result = client.post(f"/v1/signals/{signal_id}/explain", {})
    except Exception as exc:
        return f"Error: {exc}"

    lines = [
        f"Root cause analysis for signal #{signal_id}",
        "",
        "Root cause:",
        f"  {result.get('root_cause', '—')}",
        "",
    ]

    if result.get("fix_category") == "dunetrace_native":
        policy = result.get("suggested_policy", {})
        lines.append("Suggested fix: a Dunetrace runtime policy (applies directly, no code change)")
        lines.append(f"  {policy}")
        lines.append("  Apply via POST /v1/policies.")
        return "\n".join(lines)

    fix_type = result.get("fix_type", "unknown")
    apply_blocked = result.get("apply_blocked", True)

    fix_type_labels = {
        "prompt_addition": "Prompt addition (add a sentence to your system prompt)",
        "code_change": "Code / infra change required",
        "no_auto_apply": "No auto-apply available for this failure type",
    }

    lines.append(f"Fix ({fix_type_labels.get(fix_type, fix_type)}):")
    lines.append(f"  {result.get('fix_content', '—')}")
    lines.append("")

    if fix_type == "code_change" and not apply_blocked:
        lines.append(
            "Can be applied automatically: POST /v1/signals/{id}/open-pr opens a draft GitHub PR."
        )
    elif fix_type == "no_auto_apply":
        lines.append("Never auto-applied — this is a security signal. Review manually.")
    else:
        lines.append("Apply blocked: copy this fix into your system prompt or code manually.")

    return "\n".join(lines)


# ── policy tools ──────────────────────────────────────────────────────────────


@mcp.tool()
def list_policies(agent_id: str = "") -> str:
    """
    List runtime policies configured for agents.

    Policies trigger actions (stop, switch_model, inject_prompt, log) when a
    metric threshold is breached during a live run.

    Args:
        agent_id: Filter to one agent (optional; lists all agents' policies if omitted).
    """
    try:
        params: dict = {}
        if agent_id:
            params["agent_id"] = agent_id
        data = client.get("/v1/policies", **params)
    except Exception as exc:
        return f"Error: {exc}"

    policies = data if isinstance(data, list) else data.get("policies", [])

    if not policies:
        qualifier = f" for agent '{agent_id}'" if agent_id else ""
        return f"No policies found{qualifier}. Create one with create_policy or via the dashboard."

    lines = [
        f"Policies ({len(policies)}):\n",
        f"{'NAME':<25}  {'AGENT':<15}  {'TRIGGER':<12}  {'OP':<4}  {'VALUE':<10}  {'ACTION':<18}  STATUS",
        "─" * 100,
    ]
    for p in policies:
        cond = p.get("condition") or {}
        if isinstance(cond, str):
            try:
                cond = json.loads(cond)
            except Exception:
                cond = {}
        action = p.get("action") or {}
        if isinstance(action, str):
            try:
                action = json.loads(action)
            except Exception:
                action = {}

        metric = cond.get("metric", "—")[:12]
        op = cond.get("operator", "—")[:4]
        threshold = str(cond.get("threshold", "—"))[:10]
        act_type = action.get("type", "—")[:18]
        enabled = "enabled" if p.get("enabled", True) else "disabled"
        name = (p.get("name") or "—")[:25]
        p_agent = (p.get("agent_id") or "*")[:15]

        lines.append(
            f"{name:<25}  {p_agent:<15}  {metric:<12}  {op:<4}  {threshold:<10}  {act_type:<18}  {enabled}"
        )

    return "\n".join(lines)


@mcp.tool()
def create_policy(name: str, agent_id: str, condition: str, action: str) -> str:
    """
    Create a runtime policy that triggers an action when a metric threshold is
    crossed during a live agent run.

    Args:
        name:      Policy name (human-readable label).
        agent_id:  Agent to apply the policy to (use '*' for all agents).
        condition: JSON string with keys: metric, operator, threshold.
                   Example: '{"metric": "cost_usd", "operator": "gt", "threshold": 5.0}'
                   Metrics: cost_usd, tool_calls, llm_calls, retries, step_count,
                            duration_s, prompt_tokens, completion_tokens
                   Operators: gt, gte, lt, lte, eq
        action:    JSON string with key 'type' and optional params.
                   Examples:
                     '{"type": "stop"}'
                     '{"type": "switch_model", "model": "gpt-4o-mini"}'
                     '{"type": "inject_prompt", "text": "Stop looping."}'
                     '{"type": "log"}'
    """
    try:
        cond_parsed = json.loads(condition)
        act_parsed = json.loads(action)
    except json.JSONDecodeError as exc:
        return f"Error: invalid JSON in condition or action: {exc}"

    try:
        result = client.post(
            "/v1/policies",
            {
                "name": name,
                "agent_id": agent_id,
                "condition": cond_parsed,
                "action": act_parsed,
            },
        )
    except Exception as exc:
        return f"Error: {exc}"

    policy_id = result.get("id", "—")
    enabled = "enabled" if result.get("enabled", True) else "disabled"

    lines = [
        f"Policy created: {name}  (id: {policy_id})",
        "",
        f"  Name:       {result.get('name', name)}",
        f"  Agent:      {result.get('agent_id', agent_id)}",
        f"  Condition:  metric={cond_parsed.get('metric', '?')}  "
        f"operator={cond_parsed.get('operator', '?')}  "
        f"threshold={cond_parsed.get('threshold', '?')}",
        f"  Action:     {act_parsed.get('type', '?')}",
        f"  Status:     {enabled}",
    ]
    return "\n".join(lines)


@mcp.tool()
def toggle_policy(policy_id: str, enabled: bool) -> str:
    """
    Enable or disable a runtime policy.

    Args:
        policy_id: The policy ID to toggle.
        enabled:   True to enable, False to disable.
    """
    try:
        result = client.patch(f"/v1/policies/{policy_id}/toggle", {"enabled": enabled})
    except Exception as exc:
        return f"Error: {exc}"

    new_state = "enabled" if result.get("enabled", enabled) else "disabled"
    name = result.get("name", policy_id)
    return f"Policy '{name}' ({policy_id}) is now {new_state}."


@mcp.tool()
def delete_policy(policy_id: str) -> str:
    """
    Delete a runtime policy.

    Args:
        policy_id: The policy ID to delete.
    """
    try:
        client.delete(f"/v1/policies/{policy_id}")
    except Exception as exc:
        return f"Error: {exc}"

    return f"Policy {policy_id} deleted."


# ── custom detector tools ─────────────────────────────────────────────────────


@mcp.tool()
def list_custom_detectors(agent_id: str = "") -> str:
    """
    List custom detectors with their status, fire rate, and run counts.

    Status values:
      shadow  — evaluating silently; results visible via include_shadow but no alerts
      active  — firing live alerts when triggered
      paused  — evaluation suspended

    Args:
        agent_id: Filter to one agent (optional; lists all if omitted).
    """
    try:
        params: dict = {}
        if agent_id:
            params["agent_id"] = agent_id
        data = client.get("/v1/custom-detectors", **params)
    except Exception as exc:
        return f"Error: {exc}"

    detectors = data if isinstance(data, list) else data.get("detectors", [])

    if not detectors:
        qualifier = f" for agent '{agent_id}'" if agent_id else ""
        return f"No custom detectors found{qualifier}. Create one with create_custom_detector."

    status_badge = {"shadow": "[shadow]", "active": "[active]", "paused": "[paused]"}

    lines = [
        f"Custom detectors ({len(detectors)}):\n",
        f"{'NAME':<30}  {'STATUS':<9}  {'AGENT':<15}  FIRE RATE  TOTAL RUNS",
        "─" * 80,
    ]
    for d in detectors:
        name = (d.get("name") or "—")[:30]
        status = status_badge.get(d.get("status", ""), d.get("status", "—"))
        d_agent = (d.get("agent_id") or "*")[:15]
        total = d.get("total_runs") or 0
        fired = d.get("shadow_fire_count") or 0
        rate_str = f"{fired}/{total} ({fired * 100 // total}%)" if total > 0 else "—"
        lines.append(f"{name:<30}  {status:<9}  {d_agent:<15}  {rate_str:<10}  {total}")

    return "\n".join(lines)


@mcp.tool()
def create_custom_detector(description: str, agent_id: str = "*") -> str:
    """
    Create a custom detector from a plain-English description.

    This tool first translates the description to a structured config using LLM
    (preview step), then creates the detector in shadow mode so it evaluates
    without firing alerts. Use activate_custom_detector to go live.

    Args:
        description: Plain-English description of what to detect.
                     Example: "Alert when total tool calls exceed 20 in a single run"
        agent_id:    Agent to apply this detector to ('*' for all agents).
    """
    try:
        preview = client.post(
            "/v1/custom-detectors/preview",
            {"description": description, "agent_id": agent_id},
        )
    except Exception as exc:
        return f"Error during preview translation: {exc}"

    if preview.get("requires_content"):
        return (
            f"Could not translate this description: {preview.get('reason', 'unsupported')}. "
            "Rephrase using either a structural metric (tool call counts, latency, "
            "token usage, step count, retry count, etc.) or a content condition with "
            "a specific value to look for (e.g. \"when a tool error mentions 'timeout'\"). "
            "See docs/detectors.md for the supported metrics and content fields."
        )

    # Derive a name from the description
    name = re.sub(r"[^a-z0-9]+", "-", description[:50].lower().strip()).strip("-")
    name = re.sub(r"-+", "-", name)[:40]

    lines = [
        f'Preview of translated config for: "{description[:60]}"',
        "",
    ]
    conditions = preview.get("conditions") or []
    metrics = preview.get("metrics") or []
    if metrics:
        lines.append(f"  Metrics:    {', '.join(str(m) for m in metrics)}")
    if conditions:
        lines.append("  Conditions:")
        for cond in conditions:
            if isinstance(cond, dict):
                lines.append(
                    f"    {cond.get('metric', '?')} {cond.get('operator', '?')} "
                    f"{cond.get('threshold', '?')}"
                )
            else:
                lines.append(f"    {cond}")
    lines.append("")

    try:
        result = client.post(
            "/v1/custom-detectors",
            {
                "name": name,
                "description": description,
                "agent_id": agent_id,
                "config_json": preview,
            },
        )
    except Exception as exc:
        return "\n".join(lines) + f"\nError creating detector: {exc}"

    detector_id = result.get("id", "—")
    lines += [
        f"Custom detector created in shadow mode: {name}",
        f"  ID:     {detector_id}",
        f"  Agent:  {agent_id}",
        "",
        "It will evaluate on every matching run but will NOT fire live alerts",
        'until you activate it with activate_custom_detector("' + str(detector_id) + '").',
        "Monitor shadow results via list_custom_detectors or the dashboard.",
    ]
    return "\n".join(lines)


@mcp.tool()
def activate_custom_detector(detector_id: str) -> str:
    """
    Activate a custom detector so it fires live alerts.

    The detector must currently be in shadow or paused status. Once active,
    every run that matches will produce a real alert.

    Args:
        detector_id: The custom detector ID to activate.
    """
    try:
        result = client.patch(f"/v1/custom-detectors/{detector_id}", {"status": "active"})
    except Exception as exc:
        return f"Error: {exc}"

    name = result.get("name", detector_id)
    return (
        f"Custom detector '{name}' ({detector_id}) activated.\n"
        "It will now fire live alerts when triggered."
    )


@mcp.tool()
def pause_custom_detector(detector_id: str) -> str:
    """
    Pause a custom detector so it stops evaluating entirely.

    Use this to temporarily disable a detector without deleting it. Resume
    with activate_custom_detector (to go live) or the dashboard.

    Args:
        detector_id: The custom detector ID to pause.
    """
    try:
        result = client.patch(f"/v1/custom-detectors/{detector_id}", {"status": "paused"})
    except Exception as exc:
        return f"Error: {exc}"

    name = result.get("name", detector_id)
    return f"Custom detector '{name}' ({detector_id}) paused.\nIt will not evaluate until resumed."


@mcp.tool()
def delete_custom_detector(detector_id: str) -> str:
    """
    Delete a custom detector permanently.

    This also deletes all historical evaluation results for this detector.
    Use pause_custom_detector if you want to temporarily stop evaluation.

    Args:
        detector_id: The custom detector ID to delete.
    """
    try:
        client.delete(f"/v1/custom-detectors/{detector_id}")
    except Exception as exc:
        return f"Error: {exc}"

    return f"Custom detector {detector_id} deleted."


# ── issues tool ───────────────────────────────────────────────────────────────


@mcp.tool()
def list_agent_issues(agent_id: str, status: str = "open") -> str:
    """
    List open (or resolved) issues for an agent.

    Issues are aggregated across runs — the same failure type appearing across
    multiple runs is tracked as a single issue. An issue auto-resolves after 5
    consecutive clean runs.

    Args:
        agent_id: The agent ID to query.
        status:   'open', 'resolved', or 'all' (default: 'open').
    """
    try:
        data = client.get(f"/v1/agents/{agent_id}/issues", status=status)
    except Exception as exc:
        return f"Error: {exc}"

    issues = data if isinstance(data, list) else data.get("issues", [])

    if not issues:
        return f"No {status} issues for agent '{agent_id}'."

    lines = [
        f"Issues for: {agent_id}  (status={status})\n",
        f"{'FAILURE TYPE':<35}  {'OPENED':<12}  {'LAST SEEN':<12}  {'COUNT':>5}  STATUS",
        "─" * 85,
    ]
    for issue in issues:
        ft = (issue.get("failure_type") or "—")[:35]
        opened = _ago(issue.get("opened_at") or issue.get("first_seen"))
        last_seen = _ago(issue.get("last_seen") or issue.get("updated_at"))
        count = issue.get("occurrence_count") or issue.get("signal_count") or 0
        iss_status = issue.get("status", "open")
        lines.append(f"{ft:<35}  {opened:<12}  {last_seen:<12}  {count:>5}  {iss_status}")

    return "\n".join(lines)


# ── coding-agent issue tools (Phase 4.2) ───────────────────────────────────────


@mcp.tool()
def get_issue(issue_id: int) -> str:
    """
    Full context for a single issue — metadata, affected runs, root cause,
    and a suggested fix. This is the deep-dive tool for triaging one
    specific issue found via search_issues or list_agent_issues.

    Note: this may trigger an LLM call to generate the root cause/fix
    (same native root-cause analysis trigger_explain uses) and can take
    5-15 seconds. If no LLM key is configured on the backend, root_cause/
    suggested_fix are omitted but the rest of the report still returns.

    code_references (source file/line the issue maps to) is always empty
    for now — Dunetrace has no source-mapping capability yet (planned,
    not yet built).

    Args:
        issue_id: The integer issue ID (from search_issues or list_agent_issues).
    """
    try:
        data = client.get(f"/v1/issues/{issue_id}")
    except Exception as exc:
        return f"Error: {exc}"

    lines = [
        f"Issue #{data['id']}: {data['failure_type']} on {data['agent_id']}",
        f"Status: {data['status']}  (opened {_ago(data.get('first_seen'))}, "
        f"last seen {_ago(data.get('last_seen'))})",
        f"Affected runs: {data.get('affected_runs_count', 0)} total",
    ]
    if data.get("manually_resolved") and data.get("resolution_notes"):
        lines.append(f"Resolution notes: {data['resolution_notes']}")
    lines.append("")

    affected_runs = data.get("affected_runs") or []
    if affected_runs:
        lines.append("AFFECTED RUNS (most recent):")
        lines.append(f"  {'RUN ID':<15}  {'WHEN':<12}  {'STEP':>4}  CONF")
        for r in affected_runs[:10]:
            run_id = str(r.get("run_id") or "—")[:15]
            when = _ago(r.get("detected_at"))
            step = r.get("step_index", "—")
            conf = r.get("confidence")
            conf_str = f"{conf:.0%}" if conf is not None else "—"
            lines.append(f"  {run_id:<15}  {when:<12}  {str(step):>4}  {conf_str}")
        lines.append("")

    if data.get("root_cause"):
        lines.append("ROOT CAUSE:")
        lines.append(f"  {data['root_cause']}")
        lines.append("")

    if data.get("suggested_fix"):
        lines.append("SUGGESTED FIX:")
        lines.append(f"  {data['suggested_fix']}")
        lines.append("")

    if not data.get("root_cause") and not data.get("suggested_fix"):
        lines.append(
            "No root cause/fix available — configure ANTHROPIC_API_KEY or OPENAI_API_KEY "
            "on the backend, or call trigger_explain on one of the affected runs' signals."
        )
        lines.append("")

    code_refs = data.get("code_references") or []
    lines.append("CODE REFERENCES:")
    if code_refs:
        for ref in code_refs:
            lines.append(f"  {ref}")
    else:
        lines.append("  Not available yet — source mapping hasn't been configured for this agent.")
    lines.append("")

    lines.append(f'To resolve: resolve_issue(issue_id={data["id"]}, resolution_notes="...")')

    return "\n".join(lines)


@mcp.tool()
def search_issues(
    query: str = "",
    status: str = "",
    agent_id: str = "",
    failure_type: str = "",
    limit: int = 20,
) -> str:
    """
    Search issues across ALL agents in your org — for triage across an
    entire fleet rather than one agent at a time (list_agent_issues is
    scoped to a single agent).

    `query` is a plain substring match against agent_id/failure_type/
    resolution_notes — not full-text search or relevance ranking (issues
    have no free-text title/description field beyond resolution_notes,
    which is only populated once an issue has been manually resolved).

    Args:
        query:        Substring to match. Empty matches everything.
        status:       Filter: 'open', 'resolved', or 'reopened'. Empty = all.
        agent_id:     Restrict to one agent. Empty = all agents.
        failure_type: Restrict to one failure type, e.g. TOOL_LOOP. Empty = all.
        limit:        Max results (default 20).
    """
    try:
        data = client.get(
            "/v1/issues/search",
            q=query or None,
            status=status or None,
            agent_id=agent_id or None,
            failure_type=failure_type or None,
            limit=limit,
        )
    except Exception as exc:
        return f"Error: {exc}"

    issues = data.get("issues", [])
    total = data.get("page", {}).get("total", len(issues))

    if not issues:
        return "No issues match these filters."

    lines = [
        f"Issues matching filters ({total} total, showing {len(issues)}):\n",
        f"{'ID':>5}  {'AGENT':<20}  {'FAILURE TYPE':<30}  {'LAST SEEN':<12}  {'RUNS':>5}  STATUS",
        "─" * 95,
    ]
    for issue in issues:
        agent = (issue.get("agent_id") or "—")[:20]
        ft = (issue.get("failure_type") or "—")[:30]
        last_seen = _ago(issue.get("last_seen"))
        count = issue.get("affected_runs", 0)
        iss_status = issue.get("status", "open")
        lines.append(
            f"{issue.get('id'):>5}  {agent:<20}  {ft:<30}  {last_seen:<12}  {count:>5}  {iss_status}"
        )

    return "\n".join(lines)


@mcp.tool()
def resolve_issue(issue_id: int, resolution_notes: str) -> str:
    """
    Manually mark an issue resolved with notes on what fixed it — for when
    you've made a code/prompt change and want to close the loop, distinct
    from Dunetrace's automatic resolve-after-5-clean-runs detection.

    If the failure recurs later, the issue reopens automatically regardless
    of whether it was auto- or manually-resolved — resolution_notes is kept
    as a historical record either way.

    Args:
        issue_id:         The integer issue ID to resolve.
        resolution_notes: What fixed it, e.g. "Added a per-tool call limit
                           of 3 in the agent's retry loop."
    """
    try:
        client.post(f"/v1/issues/{issue_id}/resolve", {"resolution_notes": resolution_notes})
    except Exception as exc:
        return f"Error: {exc}"

    return f"Issue #{issue_id} marked resolved.\nNotes: {resolution_notes}"


# ── failure pattern detail tool ───────────────────────────────────────────────


@mcp.tool()
def get_failure_pattern_detail(agent_id: str, failure_type: str) -> str:
    """
    Deep dive into a specific failure type for an agent.

    Returns evidence aggregates, a 14-day trend (with ASCII bar chart), any
    signals that co-occur with this failure, and the top example runs.

    Args:
        agent_id:     The agent ID to analyze.
        failure_type: The detector type, e.g. TOOL_LOOP, COST_SPIKE, CONTEXT_BLOAT.
    """
    try:
        data = client.get(f"/v1/agents/{agent_id}/failure-patterns/{failure_type.upper()}")
    except Exception as exc:
        return f"Error: {exc}"

    lines = [f"Failure pattern: {failure_type.upper()} for {agent_id}\n"]

    # Overview
    total_signals = data.get("total_signals") or data.get("signal_count") or 0
    affected_runs = data.get("affected_runs") or data.get("run_count") or 0
    first_seen = data.get("first_seen")
    last_seen = data.get("last_seen")

    lines.append(f"Occurrences:   {total_signals} signals across {affected_runs} runs")
    if first_seen:
        lines.append(f"First seen:    {_ago(first_seen)}")
    if last_seen:
        lines.append(f"Last seen:     {_ago(last_seen)}")
    lines.append("")

    # 14-day trend
    trend = data.get("daily_trend") or data.get("trend") or []
    if trend:
        lines.append("14-day trend:")
        max_count = max((t.get("count", 0) for t in trend), default=1) or 1
        for t in trend[-14:]:
            day = (t.get("day") or t.get("date") or "")[-5:]  # MM-DD
            count = t.get("count", 0)
            bar = "█" * int(count / max_count * 20)
            lines.append(f"  {day}  {bar:<20}  {count}")
        lines.append("")

    # Step distribution
    step_dist = data.get("step_distribution") or {}
    if step_dist:
        lines.append("Step distribution (where failures occur):")
        for step, cnt in sorted(step_dist.items(), key=lambda x: int(x[0]))[:10]:
            bar = "█" * min(int(cnt / max(step_dist.values()) * 20), 20)
            lines.append(f"  step {step:<4}  {bar:<20}  {cnt}")
        lines.append("")

    # Co-occurring signals
    co_signals = data.get("co_occurring_signals") or data.get("co_signals") or []
    if co_signals:
        lines.append("Co-occurring signals:")
        for cs in co_signals[:8]:
            ft = cs.get("failure_type") or cs.get("type") or "—"
            cnt = cs.get("count") or cs.get("co_count") or 0
            lines.append(f"  {ft:<35}  {cnt} times")
        lines.append("")

    # Top example runs
    top_runs = data.get("top_runs") or data.get("example_runs") or []
    if top_runs:
        lines.append("Top example runs:")
        lines.append(f"  {'RUN ID':<15}  {'WHEN':<12}  {'STEP':>4}  CONF")
        for r in top_runs[:8]:
            run_id = str(r.get("run_id") or "—")[:15]
            when = _ago(r.get("detected_at") or r.get("started_at"))
            step = r.get("step_index", "—")
            conf = r.get("confidence")
            conf_str = f"{conf:.0%}" if conf is not None else "—"
            lines.append(f"  {run_id:<15}  {when:<12}  {str(step):>4}  {conf_str}")

    if not (trend or co_signals or top_runs or total_signals):
        lines.append("No pattern data available for this failure type yet.")

    return "\n".join(lines)


# ── run comparison tool ───────────────────────────────────────────────────────


@mcp.tool()
def compare_runs(run_id_1: str, run_id_2: str, agent_id: str = "") -> str:
    """
    Compare two runs side by side: duration, step count, token usage, signals,
    and exit reason.

    Useful for spotting regressions between a good run and a bad one, or
    comparing runs before and after a deploy.

    Args:
        run_id_1:  First run UUID.
        run_id_2:  Second run UUID.
        agent_id:  Agent ID (optional; informational only).
    """
    try:
        raw1 = client.get(f"/v1/runs/{run_id_1}")
        raw2 = client.get(f"/v1/runs/{run_id_2}")
    except Exception as exc:
        return f"Error: {exc}"

    def _extract(raw: dict) -> dict:
        r = raw.get("run", raw)
        sigs = raw.get("signals", [])
        dur = None
        if r.get("completed_at") and r.get("started_at"):
            dur = r["completed_at"] - r["started_at"]
        return {
            "run_id": r.get("run_id", "—"),
            "agent": f"{r.get('agent_id', '—')}  v{r.get('agent_version', '?')}",
            "started": _ago(r.get("started_at")),
            "duration": f"{dur:.1f}s" if dur is not None else "—",
            "steps": str(r.get("step_count", "—")),
            "tokens": f"{r['total_tokens']:,}" if r.get("total_tokens") else "—",
            "exit": r.get("exit_reason", "—"),
            "signals": (
                ", ".join(f"{s['failure_type']} [{s['severity']}]" for s in sigs) or "none"
            ),
        }

    a = _extract(raw1)
    b = _extract(raw2)

    rows = [
        ("Run ID", a["run_id"][:24], b["run_id"][:24]),
        ("Agent", a["agent"][:24], b["agent"][:24]),
        ("Started", a["started"], b["started"]),
        ("Duration", a["duration"], b["duration"]),
        ("Steps", a["steps"], b["steps"]),
        ("Tokens", a["tokens"], b["tokens"]),
        ("Exit reason", a["exit"][:24], b["exit"][:24]),
        ("Signals", a["signals"][:40], b["signals"][:40]),
    ]

    col_w = 26
    lines = [
        "Run comparison\n",
        f"{'':22}  {'RUN 1':<{col_w}}  {'RUN 2':<{col_w}}",
        "─" * (22 + col_w * 2 + 4),
    ]
    for label, v1, v2 in rows:
        marker = "  " if v1 == v2 else "!!"
        lines.append(f"{label:<22}  {v1:<{col_w}}  {v2:<{col_w}}  {marker}")

    lines.append("")
    lines.append("!! = differs between runs")
    return "\n".join(lines)


# ── documentation resources ───────────────────────────────────────────────────
# Exposed as MCP resources so clients can browse them and LLMs can answer
# instrumentation / how-to questions without calling any live-data tools.


def _read_doc(name: str) -> str:
    """Return a bundled doc, preferring the copy shipped inside the package.

    Two lookups, in this order, because the two install modes put the docs in
    different places:

    1. ``dunetrace_mcp/_docs/`` — written into the wheel at build time (see
       setup.py). This is the only copy a `pip install dunetrace-mcp` has;
       resolving relative to ``__file__`` lands in site-packages, where there is
       no repo and every doc resource returned "(doc not found)".
    2. ``<repo-root>/docs/`` — an editable install or a source checkout, where
       the package dir has no ``_docs`` and the real docs sit four levels up.
       Reading them live means local edits show up without a rebuild.
    """
    try:
        packaged = resources.files("dunetrace_mcp").joinpath("_docs", name)
        if packaged.is_file():
            return packaged.read_text(encoding="utf-8")
    except (ModuleNotFoundError, FileNotFoundError, OSError):
        pass  # fall through to the source checkout

    p = _DOCS / name
    if p.exists():
        return p.read_text(encoding="utf-8")
    return f"(doc not found: {name} — not packaged in this build and no repo checkout at {_DOCS})"


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


@mcp.resource("dunetrace://docs/integrate-haystack")
def doc_integrate_haystack() -> str:
    """Full guide: add Dunetrace monitoring to a Haystack 2.x pipeline using
    DunetraceHaystackTracer — covers simple pipelines, RAG, Agent component,
    AsyncPipeline, token extraction, and troubleshooting."""
    return _read_doc("integrate-haystack-agent.md")


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
            ...  # your tool implementation — args are sent as-is

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

        ## Option C — auto-instrumentation (simplest for existing code)
        Patches every installed client: openai, anthropic, langchain, crewai,
        httpx, requests. No call-site changes.
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
        run.tool_responded("my_tool", success=True, output=str(result), latency_ms=50)
        ```

        Full guide: docs/integrate-custom-python-agent.md
    """).strip(),
    "typescript": textwrap.dedent("""
        # Instrument a TypeScript / Node.js agent with Dunetrace

        ```bash
        npm install dunetrace
        ```

        ## Option A — auto-instrumentation (simplest)
        Patches the OpenAI and Anthropic SDKs plus outbound `fetch`, so every
        client is tracked — including ones constructed inside a library you
        don't control. Streaming calls are tracked too.
        ```typescript
        import { Dunetrace, autoInstrument } from "dunetrace";
        import OpenAI from "openai";

        const dt = new Dunetrace({ endpoint: "http://localhost:8001" });
        autoInstrument({ openai: OpenAI });   // add `anthropic: Anthropic` if used

        const openai = new OpenAI();          // built after the patch — still tracked

        await dt.run("my-agent", { model: "gpt-4o" }, async (run) => {
          await openai.chat.completions.create({ model: "gpt-4o", messages });
          run.finalAnswer();
        });
        await dt.shutdown();
        ```
        Pass the imported class as shown. The zero-argument `autoInstrument()`
        auto-detects via `require`, which only resolves under CommonJS.

        ## Option B — wrap one client
        ```typescript
        const openai = dt.wrapOpenAI(new OpenAI());
        const claude = dt.wrapAnthropic(new Anthropic());
        const search = dt.tool(webSearch);     // emits tool.called + tool.responded
        ```

        ## Option C — wrap a whole agent function
        ```typescript
        const agent = dt.trace(myAgent, "my-agent", { model: "gpt-4o" });
        await agent(question);                 // opens and closes its own run
        ```

        ## Vercel AI SDK
        ```typescript
        import { generateText } from "ai";
        import { traceGenerateText } from "dunetrace";

        await traceGenerateText(dt, "my-agent", {}, generateText, { model, prompt });
        ```

        ## Option D — no SDK, raw HTTP
        Only if you can't add the dependency. Post events to the ingest endpoint
        yourself:
        ```typescript
        import { randomUUID } from "crypto";

        const ENDPOINT = process.env.DUNETRACE_ENDPOINT ?? "http://localhost:8001";

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
          payload: { input_text: userQuery, available_tools: ["web_search"] },
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
          payload: { tool_name: "web_search", args: JSON.stringify(args) },
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
        Args are sent to the backend as-is.

        ## Python — context manager (explicit)
        ```python
        with dt.run("my-agent") as run:
            run.tool_called("web_search", {"query": q})
            result = web_search(q)
            run.tool_responded(
                "web_search",
                success=True,
                output_length=len(str(result)),
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

        ## TypeScript — wrap the function (automatic)
        ```typescript
        import { Dunetrace } from "dunetrace";

        const dt = new Dunetrace();
        const search = dt.tool(webSearch);          // emits tool.called + tool.responded

        await dt.run("my-agent", {}, async () => {
          const results = await search(query);      // tracked automatically
        });
        ```
        Outbound HTTP is tracked too once `autoInstrument()` has run — each
        request emits tool.called / tool.responded named by hostname.

        ## TypeScript — explicit events
        ```typescript
        await dt.run("my-agent", {}, async (run) => {
          run.toolCalled("web_search", { query });
          const result = await webSearch(query);
          run.toolResponded("web_search", true, String(result).length, 150);
        });
        ```

        ## What the detectors watch for
        - **TOOL_LOOP** — same tool called 3+ times in a 5-call window with identical args
        - **RETRY_STORM** — same tool fails 3+ times in a row
        - **CASCADING_TOOL_FAILURE** — 3+ consecutive failures across 2+ distinct tools
        - **TOOL_THRASHING** — agent oscillates between exactly two tools
        - **TOOL_AVOIDANCE** — agent gives a final answer without using any tools
        - **SLOW_STEP** — any tool call takes longer than 15 seconds
    """).strip(),
    "haystack": textwrap.dedent("""
        # Instrument a Haystack 2.x pipeline with Dunetrace

        ## Install
        ```bash
        pip install 'dunetrace[haystack]'
        ```

        ## Register the tracer (one-time, before any pipeline.run())
        ```python
        import haystack.tracing
        from dunetrace import Dunetrace
        from dunetrace.integrations.haystack import DunetraceHaystackTracer

        dt = Dunetrace(endpoint="http://localhost:8001")  # or set DUNETRACE_ENDPOINT

        haystack.tracing.enable_tracing(
            DunetraceHaystackTracer(
                dt,
                agent_id="my-pipeline",      # identifies this agent in the dashboard
                system_prompt=SYSTEM_PROMPT, # optional — helps pattern analysis
                model="gpt-4o-mini",         # optional
                tools=["web_search"],        # optional — for TOOL_AVOIDANCE detector
            )
        )
        ```

        ## Run your pipeline normally — nothing else changes
        ```python
        from haystack import Pipeline
        from haystack.components.generators.chat import OpenAIChatGenerator
        from haystack.dataclasses import ChatMessage

        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini"))

        result = pipeline.run({
            "llm": {"messages": [ChatMessage.from_user("What is the capital of France?")]}
        })
        dt.shutdown(timeout=5)
        ```

        ## What gets tracked automatically
        - Every LLM call: model, token counts (prompt + completion), latency, finish reason
        - Every retriever call: result count, top similarity score
        - Every tool invocation via ToolInvoker / ComponentTool: success/failure, latency
        - Run start / completion / error

        ## RAG pipeline (retriever + generator)
        ```python
        from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
        from haystack.components.builders import ChatPromptBuilder

        rag = Pipeline()
        rag.add_component("retriever", InMemoryBM25Retriever(document_store=store))
        rag.add_component("prompt",    ChatPromptBuilder(template=[...]))
        rag.add_component("llm",       OpenAIChatGenerator(model="gpt-4o-mini"))
        rag.connect("retriever.documents", "prompt.documents")
        rag.connect("prompt.prompt",       "llm.messages")
        # Both RETRIEVAL_CALLED/RESPONDED and LLM_CALLED/RESPONDED fire automatically.
        ```

        ## AsyncPipeline support
        Works out of the box — ContextVar-based span tracking isolates concurrent coroutines.

        ## Deploy marker (optional — correlate failures with releases)
        ```python
        dt.mark_deploy("my-pipeline", version="v2.1.0", commit="abc1234", env="production")
        ```

        Full guide: docs/integrate-haystack-agent.md
    """).strip(),
    "voice": textwrap.dedent("""
        # Instrument a voice agent with Dunetrace

        Voice hooks live on the run object — no extra install, `pip install dunetrace`.
        They sit alongside the LLM / tool / retrieval hooks so a voice turn is
        captured end to end: speech-to-text in, LLM and tool work, text-to-speech
        out, plus the VAD and turn-taking signals a voice loop runs on.

        ## Instrument a turn
        ```python
        from dunetrace import Dunetrace

        dt = Dunetrace(endpoint="http://localhost:8001")

        # conversation_id (the call id) rolls every turn up into one call
        with dt.run("voice-support", conversation_id=call_id) as run:
            # 1. Speech-to-text for the caller's turn (advances the step)
            run.transcription_received(
                transcript, confidence=0.92, latency_ms=140, audio_seconds=3.1
            )

            # 2. Your normal LLM / tool work, tracked as usual
            run.llm_called("gpt-4o", prompt_tokens=800)
            reply = generate_reply(transcript)
            run.llm_responded(completion_tokens=120, latency_ms=900)

            # 3. Text-to-speech for the agent's reply (does NOT advance the step)
            run.tts_generated(
                reply, latency_ms=210, audio_seconds=4.4,
                provider="elevenlabs", voice_id="rachel", model="eleven_turbo_v2",
            )
        ```

        ## Real-time signals (annotate the turn, never advance the step)
        ```python
        # speech_start | speech_end | silence | barge_in
        run.voice_activity_detected("barge_in", duration_ms=200)
        # agent_speaking | user_speaking | both_speaking | neither
        run.turn_taking("agent_speaking", from_agent=True)
        run.recording_metadata(
            "https://audio.example/call.wav", duration_seconds=61, format="wav",
        )
        ```

        ## Notes
        - Only `transcription_received` advances the step counter, so a full voice
          turn stays ~one step and the always-on step detectors don't false-fire.
        - `audio_seconds` is optional but enables per-minute STT cost attribution.
        - `provider` / `voice_id` / `model` on `tts_generated` let Dunetrace pull
          provider-side generation history (ElevenLabs) back to the exact turn.
        - Query the resulting calls from an MCP client with `list_voice_calls`
          and `get_call_detail`.

        Full guide: docs/integrations/voice-frameworks.md
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

        ## Also from the Python SDK (`pip install 'dunetrace[otel]'`)

        If you already run the Dunetrace Python SDK, it bridges OpenTelemetry in
        both directions in-process — no OTLP endpoint or network hop needed.

        ### Export Dunetrace's own events to your OTel backend
        Ship every run / LLM / tool / retrieval / voice event to Datadog, Grafana
        Tempo, Honeycomb, Signoz, etc. as OTel spans (GenAI semantic conventions),
        alongside Dunetrace's normal ingest. Env-driven, zero code change:
        ```bash
        export DUNETRACE_OTEL_ENABLED=1
        export DUNETRACE_OTEL_ENDPOINT=https://otlp.example:4317
        export DUNETRACE_OTEL_HEADERS="DD-API-KEY=xxxxx"   # provider auth
        export DUNETRACE_OTEL_PROTOCOL=grpc                # or http/protobuf
        ```
        Export runs on a bounded background pipeline with a circuit breaker, so a
        slow or dead collector never blocks or breaks the agent.

        ### Feed an existing OTel pipeline into Dunetrace's detectors
        Already instrumented with OpenLLMetry / Traceloop / OpenLIT? Attach the
        receiver as a second span processor — no dt.run() calls required:
        ```python
        from opentelemetry.sdk.trace import TracerProvider
        from dunetrace import Dunetrace
        from dunetrace.integrations.otel_receiver import DunetraceOTelReceiver

        dt = Dunetrace(api_key="dt_live_...")
        provider = TracerProvider()          # your existing pipeline, unchanged
        DunetraceOTelReceiver.attach(provider, dt, agent_id="my-agent")
        ```
        gen_ai.* spans are translated to Dunetrace events and run through the full
        detector suite — same result as the HTTP endpoint above, without leaving
        the process.
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
    "haystack": "haystack",
    "haystack-ai": "haystack",
    "haystack2": "haystack",
    "hs": "haystack",
    "voice": "voice",
    "voice-agent": "voice",
    "voice_agent": "voice",
    "voice-agents": "voice",
    "stt": "voice",
    "tts": "voice",
    "speech": "voice",
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
            - "typescript"  — TypeScript / Node.js agents (npm SDK, auto-instrumentation)
            - "tools"       — How to track tool calls specifically (all languages)
            - "haystack"    — Haystack 2.x pipelines (simple, RAG, Agent component)
            - "voice"       — Voice agents (STT / TTS / VAD / turn-taking hooks)
            - "otel"        — OpenTelemetry: zero-code OTLP receiver plus the SDK's
                              in-process export and receiver bridge (dunetrace[otel])

    Aliases accepted: langgraph, lc, custom-python, ts, javascript, js, node,
    tool-calls, tracking, haystack-ai, hs, voice-agent, stt, tts, speech,
    otlp, opentelemetry, langdock, dify, no-code.
    """
    key = _ALIASES.get(framework.lower().strip())
    if key is None:
        supported = "langchain, python, typescript, tools, haystack, voice, otel"
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
        "haystack": "integrate-haystack-agent.md",
        "voice": "integrations/voice-frameworks.md",
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
    Per-window token usage breakdown for an agent (1d / 7d / 30d).

    Reports three distinct spend metrics per window:
      • Failed-run tokens — all spend on runs that had a failure (attribution).
      • Excess (avoidable) — the portion above a healthy baseline; the realistic
        "waste" figure and the one worth acting on.
      • Prevented (saved) — spend actually stopped by an in-path policy/approval
        block. Post-hoc detectors prevent nothing and never count here.
    Plus a 30-day projection of avoidable waste, and (30d) a by-failure-type
    breakdown so you can prioritise which failures to fix first.

    Use this to understand the financial impact of agent failures or answer
    "how much money is my agent wasting, and how much could I actually save?".

    Args:
        agent_id: The agent ID to query.
    """
    try:
        data = client.get(f"/v1/agents/{agent_id}/token-stats")
    except Exception as exc:
        return f"Could not fetch token stats for '{agent_id}': {exc}"

    windows = data.get("windows", {})
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
            return f"{usd * 100:.4f}¢"
        if usd < 1:
            return f"${usd:.4f}"
        return f"${usd:.2f}"

    lines = [f"═══ Token stats: {agent_id} ═══", ""]

    for win in ("1d", "7d", "30d"):
        w = windows.get(win, {})
        if not w:
            continue
        label = {"1d": "Last 24 h", "7d": "Last 7 days", "30d": "Last 30 days"}[win]
        total_tok = w.get("total_tokens") or 0
        failed_tok = w.get("failed_run_tokens", w.get("wasted_tokens", 0))
        failed_cost = w.get("failed_run_cost_usd", w.get("wasted_cost_usd", 0))
        failed_run_ct = w.get("failed_run_count", w.get("wasted_run_count", 0))
        excess_tok = w.get("excess_tokens", 0)
        excess_pct = int(round(excess_tok / total_tok * 100)) if total_tok else 0
        fail_pct = int(round(failed_tok / total_tok * 100)) if total_tok else 0
        prevented_tok = w.get("prevented_tokens", 0)
        blocked_ct = w.get("blocked_run_count", 0)
        proj_cost = w.get("projected_monthly_excess_cost_usd", 0)
        lines += [
            f"── {label} ──",
            f"  Runs:              {w.get('run_count', 0):>6}  ({failed_run_ct} with failures)",
            f"  Total tokens:      {_fmt_tok(total_tok):>8}   {_fmt_cost(w.get('total_cost_usd', 0))}",
            f"  Failed-run tokens: {_fmt_tok(failed_tok):>8}  ({fail_pct}% of total, {_fmt_cost(failed_cost)}) — attribution",
            f"  Excess (avoidable):{_fmt_tok(excess_tok):>8}  ({excess_pct}% of total, {_fmt_cost(w.get('excess_cost_usd', 0))}) — the realistic waste",
            f"  Prevented (saved): {_fmt_tok(prevented_tok):>8}  ({blocked_ct} run{'s' if blocked_ct != 1 else ''} stopped in-path, {_fmt_cost(w.get('prevented_cost_usd', 0))})",
        ]
        if win != "1d":
            lines.append(
                f"  Projected/mo:      {_fmt_cost(proj_cost):>10}  avoidable waste if left unfixed"
            )
        lines.append("")

    if waste_by_ft:
        lines.append("Failed-run spend by failure type (30 days):")
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
        version="dunetrace-mcp 0.1.4",
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
        mcp.settings.port = args.port
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
