"""
Unit tests for dunetrace_mcp tools.

All tests run offline — the HTTP client is patched so no running stack is needed.

Run:
    cd packages/mcp-server && python -m pytest tests/ -v
    python -m unittest discover -s packages/mcp-server/tests
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch

import dunetrace_mcp.server as srv

# ── shared fixtures ────────────────────────────────────────────────────────────

AGENT_LIST = {
    "agents": [
        {
            "agent_id": "my-agent",
            "last_seen": time.time() - 300,
            "run_count": 50,
            "signal_count": 12,
            "critical_count": 1,
            "high_count": 8,
            "sparkline": [0, 1, 2, 0, 3, 0, 1],
            "failure_types": {"TOOL_LOOP": 8, "COST_SPIKE": 4},
        },
        {
            "agent_id": "clean-agent",
            "last_seen": time.time() - 3600,
            "run_count": 20,
            "signal_count": 0,
            "critical_count": 0,
            "high_count": 0,
            "sparkline": [0, 0, 0, 0, 0, 0, 0],
            "failure_types": {},
        },
    ],
    "page": {"total": 2, "offset": 0, "limit": 100, "has_more": False},
}

SIGNAL_LIST = {
    "signals": [
        {
            "id": 42,
            "failure_type": "TOOL_LOOP",
            "severity": "HIGH",
            "run_id": "run-abc-123",
            "agent_id": "my-agent",
            "agent_version": "v1.2.3",
            "step_index": 5,
            "confidence": 0.9,
            "detected_at": time.time() - 600,
            "evidence": {
                "tool": "web_search",
                "count": 6,
                "first_step": 2,
                "last_step": 7,
                "args_identical": True,
                "args": ["aabbccdd"] * 6,
            },
            "alerted": True,
            "shadow": False,
            "co_signal_count": 1,
            "title": "Tool loop detected: `web_search` called 6×",
            "what": "The agent called web_search 6 times with identical args.",
            "why_it_matters": "Wastes tokens and delays response.",
            "evidence_summary": "web_search called 6× in steps 2–7.",
            "suggested_fixes": [
                {
                    "description": "Deduplicate tool calls",
                    "language": "python",
                    "code": "seen = set()\nif args not in seen:\n    seen.add(args)\n    call_tool(args)",
                }
            ],
        },
        {
            "id": 43,
            "failure_type": "COST_SPIKE",
            "severity": "CRITICAL",
            "run_id": "run-def-456",
            "agent_id": "my-agent",
            "agent_version": "v1.2.3",
            "step_index": 10,
            "confidence": 0.95,
            "detected_at": time.time() - 7200,
            "evidence": {"total_tokens": 80000, "baseline_p75": 15000},
            "alerted": False,
            "shadow": False,
            "co_signal_count": 0,
            "title": "Cost spike: 80k tokens (5× baseline)",
            "what": "Token usage was 5× above the P75 baseline.",
            "why_it_matters": "Runaway cost.",
            "evidence_summary": "80k tokens vs 15k baseline.",
            "suggested_fixes": [
                {
                    "description": "Add token budget guard",
                    "language": "python",
                    "code": "",
                }
            ],
        },
    ],
    "page": {"total": 2, "offset": 0, "limit": 500, "has_more": False},
}

HEALTH_SCORE = {
    "agent_id": "my-agent",
    "score": 62,
    "components": {
        "failure_rate": {
            "score": 24,
            "max": 40,
            "value": 24.0,
            "label": "% runs with failures",
        },
        "loop_avoidance": {
            "score": 18,
            "max": 25,
            "value": 16.0,
            "label": "% runs with loops",
        },
        "token_efficiency": {
            "score": 10,
            "max": 20,
            "value": None,
            "label": "avg prompt tokens",
        },
        "latency": {
            "score": 10,
            "max": 15,
            "value": 4200.0,
            "label": "avg LLM latency ms",
        },
    },
    "sample_runs": 50,
    "baseline_ready": True,
}

RUN_DETAIL = {
    "run_id": "run-abc-123",
    "agent_id": "my-agent",
    "agent_version": "v1.2.3",
    "started_at": time.time() - 120,
    "completed_at": time.time() - 90,
    "exit_reason": "run.completed",
    "step_count": 8,
    "total_tokens": None,
    "signals": [SIGNAL_LIST["signals"][0]],
    "events": [
        {
            "event_type": "run.started",
            "step_index": 0,
            "timestamp": time.time() - 120,
            "payload": {},
            "parent_run_id": None,
        },
        {
            "event_type": "llm.called",
            "step_index": 1,
            "timestamp": time.time() - 119,
            "payload": {
                "model": "gpt-4o-mini",
                "prompt_tokens": 500,
                "completion_tokens": 100,
                "latency_ms": 800,
            },
            "parent_run_id": None,
        },
        {
            "event_type": "tool.called",
            "step_index": 2,
            "timestamp": time.time() - 115,
            "payload": {"tool_name": "web_search", "success": True, "latency_ms": 200},
            "parent_run_id": None,
        },
        {
            "event_type": "run.completed",
            "step_index": 8,
            "timestamp": time.time() - 90,
            "payload": {"exit_reason": "final_answer"},
            "parent_run_id": None,
        },
    ],
}

INSIGHTS = {
    "input_patterns": [
        {
            "input_hash": "abc123",
            "failure_type": "TOOL_LOOP",
            "triggered_count": 8,
            "total_runs": 8,
            "rate": 1.0,
        },
        {
            "input_hash": "def456",
            "failure_type": "COST_SPIKE",
            "triggered_count": 2,
            "total_runs": 5,
            "rate": 0.4,
        },
    ],
    "signal_trends": [
        {
            "failure_type": "TOOL_LOOP",
            "agent_version": "v1.2.3",
            "day": "2026-05-12",
            "count": 4,
        },
        {
            "failure_type": "TOOL_LOOP",
            "agent_version": "v1.2.3",
            "day": "2026-05-13",
            "count": 3,
        },
        {
            "failure_type": "COST_SPIKE",
            "agent_version": "v1.2.3",
            "day": "2026-05-13",
            "count": 1,
        },
    ],
    "failure_rates": [
        {
            "failure_type": "TOOL_LOOP",
            "day": "2026-05-12",
            "total_runs": 8,
            "affected_runs": 8,
            "rate": 1.0,
        },
        {
            "failure_type": "COST_SPIKE",
            "day": "2026-05-13",
            "total_runs": 5,
            "affected_runs": 2,
            "rate": 0.4,
        },
    ],
    "systemic_patterns": [
        {
            "failure_type": "TOOL_LOOP",
            "total_runs": 16,
            "affected_runs": 16,
            "rate": 1.0,
            "first_seen": time.time() - 86400 * 5,
            "last_seen": time.time() - 600,
            "is_systemic": True,
        },
    ],
    "versions": [],
    "time_to_tool": {
        "p25": None,
        "p50": None,
        "p75": None,
        "avg_steps": None,
        "runs_with_tool": 0,
        "total_runs": 0,
        "daily_trend": [],
    },
    "hourly_pattern": [],
    "deploy_events": [],
}

TOKEN_STATS = {
    "agent_id": "my-agent",
    "windows": {
        "1d": {
            "run_count": 10,
            "wasted_run_count": 3,
            "total_tokens": 50000,
            "wasted_tokens": 15000,
            "prompt_tokens": 40000,
            "completion_tokens": 10000,
            "total_cost_usd": 0.05,
            "wasted_cost_usd": 0.015,
            "wasted_pct": 0.3,
        },
        "7d": {
            "run_count": 50,
            "wasted_run_count": 12,
            "total_tokens": 250000,
            "wasted_tokens": 60000,
            "prompt_tokens": 200000,
            "completion_tokens": 50000,
            "total_cost_usd": 0.25,
            "wasted_cost_usd": 0.06,
            "wasted_pct": 0.24,
        },
        "30d": {
            "run_count": 200,
            "wasted_run_count": 45,
            "total_tokens": 1_000_000,
            "wasted_tokens": 225000,
            "prompt_tokens": 800000,
            "completion_tokens": 200000,
            "total_cost_usd": 1.0,
            "wasted_cost_usd": 0.225,
            "wasted_pct": 0.225,
        },
    },
    "waste_by_failure_type": [
        {
            "failure_type": "TOOL_LOOP",
            "wasted_tokens": 150000,
            "wasted_cost_usd": 0.15,
            "affected_runs": 30,
        },
        {
            "failure_type": "COST_SPIKE",
            "wasted_tokens": 75000,
            "wasted_cost_usd": 0.075,
            "affected_runs": 15,
        },
    ],
}


def _mock_get(path, **params):
    """Route mock API calls to the right fixture based on path."""
    if path == "/v1/agents":
        return AGENT_LIST
    if path.endswith("/signals"):
        return SIGNAL_LIST
    if path.endswith("/health-score"):
        return HEALTH_SCORE
    if path.endswith("/insights"):
        return INSIGHTS
    if path.endswith("/token-stats"):
        return TOKEN_STATS
    if "/runs/" in path:
        return RUN_DETAIL
    if path.endswith("/runs"):
        return {
            "runs": [
                {
                    "run_id": "run-abc-123",
                    "agent_id": "my-agent",
                    "agent_version": "v1.2.3",
                    "started_at": time.time() - 120,
                    "completed_at": time.time() - 90,
                    "exit_reason": "run.completed",
                    "step_count": 8,
                    "total_tokens": None,
                    "signal_count": 1,
                    "has_signals": True,
                }
            ],
            "page": {"total": 1, "offset": 0, "limit": 100, "has_more": False},
        }
    raise ValueError(f"Unmocked path: {path}")


class TestListAgents(unittest.TestCase):
    def setUp(self):
        self.patcher = patch("dunetrace_mcp.client.get", side_effect=_mock_get)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_returns_table(self):
        out = srv.list_agents()
        self.assertIn("my-agent", out)
        self.assertIn("clean-agent", out)

    def test_shows_signal_counts(self):
        out = srv.list_agents()
        self.assertIn("12", out)  # signal count for my-agent

    def test_shows_failure_types(self):
        out = srv.list_agents()
        self.assertIn("TOOL_LOOP", out)

    def test_shows_last_seen(self):
        out = srv.list_agents()
        self.assertIn("ago", out)

    def test_no_agents(self):
        with patch(
            "dunetrace_mcp.client.get",
            return_value={"agents": [], "page": {"total": 0}},
        ):
            out = srv.list_agents()
        self.assertIn("No agents found", out)


class TestGetAgentSignals(unittest.TestCase):
    def setUp(self):
        self.patcher = patch("dunetrace_mcp.client.get", side_effect=_mock_get)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_lists_signals(self):
        out = srv.get_agent_signals("my-agent")
        self.assertIn("TOOL_LOOP", out)
        self.assertIn("COST_SPIKE", out)

    def test_severity_filter(self):
        out = srv.get_agent_signals("my-agent", severity="CRITICAL")
        self.assertIn("COST_SPIKE", out)
        self.assertNotIn("TOOL_LOOP", out)

    def test_no_match_severity(self):
        out = srv.get_agent_signals("my-agent", severity="LOW")
        self.assertIn("No signals found", out)

    def test_shows_confidence(self):
        out = srv.get_agent_signals("my-agent")
        self.assertIn("90%", out)

    def test_shows_suggested_fix(self):
        out = srv.get_agent_signals("my-agent")
        self.assertIn("Deduplicate", out)

    def test_empty_agent(self):
        with patch(
            "dunetrace_mcp.client.get",
            return_value={"signals": [], "page": {"total": 0}},
        ):
            out = srv.get_agent_signals("ghost-agent")
        self.assertIn("No signals found", out)


class TestGetAgentHealth(unittest.TestCase):
    def setUp(self):
        self.patcher = patch("dunetrace_mcp.client.get", side_effect=_mock_get)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_shows_score(self):
        out = srv.get_agent_health("my-agent")
        self.assertIn("62/100", out)

    def test_shows_components(self):
        out = srv.get_agent_health("my-agent")
        self.assertIn("failure_rate", out)
        self.assertIn("loop_avoidance", out)
        self.assertIn("token_efficiency", out)
        self.assertIn("latency", out)

    def test_baseline_ready_shown(self):
        out = srv.get_agent_health("my-agent")
        self.assertIn("yes", out)

    def test_none_score(self):
        no_score = {**HEALTH_SCORE, "score": None, "sample_runs": 2}
        with patch("dunetrace_mcp.client.get", return_value=no_score):
            out = srv.get_agent_health("my-agent")
        self.assertIn("N/A", out)
        self.assertIn("need ≥3", out)


class TestGetRunDetail(unittest.TestCase):
    def setUp(self):
        self.patcher = patch("dunetrace_mcp.client.get", side_effect=_mock_get)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_shows_metadata(self):
        out = srv.get_run_detail("run-abc-123")
        self.assertIn("run-abc-123", out)
        self.assertIn("my-agent", out)
        self.assertIn("v1.2.3", out)
        self.assertIn("30.0s", out)  # duration ~30s

    def test_shows_signal(self):
        out = srv.get_run_detail("run-abc-123")
        self.assertIn("TOOL_LOOP", out)
        self.assertIn("[HIGH]", out)

    def test_shows_event_timeline(self):
        out = srv.get_run_detail("run-abc-123")
        self.assertIn("run.started", out)
        self.assertIn("llm.called", out)
        self.assertIn("tool.called", out)
        self.assertIn("web_search", out)

    def test_clean_run_no_signals(self):
        clean = {**RUN_DETAIL, "signals": []}
        with patch("dunetrace_mcp.client.get", return_value=clean):
            out = srv.get_run_detail("run-clean")
        self.assertIn("clean run", out)

    def test_large_event_list_capped(self):
        many_events = RUN_DETAIL["events"] * 15  # 60 events
        big_run = {**RUN_DETAIL, "events": many_events}
        with patch("dunetrace_mcp.client.get", return_value=big_run):
            out = srv.get_run_detail("run-big")
        self.assertIn("Timeline truncated", out)
        self.assertIn("40 of", out)


class TestSearchSignals(unittest.TestCase):
    def setUp(self):
        self.patcher = patch("dunetrace_mcp.client.get", side_effect=_mock_get)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_basic_search(self):
        out = srv.search_signals()
        self.assertIn("TOOL_LOOP", out)
        self.assertIn("COST_SPIKE", out)

    def test_severity_filter(self):
        out = srv.search_signals(severity="CRITICAL")
        self.assertIn("COST_SPIKE", out)
        self.assertNotIn("TOOL_LOOP", out)

    def test_failure_type_filter(self):
        out = srv.search_signals(failure_type="TOOL_LOOP")
        self.assertIn("TOOL_LOOP", out)

    def test_since_hours_excludes_old(self):
        # The COST_SPIKE signal is 2 hours old; within 1h cutoff it should vanish
        out = srv.search_signals(since_hours=1)
        self.assertNotIn("COST_SPIKE", out)
        self.assertIn("TOOL_LOOP", out)  # 10 min old, passes

    def test_since_hours_zero_results(self):
        # Very short window — nothing should match
        out = srv.search_signals(since_hours=1, severity="CRITICAL")
        self.assertIn("No signals found", out)

    def test_agent_id_filter(self):
        out = srv.search_signals(agent_id="my-agent")
        self.assertIn("my-agent", out)

    def test_limit(self):
        out = srv.search_signals(limit=1)
        # Should show 1 shown, 2 matched
        self.assertIn("1 shown", out)

    def test_shows_signal_id(self):
        out = srv.search_signals()
        self.assertIn("id=42", out)
        self.assertIn("id=43", out)

    def test_no_results(self):
        with patch(
            "dunetrace_mcp.client.get",
            return_value={"signals": [], "page": {"total": 0}},
        ):
            out = srv.search_signals()
        self.assertIn("No signals found", out)


class TestGetSignalDetail(unittest.TestCase):
    def setUp(self):
        self.patcher = patch("dunetrace_mcp.client.get", side_effect=_mock_get)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_found_by_id(self):
        out = srv.get_signal_detail(42, "my-agent")
        self.assertIn("Signal #42", out)
        self.assertIn("TOOL_LOOP", out)

    def test_shows_full_evidence(self):
        out = srv.get_signal_detail(42, "my-agent")
        self.assertIn("Evidence:", out)
        self.assertIn("args_identical", out)

    def test_shows_suggested_fixes(self):
        out = srv.get_signal_detail(42, "my-agent")
        self.assertIn("Suggested fixes", out)
        self.assertIn("Deduplicate", out)

    def test_shows_code_snippet(self):
        out = srv.get_signal_detail(42, "my-agent")
        self.assertIn("```python", out)
        self.assertIn("seen = set()", out)

    def test_shows_why_it_matters(self):
        out = srv.get_signal_detail(42, "my-agent")
        self.assertIn("Why it matters:", out)
        self.assertIn("Wastes tokens", out)

    def test_not_found(self):
        out = srv.get_signal_detail(9999, "my-agent")
        self.assertIn("not found", out)

    def test_not_found_without_agent(self):
        out = srv.get_signal_detail(9999)
        self.assertIn("not found", out)

    def test_code_truncated_at_20_lines(self):
        long_fix = {
            "description": "Big fix",
            "language": "python",
            "code": "\n".join(f"line_{i} = True" for i in range(30)),
        }
        sig = {**SIGNAL_LIST["signals"][0], "suggested_fixes": [long_fix]}
        sig_list = {"signals": [sig], "page": {"total": 1}}
        with patch("dunetrace_mcp.client.get", return_value=sig_list):
            out = srv.get_signal_detail(42, "my-agent")
        self.assertIn("more lines", out)

    def test_evidence_list_truncated(self):
        big_evidence = {
            **SIGNAL_LIST["signals"][0]["evidence"],
            "args": ["x"] * 20,
        }
        sig = {**SIGNAL_LIST["signals"][0], "evidence": big_evidence}
        with patch(
            "dunetrace_mcp.client.get",
            return_value={"signals": [sig], "page": {"total": 1}},
        ):
            out = srv.get_signal_detail(42, "my-agent")
        self.assertIn("+14 more", out)


class TestGetAgentPatterns(unittest.TestCase):
    def setUp(self):
        self.patcher = patch("dunetrace_mcp.client.get", side_effect=_mock_get)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_shows_systemic(self):
        out = srv.get_agent_patterns("my-agent")
        self.assertIn("SYSTEMIC", out)
        self.assertIn("TOOL_LOOP", out)

    def test_shows_daily_trend(self):
        out = srv.get_agent_patterns("my-agent")
        self.assertIn("Daily signal counts", out)
        self.assertIn("05-12", out)
        self.assertIn("05-13", out)

    def test_shows_failure_rate(self):
        out = srv.get_agent_patterns("my-agent")
        self.assertIn("Failure rate", out)
        self.assertIn("100%", out)

    def test_shows_input_patterns(self):
        out = srv.get_agent_patterns("my-agent")
        self.assertIn("Input patterns", out)
        self.assertIn("abc123", out)

    def test_low_rate_input_patterns_hidden(self):
        # def456 has rate 0.4, below 0.5 threshold — should not appear
        out = srv.get_agent_patterns("my-agent")
        self.assertNotIn("def456", out)

    def test_no_data(self):
        empty = {k: [] for k in INSIGHTS}
        with patch("dunetrace_mcp.client.get", return_value=empty):
            out = srv.get_agent_patterns("no-data-agent")
        self.assertIn("No pattern data", out)


class TestSummarizeAgent(unittest.TestCase):
    def setUp(self):
        self.patcher = patch("dunetrace_mcp.client.get", side_effect=_mock_get)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_shows_score(self):
        out = srv.summarize_agent("my-agent")
        self.assertIn("62/100", out)

    def test_shows_failure_breakdown(self):
        out = srv.summarize_agent("my-agent")
        self.assertIn("TOOL_LOOP", out)
        self.assertIn("COST_SPIKE", out)

    def test_shows_health_bar(self):
        out = srv.summarize_agent("my-agent")
        self.assertIn("failure_rate", out)
        self.assertIn("█", out)

    def test_unknown_agent(self):
        out = srv.summarize_agent("ghost-agent")
        self.assertIn("not found", out)
        self.assertIn("list_agents", out)

    def test_health_fetch_failure_graceful(self):
        def mock_get_no_health(path, **params):
            if path.endswith("/health-score"):
                raise RuntimeError("API down")
            return _mock_get(path, **params)

        with patch("dunetrace_mcp.client.get", side_effect=mock_get_no_health):
            out = srv.summarize_agent("my-agent")
        self.assertIn("my-agent", out)
        # Should still render summary without health score
        self.assertIn("Total runs", out)


class TestGetAgentRuns(unittest.TestCase):
    def setUp(self):
        self.patcher = patch("dunetrace_mcp.client.get", side_effect=_mock_get)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_shows_run(self):
        out = srv.get_agent_runs("my-agent")
        self.assertIn("run-abc-123", out)

    def test_shows_duration(self):
        out = srv.get_agent_runs("my-agent")
        self.assertIn("30.0s", out)

    def test_shows_signal_flag(self):
        out = srv.get_agent_runs("my-agent")
        self.assertIn("🔴", out)

    def test_no_runs(self):
        with patch("dunetrace_mcp.client.get", return_value={"runs": [], "page": {"total": 0}}):
            out = srv.get_agent_runs("empty-agent")
        self.assertIn("No runs found", out)


class TestGetAgentTokenStats(unittest.TestCase):
    def setUp(self):
        self.patcher = patch("dunetrace_mcp.client.get", side_effect=_mock_get)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_shows_header(self):
        out = srv.get_agent_token_stats("my-agent")
        self.assertIn("Token stats: my-agent", out)

    def test_shows_all_windows(self):
        out = srv.get_agent_token_stats("my-agent")
        self.assertIn("Last 24 h", out)
        self.assertIn("Last 7 days", out)
        self.assertIn("Last 30 days", out)

    def test_shows_run_counts(self):
        out = srv.get_agent_token_stats("my-agent")
        self.assertIn("10", out)  # 1d run_count
        self.assertIn("3 with failures", out)

    def test_shows_token_amounts(self):
        out = srv.get_agent_token_stats("my-agent")
        self.assertIn("50.0k", out)  # total_tokens 1d
        self.assertIn("15.0k", out)  # wasted_tokens 1d

    def test_token_waste_pct_uses_token_ratio(self):
        # 1d: wasted_tokens=15000 / total_tokens=50000 = 30%
        # 30d: wasted_tokens=225000 / total_tokens=1000000 = 22% (not 22.5% from cost ratio)
        out = srv.get_agent_token_stats("my-agent")
        self.assertIn("30% of total", out)  # token-based pct for 1d

    def test_cost_waste_pct_uses_cost_ratio(self):
        # 1d: wasted_pct=0.3 → 30% for cost line; same value here but computed from different field
        # 30d: wasted_pct=0.225 → 22% for cost line (int(round(0.225*100))=22)
        out = srv.get_agent_token_stats("my-agent")
        # 30d token pct: 225000/1000000 = 22%, cost pct: 22% — both round to 22 here
        self.assertIn("22% of total", out)

    def test_shows_cost_figures(self):
        out = srv.get_agent_token_stats("my-agent")
        self.assertIn("$0.0500", out)  # total_cost_usd 1d
        self.assertIn("$0.0150", out)  # wasted_cost_usd 1d

    def test_fmt_cost_sub_cent_no_dollar_prefix(self):
        # $0.0005 should render as "0.0500¢", not "$0.0500¢"
        with patch(
            "dunetrace_mcp.client.get",
            return_value={
                "windows": {
                    "1d": {
                        "run_count": 1,
                        "wasted_run_count": 1,
                        "total_tokens": 100,
                        "wasted_tokens": 100,
                        "total_cost_usd": 0.0005,
                        "wasted_cost_usd": 0.0005,
                        "wasted_pct": 1.0,
                    }
                },
                "waste_by_failure_type": [],
            },
        ):
            out = srv.get_agent_token_stats("my-agent")
        self.assertIn("¢", out)
        self.assertNotIn("$0.0500¢", out)  # the old bug

    def test_shows_waste_by_failure_type(self):
        out = srv.get_agent_token_stats("my-agent")
        self.assertIn("Waste by failure type", out)
        self.assertIn("TOOL_LOOP", out)
        self.assertIn("COST_SPIKE", out)

    def test_waste_by_ft_shows_affected_runs(self):
        out = srv.get_agent_token_stats("my-agent")
        self.assertIn("30 runs", out)
        self.assertIn("15 runs", out)

    def test_waste_by_ft_singular_run(self):
        single_run = [
            {
                "failure_type": "RETRY_STORM",
                "wasted_tokens": 5000,
                "wasted_cost_usd": 0.005,
                "affected_runs": 1,
            }
        ]
        with patch(
            "dunetrace_mcp.client.get",
            return_value={
                "windows": TOKEN_STATS["windows"],
                "waste_by_failure_type": single_run,
            },
        ):
            out = srv.get_agent_token_stats("my-agent")
        self.assertIn("1 run", out)
        self.assertNotIn("1 runs", out)

    def test_api_error_returns_error_string(self):
        with patch("dunetrace_mcp.client.get", side_effect=Exception("connection refused")):
            out = srv.get_agent_token_stats("my-agent")
        self.assertIn("Could not fetch", out)
        self.assertIn("my-agent", out)

    def test_missing_windows_shows_header_only(self):
        with patch(
            "dunetrace_mcp.client.get",
            return_value={"windows": {}, "waste_by_failure_type": []},
        ):
            out = srv.get_agent_token_stats("my-agent")
        self.assertIn("Token stats: my-agent", out)
        self.assertNotIn("Last 24 h", out)

    def test_million_token_formatting(self):
        with patch(
            "dunetrace_mcp.client.get",
            return_value={
                "windows": {
                    "30d": {
                        "run_count": 500,
                        "wasted_run_count": 100,
                        "total_tokens": 2_000_000,
                        "wasted_tokens": 500_000,
                        "total_cost_usd": 2.0,
                        "wasted_cost_usd": 0.5,
                        "wasted_pct": 0.25,
                    }
                },
                "waste_by_failure_type": [],
            },
        ):
            out = srv.get_agent_token_stats("my-agent")
        self.assertIn("2.0M", out)
        self.assertIn("500.0k", out)


class TestHelpers(unittest.TestCase):
    """Edge cases for internal helper functions."""

    def test_ts_none(self):
        self.assertEqual(srv._ts(None), "—")

    def test_ago_none(self):
        self.assertEqual(srv._ago(None), "—")

    def test_ago_seconds(self):
        out = srv._ago(time.time() - 30)
        self.assertIn("s ago", out)

    def test_ago_minutes(self):
        out = srv._ago(time.time() - 90)
        self.assertIn("m ago", out)

    def test_ago_hours(self):
        out = srv._ago(time.time() - 7200)
        self.assertIn("h ago", out)

    def test_ago_days(self):
        out = srv._ago(time.time() - 86400 * 3)
        self.assertIn("d ago", out)

    def test_sev_icons(self):
        self.assertEqual(srv._sev_icon("CRITICAL"), "🔴")
        self.assertEqual(srv._sev_icon("HIGH"), "🟠")
        self.assertEqual(srv._sev_icon("MEDIUM"), "🟡")
        self.assertEqual(srv._sev_icon("LOW"), "🟢")
        self.assertEqual(srv._sev_icon("UNKNOWN"), "⚪")

    def test_score_icon(self):
        self.assertEqual(srv._score_icon(85), "✅")
        self.assertEqual(srv._score_icon(65), "🟡")
        self.assertEqual(srv._score_icon(40), "🔴")
        self.assertEqual(srv._score_icon(None), "❓")


class TestGetInstrumentationGuide(unittest.TestCase):
    """No HTTP calls — guide content is inline + local file reads."""

    def test_langchain_guide(self):
        out = srv.get_instrumentation_guide("langchain")
        self.assertIn("LangChain", out)
        self.assertIn("DunetraceCallbackHandler", out)
        self.assertIn("pip install", out)

    def test_langgraph_alias(self):
        out = srv.get_instrumentation_guide("langgraph")
        self.assertIn("DunetraceCallbackHandler", out)

    def test_lc_alias(self):
        out = srv.get_instrumentation_guide("lc")
        self.assertIn("DunetraceCallbackHandler", out)

    def test_python_guide(self):
        out = srv.get_instrumentation_guide("python")
        self.assertIn("@dt.tool", out)
        self.assertIn("@dt.trace", out)
        self.assertIn("pip install dunetrace", out)

    def test_custom_python_alias(self):
        out = srv.get_instrumentation_guide("custom-python")
        self.assertIn("@dt.tool", out)

    def test_typescript_guide(self):
        out = srv.get_instrumentation_guide("typescript")
        self.assertIn("TypeScript", out)
        self.assertIn("sendEvent", out)
        self.assertIn("run.started", out)

    def test_ts_alias(self):
        out = srv.get_instrumentation_guide("ts")
        self.assertIn("sendEvent", out)

    def test_javascript_alias(self):
        out = srv.get_instrumentation_guide("javascript")
        self.assertIn("sendEvent", out)

    def test_node_alias(self):
        out = srv.get_instrumentation_guide("node")
        self.assertIn("sendEvent", out)

    def test_tools_guide(self):
        out = srv.get_instrumentation_guide("tools")
        self.assertIn("tool_called", out)
        self.assertIn("TOOL_LOOP", out)

    def test_tool_calls_alias(self):
        out = srv.get_instrumentation_guide("tool-calls")
        self.assertIn("tool_called", out)

    def test_tracking_alias(self):
        out = srv.get_instrumentation_guide("tracking")
        self.assertIn("tool_called", out)

    def test_unknown_framework(self):
        out = srv.get_instrumentation_guide("rails")
        self.assertIn("Unknown framework", out)
        self.assertIn("rails", out)
        self.assertIn("langchain", out)  # lists supported values

    def test_case_insensitive(self):
        out = srv.get_instrumentation_guide("LANGCHAIN")
        self.assertIn("DunetraceCallbackHandler", out)

    def test_whitespace_stripped(self):
        out = srv.get_instrumentation_guide("  python  ")
        self.assertIn("@dt.tool", out)

    def test_includes_full_doc_when_available(self):
        # When docs directory exists (it does in repo), the full markdown is appended
        import pathlib

        doc_path = (
            pathlib.Path(__file__).resolve().parents[3] / "docs" / "integrate-langchain-agent.md"
        )
        if doc_path.exists():
            out = srv.get_instrumentation_guide("langchain")
            # The full doc contains sections not in the inline snippet
            self.assertIn("## Advanced", out)

    def test_degrades_gracefully_without_docs(self):
        # Patch the docs path to a non-existent directory — should still return inline guide
        import pathlib
        from unittest.mock import patch

        fake_path = pathlib.Path("/nonexistent/docs")
        with patch.object(srv, "_DOCS", fake_path):
            out = srv.get_instrumentation_guide("python")
        self.assertIn("@dt.tool", out)


class TestMainCLI(unittest.TestCase):
    """Tests for the main() entry point CLI argument handling."""

    def test_no_args_runs_stdio(self):
        import sys

        with patch.object(sys, "argv", ["dunetrace-mcp"]):
            with patch.object(srv.mcp, "run") as mock_run:
                srv.main()
        mock_run.assert_called_once_with(transport="stdio")

    def test_sse_flag_runs_sse(self):
        import sys

        with patch.object(sys, "argv", ["dunetrace-mcp", "--sse"]):
            with patch.object(srv.mcp, "run") as mock_run:
                srv.main()
        mock_run.assert_called_once_with(transport="sse")
        assert srv.mcp.settings.port == 8000

    def test_sse_with_custom_port(self):
        import sys

        with patch.object(sys, "argv", ["dunetrace-mcp", "--sse", "--port", "9000"]):
            with patch.object(srv.mcp, "run") as mock_run:
                srv.main()
        mock_run.assert_called_once_with(transport="sse")
        assert srv.mcp.settings.port == 9000

    def test_port_without_sse_still_runs_stdio(self):
        import sys

        with patch.object(sys, "argv", ["dunetrace-mcp", "--port", "9000"]):
            with patch.object(srv.mcp, "run") as mock_run:
                srv.main()
        mock_run.assert_called_once_with(transport="stdio")

    def test_help_exits_zero(self):
        import sys

        with patch.object(sys, "argv", ["dunetrace-mcp", "--help"]):
            with self.assertRaises(SystemExit) as ctx:
                srv.main()
        self.assertEqual(ctx.exception.code, 0)

    def test_version_exits_zero(self):
        import sys

        with patch.object(sys, "argv", ["dunetrace-mcp", "--version"]):
            with self.assertRaises(SystemExit) as ctx:
                srv.main()
        self.assertEqual(ctx.exception.code, 0)


class TestResources(unittest.TestCase):
    """Verify resources are registered and readable."""

    def test_resources_registered(self):
        uris = list(srv.mcp._resource_manager._resources.keys())
        self.assertIn("dunetrace://docs/integrate-python", uris)
        self.assertIn("dunetrace://docs/integrate-langchain", uris)
        self.assertIn("dunetrace://docs/integrate-typescript", uris)
        self.assertIn("dunetrace://docs/policies", uris)
        self.assertIn("dunetrace://docs/detectors", uris)
        self.assertIn("dunetrace://docs/mcp-server", uris)

    def test_doc_resource_returns_content(self):
        content = srv.doc_integrate_python()
        self.assertIn("Dunetrace", content)
        self.assertGreater(len(content), 500)

    def test_langchain_resource_returns_content(self):
        content = srv.doc_integrate_langchain()
        self.assertIn("DunetraceCallbackHandler", content)

    def test_typescript_resource_returns_content(self):
        content = srv.doc_integrate_typescript()
        self.assertIn("toolCalled", content)

    def test_missing_doc_returns_error_string(self):
        import pathlib
        from unittest.mock import patch

        with patch.object(srv, "_DOCS", pathlib.Path("/nonexistent")):
            content = srv.doc_integrate_python()
        self.assertIn("doc not found", content)


class TestGetRunDetailExtra(unittest.TestCase):
    """Additional get_run_detail edge cases."""

    def setUp(self):
        self.patcher = patch("dunetrace_mcp.client.get", side_effect=_mock_get)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_shows_total_tokens(self):
        run_with_tokens = {**RUN_DETAIL, "total_tokens": 45000}
        with patch("dunetrace_mcp.client.get", return_value=run_with_tokens):
            out = srv.get_run_detail("run-abc-123")
        self.assertIn("45,000", out)

    def test_no_tokens_field_omitted(self):
        # Default fixture has total_tokens=None — Tokens line should not appear
        out = srv.get_run_detail("run-abc-123")
        self.assertNotIn("Tokens:", out)

    def test_parent_run_id_event_does_not_crash(self):
        events_with_parent = [
            {**RUN_DETAIL["events"][0], "parent_run_id": "parent-run-xyz"},
            *RUN_DETAIL["events"][1:],
        ]
        run_with_parent = {**RUN_DETAIL, "events": events_with_parent}
        with patch("dunetrace_mcp.client.get", return_value=run_with_parent):
            out = srv.get_run_detail("run-abc-123")
        self.assertIn("run.started", out)


class TestGetAgentHealthExtra(unittest.TestCase):
    """Additional get_agent_health edge cases."""

    def test_baseline_not_ready(self):
        not_ready = {**HEALTH_SCORE, "baseline_ready": False, "sample_runs": 15}
        with patch("dunetrace_mcp.client.get", return_value=not_ready):
            out = srv.get_agent_health("my-agent")
        self.assertIn("no (need ≥30", out)
        self.assertNotIn("yes", out)


class TestSearchSignalsExtra(unittest.TestCase):
    """Additional search_signals edge cases."""

    def setUp(self):
        self.patcher = patch("dunetrace_mcp.client.get", side_effect=_mock_get)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_combined_failure_type_and_severity(self):
        out = srv.search_signals(failure_type="TOOL_LOOP", severity="HIGH")
        self.assertIn("TOOL_LOOP", out)
        self.assertNotIn("COST_SPIKE", out)

    def test_session_latency_signal_rendered(self):
        session_sig = {
            "id": 45,
            "failure_type": "SESSION_LATENCY",
            "severity": "MEDIUM",
            "run_id": "run-slow-789",
            "agent_id": "my-agent",
            "agent_version": "v1.2.3",
            "step_index": 12,
            "confidence": 0.85,
            "detected_at": time.time() - 1800,
            "evidence": {"duration_s": 350.0, "baseline_p75": 90.0},
            "alerted": False,
            "shadow": False,
            "co_signal_count": 0,
            "title": "Session latency: 350s (3.9× baseline)",
            "what": "Run took far longer than typical.",
            "why_it_matters": "Indicates hanging tools or infrastructure.",
            "evidence_summary": "350s vs 90s baseline.",
            "suggested_fixes": [
                {"description": "Add timeout guards", "language": "python", "code": ""}
            ],
        }
        sig_list = {
            "signals": [session_sig],
            "page": {"total": 1, "offset": 0, "limit": 200, "has_more": False},
        }
        with patch(
            "dunetrace_mcp.client.get",
            side_effect=lambda path, **p: AGENT_LIST if path == "/v1/agents" else sig_list,
        ):
            out = srv.search_signals(failure_type="SESSION_LATENCY")
        self.assertIn("SESSION_LATENCY", out)
        self.assertIn("🟡", out)  # MEDIUM icon


class TestGetAgentPatternsExtra(unittest.TestCase):
    """Additional get_agent_patterns edge cases."""

    def test_occasional_pattern_label(self):
        occasional_insights = {
            **INSIGHTS,
            "systemic_patterns": [
                {
                    "failure_type": "COST_SPIKE",
                    "total_runs": 10,
                    "affected_runs": 2,
                    "rate": 0.2,
                    "first_seen": time.time() - 86400,
                    "last_seen": time.time() - 3600,
                    "is_systemic": False,
                },
            ],
        }
        with patch("dunetrace_mcp.client.get", return_value=occasional_insights):
            out = srv.get_agent_patterns("my-agent")
        self.assertIn("Occasional", out)
        self.assertNotIn("SYSTEMIC", out)


class TestSummarizeAgentExtra(unittest.TestCase):
    """Additional summarize_agent edge cases."""

    def test_no_signals_section_absent(self):
        def _no_signals(path, **params):
            if path.endswith("/signals"):
                return {"signals": [], "page": {"total": 0}}
            return _mock_get(path, **params)

        with patch("dunetrace_mcp.client.get", side_effect=_no_signals):
            out = srv.summarize_agent("clean-agent")
        self.assertIn("clean-agent", out)
        self.assertIn("Total runs", out)
        self.assertNotIn("Most recent signals", out)


class TestGetAgentRunsExtra(unittest.TestCase):
    """Additional get_agent_runs edge cases."""

    def test_custom_limit_does_not_error(self):
        with patch("dunetrace_mcp.client.get", side_effect=_mock_get):
            out = srv.get_agent_runs("my-agent", limit=5)
        self.assertIn("run-abc-123", out)

    def test_limit_capped_at_100(self):
        captured = {}

        def _capturing_get(path, **params):
            if path.endswith("/runs"):
                captured["limit"] = params.get("limit")
            return _mock_get(path, **params)

        with patch("dunetrace_mcp.client.get", side_effect=_capturing_get):
            srv.get_agent_runs("my-agent", limit=999)
        self.assertEqual(captured.get("limit"), 100)


class TestGetInstrumentationGuideExtra(unittest.TestCase):
    """Additional get_instrumentation_guide alias and edge cases."""

    def test_lc_graph_hyphen_alias(self):
        out = srv.get_instrumentation_guide("lc-graph")
        self.assertIn("DunetraceCallbackHandler", out)

    def test_lc_graph_underscore_alias(self):
        out = srv.get_instrumentation_guide("lc_graph")
        self.assertIn("DunetraceCallbackHandler", out)

    def test_crewai_unknown(self):
        out = srv.get_instrumentation_guide("crewai")
        self.assertIn("Unknown framework", out)
        self.assertIn("langchain", out)

    def test_autogen_unknown(self):
        out = srv.get_instrumentation_guide("autogen")
        self.assertIn("Unknown framework", out)

    def test_py_alias(self):
        out = srv.get_instrumentation_guide("py")
        self.assertIn("@dt.tool", out)

    def test_nodejs_alias(self):
        out = srv.get_instrumentation_guide("nodejs")
        self.assertIn("sendEvent", out)


class TestClientHelpers(unittest.TestCase):
    """Tests for the new post / patch / delete helpers in client.py."""

    def _mock_httpx_client(self, method: str, json_data: dict, status_code: int = 200):
        """Return a context-manager-compatible mock httpx.Client."""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = json_data
        mock_resp.raise_for_status = MagicMock()

        mock_c = MagicMock()
        mock_c.__enter__ = MagicMock(return_value=mock_c)
        mock_c.__exit__ = MagicMock(return_value=False)
        getattr(mock_c, method).return_value = mock_resp
        return mock_c

    def test_post_returns_json(self):
        from dunetrace_mcp import client

        mock_c = self._mock_httpx_client("post", {"id": "abc123"})
        with patch("httpx.Client", return_value=mock_c):
            result = client.post("/v1/test", {"key": "val"})
        self.assertEqual(result, {"id": "abc123"})
        mock_c.post.assert_called_once()

    def test_patch_returns_json(self):
        from dunetrace_mcp import client

        mock_c = self._mock_httpx_client("patch", {"status": "active"})
        with patch("httpx.Client", return_value=mock_c):
            result = client.patch("/v1/test/1", {"status": "active"})
        self.assertEqual(result, {"status": "active"})
        mock_c.patch.assert_called_once()

    def test_delete_returns_empty_on_204(self):
        from dunetrace_mcp import client

        mock_c = self._mock_httpx_client("delete", {}, status_code=204)
        with patch("httpx.Client", return_value=mock_c):
            result = client.delete("/v1/test/1")
        self.assertEqual(result, {})

    def test_delete_returns_json_on_200(self):
        from dunetrace_mcp import client

        mock_c = self._mock_httpx_client("delete", {"deleted": True}, status_code=200)
        with patch("httpx.Client", return_value=mock_c):
            result = client.delete("/v1/test/1")
        self.assertEqual(result, {"deleted": True})

    def test_post_connect_error_raises_runtime_error(self):
        import httpx
        from dunetrace_mcp import client

        mock_c = MagicMock()
        mock_c.__enter__ = MagicMock(return_value=mock_c)
        mock_c.__exit__ = MagicMock(return_value=False)
        mock_c.post.side_effect = httpx.ConnectError("refused")

        with patch("httpx.Client", return_value=mock_c):
            with self.assertRaises(RuntimeError) as ctx:
                client.post("/v1/test", {})
        self.assertIn("unreachable", str(ctx.exception))
        self.assertIn("docker compose", str(ctx.exception))

    def test_get_connect_error_raises_runtime_error(self):
        import httpx
        from dunetrace_mcp import client

        mock_c = MagicMock()
        mock_c.__enter__ = MagicMock(return_value=mock_c)
        mock_c.__exit__ = MagicMock(return_value=False)
        mock_c.get.side_effect = httpx.ConnectError("refused")

        with patch("httpx.Client", return_value=mock_c):
            with self.assertRaises(RuntimeError) as ctx:
                client.get("/v1/test")
        self.assertIn("unreachable", str(ctx.exception))


class TestGetFixStatus(unittest.TestCase):
    """Smoke tests for get_fix_status."""

    FIX_STATUS = {
        "verdict": "verified",
        "runs_before": 10,
        "runs_after": 15,
        "recurrence_before": 8,
        "recurrence_after": 0,
        "runs_evaluated": 25,
        "fix_applied_at": time.time() - 86400,
    }

    def test_shows_verdict(self):
        with patch("dunetrace_mcp.client.get", return_value=self.FIX_STATUS):
            out = srv.get_fix_status(42)
        self.assertIn("VERIFIED", out)
        self.assertIn("signal #42", out)

    def test_shows_recurrence_counts(self):
        with patch("dunetrace_mcp.client.get", return_value=self.FIX_STATUS):
            out = srv.get_fix_status(42)
        self.assertIn("10", out)  # runs_before
        self.assertIn("15", out)  # runs_after
        self.assertIn("25", out)  # runs_evaluated

    def test_still_occurring_verdict(self):
        data = {**self.FIX_STATUS, "verdict": "still_occurring"}
        with patch("dunetrace_mcp.client.get", return_value=data):
            out = srv.get_fix_status(42)
        self.assertIn("STILL OCCURRING", out)

    def test_insufficient_data_verdict(self):
        data = {**self.FIX_STATUS, "verdict": "insufficient_data"}
        with patch("dunetrace_mcp.client.get", return_value=data):
            out = srv.get_fix_status(42)
        self.assertIn("INSUFFICIENT DATA", out)

    def test_api_error_returns_error(self):
        with patch(
            "dunetrace_mcp.client.get", side_effect=RuntimeError("API error 404: not found")
        ):
            out = srv.get_fix_status(9999)
        self.assertIn("Error", str(out))


class TestTriggerExplain(unittest.TestCase):
    """Smoke tests for trigger_explain."""

    EXPLAIN_RESULT = {
        "signal_id": 42,
        "source": "native",
        "root_cause": "The agent repeats web_search with identical args.",
        "fix_category": "customer_code",
        "fix_content": "Track all tool calls you've made. Never repeat with identical args.",
        "fix_type": "prompt_addition",
        "apply_blocked": True,
    }

    def test_shows_root_cause(self):
        with patch("dunetrace_mcp.client.post", return_value=self.EXPLAIN_RESULT):
            out = srv.trigger_explain(42)
        self.assertIn("Root cause", out)
        self.assertIn("web_search", out)

    def test_shows_fix_content(self):
        with patch("dunetrace_mcp.client.post", return_value=self.EXPLAIN_RESULT):
            out = srv.trigger_explain(42)
        self.assertIn("Fix", out)
        self.assertIn("Track all tool calls", out)

    def test_shows_open_pr_instructions_for_unblocked_code_change(self):
        unblocked = {**self.EXPLAIN_RESULT, "fix_type": "code_change", "apply_blocked": False}
        with patch("dunetrace_mcp.client.post", return_value=unblocked):
            out = srv.trigger_explain(42)
        self.assertIn("open-pr", out)

    def test_dunetrace_native_shows_suggested_policy(self):
        native_result = {
            "signal_id": 42,
            "source": "native",
            "root_cause": "Tool loop detected.",
            "fix_category": "dunetrace_native",
            "suggested_policy": {"name": "cap-tool-calls", "agent_id": "agent-1"},
            "fix_type": "policy",
            "apply_blocked": False,
        }
        with patch("dunetrace_mcp.client.post", return_value=native_result):
            out = srv.trigger_explain(42)
        self.assertIn("runtime policy", out)
        self.assertIn("POST /v1/policies", out)

    def test_blocked_shows_blocked_message(self):
        blocked = {**self.EXPLAIN_RESULT, "fix_type": "prompt_addition", "apply_blocked": True}
        with patch("dunetrace_mcp.client.post", return_value=blocked):
            out = srv.trigger_explain(42)
        self.assertIn("Apply blocked", out)

    def test_api_error_returns_error(self):
        with patch(
            "dunetrace_mcp.client.post", side_effect=RuntimeError("API error 500: server error")
        ):
            out = srv.trigger_explain(42)
        self.assertIn("Error", str(out))


class TestListCustomDetectors(unittest.TestCase):
    """Smoke tests for list_custom_detectors."""

    DETECTORS = [
        {
            "id": "det-1",
            "name": "high-cost-detector",
            "description": "Fire when cost_usd > 10",
            "agent_id": "*",
            "status": "shadow",
            "shadow_fire_count": 3,
            "total_runs": 50,
            "created_at": time.time() - 86400,
        },
        {
            "id": "det-2",
            "name": "strict-loop-guard",
            "description": "Fire when tool_calls > 20",
            "agent_id": "my-agent",
            "status": "active",
            "shadow_fire_count": 10,
            "total_runs": 20,
            "created_at": time.time() - 3600,
        },
    ]

    def test_shows_detector_names(self):
        with patch("dunetrace_mcp.client.get", return_value=self.DETECTORS):
            out = srv.list_custom_detectors()
        self.assertIn("high-cost-detector", out)
        self.assertIn("strict-loop-guard", out)

    def test_shows_status(self):
        with patch("dunetrace_mcp.client.get", return_value=self.DETECTORS):
            out = srv.list_custom_detectors()
        self.assertIn("[shadow]", out)
        self.assertIn("[active]", out)

    def test_shows_fire_rate(self):
        with patch("dunetrace_mcp.client.get", return_value=self.DETECTORS):
            out = srv.list_custom_detectors()
        self.assertIn("3/50", out)  # shadow_fire_count/total_runs

    def test_empty_returns_helpful_message(self):
        with patch("dunetrace_mcp.client.get", return_value=[]):
            out = srv.list_custom_detectors()
        self.assertIn("No custom detectors", out)
        self.assertIn("create_custom_detector", out)


class TestListPolicies(unittest.TestCase):
    """Smoke tests for list_policies."""

    POLICIES = [
        {
            "id": "pol-1",
            "name": "budget-guard",
            "agent_id": "my-agent",
            "condition": {"metric": "cost_usd", "operator": "gt", "threshold": 5.0},
            "action": {"type": "stop"},
            "enabled": True,
        },
        {
            "id": "pol-2",
            "name": "loop-guard",
            "agent_id": "*",
            "condition": {"metric": "tool_calls", "operator": "gt", "threshold": 20},
            "action": {"type": "switch_model", "model": "gpt-4o-mini"},
            "enabled": False,
        },
    ]

    def test_shows_policy_names(self):
        with patch("dunetrace_mcp.client.get", return_value={"policies": self.POLICIES}):
            out = srv.list_policies()
        self.assertIn("budget-guard", out)
        self.assertIn("loop-guard", out)

    def test_shows_enabled_status(self):
        with patch("dunetrace_mcp.client.get", return_value={"policies": self.POLICIES}):
            out = srv.list_policies()
        self.assertIn("enabled", out)
        self.assertIn("disabled", out)

    def test_shows_action_type(self):
        with patch("dunetrace_mcp.client.get", return_value={"policies": self.POLICIES}):
            out = srv.list_policies()
        self.assertIn("stop", out)
        self.assertIn("switch_model", out)

    def test_empty_returns_helpful_message(self):
        with patch("dunetrace_mcp.client.get", return_value={"policies": []}):
            out = srv.list_policies()
        self.assertIn("No policies found", out)
        self.assertIn("create_policy", out)


class TestCompareRuns(unittest.TestCase):
    """Smoke tests for compare_runs."""

    RUN_A = {
        "run_id": "run-aaa-111",
        "agent_id": "my-agent",
        "agent_version": "v1.0",
        "started_at": time.time() - 200,
        "completed_at": time.time() - 190,
        "exit_reason": "final_answer",
        "step_count": 5,
        "total_tokens": 1000,
        "signals": [],
        "events": [],
    }
    RUN_B = {
        "run_id": "run-bbb-222",
        "agent_id": "my-agent",
        "agent_version": "v1.1",
        "started_at": time.time() - 100,
        "completed_at": time.time() - 80,
        "exit_reason": "error",
        "step_count": 12,
        "total_tokens": 8000,
        "signals": [
            {
                "failure_type": "TOOL_LOOP",
                "severity": "HIGH",
                "confidence": 0.9,
                "step_index": 8,
            }
        ],
        "events": [],
    }

    def _side_effect(self, path, **params):
        if "aaa" in path:
            return self.RUN_A
        return self.RUN_B

    def test_shows_both_run_ids(self):
        with patch("dunetrace_mcp.client.get", side_effect=self._side_effect):
            out = srv.compare_runs("run-aaa-111", "run-bbb-222")
        self.assertIn("run-aaa-111", out)
        self.assertIn("run-bbb-222", out)

    def test_shows_diff_marker(self):
        with patch("dunetrace_mcp.client.get", side_effect=self._side_effect):
            out = srv.compare_runs("run-aaa-111", "run-bbb-222")
        self.assertIn("!!", out)  # marker for differing rows

    def test_shows_step_counts(self):
        with patch("dunetrace_mcp.client.get", side_effect=self._side_effect):
            out = srv.compare_runs("run-aaa-111", "run-bbb-222")
        self.assertIn("5", out)
        self.assertIn("12", out)

    def test_shows_signals(self):
        with patch("dunetrace_mcp.client.get", side_effect=self._side_effect):
            out = srv.compare_runs("run-aaa-111", "run-bbb-222")
        self.assertIn("TOOL_LOOP", out)


class TestListAgentIssues(unittest.TestCase):
    """Smoke tests for list_agent_issues."""

    ISSUES = [
        {
            "id": "iss-1",
            "failure_type": "TOOL_LOOP",
            "status": "open",
            "occurrence_count": 46,
            "opened_at": time.time() - 86400 * 5,
            "last_seen": time.time() - 3600,
        },
        {
            "id": "iss-2",
            "failure_type": "COST_SPIKE",
            "status": "open",
            "occurrence_count": 8,
            "opened_at": time.time() - 86400 * 2,
            "last_seen": time.time() - 7200,
        },
    ]

    def test_shows_failure_types(self):
        with patch("dunetrace_mcp.client.get", return_value={"issues": self.ISSUES}):
            out = srv.list_agent_issues("my-agent")
        self.assertIn("TOOL_LOOP", out)
        self.assertIn("COST_SPIKE", out)

    def test_shows_occurrence_count(self):
        with patch("dunetrace_mcp.client.get", return_value={"issues": self.ISSUES}):
            out = srv.list_agent_issues("my-agent")
        self.assertIn("46", out)
        self.assertIn("8", out)

    def test_empty_returns_message(self):
        with patch("dunetrace_mcp.client.get", return_value={"issues": []}):
            out = srv.list_agent_issues("clean-agent")
        self.assertIn("No open issues", out)


class TestAgoISOString(unittest.TestCase):
    """Ensure _ago handles ISO-8601 strings as well as epoch floats."""

    def test_iso_string_recent(self):
        from datetime import datetime, timezone, timedelta

        iso = (datetime.now(tz=timezone.utc) - timedelta(minutes=5)).isoformat()
        out = srv._ago(iso)
        self.assertIn("m ago", out)

    def test_iso_string_with_z_suffix(self):
        from datetime import datetime, timezone, timedelta

        iso = (datetime.now(tz=timezone.utc) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        out = srv._ago(iso)
        self.assertIn("h ago", out)

    def test_float_still_works(self):
        out = srv._ago(time.time() - 90)
        self.assertIn("m ago", out)

    def test_none_still_works(self):
        self.assertEqual(srv._ago(None), "—")


if __name__ == "__main__":
    unittest.main()
