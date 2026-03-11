#!/usr/bin/env python3
"""
scripts/smoke_test_all_detectors.py

Synthetic end-to-end smoke test for every Tier 1 detector.
Injects crafted event sequences directly to the ingest API i.e.
no LangChain or OpenAI key required.

Coverage (12 of 13 detectors):
  TOOL_LOOP · TOOL_THRASHING · TOOL_AVOIDANCE · RAG_EMPTY_RETRIEVAL
  LLM_TRUNCATION_LOOP · CONTEXT_BLOAT · SLOW_STEP · RETRY_STORM
  EMPTY_LLM_RESPONSE · STEP_COUNT_INFLATION · CASCADING_TOOL_FAILURE
  FIRST_STEP_FAILURE

Skipped (not triggerable through the ingest pipeline):
  GOAL_ABANDONMENT: needs stalled run (90 s timeout)
  PROMPT_INJECTION_SIGNAL: checked pre-ingest via SDK check_input()

Usage:
    docker compose up -d
    python scripts/smoke_test_all_detectors.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
import urllib.request
import urllib.error
from pathlib import Path

INGEST_URL = os.environ.get("INGEST_URL", "http://localhost:8001")
API_URL    = os.environ.get("API_URL",    "http://localhost:8002")
API_KEY    = "dt_dev_test"

_TS = int(time.time())
# All scenarios share one agent_id (signals fetched in one call)
AGENT_ID      = f"synth-all-{_TS}"
AGENT_VERSION = "v-synth-1"

# STEP_COUNT_INFLATION needs its own agent so baseline runs don't
# interfere with (or count toward) the other scenarios.
INFL_AGENT_ID  = f"synth-infl-{_TS}"
INFL_VERSION   = "v-infl-1"


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _post(url: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _get(url: str) -> dict:
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {API_KEY}"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def wait_healthy(url: str, label: str, timeout: int = 60) -> None:
    print(f"  Waiting for {label} ...", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            _get(f"{url}/health")
            print(" ready.")
            return
        except Exception:
            print(".", end="", flush=True)
            time.sleep(2)
    raise SystemExit(f"\nERROR: {label} not healthy. Is the stack running?")


def ingest(agent_id: str, run_id: str, events: list[dict]) -> None:
    """POST a batch of events for one run to the ingest service."""
    body = {"api_key": API_KEY, "agent_id": agent_id, "events": events}
    _post(f"{INGEST_URL}/v1/ingest", body)


def fetch_signals(agent_id: str, limit: int = 100) -> list[dict]:
    url = f"{API_URL}/v1/agents/{agent_id}/signals?limit={limit}"
    return _get(url).get("signals", [])


def mk_run_id(tag: str) -> str:
    return f"synth-{tag}-{_TS}-{uuid.uuid4().hex[:6]}"


def ev(event_type: str, run_id: str, agent_id: str, step: int,
       payload: dict | None = None, ts: float | None = None) -> dict:
    """Build a single ingest event dict."""
    return {
        "event_type":    event_type,
        "run_id":        run_id,
        "agent_id":      agent_id,
        "agent_version": AGENT_VERSION if agent_id == AGENT_ID else INFL_VERSION,
        "step_index":    step,
        "timestamp":     ts if ts is not None else time.time(),
        "payload":       payload or {},
    }


# ── Scenario builders ─────────────────────────────────────────────────────────

def scenario_tool_loop() -> str:
    """TOOL_LOOP: same tool called 5× in a 5-step window."""
    rid = mk_run_id("tool-loop")
    ingest(AGENT_ID, rid, [
        ev("run.started",   rid, AGENT_ID, 0, {"tools": ["search"]}),
        ev("tool.called",   rid, AGENT_ID, 1, {"tool_name": "search", "args_hash": "h1"}),
        ev("tool.called",   rid, AGENT_ID, 2, {"tool_name": "search", "args_hash": "h1"}),
        ev("tool.called",   rid, AGENT_ID, 3, {"tool_name": "search", "args_hash": "h1"}),
        ev("tool.called",   rid, AGENT_ID, 4, {"tool_name": "search", "args_hash": "h1"}),
        ev("tool.called",   rid, AGENT_ID, 5, {"tool_name": "search", "args_hash": "h1"}),
        ev("run.completed", rid, AGENT_ID, 5, {"exit_reason": "completed"}),
    ])
    return rid


def scenario_tool_thrashing() -> str:
    """TOOL_THRASHING: A/B/A/B/A/B alternation over 6 steps."""
    rid = mk_run_id("thrash")
    ingest(AGENT_ID, rid, [
        ev("run.started",  rid, AGENT_ID, 0, {"tools": ["search", "database"]}),
        ev("tool.called",  rid, AGENT_ID, 1, {"tool_name": "search",   "args_hash": "h1"}),
        ev("tool.called",  rid, AGENT_ID, 2, {"tool_name": "database", "args_hash": "h2"}),
        ev("tool.called",  rid, AGENT_ID, 3, {"tool_name": "search",   "args_hash": "h3"}),
        ev("tool.called",  rid, AGENT_ID, 4, {"tool_name": "database", "args_hash": "h4"}),
        ev("tool.called",  rid, AGENT_ID, 5, {"tool_name": "search",   "args_hash": "h5"}),
        ev("tool.called",  rid, AGENT_ID, 6, {"tool_name": "database", "args_hash": "h6"}),
        ev("run.completed",rid, AGENT_ID, 6, {"exit_reason": "completed"}),
    ])
    return rid


def scenario_tool_avoidance() -> str:
    """TOOL_AVOIDANCE: tools available, 2 LLM calls, no tools used, final_answer."""
    rid = mk_run_id("avoidance")
    ingest(AGENT_ID, rid, [
        ev("run.started",   rid, AGENT_ID, 0, {"tools": ["search", "database"]}),
        ev("llm.called",    rid, AGENT_ID, 1, {"model": "test"}),
        ev("llm.responded", rid, AGENT_ID, 1,
           {"finish_reason": "stop", "prompt_tokens": 100, "output_length": 80}),
        ev("llm.called",    rid, AGENT_ID, 2, {"model": "test"}),
        ev("llm.responded", rid, AGENT_ID, 2,
           {"finish_reason": "stop", "prompt_tokens": 110, "output_length": 120}),
        ev("run.completed", rid, AGENT_ID, 2, {"exit_reason": "final_answer"}),
    ])
    return rid


def scenario_rag_empty_retrieval() -> str:
    """RAG_EMPTY_RETRIEVAL: retrieval returns 0 results, agent answers anyway."""
    rid = mk_run_id("rag-empty")
    ingest(AGENT_ID, rid, [
        ev("run.started",          rid, AGENT_ID, 0),
        ev("retrieval.called",     rid, AGENT_ID, 1, {"index_name": "knowledge-base"}),
        ev("retrieval.responded",  rid, AGENT_ID, 1,
           {"index_name": "knowledge-base", "result_count": 0, "top_score": 0.0}),
        ev("llm.called",    rid, AGENT_ID, 2, {"model": "test"}),
        ev("llm.responded", rid, AGENT_ID, 2,
           {"finish_reason": "stop", "prompt_tokens": 100, "output_length": 60}),
        ev("run.completed", rid, AGENT_ID, 2, {"exit_reason": "final_answer"}),
    ])
    return rid


def scenario_llm_truncation_loop() -> str:
    """LLM_TRUNCATION_LOOP: 2× finish_reason='length'."""
    rid = mk_run_id("truncation")
    ingest(AGENT_ID, rid, [
        ev("run.started",   rid, AGENT_ID, 0),
        ev("llm.called",    rid, AGENT_ID, 1, {"model": "test"}),
        ev("llm.responded", rid, AGENT_ID, 1,
           {"finish_reason": "length", "prompt_tokens": 800, "output_length": 512}),
        ev("llm.called",    rid, AGENT_ID, 2, {"model": "test"}),
        ev("llm.responded", rid, AGENT_ID, 2,
           {"finish_reason": "length", "prompt_tokens": 900, "output_length": 512}),
        ev("run.completed", rid, AGENT_ID, 2, {"exit_reason": "completed"}),
    ])
    return rid


def scenario_context_bloat() -> str:
    """CONTEXT_BLOAT: prompt tokens grow 3.2× over 3 LLM calls."""
    rid = mk_run_id("bloat")
    ingest(AGENT_ID, rid, [
        ev("run.started",   rid, AGENT_ID, 0),
        ev("llm.called",    rid, AGENT_ID, 1, {"model": "test"}),
        ev("llm.responded", rid, AGENT_ID, 1,
           {"finish_reason": "stop", "prompt_tokens": 100, "output_length": 50}),
        ev("llm.called",    rid, AGENT_ID, 2, {"model": "test"}),
        ev("llm.responded", rid, AGENT_ID, 2,
           {"finish_reason": "stop", "prompt_tokens": 200, "output_length": 50}),
        ev("llm.called",    rid, AGENT_ID, 3, {"model": "test"}),
        ev("llm.responded", rid, AGENT_ID, 3,
           {"finish_reason": "stop", "prompt_tokens": 320, "output_length": 50}),
        ev("run.completed", rid, AGENT_ID, 3, {"exit_reason": "completed"}),
    ])
    return rid


def scenario_slow_step() -> str:
    """SLOW_STEP: tool.called at step 1, next event 20 s later → 20 000 ms > 15 000 ms threshold."""
    rid  = mk_run_id("slow")
    now  = time.time()
    T    = now - 25  # "happened" 25 seconds ago
    ingest(AGENT_ID, rid, [
        ev("run.started",  rid, AGENT_ID, 0, ts=T),
        # tool.called at T — the gap to the next event drives the duration
        ev("tool.called",  rid, AGENT_ID, 1,
           {"tool_name": "slow_api", "args_hash": "h1"}, ts=T),
        # run.completed at T+20: step_durations_ms[1] = 20 000 ms > 15 000 ms threshold
        ev("run.completed",rid, AGENT_ID, 2,
           {"exit_reason": "completed"}, ts=T + 20),
    ])
    return rid


def scenario_retry_storm() -> str:
    """RETRY_STORM: same tool, 3 consecutive success=False calls (args vary)."""
    rid = mk_run_id("retry-storm")
    ingest(AGENT_ID, rid, [
        ev("run.started",   rid, AGENT_ID, 0),
        ev("tool.called",   rid, AGENT_ID, 1,
           {"tool_name": "payment_api", "args_hash": "h1"}),
        ev("tool.responded",rid, AGENT_ID, 1, {"success": False}),
        ev("tool.called",   rid, AGENT_ID, 2,
           {"tool_name": "payment_api", "args_hash": "h2"}),
        ev("tool.responded",rid, AGENT_ID, 2, {"success": False}),
        ev("tool.called",   rid, AGENT_ID, 3,
           {"tool_name": "payment_api", "args_hash": "h3"}),
        ev("tool.responded",rid, AGENT_ID, 3, {"success": False}),
        ev("run.completed", rid, AGENT_ID, 3, {"exit_reason": "error"}),
    ])
    return rid


def scenario_empty_llm_response() -> str:
    """EMPTY_LLM_RESPONSE: output_length=0 with finish_reason='stop'."""
    rid = mk_run_id("empty-llm")
    ingest(AGENT_ID, rid, [
        ev("run.started",   rid, AGENT_ID, 0),
        ev("llm.called",    rid, AGENT_ID, 1, {"model": "test"}),
        ev("llm.responded", rid, AGENT_ID, 1,
           {"finish_reason": "stop", "prompt_tokens": 150, "output_length": 0}),
        ev("run.errored",   rid, AGENT_ID, 1),
    ])
    return rid


def scenario_cascading_tool_failure() -> str:
    """CASCADING_TOOL_FAILURE: 3 consecutive fails across 2 distinct tools."""
    rid = mk_run_id("cascade")
    ingest(AGENT_ID, rid, [
        ev("run.started",   rid, AGENT_ID, 0),
        ev("tool.called",   rid, AGENT_ID, 1,
           {"tool_name": "db_lookup", "args_hash": "h1"}),
        ev("tool.responded",rid, AGENT_ID, 1, {"success": False}),
        ev("tool.called",   rid, AGENT_ID, 2,
           {"tool_name": "search_api", "args_hash": "h2"}),
        ev("tool.responded",rid, AGENT_ID, 2, {"success": False}),
        ev("tool.called",   rid, AGENT_ID, 3,
           {"tool_name": "db_lookup", "args_hash": "h3"}),
        ev("tool.responded",rid, AGENT_ID, 3, {"success": False}),
        ev("run.errored",   rid, AGENT_ID, 3),
    ])
    return rid


def scenario_first_step_failure() -> str:
    """FIRST_STEP_FAILURE: run.errored at step 1 (≤ MAX_STEP=2)."""
    rid = mk_run_id("first-fail")
    ingest(AGENT_ID, rid, [
        ev("run.started", rid, AGENT_ID, 0),
        ev("run.errored", rid, AGENT_ID, 1),
    ])
    return rid


# ── STEP_COUNT_INFLATION (separate agent, two-phase) ─────────────────────────

def _infl_ev(event_type: str, run_id: str, step: int,
             payload: dict | None = None, ts: float | None = None) -> dict:
    return {
        "event_type":    event_type,
        "run_id":        run_id,
        "agent_id":      INFL_AGENT_ID,
        "agent_version": INFL_VERSION,
        "step_index":    step,
        "timestamp":     ts if ts is not None else time.time(),
        "payload":       payload or {},
    }


def _baseline_run() -> list[dict]:
    """Minimal 5-step run (step_count = max_step+1 = 6)."""
    rid = mk_run_id("base")
    return [
        _infl_ev("run.started",   rid, 0),
        _infl_ev("llm.called",    rid, 1, {"model": "test"}),
        _infl_ev("llm.responded", rid, 1,
                 {"finish_reason": "stop", "prompt_tokens": 50, "output_length": 30}),
        _infl_ev("tool.called",   rid, 2, {"tool_name": "search", "args_hash": "h"}),
        _infl_ev("tool.responded",rid, 2, {"success": True}),
        _infl_ev("llm.called",    rid, 3, {"model": "test"}),
        _infl_ev("llm.responded", rid, 3,
                 {"finish_reason": "stop", "prompt_tokens": 60, "output_length": 30}),
        _infl_ev("llm.called",    rid, 4, {"model": "test"}),
        _infl_ev("llm.responded", rid, 4,
                 {"finish_reason": "stop", "prompt_tokens": 65, "output_length": 30}),
        _infl_ev("llm.called",    rid, 5, {"model": "test"}),
        _infl_ev("llm.responded", rid, 5,
                 {"finish_reason": "stop", "prompt_tokens": 70, "output_length": 30}),
        _infl_ev("run.completed", rid, 5, {"exit_reason": "completed"}),
    ]


def inject_inflation_baseline(n: int = 10) -> None:
    """Inject n baseline runs (separate batches to avoid MAX_BATCH_SIZE)."""
    for _ in range(n):
        events = _baseline_run()
        rid    = events[0]["run_id"]
        ingest(INFL_AGENT_ID, rid, events)


def scenario_step_count_inflation() -> str:
    """STEP_COUNT_INFLATION: 20 steps vs P75 baseline of ~6."""
    rid = mk_run_id("infl")
    events: list[dict] = [_infl_ev("run.started", rid, 0)]
    # Build 10 pairs of llm.called/responded (steps 1–10)
    for i in range(1, 11):
        events.append(_infl_ev("llm.called",    rid, i,   {"model": "test"}))
        events.append(_infl_ev("llm.responded", rid, i,
                               {"finish_reason": "stop", "prompt_tokens": 50 + i * 5,
                                "output_length": 30}))
        events.append(_infl_ev("tool.called",   rid, i,
                               {"tool_name": "search", "args_hash": f"h{i}"}))
        events.append(_infl_ev("tool.responded",rid, i, {"success": True}))
    events.append(_infl_ev("run.completed", rid, 20, {"exit_reason": "completed"}))
    ingest(INFL_AGENT_ID, rid, events)
    return rid


# ── Main ───────────────────────────────────────────────────────────────────────

EXPECTED = {
    "TOOL_LOOP",
    "TOOL_THRASHING",
    "TOOL_AVOIDANCE",
    "RAG_EMPTY_RETRIEVAL",
    "LLM_TRUNCATION_LOOP",
    "CONTEXT_BLOAT",
    "SLOW_STEP",
    "RETRY_STORM",
    "EMPTY_LLM_RESPONSE",
    "CASCADING_TOOL_FAILURE",
    "FIRST_STEP_FAILURE",
    "STEP_COUNT_INFLATION",
}

SEVERITY_COLOUR = {
    "CRITICAL": "\033[91m",
    "HIGH":     "\033[93m",
    "MEDIUM":   "\033[94m",
    "LOW":      "\033[96m",
}
RESET = "\033[0m"
GREEN = "\033[92m"
RED   = "\033[91m"


def main() -> None:
    print("=" * 65)
    print("DuneTrace — All-Detector Synthetic Smoke Test")
    print(f"  Ingest : {INGEST_URL}")
    print(f"  API    : {API_URL}")
    print(f"  Agent  : {AGENT_ID}")
    print(f"  Infl.  : {INFL_AGENT_ID}")
    print(f"  Target : {len(EXPECTED)} detectors")
    print("=" * 65)

    # ── 1. Health ────────────────────────────────────────────────────────────
    print("\n[1/5] Checking service health...")
    wait_healthy(INGEST_URL, "ingest")
    wait_healthy(API_URL,    "api")

    # ── 2. Baseline for STEP_COUNT_INFLATION ─────────────────────────────────
    print("\n[2/5] Injecting 10 baseline runs for STEP_COUNT_INFLATION...")
    inject_inflation_baseline(10)
    print("      Waiting 20 s for detector to process baseline runs...")
    time.sleep(20)

    # ── 3. Inject all scenarios ───────────────────────────────────────────────
    print("\n[3/5] Injecting 11 detector scenarios...")
    scenarios = [
        ("TOOL_LOOP",              scenario_tool_loop),
        ("TOOL_THRASHING",         scenario_tool_thrashing),
        ("TOOL_AVOIDANCE",         scenario_tool_avoidance),
        ("RAG_EMPTY_RETRIEVAL",    scenario_rag_empty_retrieval),
        ("LLM_TRUNCATION_LOOP",    scenario_llm_truncation_loop),
        ("CONTEXT_BLOAT",          scenario_context_bloat),
        ("SLOW_STEP",              scenario_slow_step),
        ("RETRY_STORM",            scenario_retry_storm),
        ("EMPTY_LLM_RESPONSE",     scenario_empty_llm_response),
        ("CASCADING_TOOL_FAILURE", scenario_cascading_tool_failure),
        ("FIRST_STEP_FAILURE",     scenario_first_step_failure),
    ]
    for name, fn in scenarios:
        try:
            fn()
            print(f"  ✓  {name}")
        except Exception as exc:
            print(f"  ✗  {name}  ERROR: {exc}")
            sys.exit(1)

    # STEP_COUNT_INFLATION goes last (needs baseline already in processed_runs)
    try:
        scenario_step_count_inflation()
        print("  ✓  STEP_COUNT_INFLATION")
    except Exception as exc:
        print(f"  ✗  STEP_COUNT_INFLATION  ERROR: {exc}")
        sys.exit(1)

    # ── 4. Poll for results ───────────────────────────────────────────────────
    print("\n[4/5] Polling for signals (up to 60 s)...")
    deadline = time.time() + 60
    found: dict[str, dict] = {}

    while time.time() < deadline:
        for agent, label in [(AGENT_ID, "main"), (INFL_AGENT_ID, "infl")]:
            try:
                sigs = fetch_signals(agent)
                for s in sigs:
                    ft = s.get("failure_type", "")
                    if ft in EXPECTED and ft not in found:
                        found[ft] = s
            except Exception:
                pass
        if found.keys() >= EXPECTED:
            break
        missing = EXPECTED - found.keys()
        print(f"  {len(found)}/{len(EXPECTED)} found, waiting... missing: {', '.join(sorted(missing))}")
        time.sleep(5)

    # ── 5. Report ─────────────────────────────────────────────────────────────
    print("\n[5/5] Results:\n")

    all_pass = True
    for ft in sorted(EXPECTED):
        sig  = found.get(ft)
        if sig:
            sev   = sig.get("severity", "?")
            conf  = sig.get("confidence", 0)
            col   = SEVERITY_COLOUR.get(sev, "")
            label = f"{col}{sev}{RESET}"
            print(f"  {GREEN}PASS{RESET}  {ft:<30}  {label:<20}  {conf:.0%}")
        else:
            all_pass = False
            print(f"  {RED}FAIL{RESET}  {ft}")

    not_tested = [
        "GOAL_ABANDONMENT        (needs stalled run i.e. 90 s wait, not suitable for smoke test)",
        "PROMPT_INJECTION_SIGNAL (triggered pre-ingest via SDK check_input, not via pipeline)",
    ]
    print(f"\n  Skipped:")
    for s in not_tested:
        print(f"    ✗  {s}")

    total = len(EXPECTED)
    passed = len(found.keys() & EXPECTED)
    print(f"\n  {passed}/{total} detectors passed.")

    if not all_pass:
        failed = sorted(EXPECTED - found.keys())
        print(f"\n  Failed detectors: {failed}")
        print("  Check detector logs:  docker compose logs detector --tail=50")
        sys.exit(1)

    print(f"\n  {GREEN}All {total} detectors firing correctly.{RESET}")
    sys.exit(0)


if __name__ == "__main__":
    main()
