#!/usr/bin/env python3
"""
End-to-end test: fire all 13 testable detectors, then call the explain endpoint
on each signal and verify the Langfuse integration + fix_type classification.

Detectors covered (13 of 15):
  TOOL_LOOP · TOOL_THRASHING · TOOL_AVOIDANCE · RAG_EMPTY_RETRIEVAL
  LLM_TRUNCATION_LOOP · CONTEXT_BLOAT · SLOW_STEP · RETRY_STORM
  EMPTY_LLM_RESPONSE · CASCADING_TOOL_FAILURE · FIRST_STEP_FAILURE
  STEP_COUNT_INFLATION · REASONING_STALL · PROMPT_INJECTION_SIGNAL (from DB)

Skipped:
  GOAL_ABANDONMENT — requires 90s stall timeout; not suitable for a smoke test.

The explain endpoint is called with a known-good Langfuse trace_id override so
every detector exercises the full LLM + fix_type path even though the synthetic
runs themselves have no Langfuse trace.

Usage:
    docker compose up -d
    python scripts/test_detectors_explain.py

Env vars:
    INGEST_URL   (default: http://localhost:8001)
    API_URL      (default: http://localhost:8002)
    LF_TRACE_ID  override the Langfuse trace used for explain calls
                 (default: the most recent TOOL_LOOP trace from langfuse-example-agent)
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
import urllib.request
import urllib.error

INGEST_URL = os.environ.get("INGEST_URL", "http://localhost:8001")
API_URL    = os.environ.get("API_URL",    "http://localhost:8002")
API_KEY    = "dt_dev_test"

_TS           = int(time.time())
AGENT_ID      = f"synth-explain-{_TS}"
AGENT_VERSION = "v-test-1"
INFL_AGENT_ID = f"synth-explain-infl-{_TS}"
INFL_VERSION  = "v-infl-1"

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"
RESET  = "\033[0m"

# Expected fix_type and apply_blocked per failure type.
# apply_blocked is True for code_change and no_auto_apply.
EXPECTED_FIX_TYPE: dict[str, tuple[str, bool]] = {
    "TOOL_LOOP":              ("prompt_addition", False),
    "TOOL_THRASHING":         ("prompt_addition", False),
    "TOOL_AVOIDANCE":         ("prompt_addition", False),
    "RETRY_STORM":            ("prompt_addition", False),
    "EMPTY_LLM_RESPONSE":     ("prompt_addition", False),
    "STEP_COUNT_INFLATION":   ("prompt_addition", False),
    "REASONING_STALL":        ("prompt_addition", False),
    "GOAL_ABANDONMENT":       ("prompt_addition", False),
    "RAG_EMPTY_RETRIEVAL":    ("code_change",     True),
    "LLM_TRUNCATION_LOOP":    ("code_change",     True),
    "CONTEXT_BLOAT":          ("code_change",     True),
    "SLOW_STEP":              ("code_change",     True),
    "CASCADING_TOOL_FAILURE": ("code_change",     True),
    "FIRST_STEP_FAILURE":     ("code_change",     True),
    "PROMPT_INJECTION_SIGNAL":("no_auto_apply",   True),
}


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _post(url: str, body: dict) -> tuple[dict, int]:
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        body_bytes = e.read()
        try:
            detail = json.loads(body_bytes)
        except Exception:
            detail = {"detail": body_bytes.decode()}
        return detail, e.code


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def ingest(agent_id: str, run_id: str, events: list[dict]) -> None:
    _post(f"{INGEST_URL}/v1/ingest",
          {"api_key": API_KEY, "agent_id": agent_id, "events": events})


def fetch_signals(agent_id: str, limit: int = 200) -> list[dict]:
    return _get(f"{API_URL}/v1/agents/{agent_id}/signals?limit={limit}").get("signals", [])


def mk(tag: str) -> str:
    return f"synth-{tag}-{_TS}-{uuid.uuid4().hex[:6]}"


def ev(event_type: str, run_id: str, agent_id: str, step: int,
       payload: dict | None = None, ts: float | None = None,
       version: str = AGENT_VERSION) -> dict:
    return {
        "event_type":    event_type,
        "run_id":        run_id,
        "agent_id":      agent_id,
        "agent_version": version,
        "step_index":    step,
        "timestamp":     ts or time.time(),
        "payload":       payload or {},
    }


def wait_healthy(url: str, label: str) -> None:
    print(f"  {label}...", end="", flush=True)
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            _get(f"{url}/health")
            print(" ready")
            return
        except Exception:
            print(".", end="", flush=True)
            time.sleep(2)
    raise SystemExit(f"\n{label} not healthy")


# ── Scenario builders ──────────────────────────────────────────────────────────

def s_tool_loop() -> str:
    rid = mk("tl")
    ingest(AGENT_ID, rid, [
        ev("run.started",   rid, AGENT_ID, 0, {"tools": ["search"]}),
        *[ev("tool.called", rid, AGENT_ID, i, {"tool_name": "search", "args_hash": "h1"})
          for i in range(1, 6)],
        ev("run.completed", rid, AGENT_ID, 5, {"exit_reason": "completed"}),
    ])
    return rid

def s_tool_thrashing() -> str:
    rid = mk("th")
    pairs = []
    for i in range(1, 7):
        tool = "search" if i % 2 else "database"
        pairs.append(ev("tool.called", rid, AGENT_ID, i,
                        {"tool_name": tool, "args_hash": f"h{i}"}))
    ingest(AGENT_ID, rid, [
        ev("run.started",   rid, AGENT_ID, 0, {"tools": ["search", "database"]}),
        *pairs,
        ev("run.completed", rid, AGENT_ID, 6, {"exit_reason": "completed"}),
    ])
    return rid

def s_tool_avoidance() -> str:
    rid = mk("ta")
    ingest(AGENT_ID, rid, [
        ev("run.started",   rid, AGENT_ID, 0, {"tools": ["search"]}),
        ev("llm.called",    rid, AGENT_ID, 1, {"model": "test"}),
        ev("llm.responded", rid, AGENT_ID, 1,
           {"finish_reason": "stop", "prompt_tokens": 100, "output_length": 80}),
        ev("llm.called",    rid, AGENT_ID, 2, {"model": "test"}),
        ev("llm.responded", rid, AGENT_ID, 2,
           {"finish_reason": "stop", "prompt_tokens": 110, "output_length": 120}),
        ev("run.completed", rid, AGENT_ID, 2, {"exit_reason": "final_answer"}),
    ])
    return rid

def s_rag_empty() -> str:
    rid = mk("rag")
    ingest(AGENT_ID, rid, [
        ev("run.started",         rid, AGENT_ID, 0),
        ev("retrieval.called",    rid, AGENT_ID, 1, {"index_name": "kb"}),
        ev("retrieval.responded", rid, AGENT_ID, 1,
           {"index_name": "kb", "result_count": 0, "top_score": 0.0}),
        ev("llm.called",    rid, AGENT_ID, 2, {"model": "test"}),
        ev("llm.responded", rid, AGENT_ID, 2,
           {"finish_reason": "stop", "prompt_tokens": 100, "output_length": 60}),
        ev("run.completed", rid, AGENT_ID, 2, {"exit_reason": "final_answer"}),
    ])
    return rid

def s_llm_truncation() -> str:
    rid = mk("trunc")
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

def s_context_bloat() -> str:
    """CONTEXT_BLOAT: last_tokens (2200) > MIN_LAST_TOKENS (2000), growth 22× > 3.0."""
    rid = mk("bloat")
    ingest(AGENT_ID, rid, [
        ev("run.started",   rid, AGENT_ID, 0),
        ev("llm.called",    rid, AGENT_ID, 1, {"model": "test"}),
        ev("llm.responded", rid, AGENT_ID, 1,
           {"finish_reason": "stop", "prompt_tokens": 100, "output_length": 50}),
        ev("llm.called",    rid, AGENT_ID, 2, {"model": "test"}),
        ev("llm.responded", rid, AGENT_ID, 2,
           {"finish_reason": "stop", "prompt_tokens": 800, "output_length": 50}),
        ev("llm.called",    rid, AGENT_ID, 3, {"model": "test"}),
        ev("llm.responded", rid, AGENT_ID, 3,
           {"finish_reason": "stop", "prompt_tokens": 2200, "output_length": 50}),
        ev("run.completed", rid, AGENT_ID, 3, {"exit_reason": "completed"}),
    ])
    return rid

def s_slow_step() -> str:
    rid = mk("slow")
    T = time.time() - 25
    ingest(AGENT_ID, rid, [
        ev("run.started",   rid, AGENT_ID, 0, ts=T),
        ev("tool.called",   rid, AGENT_ID, 1,
           {"tool_name": "slow_api", "args_hash": "h1"}, ts=T),
        ev("run.completed", rid, AGENT_ID, 2,
           {"exit_reason": "completed"}, ts=T + 20),
    ])
    return rid

def s_retry_storm() -> str:
    rid = mk("retry")
    ingest(AGENT_ID, rid, [
        ev("run.started",    rid, AGENT_ID, 0),
        ev("tool.called",    rid, AGENT_ID, 1,
           {"tool_name": "payment_api", "args_hash": "h1"}),
        ev("tool.responded", rid, AGENT_ID, 1,
           {"tool_name": "payment_api", "success": False}),
        ev("tool.called",    rid, AGENT_ID, 2,
           {"tool_name": "payment_api", "args_hash": "h2"}),
        ev("tool.responded", rid, AGENT_ID, 2,
           {"tool_name": "payment_api", "success": False}),
        ev("tool.called",    rid, AGENT_ID, 3,
           {"tool_name": "payment_api", "args_hash": "h3"}),
        ev("tool.responded", rid, AGENT_ID, 3,
           {"tool_name": "payment_api", "success": False}),
        ev("run.completed",  rid, AGENT_ID, 3, {"exit_reason": "error"}),
    ])
    return rid

def s_empty_llm() -> str:
    rid = mk("empty")
    ingest(AGENT_ID, rid, [
        ev("run.started",   rid, AGENT_ID, 0),
        ev("llm.called",    rid, AGENT_ID, 1, {"model": "test"}),
        ev("llm.responded", rid, AGENT_ID, 1,
           {"finish_reason": "stop", "prompt_tokens": 150, "output_length": 0}),
        ev("run.errored",   rid, AGENT_ID, 1),
    ])
    return rid

def s_cascading_failure() -> str:
    rid = mk("cascade")
    ingest(AGENT_ID, rid, [
        ev("run.started",    rid, AGENT_ID, 0),
        ev("tool.called",    rid, AGENT_ID, 1,
           {"tool_name": "db",     "args_hash": "h1"}),
        ev("tool.responded", rid, AGENT_ID, 1,
           {"tool_name": "db",     "success": False}),
        ev("tool.called",    rid, AGENT_ID, 2,
           {"tool_name": "search", "args_hash": "h2"}),
        ev("tool.responded", rid, AGENT_ID, 2,
           {"tool_name": "search", "success": False}),
        ev("tool.called",    rid, AGENT_ID, 3,
           {"tool_name": "db",     "args_hash": "h3"}),
        ev("tool.responded", rid, AGENT_ID, 3,
           {"tool_name": "db",     "success": False}),
        ev("run.errored",    rid, AGENT_ID, 3),
    ])
    return rid

def s_first_step_failure() -> str:
    rid = mk("first-fail")
    ingest(AGENT_ID, rid, [
        ev("run.started", rid, AGENT_ID, 0),
        ev("run.errored", rid, AGENT_ID, 1),
    ])
    return rid

def s_reasoning_stall() -> str:
    """5 LLM calls, 1 tool call → ratio 5.0 > threshold 4.0."""
    rid = mk("reason")
    events: list[dict] = [ev("run.started", rid, AGENT_ID, 0, {"tools": ["search"]})]
    for i in range(1, 6):
        events.append(ev("llm.called",    rid, AGENT_ID, i, {"model": "test"}))
        events.append(ev("llm.responded", rid, AGENT_ID, i,
                         {"finish_reason": "stop", "prompt_tokens": 100 + i * 10,
                          "output_length": 40}))
    events.append(ev("tool.called",    rid, AGENT_ID, 3, {"tool_name": "search", "args_hash": "h1"}))
    events.append(ev("tool.responded", rid, AGENT_ID, 3, {"success": True}))
    events.append(ev("run.completed",  rid, AGENT_ID, 6, {"exit_reason": "final_answer"}))
    ingest(AGENT_ID, rid, events)
    return rid

# ── STEP_COUNT_INFLATION helpers (separate agent) ─────────────────────────────

def _infl(event_type: str, run_id: str, step: int,
          payload: dict | None = None, ts: float | None = None) -> dict:
    return ev(event_type, run_id, INFL_AGENT_ID, step, payload, ts, version=INFL_VERSION)

def inject_inflation_baseline(n: int = 10) -> None:
    for _ in range(n):
        rid = mk("base")
        batch = [_infl("run.started", rid, 0)]
        for i in range(1, 6):
            batch += [
                _infl("llm.called",    rid, i, {"model": "test"}),
                _infl("llm.responded", rid, i,
                      {"finish_reason": "stop", "prompt_tokens": 50 + i * 5, "output_length": 30}),
            ]
        batch.append(_infl("run.completed", rid, 5, {"exit_reason": "completed"}))
        ingest(INFL_AGENT_ID, rid, batch)

def s_step_count_inflation() -> str:
    rid = mk("infl")
    events = [_infl("run.started", rid, 0)]
    for i in range(1, 11):
        events += [
            _infl("llm.called",    rid, i,   {"model": "test"}),
            _infl("llm.responded", rid, i,
                  {"finish_reason": "stop", "prompt_tokens": 50 + i * 5, "output_length": 30}),
            _infl("tool.called",   rid, i,   {"tool_name": "search", "args_hash": f"h{i}"}),
            _infl("tool.responded",rid, i,   {"success": True}),
        ]
    events.append(_infl("run.completed", rid, 20, {"exit_reason": "completed"}))
    ingest(INFL_AGENT_ID, rid, events)
    return rid


# ── Explain call ───────────────────────────────────────────────────────────────

def explain(signal_id: int, lf_trace_id: str) -> tuple[dict, int]:
    return _post(
        f"{API_URL}/v1/signals/{signal_id}/explain",
        {"langfuse_trace_id": lf_trace_id},
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("Dunetrace — All-detector explain + Langfuse integration test")
    print(f"  Ingest : {INGEST_URL}")
    print(f"  API    : {API_URL}")
    print("=" * 70)

    # ── Health ────────────────────────────────────────────────────────────────
    print("\n[0] Service health")
    wait_healthy(INGEST_URL, "  ingest")
    wait_healthy(API_URL,    "  api   ")

    # ── Find known-good Langfuse trace ────────────────────────────────────────
    print("\n[1] Finding known-good Langfuse trace...")
    lf_trace_id = os.environ.get("LF_TRACE_ID")
    if not lf_trace_id:
        try:
            sigs = fetch_signals("langfuse-example-agent", limit=5)
            if sigs:
                raw_id   = sigs[0]["run_id"]
                lf_trace_id = raw_id.replace("-", "")
                print(f"  Using most recent TOOL_LOOP trace: {lf_trace_id}")
            else:
                print("  WARNING: no langfuse-example-agent signals found."
                      " Set LF_TRACE_ID= to override.")
        except Exception as exc:
            print(f"  WARNING: could not fetch trace ID: {exc}")
    if not lf_trace_id:
        print(f"  {RED}ABORT: cannot run explain tests without a Langfuse trace ID.{RESET}")
        print("  Run:  SCENARIO=tool_loop python packages/sdk-py/examples/langfuse_agent.py")
        sys.exit(1)

    # ── Baseline for STEP_COUNT_INFLATION ─────────────────────────────────────
    print("\n[2] Injecting 10 baseline runs for STEP_COUNT_INFLATION...")
    inject_inflation_baseline(10)
    print("    Waiting 20s for detector to process baseline runs...")
    time.sleep(20)

    # ── Inject all scenarios ──────────────────────────────────────────────────
    print("\n[3] Injecting 12 synthetic scenarios...")
    SYNTH_SCENARIOS = [
        ("TOOL_LOOP",              s_tool_loop),
        ("TOOL_THRASHING",         s_tool_thrashing),
        ("TOOL_AVOIDANCE",         s_tool_avoidance),
        ("RAG_EMPTY_RETRIEVAL",    s_rag_empty),
        ("LLM_TRUNCATION_LOOP",    s_llm_truncation),
        ("CONTEXT_BLOAT",          s_context_bloat),
        ("SLOW_STEP",              s_slow_step),
        ("RETRY_STORM",            s_retry_storm),
        ("EMPTY_LLM_RESPONSE",     s_empty_llm),
        ("CASCADING_TOOL_FAILURE", s_cascading_failure),
        ("FIRST_STEP_FAILURE",     s_first_step_failure),
        ("REASONING_STALL",        s_reasoning_stall),
    ]
    for name, fn in SYNTH_SCENARIOS:
        try:
            fn()
            print(f"  injected  {name}")
        except Exception as exc:
            print(f"  {RED}ERROR{RESET}  {name}: {exc}")
            sys.exit(1)

    try:
        s_step_count_inflation()
        print("  injected  STEP_COUNT_INFLATION")
    except Exception as exc:
        print(f"  {RED}ERROR{RESET}  STEP_COUNT_INFLATION: {exc}")
        sys.exit(1)

    # ── Poll for signals ──────────────────────────────────────────────────────
    SYNTH_TARGETS = {name for name, _ in SYNTH_SCENARIOS} | {"STEP_COUNT_INFLATION"}
    print(f"\n[4] Polling for {len(SYNTH_TARGETS)} synthetic signals (up to 60s)...")
    deadline = time.time() + 60
    found_synth: dict[str, dict] = {}
    while time.time() < deadline:
        for agent_id in (AGENT_ID, INFL_AGENT_ID):
            for s in fetch_signals(agent_id):
                ft = s["failure_type"]
                if ft in SYNTH_TARGETS and ft not in found_synth:
                    found_synth[ft] = s
        if found_synth.keys() >= SYNTH_TARGETS:
            break
        missing = SYNTH_TARGETS - found_synth.keys()
        print(f"  {len(found_synth)}/{len(SYNTH_TARGETS)}  missing: {', '.join(sorted(missing))}")
        time.sleep(5)

    print(f"  Found: {len(found_synth)}/{len(SYNTH_TARGETS)}")

    # Also pull PROMPT_INJECTION from existing DB (example-agent)
    print("\n[5] Looking for PROMPT_INJECTION_SIGNAL in existing data...")
    found_injection: dict | None = None
    for agent_id in ("example-agent", "basic-example-agent"):
        try:
            sigs = fetch_signals(agent_id)
            found_injection = next(
                (s for s in sigs if s["failure_type"] == "PROMPT_INJECTION_SIGNAL"), None
            )
            if found_injection:
                print(f"  Found signal {found_injection['id']} on {agent_id}")
                break
        except Exception:
            pass
    if not found_injection:
        print(f"  {YELLOW}SKIP{RESET}  No PROMPT_INJECTION_SIGNAL found in DB")

    # ── Run explain on every signal ───────────────────────────────────────────
    print(f"\n[6] Running explain on each signal (Langfuse trace: {lf_trace_id[:12]}...)\n")

    results: list[tuple[str, str, str]] = []   # (failure_type, status, detail)

    def run_explain_check(sig: dict) -> None:
        ft         = sig["failure_type"]
        sig_id     = sig["id"]
        exp_type, exp_blocked = EXPECTED_FIX_TYPE.get(ft, ("?", None))

        resp, status = explain(sig_id, lf_trace_id)

        if status != 200:
            detail = resp.get("detail", str(resp))
            results.append((ft, "FAIL", f"explain returned {status}: {detail}"))
            return

        rc  = resp.get("root_cause", "")
        fc  = resp.get("fix_content", "")
        ft_ = resp.get("fix_type", "")
        ab  = resp.get("apply_blocked")

        failures = []
        if not rc:
            failures.append("root_cause is empty")
        if not fc:
            failures.append("fix_content is empty")
        if ft_ != exp_type:
            failures.append(f"fix_type={ft_!r} expected {exp_type!r}")
        if exp_blocked is not None and ab != exp_blocked:
            failures.append(f"apply_blocked={ab} expected {exp_blocked}")

        if failures:
            results.append((ft, "FAIL", " | ".join(failures)))
        else:
            snippet = rc[:80].replace("\n", " ")
            fix_snip = fc[:60].replace("\n", " ")
            results.append((ft, "PASS",
                             f"fix_type={ft_} ab={ab}  "
                             f"root_cause: {snippet}…  "
                             f"fix: {fix_snip}"))

    all_signals = list(found_synth.values())
    if found_injection:
        all_signals.append(found_injection)

    for sig in all_signals:
        run_explain_check(sig)

    # ── Test apply-fix 403 guard for PROMPT_INJECTION ─────────────────────────
    print("[7] Verifying apply-fix 403 guard for PROMPT_INJECTION_SIGNAL...")
    if found_injection:
        body, status = _post(
            f"{API_URL}/v1/signals/{found_injection['id']}/apply-fix",
            {"fix_content": "test", "langfuse_prompt_name": "test-prompt"},
        )
        if status == 403:
            results.append(("PROMPT_INJECTION_SIGNAL (apply-fix guard)", "PASS",
                             "apply-fix correctly returned 403"))
        else:
            results.append(("PROMPT_INJECTION_SIGNAL (apply-fix guard)", "FAIL",
                             f"expected 403, got {status}: {body.get('detail','')}"))
    else:
        results.append(("PROMPT_INJECTION_SIGNAL (apply-fix guard)", "SKIP",
                        "no signal in DB"))

    # ── Print results ─────────────────────────────────────────────────────────
    print()
    print(f"{'─'*70}")
    print(f"{'FAILURE TYPE':<35}  {'STATUS':<6}  DETAIL")
    print(f"{'─'*70}")

    passed = failed = skipped = 0
    for ft, status, detail in sorted(results):
        if status == "PASS":
            colour = GREEN
            passed += 1
        elif status == "SKIP":
            colour = YELLOW
            skipped += 1
        else:
            colour = RED
            failed += 1
        print(f"  {colour}{status:<6}{RESET}  {ft:<35}  {DIM}{detail[:80]}{RESET}")

    # Detectors we didn't find in synth run
    not_found = SYNTH_TARGETS - found_synth.keys()
    if not_found:
        for ft in sorted(not_found):
            print(f"  {RED}MISS  {RESET}  {ft:<35}  {DIM}signal not detected within timeout{RESET}")
            failed += 1

    print(f"\n  Skipped (needs stall timeout):  GOAL_ABANDONMENT")
    print(f"\n  {GREEN}{passed} passed{RESET}  "
          f"{(RED+str(failed)+RESET) if failed else str(failed)+' failed'}  "
          f"{YELLOW}{skipped} skipped{RESET}\n")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
