#!/usr/bin/env python3
"""
scripts/smoke_test.py

End-to-end test with a real LangChain + GPT-4o-mini agent.

  1. Waits for ingest + API services to be healthy
  2. Runs two real agent scenarios:
       - tool_loop:      agent forced to search repeatedly -> TOOL_LOOP signal
       - tool_avoidance: trivial question answered from memory -> TOOL_AVOIDANCE signal
  3. Waits for the detector to process the runs
  4. Prints every signal detected with its explanation
  5. Exits 0 on pass, 1 on failure

Usage:
    docker compose up -d --build
    python scripts/smoke_test.py

    # Override if needed:
    INGEST_URL=http://localhost:8001 API_URL=http://localhost:8002 python scripts/smoke_test.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

# ── Load .env from repo root ────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # python-dotenv optional; rely on env vars being set

# ── SDK path ───────────────────────────────────────────────────────────────────
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / "packages" / "sdk-py"))

from dunetrace import Dunetrace
from dunetrace.integrations.langchain import DunetraceCallbackHandler

# ── Config ─────────────────────────────────────────────────────────────────────
INGEST_URL = os.environ.get("INGEST_URL", "http://localhost:8001")
API_URL    = os.environ.get("API_URL",    "http://localhost:8002")
AGENT_ID   = f"real-agent-{int(time.time())}"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY not set. Add it to .env or export it.")
    sys.exit(1)

SYSTEM_PROMPT = (
    "You are a research assistant. "
    "You have access to a search tool. "
    "Use the search tool when you need to find information."
)

# ── Search tool with "incomplete results" behaviour ────────────────────────────
# Returns partial results for the first 3 searches on any query, then "complete".
# This reliably pushes the agent into a loop (TOOL_LOOP detector fires at call 5).
_search_attempts: dict[str, int] = {}

def do_search(query: str) -> str:
    # Normalise the query so quoted variants ("'foo'") count the same as unquoted ("foo")
    key = query.strip().strip("'\"")
    _search_attempts[key] = _search_attempts.get(key, 0) + 1
    n = _search_attempts[key]
    if n < 6:
        return (
            f"[Search {n}/5 for '{key}'] "
            "Results are incomplete — data source still indexing. "
            "Search again to retrieve the remaining results."
        )
    return (
        f"[Search {n}/5 for '{key}'] "
        "Complete results retrieved. "
        "Significant recent advances have been published in this area. "
        "Multiple peer-reviewed papers confirm the trend."
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": "Bearer dt_dev_test"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def wait_healthy(url: str, label: str, timeout: int = 60) -> None:
    print(f"  Waiting for {label}...", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            _get(f"{url}/health")
            print(" ready.")
            return
        except Exception:
            print(".", end="", flush=True)
            time.sleep(2)
    raise SystemExit(f"\nERROR: {label} did not become healthy. Is the stack running?")


def fetch_signals(agent_id: str) -> list[dict]:
    return _get(f"{API_URL}/v1/agents/{agent_id}/signals").get("signals", [])


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_agent
    from langchain.tools import tool as lc_tool

    print("=" * 60)
    print("DuneTrace: Real Agent Smoke Test")
    print(f"  Model  : gpt-4o-mini")
    print(f"  Ingest : {INGEST_URL}")
    print(f"  API    : {API_URL}")
    print(f"  Agent  : {AGENT_ID}")
    print("=" * 60)

    # 1. Health checks
    print("\n[1/4] Checking service health...")
    wait_healthy(INGEST_URL, "ingest")
    wait_healthy(API_URL,    "api")

    # 2. Build the real agent
    print("\n[2/4] Running real GPT-4o-mini scenarios...")

    @lc_tool
    def search(query: str) -> str:
        """Search for information on a topic. May return partial results i.e. search again if incomplete."""
        return do_search(query)

    dt = Dunetrace(
        api_key="dt_dev_test",
        endpoint=INGEST_URL,
        flush_interval_ms=100,
    )

    callback = DunetraceCallbackHandler(
        dt,
        agent_id=AGENT_ID,
        system_prompt=SYSTEM_PROMPT,
        model="gpt-4o-mini",
        tools=["search"],
    )

    llm   = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
    agent = create_agent(llm, [search], system_prompt=SYSTEM_PROMPT)

    # ── Scenario 1: TOOL_LOOP ──────────────────────────────────────────────────
    print("\n  ── Scenario 1: TOOL_LOOP ──")
    print("  (Agent will search repeatedly; detector fires when search is called 3+ times in 5 steps)")
    try:
        result = agent.invoke(
            {"messages": [("human", (
                "I need a thorough literature review on quantum computing advances in 2024. "
                "The search tool may return incomplete results i.e. keep searching with the same "
                "query until you have complete results. "
                "Query to use: 'quantum computing advances 2024'."
            ))]},
            config={"callbacks": [callback]},
        )
        output = result["messages"][-1].content if result.get("messages") else ""
        print(f"\n  Agent answer: {str(output)[:120]}")
    except Exception as exc:
        print(f"\n  Agent error: {exc}")

    # Reset search counter between scenarios
    _search_attempts.clear()

    # ── Scenario 2: TOOL_AVOIDANCE ─────────────────────────────────────────────
    print("\n  Scenario 2: TOOL_AVOIDANCE")
    print("  (Agent answers from memory without using any tool)")
    try:
        result = agent.invoke(
            {"messages": [("human", "In what year did World War II end?")]},
            config={"callbacks": [callback]},
        )
        output = result["messages"][-1].content if result.get("messages") else ""
        print(f"\n  Agent answer: {str(output)[:120]}")
    except Exception as exc:
        print(f"\n  Agent error (expected if model answered without a tool): {exc}")

    dt.shutdown(timeout=5.0)
    print("\n  All runs flushed to ingest.")

    # 3. Wait for detector
    print("\n[3/4] Waiting up to 40s for detector to process runs...")
    deadline = time.time() + 40
    signals: list[dict] = []
    while time.time() < deadline:
        try:
            signals = fetch_signals(AGENT_ID)
        except Exception as exc:
            print(f"  (API error: {exc})")
        if signals:
            break
        print("  ...", end="\r", flush=True)
        time.sleep(3)

    # 4. Report
    print("\n[4/4] Results:")
    if not signals:
        print("  FAIL  No signals detected within 40s.")
        print("        Check: docker compose logs detector")
        sys.exit(1)

    print(f"  Signals detected: {len(signals)}")
    for s in sorted(signals, key=lambda x: (x["failure_type"],)):
        print(f"\n  [{s['severity']:8s}] {s['failure_type']}  confidence={s['confidence']:.0%}")
        if s.get("what"):
            print(f"    What:  {s['what']}")
        if s.get("why_it_matters"):
            print(f"    Why:   {s['why_it_matters']}")
        if s.get("suggested_fixes"):
            fix = s["suggested_fixes"][0]
            print(f"    Fix:   {fix.get('description', '')}")

    print(f"\n  Dashboard : http://localhost:3000")
    sys.exit(0)


if __name__ == "__main__":
    main()
