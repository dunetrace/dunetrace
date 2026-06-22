"""
Hermes Agent + Dunetrace integration example.

Scenarios (set SCENARIO env var):

    SCENARIO=happy     — normal run: tool called, LLM responds, run completes
    SCENARIO=tool_loop — TOOL_LOOP: same tool called 4× with identical args
    SCENARIO=retry     — RETRY_STORM: tool fails 3 consecutive times
    SCENARIO=abandon   — GOAL_ABANDONMENT: LLM calls with no tools, no answer
    SCENARIO=real      — runs a real Hermes agent (requires OPENAI_API_KEY)
    SCENARIO=real_loop — runs a real Hermes agent that provokes a tool loop

Default is to run all synthetic scenarios.

Synthetic scenarios fire Hermes hooks directly — no LLM API key needed.
The real scenario requires: pip install hermes-agent && OPENAI_API_KEY=sk-...

    # All synthetic scenarios (no credentials needed):
    PYTHONPATH=packages/sdk-py python examples/hermes_agent.py

    # Real Hermes agent:
    SCENARIO=real OPENAI_API_KEY=sk-... python examples/hermes_agent.py
"""

from __future__ import annotations

import os
import sys
import time
import uuid

ENDPOINT = os.getenv("DUNETRACE_ENDPOINT", "http://localhost:8001")
SCENARIO = os.getenv("SCENARIO", "all")
AGENT_ID = "hermes-example-agent"

# ── Bootstrap ──────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dunetrace import Dunetrace
from dunetrace.integrations.hermes import DunetraceHermesPlugin

dt = Dunetrace(endpoint=ENDPOINT)
plugin = DunetraceHermesPlugin(
    dt,
    agent_id=AGENT_ID,
    model="hermes-3-llama-3.1-70b",
    system_prompt="You are a research assistant. Use tools before answering.",
    tools=["web_search", "calculator", "read_file"],
)
plugin.attach()

# ── Synthetic scenario helpers ─────────────────────────────────────────────────
# Fire Hermes hooks directly to test the integration logic without LLM credentials.


def make_turn_id() -> str:
    sid = str(uuid.uuid4())
    return f"{sid}:{uuid.uuid4().hex[:8]}:{uuid.uuid4().hex[:8]}"


def fire_run_start(turn_id: str, user_msg: str = "What is the capital of France?") -> None:
    plugin._pre_llm_call(
        turn_id=turn_id,
        session_id=turn_id.split(":")[0],
        user_message=user_msg,
        model="hermes-3-llama-3.1-70b",
        platform="example",
    )


def fire_llm_call(
    turn_id: str,
    api_request_id: str,
    call_count: int,
    input_tokens: int = 250,
    output_tokens: int = 80,
    finish_reason: str = "stop",
    latency_s: float = 0.4,
) -> None:
    t0 = time.time()
    plugin._pre_api_request(
        turn_id=turn_id,
        api_request_id=api_request_id,
        api_call_count=call_count,
        model="hermes-3-llama-3.1-70b",
        approx_input_tokens=input_tokens,
        started_at=t0,
    )
    time.sleep(latency_s)
    plugin._post_api_request(
        turn_id=turn_id,
        api_request_id=api_request_id,
        api_duration=latency_s,
        finish_reason=finish_reason,
        usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
    )


def fire_tool_call(
    turn_id: str,
    tool_name: str,
    args: dict,
    tool_call_id: str,
    result: str = "Result text",
    ok: bool = True,
    latency_ms: int = 200,
) -> None:
    plugin._pre_tool_call(
        turn_id=turn_id,
        tool_name=tool_name,
        args=args,
        tool_call_id=tool_call_id,
        session_id="",
        task_id="",
        api_request_id="",
    )
    time.sleep(latency_ms / 1000)
    plugin._post_tool_call(
        turn_id=turn_id,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        duration_ms=latency_ms,
        status="ok" if ok else "error",
        error_message=None if ok else "Tool raised an exception",
        result=result if ok else None,
    )


def fire_run_end(turn_id: str, completed: bool = True, interrupted: bool = False) -> None:
    plugin._on_session_end(
        turn_id=turn_id,
        session_id=turn_id.split(":")[0],
        task_id="",
        completed=completed,
        interrupted=interrupted,
        model="hermes-3-llama-3.1-70b",
        platform="example",
    )


# ── Scenarios ──────────────────────────────────────────────────────────────────


def scenario_happy() -> str:
    """Normal run: web_search + calculator, then final answer."""
    print("\n[happy] Normal run — tool calls + final answer")
    tid = make_turn_id()

    fire_run_start(tid, "What is 15% of 847, and is 847 prime?")
    fire_llm_call(tid, f"{tid}:api:1", 1, finish_reason="tool_calls", latency_s=0.3)
    fire_tool_call(tid, "calculator", {"expression": "847 * 0.15"}, "tc-1", result="127.05")
    fire_tool_call(
        tid,
        "web_search",
        {"query": "is 847 prime number"},
        "tc-2",
        result="847 = 7 × 121 = 7 × 11²  — not prime",
    )
    fire_llm_call(tid, f"{tid}:api:2", 2, finish_reason="stop", latency_s=0.25)
    fire_run_end(tid, completed=True)

    print(f"  run_id={tid.split(':')[0][:8]}…  steps=4  ✓")
    return tid


def scenario_tool_loop() -> str:
    """TOOL_LOOP: same tool called 5× with identical args (WINDOW=5, THRESHOLD=3)."""
    print("\n[tool_loop] TOOL_LOOP — same web_search query repeated 5 times")
    tid = make_turn_id()

    fire_run_start(tid, "Find the latest research on LLM alignment")
    for i in range(1, 6):
        fire_llm_call(tid, f"{tid}:api:{i}", i, finish_reason="tool_calls", latency_s=0.1)
        fire_tool_call(
            tid,
            "web_search",
            {"query": "LLM alignment latest research 2025"},
            f"tc-loop-{i}",
            result="[search result]",
        )
    fire_llm_call(tid, f"{tid}:api:6", 6, finish_reason="stop", latency_s=0.1)
    fire_run_end(tid, completed=True)

    print(f"  run_id={tid.split(':')[0][:8]}…  steps=11  → expect TOOL_LOOP signal")
    return tid


def scenario_retry_storm() -> str:
    """RETRY_STORM: read_file fails 3 consecutive times."""
    print("\n[retry] RETRY_STORM — tool fails 3× in a row")
    tid = make_turn_id()

    fire_run_start(tid, "Read the config file and summarize it")
    fire_llm_call(tid, f"{tid}:api:1", 1, finish_reason="tool_calls", latency_s=0.2)
    for i in range(1, 4):
        fire_tool_call(
            tid,
            "read_file",
            {"path": "/etc/config.yaml"},
            f"tc-fail-{i}",
            ok=False,
            latency_ms=50,
        )
        fire_llm_call(tid, f"{tid}:api:{i + 1}", i + 1, finish_reason="tool_calls", latency_s=0.1)
    fire_run_end(tid, completed=False, interrupted=False)

    print(f"  run_id={tid.split(':')[0][:8]}…  steps=6  → expect RETRY_STORM signal")
    return tid


def scenario_goal_abandonment() -> str:
    """GOAL_ABANDONMENT: agent stops using tools and keeps LLM-calling."""
    print("\n[abandon] GOAL_ABANDONMENT — 5 consecutive LLM calls, no tools")
    tid = make_turn_id()

    fire_run_start(tid, "Research and summarize the history of neural networks")
    fire_llm_call(tid, f"{tid}:api:1", 1, finish_reason="tool_calls", latency_s=0.2)
    fire_tool_call(tid, "web_search", {"query": "history neural networks"}, "tc-1", result="...")
    # Agent stops using tools but keeps calling LLM — goal abandonment
    for i in range(2, 7):
        fire_llm_call(tid, f"{tid}:api:{i}", i, finish_reason="stop", latency_s=0.15)
    fire_run_end(tid, completed=True)

    print(f"  run_id={tid.split(':')[0][:8]}…  steps=7  → expect GOAL_ABANDONMENT signal")
    return tid


def scenario_edge_missing_turn_id() -> str:
    """Edge case: hooks fire with empty turn_id — should be ignored gracefully."""
    print("\n[edge] Empty turn_id — hooks should no-op without crashing")
    plugin._pre_api_request(turn_id="", api_request_id="x", api_call_count=1)
    plugin._post_tool_call(turn_id="", tool_name="web_search", tool_call_id="x", status="ok")
    plugin._on_session_end(turn_id="", completed=True, interrupted=False)
    print("  ✓ No exceptions raised")
    return ""


def scenario_edge_interrupted() -> str:
    """Edge case: run interrupted mid-way — maps to run.errored."""
    print("\n[edge] Interrupted run — should emit run.errored")
    tid = make_turn_id()
    fire_run_start(tid, "What is the weather in San Francisco?")
    fire_llm_call(tid, f"{tid}:api:1", 1, finish_reason="tool_calls", latency_s=0.1)
    # User cancelled before tool returned
    fire_run_end(tid, completed=False, interrupted=True)
    print(f"  run_id={tid.split(':')[0][:8]}…  → run.errored (interrupted=True)")
    return tid


def scenario_edge_duplicate_attach() -> None:
    """Edge case: attach() called twice — events should not double-fire."""
    print("\n[edge] Double attach() — events should fire once per hook (Hermes appends)")
    plugin2 = DunetraceHermesPlugin(dt, agent_id=AGENT_ID + "-dup")
    plugin2.attach()
    plugin2.attach()  # second attach — would double-append, warn user in docs
    tid = make_turn_id()
    fire_run_start(tid, "Test dedup")
    fire_run_end(tid, completed=True)
    print("  ✓ No crash (note: each attach() adds another handler — use one instance)")


# ── Real Hermes scenarios ──────────────────────────────────────────────────────


def scenario_real() -> None:
    """Real Hermes agent run — requires OPENAI_API_KEY."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n[real] Skipped — set OPENAI_API_KEY to run this scenario")
        return

    try:
        from run_agent import AIAgent
    except ImportError:
        print("\n[real] Skipped — hermes-agent not installed (pip install hermes-agent)")
        return

    print("\n[real] Running a real Hermes agent with Dunetrace hooks attached...")
    agent = AIAgent(
        api_key=api_key,
        model="gpt-4o-mini",
        quiet_mode=True,
        enabled_toolsets=["core"],  # minimal toolset for the test
    )
    result = agent.run_conversation("What is 42 * 17? Use the calculator tool.")
    print(f"  Response: {result.get('final_response', '')[:120]}")
    print(f"  API calls: {result.get('api_calls', 0)}  completed={result.get('completed')}")
    print("  ✓ Real Hermes run traced")


def scenario_real_loop() -> None:
    """Real Hermes agent provoked into a tool loop."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n[real_loop] Skipped — set OPENAI_API_KEY to run this scenario")
        return

    try:
        from run_agent import AIAgent
    except ImportError:
        print("\n[real_loop] Skipped — hermes-agent not installed")
        return

    print("\n[real_loop] Provoking tool loop in real Hermes agent...")
    agent = AIAgent(
        api_key=api_key,
        model="gpt-4o-mini",
        quiet_mode=True,
        enabled_toolsets=["core"],
        max_iterations=8,
        ephemeral_system_prompt=(
            "Every time the user asks a question, you MUST call web_search "
            "at least 5 times with the exact same query before answering."
        ),
    )
    result = agent.run_conversation("What is the capital of France?")
    print(f"  API calls: {result.get('api_calls', 0)}  completed={result.get('completed')}")
    print("  ✓ Tool loop run traced — check dashboard for TOOL_LOOP signal")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Dunetrace Hermes example  endpoint={ENDPOINT}  agent={AGENT_ID}")
    print("=" * 60)

    run_ids = []

    if SCENARIO in ("all", "happy"):
        run_ids.append(scenario_happy())
    if SCENARIO in ("all", "tool_loop"):
        run_ids.append(scenario_tool_loop())
    if SCENARIO in ("all", "retry"):
        run_ids.append(scenario_retry_storm())
    if SCENARIO in ("all", "abandon"):
        run_ids.append(scenario_goal_abandonment())
    if SCENARIO in ("all", "edge"):
        scenario_edge_missing_turn_id()
        scenario_edge_interrupted()
        scenario_edge_duplicate_attach()
    if SCENARIO == "real":
        scenario_real()
    if SCENARIO == "real_loop":
        scenario_real_loop()

    # Flush all buffered events before exit
    dt.flush()
    print("\n" + "=" * 60)
    print(f"Done. {len([r for r in run_ids if r])} run(s) submitted to Dunetrace.")
    print("Open http://localhost:3000 to see signals.")
