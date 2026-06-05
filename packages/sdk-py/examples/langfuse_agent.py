"""
LangChain agent with Dunetrace + Langfuse running simultaneously.

Both callbacks receive the same LangChain run_id, which means:
  Dunetrace run_id == Langfuse trace_id

After the tool-loop run, this script queries the Dunetrace API for the
triggered signal and calls POST /v1/signals/{id}/explain to get a
Langfuse-backed root-cause explanation.

Install:
    pip install 'dunetrace[langchain,langfuse]' langchain-openai langgraph langfuse httpx anthropic

Run (happy path):
    OPENAI_API_KEY=sk-...
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    ANTHROPIC_API_KEY=sk-ant-...
    python examples/langfuse_agent.py

Run (tool loop, triggers TOOL_LOOP signal + explain):
    SCENARIO=tool_loop python examples/langfuse_agent.py
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid as uuid_mod
import warnings
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / ".env")

import httpx
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# The suggested migration target (langchain.agents.create_react_agent) doesn't
# exist in the installed langchain version, so the warning is premature.
warnings.filterwarnings(
    "ignore", message=".*create_react_agent.*", category=DeprecationWarning
)

from dunetrace import Dunetrace
from dunetrace.integrations.langchain import DunetraceCallbackHandler

try:
    try:
        from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler  # v4+
    except ImportError:
        from langfuse.callback import (
            CallbackHandler as LangfuseCallbackHandler,
        )  # v2/v3
    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False
    print("Warning: langfuse not installed. Run without Langfuse explain.")
    print("  pip install langfuse")

# ── Config ─────────────────────────────────────────────────────────────────────

DUNETRACE_ENDPOINT = os.getenv("DUNETRACE_ENDPOINT", "http://localhost:8001")
DUNETRACE_API = os.getenv("DUNETRACE_API", "http://localhost:8002")
DUNETRACE_KEY = os.getenv("DUNETRACE_KEY", "dt_dev_test")
AGENT_ID = "langfuse-example-agent"

SYSTEM_PROMPT = "You are a research assistant. Use the search tool to find information before answering."

dt = Dunetrace(endpoint=DUNETRACE_ENDPOINT)


# ── Tools ──────────────────────────────────────────────────────────────────────


@tool
def web_search(query: str) -> str:
    """Search the web for information on a topic."""
    time.sleep(0.2)
    return f"Search results for '{query}': simulated result."


tools = [web_search]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

dt_callback = DunetraceCallbackHandler(
    dt,
    agent_id=AGENT_ID,
    system_prompt=SYSTEM_PROMPT,
    model="gpt-4o-mini",
    tools=[t.name for t in tools],
)

agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)


# ── Run ────────────────────────────────────────────────────────────────────────


def run_agent(query: str) -> str:
    # Pre-generate the run_id so both Dunetrace and Langfuse use the same value.
    # LangChain / Dunetrace use standard UUID format (with dashes).
    # Langfuse v4 uses OTel 32-char hex format (no dashes) — pass uuid.hex.
    # In the fetch, we normalize by stripping dashes before querying Langfuse.
    shared_run_id = uuid_mod.uuid4()
    callbacks = [dt_callback]

    if _LANGFUSE_AVAILABLE:
        lf_callback = LangfuseCallbackHandler(
            trace_context={"trace_id": shared_run_id.hex}  # 32-char hex, no dashes
        )
        callbacks.append(lf_callback)

    result = agent.invoke(
        {"messages": [("human", query)]},
        config={"callbacks": callbacks, "run_id": shared_run_id},
    )

    dt.shutdown(timeout=5)

    lf_trace_id = None
    if _LANGFUSE_AVAILABLE:
        import langfuse as lf_module

        lf_module.get_client().flush()
        lf_trace_id = getattr(lf_callback, "last_trace_id", None)

    # Use the run_id that DunetraceCallbackHandler actually assigned (not our
    # pre-generated shared_run_id, which LangGraph ignores for callbacks).
    dt_run_id = dt_callback.last_run_id

    return result["messages"][-1].content, dt_run_id, lf_trace_id


# ── Post-run: fetch signal + explain ──────────────────────────────────────────


async def fetch_and_explain(
    dt_run_id: Optional[str], lf_trace_id: Optional[str]
) -> None:
    """
    1. Poll until a signal for the current run_id appears (up to 30s).
    2. Call POST /v1/signals/{id}/explain, passing the Langfuse trace_id override.
    """
    headers = {"Authorization": f"Bearer {DUNETRACE_KEY}"}

    async with httpx.AsyncClient(timeout=60) as client:
        top = None
        for attempt in range(6):  # up to ~30s
            await asyncio.sleep(5)
            resp = await client.get(
                f"{DUNETRACE_API}/v1/agents/{AGENT_ID}/signals",
                headers=headers,
            )
            resp.raise_for_status()
            signals = resp.json().get("signals", [])
            top = next((s for s in signals if s["run_id"] == dt_run_id), None)
            if top:
                break
            print(f"  waiting for signal (attempt {attempt + 1}/6)…")

        if not top:
            print("\nNo signal for this run found — check the dashboard.")
            return
        signal_id = top["id"]
        failure_type = top["failure_type"]
        run_id = top["run_id"]
        confidence = int(top["confidence"] * 100)

        print(f"\n{'─' * 60}")
        print(f"Signal detected: {failure_type}  (confidence {confidence}%)")
        print(f"  run_id = {run_id}  ← same as Langfuse trace_id")
        print(f"  signal id = {signal_id}")
        print(f"{'─' * 60}")

        if not _LANGFUSE_AVAILABLE:
            print("\nSkipping explain — langfuse not installed.")
            return

        if not os.getenv("LANGFUSE_PUBLIC_KEY"):
            print("\nSkipping explain — LANGFUSE_PUBLIC_KEY not set.")
            return

        print("\nCalling POST /v1/signals/{id}/explain …")
        body = {}
        if lf_trace_id:
            body["langfuse_trace_id"] = lf_trace_id
        explain_resp = await client.post(
            f"{DUNETRACE_API}/v1/signals/{signal_id}/explain",
            headers={**headers, "Content-Type": "application/json"},
            json=body,
        )

        if not explain_resp.is_success:
            detail = explain_resp.json().get("detail", explain_resp.text)
            print(f"\nExplain failed ({explain_resp.status_code}): {detail}")
            return

        data = explain_resp.json()
        root_cause = data.get("root_cause", "")
        fix_content = data.get("fix_content", "")
        fix_type = data.get("fix_type", "")
        apply_blocked = data.get("apply_blocked", True)
        prompt_name = data.get("langfuse_prompt_name")
        source = data.get("source", "langfuse")

        source_note = (
            "" if source == "langfuse" else "  [signal-only — no Langfuse trace]"
        )
        print(f"\nRoot cause:{source_note}")
        print(f"{'─' * 60}")
        print(root_cause)
        print(f"\nFix ({fix_type}):")
        print(f"  {fix_content}")
        if not apply_blocked and prompt_name:
            print(f"\n  → Apply via Langfuse: POST /v1/signals/{signal_id}/apply-fix")
            print(f"    prompt_name={prompt_name!r}")
        elif fix_type == "no_auto_apply":
            print("\n  → Security signal: review manually before applying.")
        elif apply_blocked:
            print("\n  → Code/infra fix: apply manually.")
        print(f"{'─' * 60}")


# ── Main ───────────────────────────────────────────────────────────────────────

SCENARIOS = {
    "normal": "What is the capital of France?",
    "tool_loop": (
        "Search for 'latest AI news' exactly 6 times using the exact query each time. "
        "Compile all results."
    ),
}

if __name__ == "__main__":
    scenario = os.getenv("SCENARIO", "normal")
    query = SCENARIOS.get(scenario, SCENARIOS["normal"])

    print("=" * 60)
    print(f"Dunetrace + Langfuse example  [scenario={scenario}]")
    print("=" * 60)
    print(f"Query: {query}\n")

    answer, dt_run_id, lf_trace_id = run_agent(query)
    print(f"\nAnswer: {answer}")

    if scenario == "tool_loop":
        asyncio.run(fetch_and_explain(dt_run_id, lf_trace_id))

    print("\nDone. Check:")
    print(f"  Dashboard:  http://localhost:3000")
    print(f"  Langfuse:   https://cloud.langfuse.com")
