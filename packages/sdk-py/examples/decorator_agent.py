"""
Minimal example using @dt.agent() decorator + auto_instrument().

This is the recommended starting point for pure Python agents that call
OpenAI or Anthropic directly. Zero changes to your LLM call sites.

    pip install dunetrace
    python examples/decorator_agent.py                      # happy-path runs
    SCENARIO=failures python examples/decorator_agent.py    # signal-triggering runs

Sends events to http://localhost:8001. Start the backend first:
    docker compose up
"""

import asyncio
import os

from dunetrace import Dunetrace, get_current_run

ENDPOINT = os.getenv("DUNETRACE_ENDPOINT", "http://localhost:8001")

dt = Dunetrace(endpoint=ENDPOINT)
dt.auto_instrument()  # patches openai + anthropic if installed


# ── Sync agent (normal) ───────────────────────────────────────────────────────


@dt.agent("sync-agent", model="gpt-4o", tools=["web_search"])
def run_sync_agent(query: str) -> str:
    """
    Simulate a sync agent. In a real agent you'd call:
        openai_client.chat.completions.create(...)  <-- auto-tracked
    """
    run = get_current_run()
    run.tool_called("web_search", {"query": query})
    run.tool_responded("web_search", success=True, output_length=256)
    return f"Answer to: {query}"


# ── Async agent (normal) ──────────────────────────────────────────────────────


@dt.agent("async-agent", model="claude-3-5-sonnet")
async def run_async_agent(query: str) -> str:
    """Same pattern for async agents — decorator handles both."""
    run = get_current_run()
    run.tool_called("vector_search", {"query": query})
    await asyncio.sleep(0.01)
    run.tool_responded("vector_search", success=True, output_length=128)
    return f"Async answer to: {query}"


# ── RAG agent (normal) ────────────────────────────────────────────────────────


@dt.agent("rag-agent", model="gpt-4o", input_from="question")
def run_rag_agent(context: str, question: str) -> str:
    """Use input_from when the user query is not the first argument."""
    run = get_current_run()
    run.retrieval_called("product-docs", query_hash="abc123")
    run.retrieval_responded(
        "product-docs", result_count=3, top_score=0.91, latency_ms=22
    )
    return f"RAG answer to: {question}"


# ── Failure scenarios (SCENARIO=failures) ─────────────────────────────────────


@dt.agent("sync-agent", model="gpt-4o", tools=["web_search"])
def run_tool_loop_agent(query: str) -> str:
    """Calls the same tool 5× with identical args → triggers TOOL_LOOP (threshold 3 in window 5)."""
    run = get_current_run()
    for _ in range(5):
        run.tool_called("web_search", {"query": query})
        run.tool_responded("web_search", success=True, output_length=256)
    return f"Tool loop answer to: {query}"


@dt.agent("async-agent", model="claude-3-5-sonnet", tools=["vector_search"])
async def run_retry_storm_agent(query: str) -> str:
    """Fails the same tool 3× in a row → triggers RETRY_STORM (threshold 3)."""
    run = get_current_run()
    for _ in range(3):
        run.tool_called("vector_search", {"query": query})
        await asyncio.sleep(0.01)
        run.tool_responded("vector_search", success=False, error="upstream timeout")
    return f"Retry storm (failed) for: {query}"


@dt.agent("rag-agent", model="gpt-4o", input_from="question")
def run_rag_empty_agent(context: str, question: str) -> str:
    """Returns 0 retrieval results with a low score → triggers RAG_EMPTY_RETRIEVAL."""
    run = get_current_run()
    run.retrieval_called("product-docs", query_hash="abc123")
    run.retrieval_responded(
        "product-docs", result_count=0, top_score=0.05, latency_ms=18
    )
    return f"RAG (empty) answer to: {question}"


# ── Main ──────────────────────────────────────────────────────────────────────


def run_normal() -> None:
    print("\n[sync]  (no signal expected)")
    print(f"  -> {run_sync_agent('What is the capital of France?')}")

    print("\n[async]  (no signal expected)")
    print(f"  -> {asyncio.run(run_async_agent('Explain quantum entanglement'))}")

    print("\n[rag / input_from]  (no signal expected)")
    print(
        f"  -> {run_rag_agent('Product docs context...', question='How do I configure feature X?')}"
    )


def run_failures() -> None:
    print("\n[tool_loop]  → expect TOOL_LOOP signal")
    print(f"  -> {run_tool_loop_agent('latest AI news')}")

    print("\n[retry_storm]  → expect RETRY_STORM signal")
    print(f"  -> {asyncio.run(run_retry_storm_agent('recent papers'))}")

    print("\n[rag_empty]  → expect RAG_EMPTY_RETRIEVAL signal")
    print(
        f"  -> {run_rag_empty_agent('context', question='How do I configure feature X?')}"
    )


if __name__ == "__main__":
    scenario = os.environ.get("SCENARIO", "normal")
    print("=" * 60)
    print(f"Dunetrace SDK - Decorator Agent Examples  [scenario={scenario}]")
    print("=" * 60)

    if scenario == "failures":
        run_failures()
    else:
        run_normal()

    dt.shutdown()
    print("\n" + "=" * 60)
    print("Done. Check http://localhost:3000 for runs.")
