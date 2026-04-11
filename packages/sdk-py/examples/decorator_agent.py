"""
Minimal example using @dt.agent() decorator + auto_instrument().

This is the recommended starting point for pure Python agents that call
OpenAI or Anthropic directly. Zero changes to your LLM call sites.

    pip install dunetrace
    python examples/decorator_agent.py

Sends events to http://localhost:8001. Start the backend first:
    docker compose up
"""
import asyncio
import os

from dunetrace import Dunetrace, get_current_run

ENDPOINT = os.getenv("DUNETRACE_ENDPOINT", "http://localhost:8001")

dt = Dunetrace(endpoint=ENDPOINT)
dt.auto_instrument()   # patches openai + anthropic if installed


# ── Sync agent ────────────────────────────────────────────────────────────────

@dt.agent("sync-agent", model="gpt-4o", tools=["web_search"])
def run_sync_agent(query: str) -> str:
    """
    Simulate a sync agent. In a real agent you'd call:
        openai_client.chat.completions.create(...)  <-- auto-tracked
    """
    run = get_current_run()

    # Manually instrument non-LLM steps (DB, cache, APIs, etc.)
    run.tool_called("web_search", {"query": query})
    # ... your tool logic here ...
    run.tool_responded("web_search", success=True, output_length=256)

    # openai / anthropic calls here would be tracked automatically
    # via auto_instrument() — no run.llm_called() needed

    return f"Answer to: {query}"


# ── Async agent ───────────────────────────────────────────────────────────────

@dt.agent("async-agent", model="claude-3-5-sonnet")
async def run_async_agent(query: str) -> str:
    """Same pattern for async agents — decorator handles both."""
    run = get_current_run()
    run.tool_called("vector_search", {"query": query})
    await asyncio.sleep(0.01)  # simulate async I/O
    run.tool_responded("vector_search", success=True, output_length=128)
    return f"Async answer to: {query}"


# ── Named input_from ──────────────────────────────────────────────────────────

@dt.agent("rag-agent", model="gpt-4o", input_from="question")
def run_rag_agent(context: str, question: str) -> str:
    """Use input_from when the user query is not the first argument."""
    run = get_current_run()
    run.retrieval_called("product-docs", query_hash="abc123")
    run.retrieval_responded("product-docs", result_count=3, top_score=0.91, latency_ms=22)
    return f"RAG answer to: {question}"


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Dunetrace SDK - Decorator Agent Examples")
    print("=" * 60)

    print("\n[sync]")
    result = run_sync_agent("What is the capital of France?")
    print(f"  -> {result}")

    print("\n[async]")
    result = asyncio.run(run_async_agent("Explain quantum entanglement"))
    print(f"  -> {result}")

    print("\n[rag / input_from]")
    result = run_rag_agent("Product docs context...", question="How do I configure feature X?")
    print(f"  -> {result}")

    dt.shutdown()
    print("\n" + "=" * 60)
    print("Done. Check http://localhost:3000 for runs.")
