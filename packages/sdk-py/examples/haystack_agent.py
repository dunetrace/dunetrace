"""
Haystack 2.x agent example for Dunetrace monitoring.

Registers DunetraceHaystackTracer once at startup — no pipeline changes needed.

Install:
    pip install 'dunetrace[haystack]' python-dotenv

Run:
    python examples/haystack_agent.py                       # RAG pipeline
    SCENARIO=tool_loop python examples/haystack_agent.py    # triggers TOOL_LOOP signal

Deploy markers:
    dt.mark_deploy() overlays a blue dashed line on the 30-day detector timeline
    in the dashboard, making failure spikes easy to correlate with releases.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

try:
    from dotenv import load_dotenv

    # Reads the repo-root .env — including OPENAI_API_KEY. This example makes
    # real, billable LLM calls, and will pick up a key from there even if you
    # don't pass one on the command line.
    load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / ".env")
except ImportError:
    pass  # python-dotenv is optional; fall back to the ambient environment.

try:
    import haystack.tracing
    from haystack import Pipeline
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.tools import Tool
except ImportError as exc:
    raise SystemExit(
        f"This example needs Haystack 2.x — {exc.name} is not installed.\n"
        "  pip install 'dunetrace[haystack]' python-dotenv"
    ) from None

from dunetrace import Dunetrace
from dunetrace.integrations.haystack import DunetraceHaystackTracer

# ── Dunetrace setup ────────────────────────────────────────────────────────────

AGENT_ID = "haystack-example-agent"

dt = Dunetrace(endpoint=os.environ.get("DUNETRACE_ENDPOINT", "http://localhost:8001"))

dt.mark_deploy(
    AGENT_ID,
    version=os.environ.get("APP_VERSION", "dev"),
    commit=os.environ.get("GIT_COMMIT", "local"),
    env=os.environ.get("ENV", "development"),
)

SYSTEM_PROMPT = (
    "You are a research assistant. "
    "Use the web_search tool to find information before answering. "
    "Always search at least once before giving a final answer."
)

haystack.tracing.enable_tracing(
    DunetraceHaystackTracer(
        dt,
        agent_id=AGENT_ID,
        system_prompt=SYSTEM_PROMPT,
        model="gpt-4o-mini",
        tools=["web_search"],
    )
)

# ── Tool definitions ───────────────────────────────────────────────────────────

_FACTS: dict = {
    "capital of france": "Paris is the capital of France. Population: ~2.1 million in the city, ~12 million in Greater Paris.",
    "capital of germany": "Berlin is the capital of Germany. Population: ~3.6 million.",
    "capital of japan": "Tokyo is the capital of Japan. Population: ~13.9 million.",
    "population of paris": "Paris has approximately 2.1 million residents. Greater Paris (Île-de-France) has ~12 million.",
}


def web_search(query: str) -> str:
    """Search the web for information on a topic."""
    time.sleep(0.2)
    key = query.lower()
    for k, v in _FACTS.items():
        if k in key:
            return v
    return f"Search results for '{query}': Relevant information about {query}. (simulated)"


search_tool = Tool(
    name="web_search",
    description="Search the web for information on a topic.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
        },
        "required": ["query"],
    },
    function=web_search,
)

# ── Scenarios ─────────────────────────────────────────────────────────────────

SCENARIOS = {
    "normal": "What is the capital of France and what is its population?",
    "tool_loop": (
        "Search for 'latest AI news' exactly 6 times and compile all results. "
        "Each search must use the exact query 'latest AI news' — do not vary the wording."
    ),
}

# ── Agent pipeline ─────────────────────────────────────────────────────────────

llm = OpenAIChatGenerator(model="gpt-4o-mini")
agent = Agent(
    chat_generator=llm,
    tools=[search_tool],
    system_prompt=SYSTEM_PROMPT,
    max_agent_steps=20,
)

pipeline = Pipeline()
pipeline.add_component("agent", agent)


def run(scenario: str = "normal") -> None:
    query = SCENARIOS.get(scenario, SCENARIOS["normal"])
    print(f"\nScenario : {scenario}")
    print(f"Query    : {query}\n")
    try:
        result = pipeline.run({"agent": {"messages": [ChatMessage.from_user(query)]}})
        last_msg = result["agent"]["messages"][-1]
        print(f"\nAnswer: {last_msg.text}")
    except Exception as e:
        print(f"Error: {e}")

    dt.shutdown(timeout=5)
    print("Events flushed.")


if __name__ == "__main__":
    run(os.environ.get("SCENARIO", "normal"))
