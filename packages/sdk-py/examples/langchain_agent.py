"""
LangChain agent example. One callback, nothing else changes.
Works with LangChain 1.x and LangGraph (langgraph >= 0.2).

Install:
    pip install 'dunetrace[langchain]' langchain-openai langgraph

Run:
    OPENAI_API_KEY=sk-... python examples/langchain_agent.py
    OPENAI_API_KEY=sk-... SCENARIO=tool_loop python examples/langchain_agent.py

Deploy markers:
    Call dt.mark_deploy() once at startup (or from CI) to overlay a blue dashed
    line on the 30-day detector timeline in the dashboard.  Runs after the marker
    are visually separated from runs before it, making it easy to see whether a
    failure spike started before or after a release.
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / ".env")

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# The suggested migration target (langchain.agents.create_react_agent) doesn't
# exist in the installed langchain version, so the warning is premature.
warnings.filterwarnings("ignore", message=".*create_react_agent.*", category=DeprecationWarning)

from dunetrace import Dunetrace
from dunetrace.integrations.langchain import DunetraceCallbackHandler

dt = Dunetrace(endpoint=os.environ.get("DUNETRACE_ENDPOINT", "http://localhost:8001"))

# ── Deploy marker ──────────────────────────────────────────────────────────────
# Call this once at application startup or from your CI/CD pipeline.
# Fire-and-forget: runs on a background thread, never blocks the caller.
# The dashboard overlays a blue dashed line at this point on the detector timeline.
dt.mark_deploy(
    "langchain-example-agent",
    version=os.environ.get("APP_VERSION", "dev"),
    commit=os.environ.get("GIT_COMMIT", "local"),
    env=os.environ.get("ENV", "development"),
)

SYSTEM_PROMPT = (
    "You are a research assistant. "
    "Use the search tool to find information before answering. "
    "Always search at least once before giving a final answer."
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


@tool
def web_search(query: str) -> str:
    """Search the web for information on a topic."""
    time.sleep(0.3)
    return f"Search results for '{query}': Found relevant information about {query}. This is a simulated result."


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Input: a valid Python arithmetic expression."""
    try:
        allowed = set("0123456789+-*/()., ")
        if all(c in allowed for c in expression):
            return f"{expression} = {eval(expression)}"  # noqa: S307
        return f"Cannot evaluate '{expression}' — only basic arithmetic supported."
    except Exception as e:
        return f"Error: {e}"


tools = [web_search, calculator]

callback = DunetraceCallbackHandler(
    dt,
    agent_id="langchain-example-agent",
    system_prompt=SYSTEM_PROMPT,
    model="gpt-4o-mini",
    tools=[t.name for t in tools],
)

agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

SCENARIOS = {
    "normal": "What is the capital of France and what is its population?",
    "tool_loop": (
        "Search for 'latest AI news' exactly 6 times and compile all results. "
        "Each search must use the exact query 'latest AI news'."
    ),
}


def run(scenario: str = "normal") -> None:
    query = SCENARIOS.get(scenario, SCENARIOS["normal"])
    print(f"\nScenario: {scenario}")
    print(f"Query: {query}\n")
    try:
        result = agent.invoke(
            {"messages": [("human", query)]},
            config={"callbacks": [callback]},
        )
        output = result["messages"][-1].content
        print(f"\nAnswer: {output}")
    except Exception as e:
        print(f"Error: {e}")
    dt.shutdown(timeout=5)
    print("Events flushed.")


if __name__ == "__main__":
    run(os.environ.get("SCENARIO", "normal"))
