"""
CrewAI agent example: Dunetrace monitoring for a multi-agent crew (CrewAI 1.x).

Shows a research crew with two agents (Researcher + Writer) monitored via
DunetraceCrewCallback. Hooks into CrewAI 1.x's global LLM/tool hook system —
no changes to agent or crew definitions needed.

Install:
    pip install 'dunetrace' crewai python-dotenv

Run (happy path):
    OPENAI_API_KEY=sk-... python examples/crewai_agent.py

Run (tool loop — triggers TOOL_LOOP signal):
    SCENARIO=tool_loop OPENAI_API_KEY=sk-... python examples/crewai_agent.py
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / ".env")

from crewai import Agent, Crew, Task, Process
from crewai.tools import tool

from dunetrace import Dunetrace
from dunetrace.integrations.crewai import DunetraceCrewCallback

# ── Config ─────────────────────────────────────────────────────────────────────

AGENT_ID   = "crewai-example-crew"
MODEL      = "gpt-4o-mini"
TOOLS_LIST = ["web_search"]

SCENARIOS = {
    "normal":    "Search for information about the Eiffel Tower and write a one-paragraph summary.",
    "tool_loop": (
        "Search for 'latest AI news' exactly 6 times in a row, each time using "
        "the exact query 'latest AI news'. After all searches, compile the results."
    ),
}

# ── Simulated tool ─────────────────────────────────────────────────────────────

@tool("web_search")
def web_search(query: str) -> str:
    """Search the web for information on a topic. Use this for any research task."""
    return f"Search results for '{query}': simulated result about {query}."

# ── Setup ──────────────────────────────────────────────────────────────────────

dt = Dunetrace(endpoint=os.environ.get("DUNETRACE_ENDPOINT", "http://localhost:8001"))
dt.mark_deploy(AGENT_ID, version=os.environ.get("APP_VERSION", "dev"))

scenario = os.environ.get("SCENARIO", "normal")
query    = SCENARIOS.get(scenario, SCENARIOS["normal"])

print("=" * 60)
print(f"Dunetrace + CrewAI example  [scenario={scenario}]")
print("=" * 60)
print(f"Query: {query}\n")

# Install global LLM/tool hooks once before the crew runs
cb = DunetraceCrewCallback(dt, agent_id=AGENT_ID, model=MODEL, tools=TOOLS_LIST)
cb.install()

# CrewAI 1.x: pass model as a plain string; it routes via LiteLLM internally
researcher = Agent(
    role="Senior Research Analyst",
    goal="Research topics thoroughly using the web_search tool.",
    backstory=(
        "You are an expert researcher who always uses the web_search tool "
        "to find information before writing anything."
    ),
    llm=MODEL,
    tools=[web_search],
    verbose=True,
    max_iter=8,
)

writer = Agent(
    role="Content Writer",
    goal="Write clear, concise summaries based on research findings.",
    backstory="You are a skilled writer who synthesizes research into readable content.",
    llm=MODEL,
    verbose=True,
)

research_task = Task(
    description=f"Use the web_search tool to research: {query}",
    expected_output="A research report with key findings from the search results.",
    agent=researcher,
)

write_task = Task(
    description="Write a concise one-paragraph summary of the research report.",
    expected_output="A well-structured summary paragraph of 80-120 words.",
    agent=writer,
    context=[research_task],
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True,
)

# ── Run inside dt.run() so hooks emit to this context ─────────────────────────

with dt.run(AGENT_ID, user_input=query, model=MODEL, tools=TOOLS_LIST) as run:
    result = crew.kickoff()
    run.final_answer()

cb.uninstall()

print("\n" + "=" * 60)
print("Final output:")
print("=" * 60)
print(result)

dt.shutdown(timeout=5)
print("\nDone. Check:")
print("  Dashboard: http://localhost:3000")
print(f"  Agent: {AGENT_ID}")
