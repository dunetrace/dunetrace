"""
AutoGen agent example: Dunetrace monitoring for a multi-agent conversation.

Shows an AssistantAgent + RoundRobinGroupChat monitored via DunetraceAutoGenObserver.
Every LLM call is tracked by wrapping the model client with DunetraceModelClient.

Install:
    pip install 'dunetrace' autogen-agentchat autogen-ext python-dotenv

Run (happy path):
    OPENAI_API_KEY=sk-... python examples/autogen_agent.py

Run (tool loop — triggers TOOL_LOOP signal):
    SCENARIO=tool_loop OPENAI_API_KEY=sk-... python examples/autogen_agent.py

DUNETRACE_ENDPOINT defaults to http://localhost:8001.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / ".env")

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool

from dunetrace import Dunetrace
from dunetrace.integrations.autogen import DunetraceAutoGenObserver

# ── Config ─────────────────────────────────────────────────────────────────────

AGENT_ID = "autogen-example-agent"
MODEL = "gpt-4o-mini"
TOOLS_LIST = ["web_search"]

SCENARIOS = {
    "normal": "What is the capital of France? Answer in one sentence.",
    "tool_loop": (
        "Use web_search exactly 6 times with the exact query 'latest AI news'. "
        "After all 6 searches, compile the results into a brief report."
    ),
}

# ── Simulated tool ─────────────────────────────────────────────────────────────


def web_search(query: str) -> str:
    """Search the web for information on a topic."""
    return f"Search results for '{query}': simulated AI news content."


# ── Setup ──────────────────────────────────────────────────────────────────────

dt = Dunetrace(endpoint=os.environ.get("DUNETRACE_ENDPOINT", "http://localhost:8001"))
dt.mark_deploy(AGENT_ID, version=os.environ.get("APP_VERSION", "dev"))

scenario = os.environ.get("SCENARIO", "normal")
query = SCENARIOS.get(scenario, SCENARIOS["normal"])

print("=" * 60)
print(f"Dunetrace + AutoGen example  [scenario={scenario}]")
print("=" * 60)
print(f"Query: {query}\n")

observer = DunetraceAutoGenObserver(
    dt,
    agent_id=AGENT_ID,
    model=MODEL,
    tools=TOOLS_LIST,
)

# ── Agent setup ────────────────────────────────────────────────────────────────


async def main() -> None:
    base_client = OpenAIChatCompletionClient(model=MODEL)
    dt_client = observer.wrap_client(base_client)  # instruments LLM calls

    search_tool = FunctionTool(
        web_search, description="Search the web for information."
    )

    assistant = AssistantAgent(
        name="assistant",
        model_client=dt_client,
        tools=[search_tool],
        system_message=(
            "You are a helpful AI assistant. "
            "Use the web_search tool when asked to search. "
            "Terminate with 'TERMINATE' when done."
        ),
    )

    team = RoundRobinGroupChat(
        [assistant],
        termination_condition=MaxMessageTermination(10),
    )

    # ── Run inside observer context ────────────────────────────────────────────
    async with observer.run(user_input=query):
        result = await team.run(task=query)

    final_msg = result.messages[-1].content if result.messages else ""
    print("\n" + "=" * 60)
    print("Final reply:")
    print("=" * 60)
    print(final_msg)

    await base_client.close()


asyncio.run(main())

dt.shutdown(timeout=5)
print("\nDone. Check:")
print("  Dashboard: http://localhost:3000")
print(f"  Agent: {AGENT_ID}")
