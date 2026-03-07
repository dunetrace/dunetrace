"""
examples/langchain_agent.py

LangChain agent instrumented with Dunetrace. One callback, zero agent changes.

Install:
    pip install 'dunetrace[langchain]' langchain-openai

Run:
    OPENAI_API_KEY=sk-... python examples/langchain_agent.py
    OPENAI_API_KEY=sk-... SCENARIO=tool_loop python examples/langchain_agent.py
"""
from __future__ import annotations

import os
import time

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from dunetrace import Dunetrace
from dunetrace.integrations.langchain import DunetraceCallbackHandler

dt = Dunetrace(endpoint=os.environ.get("DUNETRACE_ENDPOINT", "http://localhost:8001"))

SYSTEM_PROMPT = (
    "You are a research assistant. "
    "Use the search tool to find information before answering. "
    "Always search at least once before giving a final answer."
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.environ["OPENAI_API_KEY"],
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

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=[callback],
    verbose=True,
    max_iterations=12,
    handle_parsing_errors=True,
)

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
        result = agent_executor.invoke({"input": query})
        print(f"\nAnswer: {result['output']}")
    except Exception as e:
        print(f"Error: {e}")
    dt.shutdown(timeout=5)
    print("Events flushed.")


if __name__ == "__main__":
    run(os.environ.get("SCENARIO", "normal"))
