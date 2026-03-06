"""
examples/langchain_agent.py

LangChain agent instrumented with Dunetrace. One callback, zero agent changes.

Install:
    pip install 'dunetrace[langchain]' langchain-openai langchain-community

Run:
    OPENAI_API_KEY=sk-... python examples/langchain_agent.py
    OPENAI_API_KEY=sk-... SCENARIO=tool_loop python examples/langchain_agent.py
"""
from __future__ import annotations

import os

from dunetrace import DunetraceClient
from dunetrace.integrations.langchain import DunetraceCallbackHandler

dt = DunetraceClient(
    api_key=os.environ.get("DUNETRACE_API_KEY", "dt_dev_local"),
    endpoint=os.environ.get("DUNETRACE_ENDPOINT", "http://localhost:8001"),
)

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub

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

from langchain_community.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()

tools = [
    Tool(name="web_search", func=search.run,
         description="Search the web. Input: a search query string."),
    Tool(name="calculator",
         func=lambda x: str(eval(x)),  # noqa: S307 — demo only, do not use eval in production
         description="Evaluate a math expression. Input: a valid Python expression."),
]
tool_names = [t.name for t in tools]

callback = DunetraceCallbackHandler(
    dt,
    agent_id="langchain-example-agent",
    system_prompt=SYSTEM_PROMPT,
    model="gpt-4o-mini",
    tools=tool_names,
)

prompt = hub.pull("hwchase17/react")
agent  = create_react_agent(llm, tools, prompt)
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
        result = agent_executor.invoke({"input": query}, config={"callbacks": [callback]})
        print(f"\nAnswer: {result['output']}")
    except Exception as e:
        print(f"Error: {e}")
    dt.shutdown(timeout=5)
    print("Events flushed.")


if __name__ == "__main__":
    run(os.environ.get("SCENARIO", "normal"))
