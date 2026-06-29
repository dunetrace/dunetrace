"""
OpenAI Agents SDK example. One trace processor, nothing else changes.

The Agents SDK's built-in tracing fires for every run, LLM generation, and
function-tool call, so registering a single Dunetrace processor instruments the
whole agent with no changes to the agent definition.

Install:
    pip install 'dunetrace[openai-agents]'
    pip install python-dotenv  # optional, only for loading a .env file

Run:
    OPENAI_API_KEY=sk-... python examples/openai_agents_agent.py
    OPENAI_API_KEY=sk-... SCENARIO=tool_loop python examples/openai_agents_agent.py
"""

from __future__ import annotations

import ast
import operator as op
import os
import time
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / ".env")
except ImportError:
    # python-dotenv is optional; fall back to the ambient environment.
    pass

from agents import Agent, Runner, function_tool

from dunetrace import Dunetrace
from dunetrace.integrations.openai_agents import add_dunetrace_processor

dt = Dunetrace(endpoint=os.environ.get("DUNETRACE_ENDPOINT", "http://localhost:8001"))

SYSTEM_PROMPT = (
    "You are a research assistant. "
    "Use the search tool to find information before answering. "
    "Always search at least once before giving a final answer."
)


@function_tool
def web_search(query: str) -> str:
    """Search the web for information on a topic."""
    time.sleep(0.3)
    return f"Search results for '{query}': Found relevant information about {query}."


_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
}


def _safe_eval_arithmetic(expression: str) -> float:
    """Evaluate a basic arithmetic expression without calling eval()."""
    node = ast.parse(expression, mode="eval").body

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -_eval(n.operand)
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN_OPS:
            return _BIN_OPS[type(n.op)](_eval(n.left), _eval(n.right))
        raise ValueError("unsupported expression")

    return _eval(node)


@function_tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression."""
    allowed = set("0123456789+-*/()., ")
    if not all(c in allowed for c in expression):
        return f"Cannot evaluate '{expression}' — only basic arithmetic supported."
    try:
        return f"{expression} = {_safe_eval_arithmetic(expression)}"
    except (SyntaxError, ValueError, ZeroDivisionError):
        return f"Cannot evaluate '{expression}' — only basic arithmetic supported."


agent = Agent(
    name="openai-agents-example",
    instructions=SYSTEM_PROMPT,
    model="gpt-4o-mini",
    tools=[web_search, calculator],
)

# Register the Dunetrace processor once at startup. It composes with the SDK's
# default exporter, so existing tracing keeps working.
add_dunetrace_processor(
    dt,
    agent_id="openai-agents-example",
    system_prompt=SYSTEM_PROMPT,
    model="gpt-4o-mini",
    tools=["web_search", "calculator"],
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
        result = Runner.run_sync(agent, query)
        print(f"\nAnswer: {result.final_output}")
    except Exception as e:
        print(f"Error: {e}")
    dt.shutdown(timeout=5)
    print("Events flushed.")


if __name__ == "__main__":
    run(os.environ.get("SCENARIO", "normal"))
