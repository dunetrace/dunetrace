"""
agent.py
========
A LangChain research agent instrumented with Dunetrace.

Setup:
    pip install 'dunetrace[langchain]' langchain langchain-openai

    # Start the backend first:
    docker compose up -d

    export OPENAI_API_KEY=sk-...

Run:
    python agent.py
    python agent.py --task "Compare Python and Go for backend services"
    python agent.py --task "What are the best practices for RAG pipelines?"
"""

import os
import sys
import time
import argparse
import random
from datetime import datetime

# ── Check deps ────────────────────────────────────────────────────────────────
try:
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain_openai import ChatOpenAI
    from langchain.tools import tool
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
except ImportError:
    print("\nMissing dependencies. Run:\n")
    print("  pip install 'dunetrace[langchain]' langchain langchain-openai\n")
    sys.exit(1)

from dunetrace import Dunetrace
from dunetrace.integrations.langchain import DunetraceCallbackHandler

dt = Dunetrace()

if not os.getenv("OPENAI_API_KEY"):
    print("\nError: OPENAI_API_KEY not set\n")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS
# Realistic tools with simulated latency and real-ish responses.
# Swap these out for actual API calls when you want real data.
# ══════════════════════════════════════════════════════════════════════════════

# Fake knowledge base — enough variety to make the agent actually think
KNOWLEDGE_BASE = {
    "python": """
        Python is a high-level, interpreted language known for readability and rapid development.
        Strengths: vast ecosystem (pip), ML/data science libraries (numpy, pandas, pytorch),
        scripting, automation. Weaknesses: GIL limits true thread parallelism, slower than
        compiled languages. Popular frameworks: FastAPI, Django, Flask. Used heavily at
        Instagram, Dropbox, Netflix.
    """,
    "go": """
        Go (Golang) is a statically typed, compiled language by Google (2009). Strengths:
        excellent concurrency via goroutines, fast compilation, single binary deploys,
        low memory footprint. Weaknesses: verbose error handling, no generics until 1.18,
        smaller ML ecosystem. Popular for: CLI tools, microservices, infrastructure (Docker,
        Kubernetes written in Go). Used at Uber, Cloudflare, Dropbox.
    """,
    "rag": """
        Retrieval-Augmented Generation (RAG) combines a vector database with an LLM.
        Pipeline: chunk documents → embed → store in vector DB → at query time, embed query
        → retrieve top-k chunks → pass to LLM with context. Best practices: chunk size 256-512
        tokens with 10-20% overlap, use hybrid search (dense + BM25), rerank retrieved chunks
        before passing to LLM (Cohere Rerank, cross-encoders), evaluate with RAGAS metrics.
        Common pitfalls: chunks too large (lose precision), no reranking (irrelevant context
        degrades answers), no eval pipeline.
    """,
    "langchain": """
        LangChain is a framework for building LLM-powered applications. Key abstractions:
        chains, agents, memory, tools, callbacks. AgentExecutor runs a loop: LLM decides
        which tool to call → tool runs → result fed back to LLM → repeat until final answer.
        LangChain Expression Language (LCEL) is the newer composition model. Alternatives:
        LlamaIndex (better for document Q&A), LangGraph (better for stateful multi-agent),
        bare API calls (when you don't need the framework overhead).
    """,
    "vector_database": """
        Vector databases store and search embeddings efficiently. Options: Pinecone (managed,
        fast), Weaviate (open source, hybrid search built in), Qdrant (open source, written
        in Rust, very fast), Chroma (easiest to self-host, great for dev), pgvector (Postgres
        extension, good for existing Postgres users). Evaluation criteria: query latency at
        scale, filtering support, hybrid search, managed vs self-hosted, cost.
    """,
    "ai_agents": """
        AI agents are LLM-powered systems that use tools to complete multi-step tasks.
        Key patterns: ReAct (reason then act), Plan-and-Execute (plan all steps upfront then
        execute), multi-agent (multiple specialized agents collaborate). Common failure modes:
        tool loops (same tool called repeatedly), context bloat (prompt grows unbounded),
        goal abandonment (agent stops using tools and hallucinates), slow steps (tool
        timeouts with no circuit breaker). Observability is critical for production agents.
    """,
    "microservices": """
        Microservices architecture splits an application into small, independently deployable
        services. Benefits: independent scaling, technology diversity, fault isolation.
        Challenges: network latency, distributed tracing complexity, service discovery,
        eventual consistency. Best suited for: large teams (Conway's Law), services with
        very different scaling needs. Not recommended for: small teams, early-stage products
        where the domain model is still evolving.
    """,
    "observability": """
        Observability in distributed systems = metrics + logs + traces (the three pillars).
        Tools: Datadog (full stack, expensive), Grafana+Prometheus (open source metrics),
        Jaeger/Zipkin (distributed tracing), ELK stack (logging). For AI agents specifically,
        standard APM tools miss behavioral failures — they monitor the server layer but not
        whether the agent is making progress. Agent-specific observability needs to track:
        tool call patterns, context growth, step latency, completion signals.
    """,
}


@tool
def web_search(query: str) -> str:
    """
    Search the web for current information on a topic.
    Use for: recent events, current best practices, comparisons, factual questions.
    Returns a summary of search results.
    """
    time.sleep(random.uniform(0.3, 1.2))  # realistic network latency

    # Find relevant knowledge base entries
    query_lower = query.lower()
    results = []

    for key, content in KNOWLEDGE_BASE.items():
        if any(word in query_lower for word in key.split("_")) or key in query_lower:
            results.append(content.strip())

    if results:
        return f"Search results for '{query}':\n\n" + "\n\n---\n\n".join(results)

    # Generic fallback so the agent can still make progress
    return (
        f"Search results for '{query}':\n\n"
        f"Found general information about this topic. "
        f"The query relates to software engineering and AI/ML systems. "
        f"Key considerations include performance, scalability, developer experience, "
        f"and ecosystem maturity. Best practices vary by use case and team size."
    )


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression or do unit conversions.
    Examples: '2 + 2', '1000 / 8', '(50 * 1.2) - 15'
    """
    time.sleep(0.05)
    try:
        # Safe eval — only allow basic math
        allowed = set("0123456789+-*/()., ")
        if all(c in allowed for c in expression):
            result = eval(expression)  # noqa: S307
            return f"{expression} = {result}"
        return f"Cannot evaluate '{expression}' — only basic arithmetic supported."
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def get_date() -> str:
    """
    Get the current date and time.
    Use when the task requires knowing today's date.
    """
    time.sleep(0.02)
    return f"Current date and time: {datetime.now().strftime('%A, %B %d, %Y at %H:%M:%S')}"


@tool
def summarise_text(text: str) -> str:
    """
    Summarise a long piece of text into 3-5 bullet points.
    Use when you have retrieved content that needs condensing before including in your answer.
    """
    time.sleep(random.uniform(0.1, 0.4))
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 20]
    bullets = sentences[:5]
    if not bullets:
        return "Could not extract key points from the provided text."
    return "Key points:\n" + "\n".join(f"• {s}." for s in bullets)


@tool
def compare(option_a: str, option_b: str) -> str:
    """
    Produce a structured comparison between two technologies, approaches, or concepts.
    Use when the task explicitly asks to compare two things.
    """
    time.sleep(random.uniform(0.2, 0.6))

    # Try to look both up
    a_info = KNOWLEDGE_BASE.get(option_a.lower().replace(" ", "_"), "")
    b_info = KNOWLEDGE_BASE.get(option_b.lower().replace(" ", "_"), "")

    if a_info and b_info:
        return (
            f"Comparison: {option_a} vs {option_b}\n\n"
            f"{option_a}:\n{a_info.strip()}\n\n"
            f"{option_b}:\n{b_info.strip()}"
        )

    return (
        f"Comparison: {option_a} vs {option_b}\n\n"
        f"Both are established options with different trade-offs. "
        f"Choice depends on: team familiarity, existing infrastructure, "
        f"performance requirements, and ecosystem needs. "
        f"Recommend evaluating both with a small proof-of-concept."
    )


# ══════════════════════════════════════════════════════════════════════════════
# AGENT SETUP
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a thorough technical research assistant.

Your job is to answer technical questions accurately and completely using your tools.

Rules:
- Always search for information before answering — don't rely on memory alone.
- For comparisons, use the compare tool.
- Summarise long retrieved content before including it in your answer.
- If a question involves multiple topics, search for each one separately.
- Provide a clear, structured final answer with your reasoning.
- Never make up specific numbers, dates, or facts — use the tools.
"""

def build_agent(model: str = "gpt-4o-mini", temperature: float = 0.0, verbose: bool = True):
    """Build and return the AgentExecutor."""
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        streaming=False,
    )

    tools = [web_search, calculate, get_date, compare, summarise_text]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=[DunetraceCallbackHandler(dt, agent_id="research-agent")],
        verbose=verbose,
        max_iterations=10,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

DEMO_TASKS = [
    "What are the best practices for building a RAG pipeline? Include common pitfalls.",
    "Compare Python and Go for building microservices. Which should I choose?",
    "What is LangChain and what are its alternatives? When would I use each?",
    "Explain AI agent failure modes and how to observe them in production.",
    "What vector database should I use for a production RAG system?",
]


def run(task: str, verbose: bool = True):
    print("\n" + "═" * 70)
    print(f"  TASK: {task}")
    print("═" * 70 + "\n")

    agent = build_agent(verbose=verbose)

    start = time.time()
    try:
        result = agent.invoke({"input": task, "chat_history": []})
        elapsed = time.time() - start

        print("\n" + "─" * 70)
        print("  FINAL ANSWER")
        print("─" * 70)
        print(result["output"])
        print("\n" + "─" * 70)
        print(f"  Completed in {elapsed:.1f}s | "
              f"{len(result.get('intermediate_steps', []))} tool calls")
        print("─" * 70 + "\n")

        return result

    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  ERROR after {elapsed:.1f}s: {e}\n")
        raise


def main():
    parser = argparse.ArgumentParser(description="Dunetrace test agent")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task to run. If omitted, runs all demo tasks.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide LangChain's verbose step-by-step output",
    )
    args = parser.parse_args()

    verbose = not args.quiet

    if args.task:
        run(args.task, verbose=verbose)
    else:
        print(f"\nRunning {len(DEMO_TASKS)} demo tasks...\n")
        for i, task in enumerate(DEMO_TASKS, 1):
            print(f"\n[{i}/{len(DEMO_TASKS)}]", end="")
            run(task, verbose=verbose)
            if i < len(DEMO_TASKS):
                time.sleep(1)  # small pause between runs

        print("\nAll tasks complete.\n")

    dt.shutdown(timeout=5)


if __name__ == "__main__":
    main()
