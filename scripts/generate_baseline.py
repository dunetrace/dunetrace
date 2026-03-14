#!/usr/bin/env python3
"""
Generates 100 GPT-4o-mini agent runs for detector calibration.
Uses three tools that produce organic failures: web_search (pagination forces loops),
calculator (AST eval errors trigger retries), and doc_lookup (long outputs bloat context).

Task mix: 30 research, 20 math, 15 factual, 20 multi_tool, 15 adversarial.

Usage:
    docker compose up -d --build
    python scripts/generate_baseline.py

    # Smaller run for quick validation:
    MAX_RUNS=20 python scripts/generate_baseline.py
"""
from __future__ import annotations

import ast
import json
import operator as op_module
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

# ── Env ────────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / "packages" / "sdk-py"))

from dunetrace import Dunetrace
from dunetrace.integrations.langchain import DunetraceCallbackHandler

# ── Config ─────────────────────────────────────────────────────────────────────
INGEST_URL      = os.environ.get("INGEST_URL",    "http://localhost:8001")
API_URL         = os.environ.get("API_URL",        "http://localhost:8002")
AGENT_ID        = os.environ.get("BASELINE_AGENT_ID", "baseline-v1")
MAX_RUNS        = int(os.environ.get("MAX_RUNS",   "100"))
BATCH_SIZE      = int(os.environ.get("BATCH_SIZE", "10"))
BATCH_DELAY     = float(os.environ.get("BATCH_DELAY", "8"))   # s between batches
RUN_DELAY       = float(os.environ.get("RUN_DELAY",   "1.5")) # s between runs

OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY not set. Add it to .env or export it.")
    sys.exit(1)

SYSTEM_PROMPT = (
    "You are a helpful research assistant with access to web_search, calculator, "
    "and doc_lookup tools. Use tools when you need information you don't already know. "
    "For web searches that return paginated results, search again with 'page=N' appended "
    "to your query to retrieve subsequent pages until results are complete. "
    "Show step-by-step reasoning before each action."
)


# ── Tool 1: web_search ─────────────────────────────────────────────────────────
# Deep/recent queries get paginated responses → agent naturally loops.
_search_call_counts: dict[str, int] = {}

_DEEP_KEYWORDS = {
    "2024", "2025", "latest", "recent", "advance", "breakthrough",
    "trend", "statistic", "report", "study", "research", "survey",
    "annual", "comprehensive", "complete", "all",
}

_SNIPPETS = {
    "quantum":   "Quantum processors exceeded 1000 qubits. Error correction milestones reached.",
    "climate":   "Global temps rose 1.1 °C above pre-industrial. CO₂ at 421 ppm. Arctic ice at record low.",
    "ai":        "LLM benchmarks pushed further. Multimodal models matched human on key tasks.",
    "renewable": "Solar capacity grew 40% YoY. Wind reached 2 TW globally.",
    "vaccine":   "mRNA platforms extended to RSV and cancer. Three FDA approvals.",
    "battery":   "Solid-state EV batteries hit 400 Wh/kg in lab. First commercial lines announced.",
    "cyber":     "Ransomware attacks up 35%. Critical infrastructure incidents doubled.",
}


def web_search(query: str) -> str:
    """Search the web. For paginated results append 'page=N' to your query string."""
    page_m = re.search(r"\bpage[=:]?\s*(\d+)", query, re.IGNORECASE)
    page   = int(page_m.group(1)) if page_m else 1
    clean  = re.sub(r"\bpage[=:]?\s*\d+", "", query, flags=re.IGNORECASE).strip().strip("'\"")
    key    = clean.lower()

    _search_call_counts[key] = _search_call_counts.get(key, 0) + 1
    call_n = _search_call_counts[key]

    is_deep = any(kw in key for kw in _DEEP_KEYWORDS)

    if is_deep and page < 4 and call_n <= 7:
        total = random.choice([4, 5, 6])
        snippet = next((v for k, v in _SNIPPETS.items() if k in key), None)
        body = (
            f"Page {page} results: {snippet} " if snippet
            else f"Page {page}: {random.randint(15, 45)} relevant sources found. Key data summarised. "
        )
        return (
            f"[Page {page}/{total} — '{clean}']\n"
            f"{body}\n"
            f"Results incomplete. Search with 'page={page + 1}' to retrieve the next page."
        )

    snippet = next((v for k, v in _SNIPPETS.items() if k in key), None)
    return (
        f"Complete results for '{clean}': "
        + (snippet if snippet else
           "Multiple peer-reviewed sources confirm the trend. "
           "Statistical significance p<0.01. Full citations available on request.")
    )


# ── Tool 2: calculator ─────────────────────────────────────────────────────────
_SAFE_OPS: dict[type, Any] = {
    ast.Add:      op_module.add,
    ast.Sub:      op_module.sub,
    ast.Mult:     op_module.mul,
    ast.Div:      op_module.truediv,
    ast.Pow:      op_module.pow,
    ast.USub:     op_module.neg,
    ast.Mod:      op_module.mod,
    ast.FloorDiv: op_module.floordiv,
}


def _safe_eval(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported literal: {node.value!r}")
    if isinstance(node, ast.BinOp):
        fn = _SAFE_OPS.get(type(node.op))
        if not fn:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("Division by zero")
        return fn(left, right)
    if isinstance(node, ast.UnaryOp):
        fn = _SAFE_OPS.get(type(node.op))
        if not fn:
            raise ValueError(f"Unsupported unary: {type(node.op).__name__}")
        return fn(_safe_eval(node.operand))
    raise ValueError(f"Unsupported node: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression. Input must be a valid expression string."""
    expr = expression.strip().rstrip(".")
    try:
        tree   = ast.parse(expr, mode="eval")
        result = _safe_eval(tree.body)
        if isinstance(result, float) and result == int(result):
            result = int(result)
        # Round floats to 6 dp
        if isinstance(result, float):
            result = round(result, 6)
        return str(result)
    except ZeroDivisionError:
        raise ValueError("Cannot divide by zero")
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid expression '{expr}': {exc}") from exc


# ── Tool 3: doc_lookup ─────────────────────────────────────────────────────────
# Known topics return long text (triggers CONTEXT_BLOAT for multi-step runs).
_DOCS: dict[str, str] = {
    "machine learning": (
        "Machine Learning Technical Reference\n\n"
        "ML is a subfield of AI enabling systems to learn from data without explicit programming. "
        "Supervised learning trains on labelled pairs (X, y); algorithms: linear/logistic regression, "
        "decision trees, random forests, gradient boosting (XGBoost, LightGBM), SVMs, k-NN, neural nets. "
        "Unsupervised learning discovers structure: k-means, DBSCAN, GMMs, PCA, t-SNE, autoencoders. "
        "Reinforcement learning uses reward signals: Q-learning, PPO, SAC, DDPG. "
        "Deep learning uses multi-layer nets: CNNs (image), RNNs/LSTMs (sequence), "
        "Transformers (attention, 2017, 'Attention Is All You Need'). "
        "Training: SGD, Adam, AdamW optimisers; batch norm, dropout, residual connections. "
        "Evaluation: accuracy, precision, recall, F1, AUC-ROC, cross-entropy, MSE, MAE. "
        "Frameworks: PyTorch, TensorFlow, JAX, scikit-learn, Keras. "
        "Challenges: overfitting, data quality, interpretability (SHAP, LIME), fairness, privacy (federated, DP). "
        "Recent: LLMs (GPT-4, Claude, Gemini), diffusion models, RLHF, LoRA fine-tuning, RAG. "
        "History: Perceptron 1958 → expert systems → connectionism → SVM 1995 → deep learning 2012 (AlexNet)."
    ),
    "climate change": (
        "Climate Science Summary\n\n"
        "Global mean surface temperature: +1.1 °C vs pre-industrial (2023). "
        "CO₂: 421 ppm (2023) vs 280 ppm (pre-industrial) — 50% increase. "
        "Sea level rise: ~3.7 mm/year (satellite era). Total rise since 1900: ~20 cm. "
        "Arctic sea ice: −13% per decade (minimum extent, 1979–present). "
        "Ocean heat content: >90% of excess energy stored in oceans. "
        "IPCC AR6 (2021-22): 'unequivocal human influence'. Scenarios (SSP1-2.6 to SSP5-8.5). "
        "Tipping elements: WAIS collapse, Amazon dieback, permafrost carbon, AMOC weakening. "
        "Paris Agreement: limit to 1.5–2 °C; NDCs currently on track for ~2.5 °C. "
        "Mitigation: renewables scale-up, electrification, carbon capture (DAC, BECCS), methane cuts, "
        "land-use change (afforestation). "
        "Adaptation: sea walls, heat action plans, drought-resistant crops, early warning systems. "
        "Carbon budget to 1.5 °C (2023): ~250 GtCO₂ remaining at current ~40 GtCO₂/year."
    ),
    "python": (
        "Python Language Reference\n\n"
        "Created 1991 by Guido van Rossum. Python 3.0: 2008. Latest stable: 3.12 (2023). "
        "Philosophy: readability, explicitness, simplicity (Zen of Python, PEP 20). "
        "Dynamic typing, GC, CPython reference impl; also PyPy (JIT), Jython, MicroPython. "
        "Types: int, float, complex, str, bytes, list, tuple, dict, set, bool, None. "
        "Control: if/elif/else, for, while, try/except/finally/else, with (context managers). "
        "Functions: def, lambda, *args/**kwargs, decorators, generators (yield), async/await (asyncio). "
        "OOP: class, inheritance, MRO (C3 linearisation), dunders (__init__, __repr__, __len__, …). "
        "Stdlib: os, pathlib, sys, json, re, datetime, collections, itertools, functools, "
        "threading, multiprocessing, asyncio, http, urllib, csv, sqlite3, logging, unittest. "
        "Ecosystem: pip, PyPI (480k+ packages), venv, conda. "
        "Frameworks: Django, Flask, FastAPI; NumPy, pandas, matplotlib; PyTorch, TensorFlow; pytest. "
        "Python 3.12: 5× faster tracebacks, per-subinterpreter GIL, improved f-strings. "
        "Adoption: #1 TIOBE, #1 Stack Overflow survey 2023. ~8 M developers worldwide."
    ),
    "quantum computing": (
        "Quantum Computing Overview\n\n"
        "Quantum computers use qubits (superposition, entanglement, interference) vs classical bits. "
        "Key algorithms: Shor's (factoring, exp speedup), Grover's (search, quadratic speedup), "
        "VQE/QAOA (optimisation), HHL (linear systems). "
        "Hardware: superconducting (IBM, Google), trapped ions (IonQ, Quantinuum), photonic, neutral atoms. "
        "IBM: 1121-qubit Condor (2023), 1386-qubit Heron (2024). Google: Sycamore quantum advantage 2019. "
        "Error rates: physical error ~0.1–1%; fault tolerance requires ~1000 physical qubits per logical qubit. "
        "NISQ era: noisy intermediate-scale quantum — 50–1000 qubits, no full error correction yet. "
        "Quantum volume (IBM metric): measures effective performance; exceeded 2^20 in 2024. "
        "Applications: cryptography (threatens RSA), drug discovery, logistics, finance, ML. "
        "Post-quantum cryptography: NIST standardised CRYSTALS-Kyber, CRYSTALS-Dilithium (2024). "
        "Timeline: ~10 years to fault-tolerant advantage on practical problems (consensus estimate)."
    ),
}


def doc_lookup(topic: str) -> str:
    """Look up detailed technical documentation on a topic."""
    t = topic.lower().strip()
    for key, content in _DOCS.items():
        if key in t or any(w in t for w in key.split()):
            return content
    return (
        f"Documentation for '{topic}':\n"
        f"Foundational work established 1970–2010. Active open-source development continues. "
        f"Key references: relevant IEEE/ACM proceedings and standard textbooks. "
        f"No detailed internal documentation available for this specific topic."
    )


# ── Tasks ──────────────────────────────────────────────────────────────────────
TASKS: list[dict] = [
    # ── Research (30) — paginated search, loop-prone ───────────────────────────
    {"cat": "research", "input": "What are the latest breakthroughs in quantum computing in 2024? Search thoroughly — retrieve all pages."},
    {"cat": "research", "input": "Find comprehensive 2024 statistics on global renewable energy adoption across all major regions."},
    {"cat": "research", "input": "Research the complete history and recent advances in mRNA vaccine technology through 2024."},
    {"cat": "research", "input": "What are the most recent AI research breakthroughs from 2024? Compile a complete list."},
    {"cat": "research", "input": "Find all recent studies on microplastics in the human body — search multiple pages to ensure completeness."},
    {"cat": "research", "input": "Comprehensive research on electric vehicle battery technology trends in 2024."},
    {"cat": "research", "input": "What does recent research say about the impact of social media on mental health? Compile all findings."},
    {"cat": "research", "input": "Find the latest research on CRISPR gene editing applications in 2024."},
    {"cat": "research", "input": "What are the comprehensive 2024 reports on global ocean temperature changes?"},
    {"cat": "research", "input": "Research all recent advances in fusion energy research through 2024."},
    {"cat": "research", "input": "Find complete statistics on global deforestation rates — retrieve all pages."},
    {"cat": "research", "input": "What are the latest 2024 findings on Alzheimer's disease treatment research?"},
    {"cat": "research", "input": "Comprehensive survey of large language model benchmarks and performance in 2024."},
    {"cat": "research", "input": "Find all recent reports on antibiotic resistance trends globally through 2024."},
    {"cat": "research", "input": "Research the complete landscape of quantum cryptography developments in 2024."},
    {"cat": "research", "input": "What are the most recent comprehensive studies on urban heat islands and mitigation?"},
    {"cat": "research", "input": "Find all 2024 annual reports on global food security and hunger statistics."},
    {"cat": "research", "input": "Research the complete picture of autonomous vehicle safety statistics through 2024."},
    {"cat": "research", "input": "Comprehensive review of 2024 breakthroughs in materials science and nanotechnology."},
    {"cat": "research", "input": "Find all recent research on the gut microbiome's role in mental health — complete findings only."},
    {"cat": "research", "input": "What are the latest comprehensive studies on dark matter detection experiments?"},
    {"cat": "research", "input": "Research all recent 2024 developments in neuromorphic computing advances."},
    {"cat": "research", "input": "Find complete 2024 data on global cybersecurity incidents and trends."},
    {"cat": "research", "input": "Comprehensive research on mRNA therapeutics beyond vaccines in 2024."},
    {"cat": "research", "input": "What are the latest advances in carbon capture technology? Find all recent studies."},
    {"cat": "research", "input": "Research the complete 2024 developments in room-temperature superconductors."},
    {"cat": "research", "input": "Find all recent comprehensive reports on space debris and orbital crowding."},
    {"cat": "research", "input": "What does the complete 2024 research say about psychedelic-assisted therapy efficacy?"},
    {"cat": "research", "input": "Research all recent advances in solid-state battery technology through 2024."},
    {"cat": "research", "input": "Find comprehensive 2024 statistics on global AI investment and startup activity."},

    # ── Math (20) — calculator-heavy ──────────────────────────────────────────
    {"cat": "math", "input": "What is 15% of 847? Use the calculator."},
    {"cat": "math", "input": "Calculate compound interest: $10,000 principal at 5% annual rate for 10 years."},
    {"cat": "math", "input": "A train travels 450 km in 3.5 hours. What is its average speed in km/h?"},
    {"cat": "math", "input": "Calculate the area of a circle with radius 7.5 metres. Use π ≈ 3.14159265."},
    {"cat": "math", "input": "Calculate: (2**10 - 1) / 3"},
    {"cat": "math", "input": "A rectangle has perimeter 56 cm and length 18 cm. Calculate the width and area."},
    {"cat": "math", "input": "Calculate 17 divided by 7 to 4 decimal places."},
    {"cat": "math", "input": "What is the volume of a sphere with diameter 12 cm? Use V = (4/3)πr³."},
    {"cat": "math", "input": "A store offers 30% off, then a further 15% off the discounted price. What is the effective total discount?"},
    {"cat": "math", "input": "Calculate 2 to the power of 32."},
    {"cat": "math", "input": "What is the Euclidean distance between points (3, 4) and (9, 12)?"},
    {"cat": "math", "input": "If a population grows at 2.3% per year, how many years until it doubles? Use ln(2)/ln(1.023)."},
    {"cat": "math", "input": "Calculate the sum of integers from 1 to 100 using the formula n*(n+1)/2."},
    {"cat": "math", "input": "What is 1234 multiplied by 5678?"},
    {"cat": "math", "input": "A car depreciates 15% per year. Calculate its value after 5 years if it cost $35,000."},
    {"cat": "math", "input": "Calculate BMI for someone 1.75 m tall and 80 kg. Formula: weight / height²."},
    {"cat": "math", "input": "What is 456789 divided by 123?"},
    {"cat": "math", "input": "Calculate (100 - 37) * (100 + 37)."},
    {"cat": "math", "input": "If 40% of a number is 68, what is the number?"},
    {"cat": "math", "input": "Calculate the monthly payment for a $300,000 loan at 6.5% annual interest for 30 years. Use M = P*(r*(1+r)^n)/((1+r)^n - 1) where r is monthly rate and n=360."},

    # ── Factual quick (15) — agent knows from training → TOOL_AVOIDANCE ────────
    {"cat": "factual", "input": "What year was the Python programming language first released? Answer directly."},
    {"cat": "factual", "input": "Who wrote the play Hamlet? Reply with Final Answer directly."},
    {"cat": "factual", "input": "What is the capital of Australia? Answer directly without tools."},
    {"cat": "factual", "input": "How many sides does a regular hexagon have? Reply directly."},
    {"cat": "factual", "input": "What is the chemical symbol for gold? Answer directly."},
    {"cat": "factual", "input": "In what year did World War II end? Answer directly."},
    {"cat": "factual", "input": "What is the speed of light in a vacuum in km/s? Reply directly."},
    {"cat": "factual", "input": "What programming language was created by Bjarne Stroustrup? Answer directly."},
    {"cat": "factual", "input": "How many bones are in the adult human body? Reply directly."},
    {"cat": "factual", "input": "What is the largest planet in our solar system? Answer directly."},
    {"cat": "factual", "input": "What does HTTP stand for? Reply directly."},
    {"cat": "factual", "input": "In what year was the World Wide Web invented? Answer directly."},
    {"cat": "factual", "input": "What is the atomic number of carbon? Reply directly."},
    {"cat": "factual", "input": "Who invented the telephone? Answer directly."},
    {"cat": "factual", "input": "What is the square root of 144? Reply directly."},

    # ── Multi-tool (20) — chains search + calc or search + lookup ─────────────
    {"cat": "multi", "input": "Look up machine learning documentation, then calculate how many years ago 2012 was (from 2024)."},
    {"cat": "multi", "input": "Search for the current global EV market size in 2024, then calculate 15% annual growth for 3 years."},
    {"cat": "multi", "input": "Look up climate change data, then calculate the percentage CO₂ increase from 280 ppm to 421 ppm."},
    {"cat": "multi", "input": "Search for Python 3.12 features, then look up Python documentation to give a complete overview."},
    {"cat": "multi", "input": "Find recent AI breakthroughs in 2024, then look up machine learning documentation for context."},
    {"cat": "multi", "input": "Look up climate change documentation, then calculate years until 1 metre sea level rise at 3.7 mm/year."},
    {"cat": "multi", "input": "Search for quantum computing advances in 2024, then look up quantum computing documentation."},
    {"cat": "multi", "input": "Look up Python documentation, then calculate how many years since Python was released in 1991 (from 2024)."},
    {"cat": "multi", "input": "Search for renewable energy capacity statistics 2024, then calculate compound annual growth if capacity doubled over 7 years."},
    {"cat": "multi", "input": "Look up machine learning documentation and search for 2024 AI benchmark results to compare."},
    {"cat": "multi", "input": "Find climate change research, then calculate: if we reach net-zero by 2050, how many years from 2024?"},
    {"cat": "multi", "input": "Look up Python documentation then search for Python developer adoption statistics 2024."},
    {"cat": "multi", "input": "Search for solid-state battery energy density improvements in 2024, then calculate 20% annual growth for 5 years starting at 400 Wh/kg."},
    {"cat": "multi", "input": "Look up climate change data for CO₂ levels, then calculate average annual increase since 1850 (174 years, from 280 to 421 ppm)."},
    {"cat": "multi", "input": "Search for global AI investment in 2024, then calculate what 25% YoY growth gives for 2025."},
    {"cat": "multi", "input": "Look up machine learning documentation, then search for the latest ML hardware accelerators 2024."},
    {"cat": "multi", "input": "Search for mRNA vaccine development timeline, then calculate years from initial mRNA research (1989) to first COVID vaccine approval (2020)."},
    {"cat": "multi", "input": "Look up quantum computing documentation, then calculate how many logical qubits you get from 1000 physical qubits at 1000:1 ratio."},
    {"cat": "multi", "input": "Search for 2024 global temperature anomaly, then calculate average annual warming since 1850 (174 years, 1.1 °C total)."},
    {"cat": "multi", "input": "Look up climate change documentation, then calculate remaining CO₂ budget: 250 GtCO₂ at 40 GtCO₂/year gives how many years?"},

    # ── Adversarial (15) — ambiguous / failure-prone ───────────────────────────
    {"cat": "adversarial", "input": "Search for the single objectively best programming language. Provide a definitive Final Answer."},
    {"cat": "adversarial", "input": "Calculate the exact probability it will rain tomorrow in New York City."},
    {"cat": "adversarial", "input": "Find all AI research published anywhere in 2024 — retrieve every paper."},
    {"cat": "adversarial", "input": "Look up documentation for every programming language and summarise all of them."},
    {"cat": "adversarial", "input": "What is 1 divided by 0? Use the calculator tool to compute this precisely."},
    {"cat": "adversarial", "input": "Search for the single most important event in all of human history and give its complete significance."},
    {"cat": "adversarial", "input": "Use the calculator to compute the meaning of life. Show your mathematical work."},
    {"cat": "adversarial", "input": "Find comprehensive detailed economic data for all 195 countries in 2024."},
    {"cat": "adversarial", "input": "Search recent news AND look up documentation AND calculate statistics AND write a complete quantum computing annual report."},
    {"cat": "adversarial", "input": "What is the current real-time stock price of Apple Inc? Search for live data."},
    {"cat": "adversarial", "input": "Use the calculator to find the largest possible number."},
    {"cat": "adversarial", "input": "Search for a definitive peer-reviewed answer on whether free will exists. Provide a Final Answer with 100% certainty."},
    {"cat": "adversarial", "input": "Find complete medical advice for diagnosing and treating all possible diseases. Be comprehensive."},
    {"cat": "adversarial", "input": "Look up Python, machine learning, and climate change documentation, then synthesise a unified grand theory."},
    {"cat": "adversarial", "input": "Calculate the exact number of atoms in the observable universe. Use the calculator with a precise formula."},
]

assert len(TASKS) == 100, f"Expected 100 tasks, got {len(TASKS)}"


# ── Helpers ────────────────────────────────────────────────────────────────────
def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": "Bearer dt_dev_test"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def wait_healthy(url: str, label: str, timeout: int = 60) -> None:
    print(f"  Waiting for {label}…", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            _get(f"{url}/health")
            print(" ready.")
            return
        except Exception:
            print(".", end="", flush=True)
            time.sleep(2)
    raise SystemExit(f"\nERROR: {label} did not become healthy within {timeout}s")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools import Tool
    from langchain import hub

    print("=" * 65)
    print("DuneTrace — Baseline Data Generator")
    print(f"  Agent ID : {AGENT_ID}")
    print(f"  Runs     : {MAX_RUNS}")
    print(f"  Ingest   : {INGEST_URL}")
    print(f"  API      : {API_URL}")
    print("=" * 65)

    print("\n[1/4] Health checks…")
    wait_healthy(INGEST_URL, "ingest")
    wait_healthy(API_URL,    "api")

    print("\n[2/4] Building agent…")

    tools = [
        Tool(
            name="web_search",
            func=web_search,
            description=(
                "Search the web for information on a topic. "
                "Input: a search query string. "
                "For deep research queries, results may be paginated — "
                "if the response says 'page N/M', append 'page=N+1' to your query to fetch the next page."
            ),
        ),
        Tool(
            name="calculator",
            func=calculator,
            description=(
                "Evaluate an arithmetic expression. "
                "Input: a valid arithmetic expression (e.g. '2**10', '100 * 1.05', '(3 + 4) * 2'). "
                "Returns the numeric result. Raises an error on invalid input."
            ),
        ),
        Tool(
            name="doc_lookup",
            func=doc_lookup,
            description=(
                "Look up detailed technical documentation on a topic. "
                "Input: a topic name (e.g. 'machine learning', 'climate change', 'python'). "
                "Returns comprehensive reference text."
            ),
        ),
    ]
    tool_names = [t.name for t in tools]

    dt = Dunetrace(
        api_key="dt_dev_test",
        endpoint=INGEST_URL,
        flush_interval_ms=200,
    )

    callback = DunetraceCallbackHandler(
        dt,
        agent_id=AGENT_ID,
        system_prompt=SYSTEM_PROMPT,
        model="gpt-4o-mini",
        tools=tool_names,
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,           # slight randomness for varied behaviour
        openai_api_key=OPENAI_API_KEY,
        callbacks=[callback],
    )

    prompt   = hub.pull("hwchase17/react")
    agent    = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=[callback],
        verbose=False,             # suppress per-step noise; we show our own progress
        max_iterations=12,
        handle_parsing_errors=False,
    )

    print("\n[3/4] Running tasks…\n")

    tasks = TASKS[:MAX_RUNS]
    random.shuffle(tasks)          # mix categories across batches

    results: list[dict] = []

    for i, task in enumerate(tasks, 1):
        # ── Batch pause ────────────────────────────────────────────────────────
        if i > 1 and (i - 1) % BATCH_SIZE == 0:
            print(f"\n  — Batch complete. Pausing {BATCH_DELAY:.0f}s to respect rate limits —\n")
            time.sleep(BATCH_DELAY)

        cat   = task["cat"]
        inp   = task["input"]
        label = inp[:65] + "…" if len(inp) > 65 else inp

        print(f"  [{i:3d}/{MAX_RUNS}] [{cat:10s}]  {label}")

        # Reset per-run state
        _search_call_counts.clear()

        t_start = time.time()
        outcome = "ok"
        steps   = 0
        answer  = ""

        try:
            result  = executor.invoke({"input": inp})
            answer  = str(result.get("output", ""))[:80]
            elapsed = time.time() - t_start
            # Count steps from callback (imprecise but useful)
            steps   = callback._step   # _step is reset after on_chain_end
        except Exception as exc:
            outcome = f"err:{type(exc).__name__}"
            elapsed = time.time() - t_start

        results.append({
            "n": i, "cat": cat, "outcome": outcome, "elapsed": elapsed, "answer": answer,
        })

        status = "✓" if outcome == "ok" else f"✗ {outcome}"
        print(f"           {status}  ({elapsed:.1f}s)")

        time.sleep(RUN_DELAY)

    # ── Flush & shutdown ───────────────────────────────────────────────────────
    print("\n  Flushing events to ingest…")
    dt.shutdown(timeout=10.0)
    print("  Done.\n")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("[4/4] Summary\n")
    cats     = ["research", "math", "factual", "multi", "adversarial"]
    ok_cnt   = sum(1 for r in results if r["outcome"] == "ok")
    err_cnt  = len(results) - ok_cnt
    avg_time = sum(r["elapsed"] for r in results) / len(results)

    print(f"  Total runs  : {len(results)}")
    print(f"  Completed   : {ok_cnt}  ({ok_cnt/len(results):.0%})")
    print(f"  Errored     : {err_cnt}")
    print(f"  Avg latency : {avg_time:.1f}s/run")
    print()
    print(f"  {'Category':<12}  {'Runs':>5}  {'OK':>5}  {'Err':>5}  {'Avg(s)':>7}")
    print(f"  {'─'*12}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*7}")
    for cat in cats:
        cr = [r for r in results if r["cat"] == cat]
        if not cr:
            continue
        cok  = sum(1 for r in cr if r["outcome"] == "ok")
        cerr = len(cr) - cok
        cavg = sum(r["elapsed"] for r in cr) / len(cr)
        print(f"  {cat:<12}  {len(cr):>5}  {cok:>5}  {cerr:>5}  {cavg:>7.1f}")

    print()
    print("  Waiting 45s for detector to process all runs…")
    time.sleep(45)

    try:
        data     = _get(f"{API_URL}/v1/agents/{AGENT_ID}/signals")
        signals  = data.get("signals", [])
        by_type: dict[str, int] = {}
        for s in signals:
            by_type[s["failure_type"]] = by_type.get(s["failure_type"], 0) + 1

        print(f"\n  Signals detected: {len(signals)}\n")
        print(f"  {'Failure type':<30}  {'Count':>6}  {'Severity':<8}")
        print(f"  {'─'*30}  {'─'*6}  {'─'*8}")
        seen: set[str] = set()
        for s in sorted(signals, key=lambda x: (x["failure_type"], x["severity"])):
            ft = s["failure_type"]
            if ft in seen:
                continue
            seen.add(ft)
            print(f"  {ft.replace('_',' '):<30}  {by_type[ft]:>6}  {s['severity']:<8}")
    except Exception as exc:
        print(f"  (Could not fetch signals: {exc})")

    print()
    print("  Dashboard:")
    print("    python -m http.server 3000 -d dashboard")
    print("    open http://localhost:3000")
    print()
    print(f"  Agent ID for further queries: {AGENT_ID}")
    print()


if __name__ == "__main__":
    main()
