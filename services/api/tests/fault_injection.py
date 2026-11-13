"""
Fault injection for event streams — the inverse of replay's repair modifications.

``api_svc/routers/replay.py`` can take a broken run and make it well: every
entry in its ``_VALID_MODS`` allowlist (``break_tool_loop``, ``fix_truncation``,
``reduce_context``, …) removes a fault. Nothing did the opposite, so the
detector battery had no corpus of synthesized faults to regress against — each
detector's tests built their own bespoke ``RunState`` by hand, which tests the
detector's ``check()`` but never the path a real run takes: raw events through
``build_run_state`` and out the other side.

This module supplies that half. ``clean_run()`` is a baseline event stream that
fires nothing; each ``inject_*`` function takes an event list and returns a new
one carrying exactly one fault, shaped the way the SDK would have emitted it.
Together they give three things the suite did not have:

  * a *negative* control — a realistic run that must stay silent. A detector
    that starts firing on clean traffic is the expensive kind of regression,
    and nothing caught it before.
  * event-level coverage — the fault is injected as events, so
    ``build_run_state`` is exercised too. The API and detector copies of that
    function had already drifted apart once (see CLAUDE.md's "Run
    reconstruction"), and replay silently ran on the broken copy.
  * a round trip — ``REPAIRS`` names the replay mod that undoes each fault, so
    a test can assert inject → fires → repair → clears. That pins the two
    halves against each other instead of each against its own fixtures.

Determinism is the point: no clocks, no randomness, no network. Timestamps are
derived from the step index, so the same call always produces byte-identical
events and a failure is always reproducible from the test name alone.

The three faults named in the original proposal map as follows: tool timeouts →
``inject_tool_timeout``, partial results → ``inject_silent_truncation`` (a
truncated response the agent builds on) and ``inject_empty_retrieval``, model
regressions → ``inject_model_downgrade``.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

Event = Dict[str, Any]

RUN_ID = "fault-injection-run"
AGENT_ID = "fault-injection-agent"
AGENT_VERSION = "v1"

# Distinct, boring tool names. Deliberately not "send_*"/"email_*"/"post_*":
# UngroundedDestinationDetector scopes its scan to send-shaped tool names, so a
# baseline built out of those would be testing that detector's tool-name filter
# rather than providing a neutral control.
CLEAN_TOOLS = ("lookup_order", "read_inventory", "check_price", "format_reply")

STABLE_PROMPT_TOKENS = 1200


def _ev(event_type: str, step: int, **payload: Any) -> Event:
    """One event, shaped like the ingest wire format. `timestamp` tracks
    `step` so runs stay short enough that SESSION_LATENCY never fires on a
    baseline and long enough that durations are non-zero."""
    return {
        "event_type": event_type,
        "run_id": RUN_ID,
        "agent_id": AGENT_ID,
        "agent_version": AGENT_VERSION,
        "step_index": step,
        "timestamp": float(step),
        "payload": payload,
        "parent_run_id": None,
    }


def _last_step(events: List[Event]) -> int:
    return max((e.get("step_index") or 0) for e in events) if events else 0


def _without_terminal(events: List[Event]) -> List[Event]:
    return [e for e in events if e["event_type"] not in ("run.completed", "run.errored")]


def _terminal(events: List[Event]) -> List[Event]:
    return [e for e in events if e["event_type"] in ("run.completed", "run.errored")]


def clean_run(steps: int = 3) -> List[Event]:
    """A well-behaved run: distinct successful tools, flat prompt growth, a
    real final answer. Must produce zero signals — `test_fault_injection.py`
    asserts exactly that, so this doubles as the suite's negative control."""
    events: List[Event] = [
        _ev("run.started", 0, input_text="Look up order 4471 and tell me the total.")
    ]
    step = 0
    for i in range(steps):
        step = i + 1
        events.append(
            _ev("llm.called", step, model="gpt-4o", prompt_tokens=STABLE_PROMPT_TOKENS, call_id=i)
        )
        events.append(
            _ev(
                "llm.responded",
                step,
                model="gpt-4o",
                call_id=i,
                prompt_tokens=STABLE_PROMPT_TOKENS + i * 20,
                completion_tokens=120,
                output_text=f"Step {i}: calling {CLEAN_TOOLS[i % len(CLEAN_TOOLS)]} to continue.",
                finish_reason="stop",
                latency_ms=900,
            )
        )
        tool = CLEAN_TOOLS[i % len(CLEAN_TOOLS)]
        events.append(
            _ev("tool.called", step, tool_name=tool, args={"order_id": 4471, "page": i + 1})
        )
        events.append(
            _ev(
                "tool.responded",
                step,
                tool_name=tool,
                success=True,
                output=f"{tool} returned 3 rows for order 4471.",
                latency_ms=300,
            )
        )

    step += 1
    events.append(
        _ev("llm.called", step, model="gpt-4o", prompt_tokens=STABLE_PROMPT_TOKENS, call_id=steps)
    )
    events.append(
        _ev(
            "llm.responded",
            step,
            model="gpt-4o",
            call_id=steps,
            prompt_tokens=STABLE_PROMPT_TOKENS + 60,
            completion_tokens=140,
            output_text="Order 4471 totals $184.20 across 3 line items, shipped on the 4th.",
            finish_reason="stop",
            latency_ms=1000,
        )
    )
    events.append(_ev("final.answer", step, text="Order 4471 totals $184.20."))
    events.append(_ev("run.completed", step + 1, exit_reason="final_answer"))
    return events


# ── Injectors ────────────────────────────────────────────────────────────────
# Each takes an event list and returns a NEW one. The terminal event is lifted
# off, the fault spliced onto the body, and the terminal re-stamped after it —
# a fault appended past run.completed would be dropped by build_run_state, and
# a run with no terminal is itself a fault (GOAL_ABANDONMENT), not a neutral
# carrier for whichever one the caller asked for.


def _splice(events: List[Event], extra: List[Event]) -> List[Event]:
    events = copy.deepcopy(events)
    body, terminal = _without_terminal(events), _terminal(events)
    body.extend(extra)
    last = _last_step(body)
    for t in terminal:
        t["step_index"] = last + 1
        t["timestamp"] = float(last + 1)
    return body + terminal


def inject_tool_timeout(
    events: List[Event], tool: str = "lookup_order", count: int = 4
) -> List[Event]:
    """`count` consecutive timeouts on ONE tool — the retry-storm shape: an
    agent hammering a dependency that is down. → RETRY_STORM."""
    start = _last_step(_without_terminal(events)) + 1
    extra: List[Event] = []
    for i in range(count):
        step = start + i
        extra.append(_ev("tool.called", step, tool_name=tool, args={"order_id": 4471}))
        extra.append(
            _ev(
                "tool.responded",
                step,
                tool_name=tool,
                success=False,
                error=f"TimeoutError: {tool} timed out after 30s (attempt {i + 1})",
                latency_ms=30_000,
            )
        )
    return _splice(events, extra)


def inject_cascading_failure(events: List[Event], count: int = 4) -> List[Event]:
    """Consecutive failures across DIFFERENT tools — a shared dependency
    falling over rather than one flaky tool. → CASCADING_TOOL_FAILURE."""
    start = _last_step(_without_terminal(events)) + 1
    extra: List[Event] = []
    for i in range(count):
        step = start + i
        tool = f"downstream_{i}"
        extra.append(_ev("tool.called", step, tool_name=tool, args={"order_id": 4471}))
        extra.append(
            _ev(
                "tool.responded",
                step,
                tool_name=tool,
                success=False,
                error="ConnectionError: upstream gateway returned 503",
                latency_ms=1200,
            )
        )
    return _splice(events, extra)


def inject_tool_loop(events: List[Event], tool: str = "check_price", count: int = 6) -> List[Event]:
    """The same tool called over and over with varying args — the arg-varying
    loop that dedup-by-(name, args) used to miss. → TOOL_LOOP."""
    start = _last_step(_without_terminal(events)) + 1
    extra: List[Event] = []
    for i in range(count):
        step = start + i
        # Args vary (so this is the arg-varying loop break_tool_loop has to
        # thin by NAME) without inventing an entity: 4471 is from the input
        # and the attempt counter is inside the skipped small-int range.
        extra.append(
            _ev("tool.called", step, tool_name=tool, args={"order_id": 4471, "attempt": i + 1})
        )
        extra.append(_ev("tool.responded", step, tool_name=tool, success=True, output=f"price {i}"))
    return _splice(events, extra)


def inject_truncation_loop(events: List[Event], count: int = 3) -> List[Event]:
    """Repeated `finish_reason=length` — the model hitting its cap again and
    again while the agent keeps asking. → LLM_TRUNCATION_LOOP."""
    start = _last_step(_without_terminal(events)) + 1
    extra: List[Event] = []
    for i in range(count):
        step = start + i
        extra.append(_ev("llm.called", step, model="gpt-4o", prompt_tokens=1400, call_id=100 + i))
        extra.append(
            _ev(
                "llm.responded",
                step,
                model="gpt-4o",
                call_id=100 + i,
                prompt_tokens=1400,
                completion_tokens=4096,
                output_text="Here is the full breakdown of every line item, beginning with",
                finish_reason="length",
                latency_ms=5000,
            )
        )
    return _splice(events, extra)


def inject_silent_truncation(events: List[Event]) -> List[Event]:
    """A PARTIAL RESULT: one response cut off at the token cap that the agent
    never retries and builds on regardless — half a plan passed downstream as
    if whole. Injected as the run's last LLM call, so nothing follows that
    could be read as a recovery. → SILENT_TRUNCATION."""
    events = copy.deepcopy(events)
    body, terminal = _without_terminal(events), _terminal(events)
    step = _last_step(body) + 1
    body.append(_ev("llm.called", step, model="gpt-4o", prompt_tokens=1500, call_id=200))
    body.append(
        _ev(
            "llm.responded",
            step,
            model="gpt-4o",
            call_id=200,
            prompt_tokens=1500,
            completion_tokens=4096,
            output_text=("The refund breakdown is: line 1 $42.00, line 2 $61.20, line 3 $8"),
            finish_reason="length",
            latency_ms=6000,
        )
    )
    for t in terminal:
        t["step_index"] = step + 1
        t["timestamp"] = float(step + 1)
    return body + terminal


def inject_empty_retrieval(events: List[Event], count: int = 2) -> List[Event]:
    """A PARTIAL RESULT of the RAG kind: retrieval returning nothing while the
    agent answers anyway. → RAG_EMPTY_RETRIEVAL."""
    start = _last_step(_without_terminal(events)) + 1
    extra: List[Event] = []
    for i in range(count):
        step = start + i
        extra.append(_ev("retrieval.called", step, query="refund policy for damaged goods"))
        extra.append(_ev("retrieval.responded", step, result_count=0, top_score=0.0, content=""))
    return _splice(events, extra)


def inject_context_growth(events: List[Event], factor: float = 9.0) -> List[Event]:
    """Prompt tokens climbing `factor`x across the run — history accumulating
    unbounded. Rewrites the existing calls rather than appending, since
    CONTEXT_BLOAT reads the last/first RATIO. → CONTEXT_BLOAT."""
    events = copy.deepcopy(events)
    calls = [e for e in events if e["event_type"] in ("llm.called", "llm.responded")]
    n = max(1, len(calls) - 1)
    for i, e in enumerate(calls):
        scale = 1.0 + (factor - 1.0) * (i / n)
        e["payload"]["prompt_tokens"] = int(STABLE_PROMPT_TOKENS * scale)
    return events


def inject_model_downgrade(
    events: List[Event], downgrade_to: str = "gpt-4o-mini", count: int = 2
) -> List[Event]:
    """A MODEL REGRESSION: the run silently continues on a weaker model, the
    shape of an SDK retrying on a cheaper tier under rate limiting.
    → MODEL_FALLBACK_DRIFT."""
    start = _last_step(_without_terminal(events)) + 1
    extra: List[Event] = []
    for i in range(count):
        step = start + i
        extra.append(
            _ev("llm.called", step, model=downgrade_to, prompt_tokens=1300, call_id=300 + i)
        )
        extra.append(
            _ev(
                "llm.responded",
                step,
                model=downgrade_to,
                call_id=300 + i,
                prompt_tokens=1300,
                completion_tokens=110,
                output_text="Continuing with the order lookup.",
                finish_reason="stop",
                latency_ms=700,
            )
        )
    return _splice(events, extra)


def inject_first_step_failure(events: List[Event]) -> List[Event]:
    """The run's very first tool call fails — a config or credentials problem,
    not agent behaviour. Spliced at the front. → FIRST_STEP_FAILURE."""
    events = copy.deepcopy(events)
    started = [e for e in events if e["event_type"] == "run.started"]
    rest = [e for e in events if e["event_type"] != "run.started"]
    for e in rest:
        e["step_index"] = (e.get("step_index") or 0) + 1
        e["timestamp"] = float(e["step_index"])
    fault = [
        _ev("tool.called", 1, tool_name="lookup_order", args={"order_id": 4471}),
        _ev(
            "tool.responded",
            1,
            tool_name="lookup_order",
            success=False,
            error="AuthenticationError: invalid API credentials for orders service",
        ),
    ]
    return started + fault + rest


def inject_goal_abandonment(events: List[Event], stall_steps: int = 6) -> List[Event]:
    """The agent stops acting — LLM turns with no tool calls and no final
    answer, the run just stopping. Drops the terminal and the final answer,
    which is what abandonment looks like on the wire. → GOAL_ABANDONMENT."""
    events = copy.deepcopy(events)
    body = [e for e in _without_terminal(events) if e["event_type"] != "final.answer"]
    start = _last_step(body) + 1
    for i in range(stall_steps):
        step = start + i
        body.append(_ev("llm.called", step, model="gpt-4o", prompt_tokens=1300, call_id=400 + i))
        body.append(
            _ev(
                "llm.responded",
                step,
                model="gpt-4o",
                call_id=400 + i,
                prompt_tokens=1300,
                completion_tokens=90,
                output_text="Let me think about how to approach this.",
                finish_reason="stop",
                latency_ms=800,
            )
        )
    return body


# name -> (injector, the failure type it must produce, the replay mod that
# undoes it or None where replay has no repair for that fault)
INJECTORS = {
    "tool_timeout": (inject_tool_timeout, "RETRY_STORM", "fix_tool_failures"),
    "cascading_failure": (inject_cascading_failure, "CASCADING_TOOL_FAILURE", "fix_tool_failures"),
    "tool_loop": (inject_tool_loop, "TOOL_LOOP", "break_tool_loop"),
    "truncation_loop": (inject_truncation_loop, "LLM_TRUNCATION_LOOP", "fix_truncation"),
    "silent_truncation": (inject_silent_truncation, "SILENT_TRUNCATION", "fix_truncation"),
    "empty_retrieval": (inject_empty_retrieval, "RAG_EMPTY_RETRIEVAL", "fix_rag_retrieval"),
    "context_growth": (inject_context_growth, "CONTEXT_BLOAT", "reduce_context"),
    "model_downgrade": (inject_model_downgrade, "MODEL_FALLBACK_DRIFT", "fix_model_downgrade"),
    "first_step_failure": (inject_first_step_failure, "FIRST_STEP_FAILURE", None),
    "goal_abandonment": (inject_goal_abandonment, "GOAL_ABANDONMENT", "add_final_answer"),
}
