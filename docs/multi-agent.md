# Multi-agent runs

An agent that delegates to other agents — an orchestrator calling a researcher,
a supervisor handing work to a coder, a crew of specialists — is a **multi-agent
system**. Dunetrace models it as a graph of runs linked by `parent_run_id`: when
one `dt.run()` opens inside another, the inner run is a *child* of the outer.

That parent/child structure is what two cross-run detectors read:

- [`HANDOFF_CONTEXT_LOSS`](detectors.md#handoff_context_loss) — a handoff drops a
  large chunk of what the parent agent had learned.
- [`DELEGATION_LOOP`](detectors.md#delegation_loop) — agents delegate to each
  other in a cycle that never converges.

---

## `parent_run_id` is auto-threaded

You don't thread ids by hand. When a `dt.run()` opens while another run is
already active on the same task, the new run inherits the active run's id as its
`parent_run_id`:

```python
with dt.run("orchestrator") as parent:          # top-level: no parent
    ...
    with dt.run("researcher") as child:         # parent_run_id = parent.run_id
        ...                                      # automatically
```

The rule is simple and the id is on the wire (in the `run.started` event), not
just in memory:

- **Top-level run** → no parent.
- **Nested run** → parent is the run that was active when it opened.
- **Explicit `parent_run_id=`** always wins, if you ever need to set it by hand
  (e.g. threading across a boundary the contextvar can't cross).

### What propagates, and what doesn't

Threading rides Python `contextvars`, so:

| Pattern | Auto-threaded? |
|---|---|
| Synchronous nested `dt.run()` | ✅ |
| `dt.run()` in an awaited `async` child task | ✅ (asyncio copies context) |
| Sub-agent run started in a **bare thread** (`Thread`, `ThreadPoolExecutor`) | ❌ — the thread starts with a fresh context |

For the thread case, either open the child run before handing off to the thread,
copy the context (`contextvars.copy_context()`), or pass `parent_run_id=` into the
child run explicitly.

---

## The multi-agent pattern

Instrument each agent as its own `dt.run()`. Nesting the runs the way your agents
actually delegate is all it takes to build the graph:

```python
def researcher(topic):
    with dt.run("researcher", user_input=topic):
        ...                                  # child of whoever called it

def orchestrator(task):
    with dt.run("orchestrator", user_input=task):
        findings = researcher(task)          # nested -> auto-linked
        ...
```

A **delegation loop** is what you get when this goes wrong — two agents calling
each other without converging:

```python
def planner(goal, depth=0):
    with dt.run("planner", user_input=goal):
        return executor(goal, depth)         # planner -> executor

def executor(goal, depth=0):
    with dt.run("executor", user_input=goal):
        if not done(goal):
            return planner(goal, depth + 1)  # executor -> planner -> ... loop
```

Each recursive call opens a nested run, so the chain becomes
`planner → executor → planner → executor → …`. Once ~2.5 round trips accumulate
(5 runs, the calibrated default), `DELEGATION_LOOP` fires. See
[`examples/delegation_loop_agent.py`](../examples/delegation_loop_agent.py).

---

## A note on frameworks

`dt.auto_instrument()` (LangChain, CrewAI) opens **at most one** top-level run per
invocation — CrewAI's kickoff patch returns to the original when a run is already
active, and LangChain's leaf patch only ever attaches to an existing run. So a
whole crew or graph is captured as a single run, and within-crew delegation does
*not* produce a multi-run graph on its own.

Cross-agent delegation graphs come from the per-agent nested-`dt.run()` pattern
above — which the auto-threading links regardless of whether the runs are opened
by your own code or by a framework integration. If your framework collapses
delegation into one run, `HANDOFF_CONTEXT_LOSS` and `DELEGATION_LOOP` have no
graph to walk and simply never fire (a disclosed limitation, not a bug).
