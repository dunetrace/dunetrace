# Auto-Instrumentation

`dt.auto_instrument()` / `dt.init(agent_id=...)` monkey-patch supported AI
framework clients so their calls are tracked automatically — no manual
`run.llm_called()` / `run.tool_called()`, and for LangChain/CrewAI, no manual
`DunetraceCallbackHandler` construction or `callbacks=[...]` wiring either.

```python
from dunetrace import Dunetrace

dt = Dunetrace(api_key="dt_live_...")
dt.init(agent_id="my-agent")   # patches every installed supported framework
```

Supported frameworks: `openai`, `anthropic`, `httpx`, `requests`, `langchain`
(covers LangGraph), `crewai`. Pass `frameworks=[...]` to either call to patch
a subset. Patching is idempotent and permanent for the life of the process —
calling it twice, or from multiple places, is safe and cheap.

---

## Why you might see an unexpected agent_id in the dashboard

This is the page to send a teammate who opens the dashboard, sees a run
attributed to `unattributed-agent` or a `langchain`/`crewai` default they
didn't expect, and wants to know what happened.

`openai`, `anthropic`, `httpx`, and `requests` never decide an agent_id
themselves — they only react to whatever `dt.run()` is already open, and do
nothing at all outside one. `langchain` and `crewai` are different: LangChain
can *only* attach to an already-open `dt.run()` (see the next section for
why), while CrewAI can open its own run when none is open — and when it does,
it has to pick an agent_id from somewhere.

### Resolution order

| Tier | Source | LangChain | CrewAI | Notes |
|---|---|---|---|---|
| 1 | Ambient `dt.run(agent_id=...)` | ✅ required | ✅ if present | The only tier LangChain actually uses through `auto_instrument()`. If a `dt.run()` block is open, its `agent_id` wins outright — no other tier is even consulted. |
| 2 | Per-call override | `config={"metadata": {"agent_id": "..."}}` | `kickoff(inputs={"agent_id": "..."})` | Only reachable when tier 1 doesn't apply. For LangChain that means: the caller separately passed `callbacks=[handler]` to a chain (the pre-`auto_instrument()` manual pattern), which is what makes `on_chain_start` fire per-call metadata in the first place. |
| 3 (CrewAI bonus) | Framework-native identity | — | `Crew.name` (if set to something other than the literal default `"crew"`), or `Agent.role` for a directly-kicked-off `Agent` | LangChain has no equivalent — a `Runnable` has no built-in notion of "whose agent is this". |
| 4 | `default_agent_id` | ✅ (only reachable alongside tier 2's conditions) | ✅ | Set once via `dt.init(agent_id="my-agent")` or the `DUNETRACE_AGENT_ID` environment variable. |
| — | Loud fallback | ✅ | ✅ | If nothing above resolves, the run is still recorded (rather than crashing your agent) under `unattributed-agent`, and a `WARNING`-level log line names the integration and links back to this doc. Search your logs for `could not determine an agent_id` if you see this id in the dashboard. |

Example:

```python
dt.init(agent_id="fallback-agent")   # tier 4

# LangChain — tier 1 is the only one that fires through auto_instrument():
with dt.run("checkout-agent"):
    result = my_langgraph_agent.invoke({"messages": [...]})
    # -> agent_id = "checkout-agent" (tier 1)

# CrewAI — no dt.run() needed, tiers 2-4 all apply:
crew.kickoff(inputs={"agent_id": "explicit-crew"})
    # -> agent_id = "explicit-crew" (tier 2, wins over Crew.name and default)

research_crew.kickoff(inputs={"topic": "AI trends"})
    # research_crew.name == "research-crew" -> agent_id = "research-crew" (tier 3)

Agent(role="researcher").kickoff("find the latest AI news")
    # -> agent_id = "researcher" (tier 3, framework-native Agent.role)
```

---

## Why LangChain needs `dt.run()` but CrewAI doesn't

Both integrations reuse an existing manual integration (`DunetraceCallbackHandler`
for LangChain, `DunetraceCrewCallback`'s global hooks for CrewAI) rather than
re-implementing event emission — `auto_instrument()`'s job is just to make sure
that existing machinery gets wired in without you doing it by hand. But the two
frameworks expose a different kind of hook, which changes what's possible:

- **CrewAI** patches the true top-level call: `Crew.kickoff` / `Agent.kickoff`
  (and the `_async` variants). This is the same call the framework's own
  execution starts from, so `auto_instrument()` can open a `dt.run()` itself,
  around the whole invocation, and everything nested inside it (every LLM and
  tool call CrewAI's global hooks report) correctly attaches to that one run.

- **LangChain** has no single top-level call to patch — an agent might be a
  raw `AgentExecutor`, a compiled LangGraph `StateGraph`, or a hand-built
  `RunnableSequence`, and there's no common base class among them worth
  patching. What *is* common to every LangChain agent, regardless of how it's
  built, is that its LLM calls go through `BaseChatModel.invoke/ainvoke/
  stream/astream` and its tool calls go through `BaseTool.run/arun` — so
  that's what `auto_instrument()` patches instead.

  The cost of patching at that lower level: `DunetraceCallbackHandler`'s own
  run-creation logic hangs off LangChain's `on_chain_start` callback, which
  only fires when a callback is attached at the *top-level* chain/agent
  invoke — not when it's attached deeper, at the LLM/tool leaf, which is all
  `auto_instrument()` can reach. So `on_chain_start` never fires through this
  patch, and the handler can never open its own run this way.

  What it *can* do is attach to a run that's already open — every LLM/tool
  call your agent makes while a `dt.run()` block is active resolves the same
  ambient `RunContext`, correctly correlating a whole multi-turn agent loop
  into one run without needing any root-tracking bookkeeping at all. Hence
  the requirement: wrap the top-level call.

```python
dt.init(agent_id="fallback")

# Won't be tracked — no ambient dt.run(), and on_chain_start never fires
# for a bare leaf-level call:
result = my_agent.invoke({"messages": [...]})

# Tracked correctly — every LLM/tool call in this invocation attaches to
# the same run:
with dt.run("my-agent"):
    result = my_agent.invoke({"messages": [...]})
```

If you need `on_chain_start` to fire on its own (e.g. you want tier 2's
per-call `metadata={"agent_id": ...}` override without wrapping in `dt.run()`),
fall back to the manual pattern instead of `auto_instrument()`: construct
`DunetraceCallbackHandler` yourself and pass `callbacks=[handler]` to the
top-level chain/agent invoke — see
[integrate-langchain-agent.md](../integrate-langchain-agent.md).

---

## Avoiding double-counted events

If your LangChain agent uses `ChatOpenAI` (which calls the raw `openai` SDK
internally) and you've also patched `openai` directly, a single LLM call would
otherwise be counted twice: once by the LangChain integration, once by the
`openai` patch underneath it. `auto_instrument()` avoids this with a
re-entrancy flag (`dunetrace.context._in_framework_call`) that LangChain and
CrewAI's patches set for the duration of the underlying call — the
`openai`/`anthropic`/`httpx`/`requests` patches check it and skip their own
emission (but still make the real call) whenever it's set. You don't need to
do anything for this — it's automatic whenever you patch both layers together.
