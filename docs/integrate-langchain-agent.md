# Integrating a LangChain Agent with Dunetrace

<!--dunetrace:instrument
framework: langchain
language: python
install: pip install 'dunetrace[langchain]' langchain-openai
primary_symbol: dunetrace.integrations.langchain.DunetraceCallbackHandler
mechanism: callback-handler
opens_own_run: true
requires_run_context: false
target_file: the module that builds and invokes the graph or executor
target_example: `app/graph.py`, `src/agent.py`
target_hints: ["create_agent", "create_react_agent", "create_deep_agent", "define_deep_agent", "StateGraph", "AgentExecutor", "builder.compile(", ".invoke(", ".ainvoke("]
emits: [run.started, run.completed, run.errored, llm.called, llm.responded, tool.called, tool.responded, retrieval.called, retrieval.responded, memory.written, memory.read, memory.cleared]
verify_cmd: SCENARIO=tool_loop python packages/sdk-py/examples/langchain_agent.py
-->

> **Looking for `dt.auto_instrument()` instead?** It patches LangChain for you, no callback object needed — but requires wrapping the top-level call in `dt.run(...)`. See [auto-instrumentation.md](./integrations/auto-instrumentation.md).

## Quick Start

```bash
pip install 'dunetrace[langchain]' langchain-openai
```

```python
from dunetrace import Dunetrace
from dunetrace.integrations.langchain import DunetraceCallbackHandler
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

dt = Dunetrace()   # local dev, no API key needed
callback = DunetraceCallbackHandler(dt, agent_id="my-agent", model="gpt-4o")

agent = create_agent(ChatOpenAI(model="gpt-4o"), tools=[])
result = agent.invoke(
    {"messages": [("human", "What is the capital of France?")]},
    config={"callbacks": [callback]},   # <-- this is the whole integration
)

dt.shutdown()
```

Start the backend once, locally, before running this: `docker compose up -d`.

## Where this goes

The handler is passed at the **top-level** `.invoke()` / `.ainvoke()`,
the one call that runs the whole agent. That is the only place it can open a run
from, because run creation hangs off LangChain's `on_chain_start`, which fires
only for a callback attached at the top-level chain or agent invoke.

Find it with:

```bash
grep -rn "create_agent\|create_react_agent\|AgentExecutor\|StateGraph\|\.invoke(" --include=*.py .
```

Typically `app/graph.py` or `src/agent.py`. Attach the handler where the graph
is invoked, not where a node is defined. A handler passed to a node's own
`config` records that node's LLM call but never opens a run, so the events have
nothing to attach to.

Two call sites to be careful about: a `.stream()` / `.astream()` loop takes the
same `config={"callbacks": [...]}`, and a LangServe or FastAPI route invokes the
graph inside the request handler, which is still the right place.

## What this does

`DunetraceCallbackHandler` plugs into LangChain's callback system and translates every LLM call, tool call, and retriever call into Dunetrace events automatically — no changes to your agent logic. Works with LangChain v1's `create_agent`, LangGraph custom graphs, and the older `create_react_agent` / `AgentExecutor` the same way; just pass the same `callback` in `config={"callbacks": [...]}`. Async (`ainvoke`) works identically to sync (`invoke`).

## Constructor options

| Parameter | Required | Description |
|---|---|---|
| `agent_id` | Yes | Identifier shown in the dashboard |
| `system_prompt` | No | Used to compute a version fingerprint when your prompt changes |
| `model` | No | Model name for display and detector context |
| `tools` | No | Tool name list — used by `TOOL_AVOIDANCE` |

## Known limitations

**`get_current_run()` returns `None` inside a tool on the async path.** Verified
against langchain-core 1.4.8: it works under `invoke()`, returns `None` under
`ainvoke()`. Always write `if run:` before using it, or your tool raises
`AttributeError` and your agent fails. Event capture is unaffected. Full
explanation under [Accessing the run inside a
tool](#accessing-the-run-inside-a-tool).

**The stale sweep can drop one invocation on the async path.** Only after a run
is abandoned without completing for 30 minutes, and it self-heals on the next
invocation. See [Concurrent invocations](#concurrent-invocations).

**`auto_instrument()` never opens its own run for LangChain.** Wrap the
top-level call in `dt.run(...)`, or use the callback handler shown above, which
does open one.

## Verification

```bash
SCENARIO=tool_loop python packages/sdk-py/examples/langchain_agent.py
```

Calls `web_search` six times in one run, triggering `TOOL_LOOP`. Check the dashboard at `http://localhost:3000` — the signal should appear within ~15 seconds.


### If nothing arrives

Work down this list. The first two cover almost every case.

1. **Is a run open?** This integration opens its own run, so there is nothing to wrap. Confirm the registration call (`DunetraceCallbackHandler`) actually ran, and ran **before** the first agent invocation.

2. **Did the process flush?** Events ship from a background thread. Call
   `dt.shutdown()` before the process exits, or the buffer dies with it.

3. **Is the backend reachable?** Both should return `{"status":"ok",...}`:

   ```bash
   curl -s localhost:8001/ready   # ingest
   curl -s localhost:8002/ready   # customer API
   ```

4. **Did anything land?** If your agent id is listed here, instrumentation is
   working and the problem is downstream:

   ```bash
   curl -s localhost:8002/v1/agents
   curl -s "localhost:8002/v1/agents/<your-agent-id>/runs?limit=5"
   ```

5. **Turn on debug logging.** `Dunetrace(debug=True)` logs every event as it is buffered and
   every batch as it ships.

**Runs appear but no signals?** That is usually correct, not a fault. The
detector polls every 5 seconds, and a healthy run produces no signals. Several
detectors also need cross-run baselines and stay dormant until the agent has
run history. Check `docker compose logs detector` if you expected one.

---

## Advanced (optional)

### RAG agents

Retriever calls are captured automatically — no extra code. `result_count` and `top_score` are extracted from document metadata (`score`, `relevance_score`, or `similarity`), feeding `RAG_EMPTY_RETRIEVAL`.

### Accessing the run inside a tool

> **`get_current_run()` returns `None` inside a tool on the async path.** Verified
> against langchain-core 1.4.8: it works under `invoke()` and returns `None`
> under `ainvoke()`. **Always guard with `if run:`.** Without the guard the
> example below raises `AttributeError: 'NoneType' object has no attribute
> 'external_signal'` inside your tool, which fails your agent, not just its
> telemetry.

```python
from dunetrace import get_current_run

@tool
def web_search(query: str) -> str:
    run = get_current_run()
    if run and rate_limited():          # `if run` is required, not defensive
        run.external_signal("rate_limit", source="serpapi")
    return do_search(query)
```

Why: LangChain dispatches a synchronous callback handler through
`run_in_executor(None, copy_context().run, ...)` on the async path, giving every
callback a fresh context copy. The handler sets the run into a copy that is
discarded the moment the callback returns, so nothing reaches your tool. The
sync path calls the handler directly and is unaffected.

Everything else is unaffected. LLM, tool and retrieval events are correlated
through the handler's own root tracking, not through the context variable, so
they are captured correctly on both paths.

If you need the run object inside a tool under `ainvoke()`, open an explicit
`with dt.run(...)` around the top-level call. The handler then attaches to that
ambient run, and `get_current_run()` resolves in your own context.

### Concurrent invocations

The handler is thread-safe — one instance can be shared across concurrent `invoke()` calls, each tracked independently by LangChain's own root `run_id`. Stale runs (never completed after 30 minutes) are pruned automatically.

> **On the async path the stale sweep can drop an invocation.** The sweep runs
> at the start of a new invocation and cleans up runs older than 30 minutes.
> Cleaning up a run recorded on the async path raises internally, and the
> exception aborts the invocation that triggered the sweep before it is
> registered, so that one invocation produces no events at all. It only happens
> after a run is abandoned without completing (a cancelled task, a hard
> timeout), and it self-heals: the stale entry is removed before the failure, so
> the next invocation is fine. Completing or cancelling runs cleanly avoids it.

### What's captured

Every LLM call (model, tokens, latency, raw prompt/completion), every tool call (name, success/failure, raw args/output — including framework-handled errors via `handle_tool_error=True`), every retriever call (index, count, score, raw query), and run-level totals. Not captured: intermediate sub-chain inputs/outputs (only the root chain boundary is a run), and streaming token counts in some provider/version combinations.

### Troubleshooting

- **No runs appear** — confirm `dt.shutdown()` was called; try `Dunetrace(debug=True)` for verbose logs
- **Token counts missing** — some providers/LangChain versions omit `token_usage`; the handler falls back to `usage_metadata`, and omits the fields entirely if both are absent (doesn't break detectors)
- **Detectors fire too aggressively** — tune thresholds in `detectors.yml` and restart the detector service
