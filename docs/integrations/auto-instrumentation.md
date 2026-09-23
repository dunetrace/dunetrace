# Auto-Instrumentation

`dt.auto_instrument()` / `dt.init(agent_id=...)` monkey-patch supported AI
framework clients so their calls are tracked automatically — no manual
`run.llm_called()` / `run.tool_called()`, and for LangChain/CrewAI, no manual
`DunetraceCallbackHandler` construction or `callbacks=[...]` wiring either.

```python
from dunetrace import Dunetrace

dt = Dunetrace(api_key="dt_...")
dt.init(agent_id="my-agent")   # patches every installed supported framework

# Patching alone emits nothing. The patches attach to an OPEN run, so the
# top-level call has to be wrapped:
with dt.run("my-agent", user_input=question):
    answer = my_agent(question)
```

> **Patching is half the job.** Every target except `crewai` is a silent no-op
> outside an open run: the call succeeds, returns normally, and emits **no
> events and no warning**. If you add `auto_instrument()` and see nothing in the
> dashboard, this is almost always why. `crewai` is the one exception, because
> it patches `Crew.kickoff`/`Agent.kickoff` and can open its own run. Full
> reasoning in [Why LangChain needs `dt.run()` but CrewAI
> doesn't](#why-langchain-needs-dtrun-but-crewai-doesnt) below.
>
> A run context comes from any one of: `with dt.run(...)`, the `@dt.agent` /
> `@dt.trace` decorators, or `DunetraceASGIMiddleware` /
> `DunetraceWSGIMiddleware`.

Supported frameworks: `openai`, `anthropic`, `mistral`, `botocore` (AWS
Bedrock), `httpx`, `requests`, `langchain` (covers LangGraph), `crewai`. Pass `frameworks=[...]` to either call
to patch a subset. Patching is idempotent and permanent for the life of the
process — calling it twice, or from multiple places, is safe and cheap.

`mistral` requires `mistralai>=2.0` (`pip install 'dunetrace[mistral]'`) and
covers chat, embeddings and FIM, including the Azure- and GCP-hosted clients —
see [mistral.md](mistral.md) for the full surface and its limits.

`botocore` covers **AWS Bedrock**, and so every model Bedrock hosts rather than
just one vendor. Bedrock is reached through boto3 rather than any vendor SDK,
and botocore rides on urllib3 rather than httpx or requests, so it is invisible
to every other patch here. `Converse` and `ConverseStream` report exact token
counts, text and stop reason. `InvokeModel` reports token counts from the
response headers and no output text: its payload is a streaming body in a
model-specific format, and reading it here would hand your code an empty
stream. `InvokeModelWithResponseStream` reads the model-agnostic
`amazon-bedrock-invocationMetrics` object Bedrock appends to the final chunk.
Non-Bedrock boto3 calls (S3, SQS, …) pass straight through.

---

## Confirming the patch is actually recording

Patching succeeds silently whether or not it ends up emitting anything, so check
for events rather than for the absence of an error. This prints what the SDK
would ship, with no backend and no LLM account:

```python
from dunetrace import Dunetrace, CallableExporter

seen = []
dt = Dunetrace(endpoint=None, exporters=[CallableExporter(lambda e: seen.append(e.event_type.value))])
dt.auto_instrument()

with dt.run("smoke-test", model="gpt-4o"):
    your_agent_entry_point("hello")

print(seen)
```

A working setup prints at least `['run.started', ..., 'run.completed']` with
`llm.called` / `llm.responded` in between. If you see only the two run events,
the patch is installed but your LLM client is not one of the patched targets, or
the call happens in a different process.

If you see nothing at all, no run was open. That is the failure this page exists
to describe.

---

## Why you might see an unexpected agent_id in the dashboard

This is the page to send a teammate who opens the dashboard, sees a run
attributed to `unattributed-agent` or a `langchain`/`crewai` default they
didn't expect, and wants to know what happened.

`openai`, `anthropic`, `mistral`, `httpx`, and `requests` never decide an agent_id
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
`openai`/`anthropic`/`mistral`/`httpx`/`requests` patches check it and skip their
own emission (but still make the real call) whenever it's set. You don't need to
do anything for this — it's automatic whenever you patch both layers together.

There is a second layer of the same problem, and a second flag for it. Every
vendor LLM SDK here is itself built on `httpx` (or `requests`), so with both the
provider patch and the HTTP patch on — which is what a bare
`dt.auto_instrument()` gives you — one LLM call would be recorded twice: once as
`llm.called`, and once as a `tool.called` named after the provider's hostname.
That inflates `tool_call_count`, which is both a policy trigger and what
`TOOL_LOOP` counts. The provider patches set
`dunetrace.context._http_suppressed` for the duration of the underlying request,
and the `httpx`/`requests` patches skip their emit while it's set. Suppression is
scoped to the provider call only, so HTTP your agent makes as a genuine tool is
still recorded normally.

---

## Streaming

Streamed LLM calls are recorded too, for `openai`, `anthropic` and `mistral`
alike. `llm.called` is emitted when the call is made; `llm.responded` when the
stream finishes, carrying the accumulated token counts, output text and latency.
The output text is subject to the SDK's per-field cap (`max_field_chars`, default
8192, env `DUNETRACE_MAX_FIELD_CHARS`); a cut field carries `output_truncated`
and `output_original_length` beside it, and `output_length` is always the real
size — see the SDK README's "What leaves the process".

The stream you get back is a transparent proxy: a real iterator and a context
manager, with everything else falling through to the underlying stream, so
`next(stream)`, `with`/`async with`, `.close()`, and provider-specific helpers
(Anthropic's `text_stream`, for instance) all behave as they do un-instrumented.

Because the response isn't known until the stream ends, *when* the event lands
depends on how you consume it:

| How you consume it | When `llm.responded` lands |
|---|---|
| Drain it fully | At the end of the stream |
| `break` out early | When the run block exits, at the latest — you still get an event for the tokens you read |
| `with` / `async with` | On context exit |
| Never touch it again | When the run block exits |

**Token counts on a streamed OpenAI call are estimated, not measured** — unless
you pass `stream_options={"include_usage": True}`, in which case the exact totals
are used. Dunetrace deliberately does not inject that option for you: it makes
the API append a final chunk with an empty `choices` list, which breaks any
caller doing `chunk.choices[0]`. Without it, completion tokens are estimated from
the streamed output at roughly 4 characters per token — including streamed
tool-call arguments, which are billed output but arrive with no text content at
all. Treat those numbers as approximate when setting a tight `cost_usd` policy
threshold. Anthropic and Mistral both report real usage on a stream by default,
so this caveat is OpenAI-only.

A stream that fails partway through is recorded with `finish_reason="error"` and
the exception text, rather than as a clean `stop`.

## Runs that open themselves

Every event belongs to a run, and a run needs a start and an end. `dt.run()`,
the decorators and the middleware give both explicitly. Since this feature,
`dt.init()` alone is enough: the SDK decides where a run starts and, the
harder half, where it ends, for code that never calls Dunetrace.

### How a run ends, in the order the SDK tries them

1. **A framework says so.** Entry points with a natural end open an exact run
   and close it when they return. Nothing is guessed; these are ordinary runs.

   | Entry point | Opens a run when | Closes it when |
   |---|---|---|
   | FastAPI / Flask app (middleware auto-installed on apps built after `dt.init()`) | a request arrives | the response is sent |
   | LangGraph `invoke` / `ainvoke` / `stream` / `astream` on a compiled graph | called with no run active | the call returns or the stream is exhausted |
   | CrewAI `kickoff` | called with no run active | it returns |
   | OpenAI Agents `Runner` (trace processor) | the trace starts | the trace ends |
   | Vercel AI `wrapGenerateText` / `wrapStreamText` (TypeScript) | called with no run active | the promise settles / `onFinish` or `onError` fires |

   Nested calls attach to the run that is already open, which is how subgraphs
   and sub-agents land under their parent instead of opening their own.

2. **Your own `dt.run()`** closes an implicit run that was open in the same
   context (`exit_reason: explicit_run_opened`), so a declared run never nests
   under a guessed one.

3. **Idle.** When a patched LLM call (`openai`, `anthropic`, `mistral`,
   Bedrock) arrives with no run active and no framework boundary in sight, the
   SDK opens an *implicit* run and attaches the calls that follow in the same
   thread or task to it. That run closes after 30 seconds with no events
   (`DUNETRACE_IMPLICIT_RUN_IDLE_S`, or `implicit_run_idle_s=` on the client).
   This is the rule for scripts and notebooks. It is a guess, and the run says
   so: `run.started` carries `implicit: true` and `opened_by` (the call that
   opened it), and the terminal event's `exit_reason` is `idle`. HTTP-only
   activity never opens a run; it attaches when one exists.

4. **Process exit.** `dt.shutdown()` and the at-exit flush close whatever
   implicit runs are still open (`exit_reason: process_exit`).

5. **A crash.** After `dt.init()`, an unhandled exception on the main thread
   or a worker thread becomes `run.errored` on the run active in that
   context, or on a one-event run when none is open, so a crash is never
   silent. The traceback still prints exactly as before.

### What the detector does with an implicit run

A guessed boundary is not a declared one. The detector treats an implicit run
the way it treats a run whose events were shed under overload: every signal is
stored in shadow with confidence capped at 0.5 and severity at MEDIUM, evidence
carries `incomplete_data: {"reason": "implicit_run"}`, and the run feeds neither
issue tracking nor baselines. Implicit runs show up in the dashboard's shadow
section and never alert. Runs opened by a framework entry point are ordinary
runs and alert normally.

The known weak spot is a synchronous server whose worker threads outlive
requests and that has no request boundary: without the middleware, the calls
of many requests on one thread would share an implicit run until the thread
goes idle. That is why implicit runs are shadowed, why the middleware is
auto-installed, and why the entry-point rules run first.

### Turning it off

`Dunetrace(implicit_runs=False)` or `DUNETRACE_IMPLICIT_RUNS=0` restores the
attach-only behaviour: a patched call outside a run records nothing, and no
crash hooks are installed. Entry-point runs and the middleware are unaffected.

### One value to configure: the DSN

```bash
export DUNETRACE_DSN=https://<api_key>@ingest.example.com
```

`Dunetrace()` reads the endpoint and the key from it. An explicit `endpoint`
or `api_key` argument, or `DUNETRACE_ENDPOINT` / `DUNETRACE_API_KEY`, wins over
the corresponding part of the DSN. The key still travels as a bearer token;
the DSN is a convenience, and the SDK never logs it whole.
