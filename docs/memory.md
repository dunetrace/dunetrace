# Agent memory channel

Agents remember things between steps and between turns — a conversation buffer, a
scratchpad, a long-term vector store of "facts" about the user. That memory is
also an **attack surface**: content from an untrusted channel (a retrieved
document, a tool response, an external feed) can get summarized and persisted
verbatim, injection and all, and then quietly re-steer the agent every time the
memory is loaded back — long after the step that wrote it.

The **memory channel** instruments what an agent writes to and reads from its own
memory, so Dunetrace can see that content and flag poisoning. It's opt-in and
purely additive: an agent that never touches the channel is unaffected, and older
event streams (recorded before the channel existed) still ingest cleanly.

The [`MEMORY_POISONING`](detectors.md#memory_poisoning) detector consumes this
channel.

---

## The three operations

Three annotation methods on the run handle, none of which advance the step
counter (they annotate the current step, like `external_signal`):

```python
with dt.run("support-agent", user_input="summarize this doc") as run:
    doc = fetch_document(url)                       # untrusted content
    run.memory_written("doc_summary", summarize(doc), source="tool_output")
    ...
    prefs = run.memory_read("doc_summary")          # loaded back later
    ...
    run.memory_cleared("doc_summary")               # or memory_cleared() to clear all
```

| Method | Meaning | Payload |
|---|---|---|
| `memory_written(key, value, source=None)` | Something was persisted to memory under `key` | `key`, `value`, optional `source` |
| `memory_read(key)` | Memory was loaded back at `key` | `key` |
| `memory_cleared(key=None)` | A key was cleared, or all memory when `key` is `None` | `key` (may be `null`) |

### The `source` argument

`source` names where a written value **originated** — it's optional but strongly
encouraged, because it's what lets the detector weigh injection risk. Content from
an attacker-controllable channel persisted to memory is materially higher risk
than the agent persisting its own reasoning.

Valid values (anything else raises `ValueError`):

| `source` | Meaning | Trust |
|---|---|---|
| `user_input` | Straight from the end user | untrusted |
| `retrieval` | A RAG / vector-store result | **attacker-controllable** |
| `tool_output` | A tool or API response | **attacker-controllable** |
| `external` | An external feed, webhook, queue | **attacker-controllable** |
| `llm_output` | The model's own generation | agent-derived |
| `agent_reasoning` | The agent's own scratch reasoning | agent-derived |

A write whose `source` is one of the three **attacker-controllable** channels
escalates a `MEMORY_POISONING` match to CRITICAL. Omitting `source` is allowed —
the detector still fires on a marker match (at HIGH, or CRITICAL if the poisoned
key is later read back), it just can't apply the source-based escalation.

---

## Automatic instrumentation

`dt.auto_instrument()` (or `dt.init(...)`) captures the memory channel for
frameworks that have a memory abstraction, no manual calls needed:

- **LangGraph** — `BaseStore.put`/`get`/`delete` (and the async variants).
  `put` → `memory_written`, `get` → `memory_read`, `delete` → `memory_cleared`.
  One patch covers every store backend (InMemoryStore, PostgresStore, …).
- **CrewAI** — short-term, long-term, and entity memory `save`/`search`/`reset`.
  `save` → `memory_written`, `search` → `memory_read`, `reset` → `memory_cleared`.

```python
import dunetrace as dt

dt.init(agent_id="support-agent")   # patches installed frameworks, including memory

# ... your existing LangGraph / CrewAI agent runs unchanged; memory ops are
#     now captured automatically whenever a dt.run() is active.
```

Framework memory APIs don't expose the **provenance** of a written value, so
auto-captured writes carry **no `source`**. That's deliberate: a guessed source
is worse than none for the detector's risk weighting. When you know the
provenance — you're the one persisting a retrieved document — prefer the manual
`run.memory_written(..., source="retrieval")` call so the detector can escalate.

Auto-captured values are capped at 4000 characters to keep events small; the
manual API is uncapped.

---

## What Dunetrace does with it

Server-side, the memory events are reconstructed into a typed `memory_events`
view on the run's state, and the `MEMORY_POISONING` detector scans each written
value for injection/override signatures. See the
[detector reference](detectors.md#memory_poisoning) for the marker set, the
HIGH-vs-CRITICAL rules, the `detectors.yml` tuning, and the calibration results.

Nothing else about your instrumentation changes — the memory channel rides the
same ingest path, wire format, and event pipeline as every other event type.

---

## Full example

See [`examples/memory_agent.py`](../examples/memory_agent.py) for a runnable
end-to-end example: an agent that fetches an untrusted document, persists a
poisoned summary to memory, reads it back, and the resulting CRITICAL
`MEMORY_POISONING` signal.
