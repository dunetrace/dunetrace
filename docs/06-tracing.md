# How Tracing Works

Dunetrace's tracing model is designed around one constraint: **the agent thread must never wait for observability infrastructure**.

---

## The Hot Path

When your agent calls `run.tool_called(...)`, this is the entire execution path:

```
run.tool_called("web_search", {"query": "..."})
    │
    └─▶ RunContext._emit(AgentEvent(...))
            │
            └─▶ Dunetrace._emit(event)
                    │
                    └─▶ self._buffer.append(event)   ← O(1), no lock in CPython
                                                        returns in <1μs
```

That's it. Your agent thread does a single `deque.append()` and continues. There is no I/O, no network call, no lock acquisition on the hot path.

---

## The Drain Thread

A background daemon thread (`dunetrace-drain`) runs independently of your agent. It:

1. Wakes up every 200ms (or when there are events to ship)
2. Pops up to 100 events from the ring buffer
3. POSTs them as a JSON batch to the ingest API
4. Sleeps until the next interval

```python
def _drain_loop(self) -> None:
    while not self._stop_evt.is_set():
        batch = []
        while self._buffer and len(batch) < 100:
            batch.append(self._buffer.popleft())
        if batch:
            self._ship(batch)
        else:
            time.sleep(self._flush_interval)
    # Final flush on shutdown
    ...
```

The drain thread is a **daemon thread** — it does not prevent process exit. Call `dt.shutdown()` to flush remaining events before your process terminates.

---

## The Ring Buffer

The buffer is a `collections.deque(maxlen=10_000)`. This gives it two important properties:

**It never blocks.** `deque.append()` in CPython is O(1) and GIL-protected. No explicit lock is needed.

**It drops oldest events under backpressure.** If the ingest API is down and the buffer fills up, new events push out the oldest ones. This is intentional — losing old events is better than blocking your agent.

At 200ms flush intervals shipping 100 events per batch, the buffer handles up to ~500 events/second before it fills. Most agents emit far fewer than that.

---

## What Gets Sent

A batch POST looks like this:

```json
{
  "api_key":  "dt_live_...",
  "agent_id": "research-agent",
  "events": [
    {
      "event_type":    "tool.called",
      "run_id":        "f4a9b2c1-...",
      "agent_id":      "research-agent",
      "agent_version": "a7f3d9b2",
      "step_index":    3,
      "timestamp":     1771632965.239,
      "payload": {
        "tool_name": "web_search",
        "args_hash": "e3b0c44298fc"    ← SHA-256 of args, truncated
      }
    }
  ]
}
```

**Never sent:** raw prompts, raw tool arguments, raw outputs, user messages, system prompts. Every content field is hashed.

**Always sent in plain text:** event types, tool names, model names, step indices, timestamps, token counts, output lengths.

---

## Content Hashing

The privacy model is simple: hash everything that could be sensitive.

```python
def hash_content(text: str) -> str:
    """SHA-256, truncated to 16 hex chars."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]
```

The hash is used for **structural detection** — the detectors need to know if the same content appeared twice (loop detection), not what the content was. A 16-character SHA-256 prefix has a collision probability of ~1 in 18 quintillion, which is sufficient.

---

## Step Index

Every event in a run has a `step_index` — a monotonically increasing integer. Steps are assigned by the `RunContext` and start at 0.

```
step 0  →  run.started
step 1  →  llm.called
step 2  →  llm.responded
step 3  →  tool.called      (step increments before emitting)
step 4  →  tool.responded
step 5  →  run.completed
```

Step indices are what the timeline visualization renders on the x-axis, and what the `FailureSignal.step_index` refers to when pinpointing where in a run a failure was detected.

---

## Agent Versioning

The `agent_version` is a deterministic fingerprint of your agent's configuration, computed once when the context manager is entered:

```python
def agent_version(system_prompt: str, model: str, tools: List[str]) -> str:
    content = system_prompt + model + "".join(sorted(tools))
    return hashlib.sha256(content.encode()).hexdigest()[:8]
```

Every event in a run is stamped with the version hash. This means:

- You can query signals by version to see if a prompt change fixed a problem
- The detector never compares runs across different configurations
- If you A/B test two system prompts, their signals are cleanly separated

---

## Run Reconstruction

The detector worker never receives events in real-time. It polls the database for completed runs and reconstructs them:

```python
events = await fetch_run_events(run_id)   # ordered by step_index
state  = RunState(run_id=run_id, ...)

for event in events:
    if event.event_type == "tool.called":
        state.tool_calls.append(ToolCall(
            tool_name  = event.payload["tool_name"],
            args_hash  = event.payload["args_hash"],
            step_index = event.step_index,
            timestamp  = event.timestamp,
        ))
    elif event.event_type == "retrieval.completed":
        state.retrieval_results.append(...)
    # ... etc
```

The reconstructed `RunState` is what detectors operate on. They never touch raw events.

---

## Multi-Agent Tracing

Sub-agents are linked to their parent via `parent_run_id`:

```python
with dt.run(user_input) as parent_run:
    # ... orchestration logic ...
    with dt.run(sub_task, parent_run_id=parent_run.run_id) as child_run:
        # child events are linked to parent in the events table
        ...
```

The `parent_run_id` field is stored on every event in the child run. The customer API exposes this so the dashboard can reconstruct the full call tree for multi-agent workflows.

---

## Overhead Summary

| Operation | Time | Notes |
|---|---|---|
| `_emit()` hot path | <1μs | deque append, no I/O |
| Event serialisation | ~10μs | In drain thread, not agent thread |
| Batch HTTP POST | ~20ms | In drain thread, every 200ms |
| Total agent overhead | <500μs | Per run, worst case |

The 500μs figure is a conservative upper bound accounting for the deque append across all events in a typical 10-step run. The actual measured overhead is closer to 50μs.
