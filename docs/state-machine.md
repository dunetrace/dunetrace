# Agent state machine

Agents don't emit "states" — they emit paired events (an LLM call and its
response, a tool call and its result). Dunetrace reconstructs the **states** an
agent moved through from those events, so you can see where a run spent its
time and which runs behaved anomalously. Nothing new is instrumented; this is a
derived view of events runs already send.

## The states

| State | The agent is… | Reconstructed from |
|---|---|---|
| `thinking` | waiting on an LLM | `llm.called` → `llm.responded` |
| `acting` | running a tool | `tool.called` → `tool.responded` |
| `retrieving` | doing a RAG lookup | `retrieval.called` → `retrieval.responded` |
| `waiting_approval` | blocked on a [human approval](approvals.md) | `approval.requested` → `approval.granted`/`denied`/`timeout` |

A state spans from its opening event to its matching closing event. Point events
(voice activity, external signals, policy triggers) aren't states. Reconstruction
is forgiving: a run that crashed mid-tool still shows the `acting` state (marked
as not closed), running to the run's end.

## Per-run timeline

Open any run and select the **State machine** tab. Each segment is one state
the run occupied, with its duration; the bar length is proportional to the
longest segment. A summary line shows total time per state. Segments marked `*`
never closed — the run ended, or the response event is missing.

Behind it:

```
GET /v1/runs/{run_id}/states
```

returns `{ "segments": [...], "summary": { "total_ms", "by_state", "segment_count" } }`,
reconstructed on demand from the run's events. Each segment carries
`state`, `label` (the tool/model name), `step_index`, `start_ts`, `end_ts`,
`duration_ms`, and `closed`.

## Cross-run analytics

Open an agent and select the **States** tab (7 / 30 / 90-day window). It shows,
for each state:

- **average** and **p95** time per run,
- how many runs entered that state,
- **outlier runs** — runs that spent anomalously long in a state (more than 3×
  that state's median, with the median above a floor so a normally near-zero
  state doesn't flag every run on noise). Each outlier links to its run detail.

Behind it:

```
GET /v1/agents/{agent_id}/state-analytics?window=30
```

returns per-state aggregates (`avg_ms`, `p50_ms`, `p95_ms`, `total_ms`,
`run_count`), a zero-filled daily `trend` per state, and the `outliers` list.

### How the analytics stay cheap

The per-run timeline reconstructs from raw events every time (so it works even
for a run the pipeline hasn't finished processing). The cross-run analytics
don't — that would mean re-reading thousands of runs' events per query. Instead
the **detector worker precomputes** per-run, per-state time totals into a
`run_state_metrics` table as each run completes, and the analytics endpoint
aggregates those small rows. The precompute runs in its own error boundary: a
bug in it never blocks failure detection.

## What this is not

- **Not new instrumentation.** States are derived from the events your agent
  already sends. An event Dunetrace never sees isn't part of the reconstruction.
- **Not a workflow definition.** This reflects what a run *did*, not a state
  machine you declare up front. Dunetrace doesn't constrain your agent to these
  states — it observes them.
- **Not cross-org.** Analytics compare an agent against its own history, never
  another org's — same as [performance trends](dashboard.md).

## See also

- [Approvals](approvals.md) — the source of the `waiting_approval` state
- [Policies](policies.md)
- [Dashboard](dashboard.md)
