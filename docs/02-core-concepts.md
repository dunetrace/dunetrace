# Core Concepts

Understanding these six concepts is enough to understand everything Dunetrace does.

---

## Agent Event

The atomic unit of data. Every meaningful thing your agent does — starting a run, calling a tool, invoking an LLM, completing — emits one event.

```python
AgentEvent(
    event_type   = "tool.called",
    run_id       = "run-f4a9b2c1",
    agent_id     = "research-agent",
    agent_version= "a7f3d9b2",          # deterministic hash of config
    step_index   = 3,                   # monotonically increasing
    payload      = {
        "tool_name": "web_search",
        "args_hash": "e3b0c442",        # SHA-256 of args, never raw args
    }
)
```

**There are eight event types:**

| Event Type | Emitted When |
|---|---|
| `run.started` | Agent begins processing a user input |
| `run.completed` | Agent produces a final answer |
| `run.errored` | Agent raises an exception |
| `llm.called` | A model inference is requested |
| `llm.responded` | Model inference completes |
| `tool.called` | Agent invokes a tool |
| `tool.responded` | Tool returns a result |
| `retrieval.completed` | RAG retrieval returns results |

---

## Run

A run is the complete lifecycle of one agent response — from receiving user input to producing a final answer (or erroring). It has a unique `run_id`, a start time, and an end time.

Runs contain an ordered sequence of events. The detector worker reconstructs this sequence from the database to produce a `RunState` — the data structure detectors actually operate on.

A run can have zero or more **failure signals**.

---

## RunState

The in-memory representation of everything that happened in a run, built by replaying its events. Detectors receive a `RunState`, not raw events.

```python
@dataclass
class RunState:
    run_id:          str
    agent_id:        str
    agent_version:   str
    available_tools: List[str]
    tool_calls:      List[ToolCall]          # ordered history
    retrieval_results: List[RetrievalResult]
    events:          List[AgentEvent]        # all events
    input_text_hash: str
    completed:       bool
    exit_reason:     Optional[str]
```

---

## Failure Signal

The output of a detector. A `FailureSignal` says: "I detected failure type X in run Y with Z% confidence."

```python
@dataclass
class FailureSignal:
    failure_type:  FailureType     # e.g. TOOL_LOOP
    severity:      Severity        # CRITICAL | HIGH | MEDIUM | LOW
    run_id:        str
    agent_id:      str
    agent_version: str
    step_index:    int             # where in the run it was detected
    confidence:    float           # 0.0–1.0
    evidence:      dict            # detector-specific raw evidence
    shadow:        bool            # True = stored but never alerted
    alerted:       bool            # True = Slack/webhook delivered
```

Signals are stored in Postgres and never deleted. This gives you a complete history of every failure ever detected, which feeds into precision tracking and detector graduation.

---

## Explanation

The explain layer converts a `FailureSignal` into an `Explanation` — a human-readable breakdown with a concrete fix. Generated in under 1ms using deterministic templates (no LLM calls).

```python
@dataclass
class Explanation:
    failure_type:     str
    severity:         str
    title:            str           # one-line summary
    what:             str           # plain-English description
    why_it_matters:   str           # business impact
    evidence_summary: str           # interpolated evidence values
    suggested_fixes:  List[CodeFix] # 1–3 copy-pasteable fixes
    confidence:       float
```

---

## Agent Version

A deterministic hash of your agent's configuration: `SHA-256(system_prompt + model + sorted(tools))[:8]`.

Every time you change your system prompt, swap models, or add/remove tools, the version changes automatically. This matters for two reasons:

1. Signals are tagged with the agent version that produced them, so you can see whether a fix actually improved things.
2. The detector worker can attribute changes in signal rates to specific deploys.

```python
# Same config always produces the same version
version = agent_version(
    system_prompt="You are a research assistant...",
    model="gpt-4o-mini",
    tools=["web_search", "calculator"],
)
# → "a7f3d9b2"
```

---

## Shadow Mode

Every detector starts in shadow mode (`shadow=true`). Shadow signals are:

- Stored in the database ✓
- Visible in the API and dashboard ✓  
- **Not sent to Slack or webhook** ✗

Shadow mode lets you measure a detector's precision on real traffic before committing to alerting. Once you've spot-checked the signals and confirmed they represent real failures (target: ≥80% true positive rate), you graduate the detector by adding it to `LIVE_DETECTORS` in the detector worker config.

This prevents the most common failure mode of observability tools: noisy alerts that get ignored or disabled.
