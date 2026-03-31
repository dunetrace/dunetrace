# Detectors

Dunetrace runs 15 structural detectors against every completed agent run. All thresholds are configurable — no code changes required.

---

## What each detector catches

| Detector | What it catches | Severity |
|---|---|---|
| `SLOW_STEP` | Tool call >15s or LLM call >30s | MEDIUM/HIGH |
| `TOOL_AVOIDANCE` | Final answer given without calling available tools | MEDIUM |
| `GOAL_ABANDONMENT` | Tool use stops, then ≥4 consecutive LLM calls with no exit | MEDIUM |
| `RAG_EMPTY_RETRIEVAL` | Retrieval returned 0 results or relevance <0.3, but agent answered | MEDIUM |
| `CONTEXT_BLOAT` | Prompt tokens grow 3× from first to last LLM call | MEDIUM |
| `STEP_COUNT_INFLATION` | Run used >2× the P75 step count for this agent ¹ | MEDIUM |
| `FIRST_STEP_FAILURE` | Error or empty output at step ≤2 | MEDIUM |
| `REASONING_STALL` | LLM:tool-call ratio ≥4× — agent reasoning without acting | MEDIUM |
| `TOOL_LOOP` | Same tool called ≥3× in a 5-tool-call window | HIGH |
| `TOOL_THRASHING` | Agent alternates between exactly two tools | HIGH |
| `LLM_TRUNCATION_LOOP` | `finish_reason=length` fires ≥2 times | HIGH |
| `RETRY_STORM` | Same tool fails 3+ times in a row | HIGH |
| `EMPTY_LLM_RESPONSE` | Model returned zero-length output with `finish_reason=stop` | HIGH |
| `CASCADING_TOOL_FAILURE` | 3+ consecutive failures across 2+ distinct tools | HIGH |
| `PROMPT_INJECTION_SIGNAL` | Input matches known injection / jailbreak patterns | CRITICAL |

¹ **`STEP_COUNT_INFLATION` requires a warm baseline.** P75 is computed from the last 50 successfully completed runs (errored runs excluded) for the same `agent_id` + `agent_version` pair. The detector produces no signal until at least 10 such runs exist, then activates automatically.

---

## Tuning thresholds

Edit `detectors.yml` in the repo root. No code change or rebuild needed.

```yaml
default:
  tool_loop:
    threshold: 2        # lower = catch loops sooner
  context_bloat:
    growth_factor: 4.0  # raise for agents that intentionally accumulate context

web-research:
  tool_loop:
    threshold: 5        # search agents legitimately repeat queries across pages
```

Named sections match the `agent_id` and inherit from `default`, overriding only what you specify. Restart the detector to apply:

```bash
docker compose restart detector
```

---

## Shadow mode

Every signal is stored with a `shadow` flag. The alerts worker only delivers signals where `shadow = false`.

All 15 built-in detectors are live (`shadow = false`) by default. Custom detectors start in shadow mode — signals are stored and visible in the dashboard, but no Slack/webhook alert fires — until you add them to `LIVE_DETECTORS` in `services/detector/detector_svc/db.py`:

```python
LIVE_DETECTORS: set[str] = {
    "TOOL_LOOP",
    "YOUR_NEW_DETECTOR",   # promote once precision > 80%
    ...
}
```

This lets you validate a new detector against real traffic before it pages anyone.

---

## How detection works

The detector worker polls Postgres every 5 seconds for completed or stalled runs. For each run it:

1. Fetches all events for that run from the `events` table
2. Replays them into a `RunState` (tool calls, LLM calls, retrievals, durations)
3. Runs all 15 detectors against the `RunState`
4. Writes any triggered `FailureSignal` rows to `failure_signals`
5. Marks the run as processed in `processed_runs`

Detection adds zero latency to the agent — it runs entirely after the run completes.
