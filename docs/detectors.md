# Detectors

Dunetrace runs 17 structural detectors against every completed agent run — 16 Tier 1 detectors plus prompt injection signal detection. All thresholds are configurable i.e. no code changes required.

---

## What each detector catches

| Detector | What it catches | Severity |
|---|---|---|
| `SLOW_STEP` | Step duration exceeds threshold — 2× P75 baseline ¹ or static fallback (tool >15s, LLM >30s) | MEDIUM/HIGH |
| `TOOL_AVOIDANCE` | Final answer given without calling available tools | MEDIUM |
| `GOAL_ABANDONMENT` | Tool use stops, then ≥4 consecutive LLM calls with no exit | MEDIUM |
| `RAG_EMPTY_RETRIEVAL` | Retrieval returned 0 results or relevance <0.3, but agent answered | MEDIUM |
| `CONTEXT_BLOAT` | Prompt tokens grow beyond 2× P75 baseline ¹ or static fallback (3× from first to last call) | MEDIUM |
| `STEP_COUNT_INFLATION` | Run used >2× the P75 step count for this agent ¹ | MEDIUM |
| `FIRST_STEP_FAILURE` | Error or empty output at step ≤2 | MEDIUM |
| `REASONING_STALL` | LLM:tool-call ratio exceeds 2× P75 baseline ¹ or static fallback (≥4×) — MEDIUM if run finished, HIGH if it stalled | MEDIUM/HIGH |
| `COST_SPIKE` | Total token consumption exceeds 3× P75 baseline ¹ or static fallback (>50,000 tokens) | MEDIUM |
| `SESSION_LATENCY` | Total wall-clock run duration exceeds 3× P75 baseline ¹ or static fallback (>5 min) | MEDIUM |
| `TOOL_LOOP` | Same tool called ≥3× in a 5-tool-call window | HIGH |
| `TOOL_THRASHING` | Agent alternates between exactly two tools | HIGH |
| `LLM_TRUNCATION_LOOP` | `finish_reason=length` fires ≥2 times | HIGH |
| `RETRY_STORM` | Same tool fails 3+ times in a row without subsequent recovery | HIGH |
| `EMPTY_LLM_RESPONSE` | Model returned zero-length output with `finish_reason=stop` | HIGH |
| `CASCADING_TOOL_FAILURE` | 3+ consecutive failures across 2+ distinct tools | HIGH |
| `PROMPT_INJECTION_SIGNAL` | Input matches known injection / jailbreak patterns | CRITICAL |

¹ **Six detectors use per-agent learned baselines.** `STEP_COUNT_INFLATION`, `SLOW_STEP`, `CONTEXT_BLOAT`, `REASONING_STALL`, `COST_SPIKE`, and `SESSION_LATENCY` compute a P75 from the last 50 successfully completed runs (errored runs excluded) for the same `agent_id` + `agent_version` pair. The threshold fires at **2× that baseline** (3× for COST_SPIKE and SESSION_LATENCY). Each detector falls back to its static threshold until at least **20** historical runs exist — below that the P75 estimate is too sensitive to individual outliers to be useful — then switches to the adaptive baseline automatically. Tune the multiplier per agent category with `inflation_factor` in `detectors.yml`.

---

## Tuning thresholds

Edit `detectors.yml` in the repo root. No code change or rebuild needed.

```yaml
default:
  tool_loop:
    threshold: 2        # lower = catch loops sooner
  context_bloat:
    growth_factor: 4.0  # static fallback when no baseline; raise for context-heavy agents
    inflation_factor: 2.0  # P75-baseline multiplier; raise for high-variance agents

  slow_step:
    inflation_factor: 2.0  # fire when step > P75_latency × this

  reasoning_stall:
    ratio_threshold: 4.0   # static fallback
    inflation_factor: 2.0  # fire when LLM:tool ratio > P75_ratio × this

  step_count_inflation:
    inflation_factor: 1.5  # tighter threshold for predictable coding agents

web-research:
  tool_loop:
    threshold: 5           # search agents legitimately repeat queries across pages
  reasoning_stall:
    inflation_factor: 3.0  # research agents naturally reason more before acting
```

Named sections match the `agent_id` and inherit from `default`, overriding only what you specify. Restart the detector to apply:

```bash
docker compose restart detector
```

---

## Shadow mode

Every signal is stored with a `shadow` flag. The alerts worker only delivers signals where `shadow = false`.

All 17 built-in detectors are live (`shadow = false`) by default. Custom detectors start in shadow mode — signals are stored and visible in the dashboard, but no Slack/webhook alert fires — until you add them to `LIVE_DETECTORS` in `services/detector/detector_svc/db.py`:

```python
LIVE_DETECTORS: set[str] = {
    "TOOL_LOOP",
    "YOUR_NEW_DETECTOR",   # promote once precision > 80%
    ...
}
```

This lets you validate a new detector against real traffic before it pages anyone.

### Shadow signals in the dashboard

The **Alerts** page surfaces shadow signals in a dedicated section below the live alert groups. Shadow signals are rendered with a dashed border, reduced opacity, and a `SHADOW` badge so they're visually distinct from alerted signals. The section only appears when at least one shadow signal exists.

The API exposes shadow signals via `?include_shadow=true` on the signals endpoint. Each signal object includes a `shadow: bool` field:

```bash
curl "http://localhost:8002/v1/agents/my-agent/signals?include_shadow=true" \
  -H "Authorization: Bearer dt_dev_test"
```

Use this to evaluate detector precision before graduating — compare shadow signal rate against run outcomes manually, then promote to `LIVE_DETECTORS` when confidence is high enough.

---

## Confidence scoring

**Dynamic confidence** — count and ratio detectors scale confidence based on how far the observation exceeds the trigger threshold:

```
confidence = min(1.0, 0.5 + (observed / threshold − 1.0) × 0.4)
```

Barely at the threshold → 0.5. Twice the threshold → 0.9. Beyond 3.25× → capped at 1.0. Binary and pattern detectors (TOOL_THRASHING, TOOL_AVOIDANCE, RAG_EMPTY_RETRIEVAL, EMPTY_LLM_RESPONSE, FIRST_STEP_FAILURE) retain static values because there is no meaningful "degree of excess" to measure.

When multiple independent signals fire on the same run, two additional post-processing steps run before signals are written to the database:

**Co-occurrence boost** — further reduces false positives without touching individual thresholds:

| Co-firing signals | Multiplier |
|---|---|
| 1 | ×1.0 (no change) |
| 2 | ×1.15 |
| 3 | ×1.30 |
| 4+ | ×1.40 |

All signals in the run receive the same multiplier (capped at 1.0). The `co_signal_count` field on each `failure_signals` row records how many signals co-fired. The dashboard shows an amber **"confidence boosted · N co-occurring signals"** badge on boosted signal cards.

**Hard overrides** — deterministic cuts for structurally unambiguous failures that bypass continuous scoring:

| Condition | Result |
|---|---|
| Same tool called ≥8× with ≥90% identical args | All signals → CRITICAL at confidence 0.98 |
| ≥5 consecutive tool failures (tail of run) | All signals → HIGH at confidence 0.95 |

Hard overrides fire before the co-occurrence boost, so the final confidence on a CRITICAL override is exactly 0.98 regardless of base scores.

---

## How detection works

The detector worker polls Postgres every 5 seconds for completed or stalled runs. For each run it:

1. Fetches all events for that run from the `events` table
2. Replays them into a `RunState` (tool calls, LLM calls, retrievals, durations)
3. Fetches per-agent P75 baselines from run history in parallel (step count, tool latency, LLM latency, token growth, LLM:tool ratio, total tokens, session duration) and attaches them to the `RunState`
4. Runs all 16 Tier 1 detectors against the `RunState`
5. Applies confidence boosting: co-occurrence multiplier + hard overrides
6. Writes any triggered `FailureSignal` rows to `failure_signals`
7. Marks the run as processed in `processed_runs`

Detection adds zero latency to the agent — it runs entirely after the run completes.

**Self-correction suppression** — `RETRY_STORM` checks whether the tool that hit the failure streak subsequently succeeded in the same run. If it did, the agent recovered on its own and no signal is emitted.

**REASONING_STALL severity** — fires MEDIUM when the run finished with a `final_answer` (CoT-heavy but converged) and HIGH when the run stalled without ever converging (the ratio is likely the reason for failure).
