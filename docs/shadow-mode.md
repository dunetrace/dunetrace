# Shadow Mode

Shadow mode lets you validate detector precision on real production traffic before any alerts fire. It prevents false alarms from eroding trust in the alert system.

## How it works

While `shadow=true`, signals are:

- **Stored** in the database — queryable via the API
- **Counted** on the dashboard — you can measure precision
- **Never delivered** — no Slack/webhook alerts

This lets you answer: *"Of the last 100 TOOL_LOOP signals on real traffic, how many were actual problems vs. false positives?"* before committing to waking anyone up.

## Graduating a detector

The detector worker's `db.py` controls which detectors are live:

```python
LIVE_DETECTORS: set = set()  # empty = all shadow
```

Add a detector here once you've validated precision > 80% on real traffic:

```python
LIVE_DETECTORS = {"TOOL_LOOP", "PROMPT_INJECTION_SIGNAL"}
```

`TOOL_LOOP` and `PROMPT_INJECTION` are good first candidates — they have tight structural signals with naturally high precision. `TOOL_AVOIDANCE` is trickier because sometimes an agent legitimately skips tools.

## Recommended workflow

1. Instrument the agent and run in full shadow mode for one week
2. Query what's firing:
   ```sql
   SELECT failure_type, COUNT(*)
   FROM failure_signals
   WHERE agent_id = 'your-agent'
   GROUP BY failure_type;
   ```
3. Spot-check 10–20 signals of each type — were they real failures?
4. Graduate detectors with clean precision; keep the rest in shadow
5. Only then do engineers start receiving pages

## Why this matters

Every APM tool (New Relic, Datadog in their early days) learned this the hard way: one false alarm that pages an engineer at 2am, finds nothing wrong, and they turn off alerts entirely. Shadow mode exists to build trust before going live.
