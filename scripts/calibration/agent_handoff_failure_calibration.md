# AGENT_HANDOFF_FAILURE calibration

Calibration of the `AGENT_HANDOFF_FAILURE` detector. Reproduce with
`python scripts/calibrate_agent_handoff_failure.py` (`--write` regenerates the
scores JSON). Structural (deterministic) detector — no LLM, no API key.

## What it detects

A handoff tool (name ends in `_agent`, or starts with `delegate_` / `handoff_` /
`transfer_to_`, and not in the `excluded_tool_names` stop-list) that reports
`success=False`, or reports `success=True` but returns an empty/terse payload
(`done`/`ok`/`complete`/…) or an output shorter than `min_output_length`
(default 10). Single-run, tool-output-based — distinct from `HANDOFF_CONTEXT_LOSS`
(cross-run context comparison).

## Method

23 labeled handoff/tool-call samples run through the real detector:

- **12 positives** — handoff tools that failed or returned an empty/short payload.
- **11 negatives** — legitimate handoffs with a real payload, ordinary non-handoff
  tools, and the two **convention-collision** cases: `transfer_funds` (a money
  transfer that starts with `transfer_` but not `transfer_to_`) and `user_agent`
  (ends in `_agent` but is a known non-handoff).

## Results

| Metric | Value | Bar |
|---|---|---|
| Recall | **100%** (12/12) | — |
| False-positive rate | **0%** (0/11) | < 15% for live |
| Precision | **100%** | — |

**Verdict: live-ready** (calibration-clean).

## The tightening

An earlier version of the detector matched a bare `transfer_` prefix and had no
stop-list, which produced an 18% FP: `transfer_funds` and `user_agent` both
matched. Two changes fixed it with no recall loss:

1. **`transfer_` → `transfer_to_`** — the OpenAI Swarm / Agents SDK handoff
   convention is `transfer_to_<agent>`, so `transfer_to_` catches real handoffs
   (`transfer_to_billing`, `transfer_to_support`) while excluding a
   `transfer_funds` money-transfer tool.
2. **`excluded_tool_names` stop-list** (default `{user_agent}`) — a small,
   YAML-extensible allowlist for names that match a pattern but are known
   non-handoffs.

Both `handoff_patterns` and `excluded_tool_names` are tunable via `detectors.yml`
so operators can extend them as real traffic surfaces new collisions.

## Ship status

Shadow by default (not in `LIVE_DETECTORS`), matching the convention that new
detectors gather real-traffic data before promotion — even calibration-clean
ones. A strong candidate for promotion once shadow traffic confirms the 0% FP
holds outside the corpus.
