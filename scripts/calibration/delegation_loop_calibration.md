# DELEGATION_LOOP calibration

Calibration of the `DELEGATION_LOOP` detector before ship. Reproduce with
`python scripts/calibrate_delegation_loop.py` (`--write` regenerates the cached
scores JSON). Structural (deterministic) detector — no LLM, no API key. It runs
the real graph pipeline (ancestor chain → agent-delegation edges → DFS cycle
detection → detector) over a labeled corpus, then sweeps `MIN_LOOP_RUNS`.

## What it detects

Two or more agents delegating to each other in a cycle that keeps going around
instead of converging — A hands off to B, B hands back to A, A to B again, and so
on. The *run* graph (linked by `parent_run_id`) is a forest and can never cycle;
the cycle is in the *agent* dimension. The worker walks a run's parent chain,
derives the directed agent graph, and runs three-colour DFS cycle detection.

## The precision tension

A pathological delegation *loop* (A→B→A→B→…) and a legitimate iterative
*supervisor* exchange (A delegates, B returns, A delegates again, then the run
finishes) look **identical** in the chain — the only structural difference is how
many times it goes around. There is no content signal to lean on; the sole knob
is `MIN_LOOP_RUNS`, the number of chain runs that must be caught in a detected
cycle before firing.

So the negatives deliberately include short, legitimate supervisor / hand-back
exchanges (2–4 runs), and the sweep finds the lowest threshold that keeps them
below the fire line while still catching the sustained loops.

## Method

- **14 positives** — sustained loops: 2-agent oscillations (6–10 runs), 3-agent
  cycles (6–9 runs), plus two borderline short loops (4–5 runs) kept as positives
  so recall is measured honestly at every threshold.
- **14 negatives** — legitimate multi-agent patterns: deep linear hierarchies
  (all agents distinct), single hand-backs (A→B→A), a repeated-but-not-looping
  chain (B→C→B once), and — the hard negatives — two-iteration supervisor
  exchanges (A→B→A→B, 4 runs) that terminate.

Each sample is a root-first agent sequence run through the real graph pipeline.

## Results — threshold sweep

| `MIN_LOOP_RUNS` | FP rate | TP recall | ships (FP<15%) |
|---|---|---|---|
| 3 | 50% | 100% | no |
| **5** | **0%** | **93%** | **yes ← shipped** |
| 4 | 14% | 100% | yes |
| 6 | 0% | 86% | yes |
| 7 | 0% | 43% | yes |
| 8 | 0% | 43% | yes |

**Shipped: `MIN_LOOP_RUNS = 5`** (FP 0%, TP recall 93%).

## Decision

Threshold 4 reaches 100% recall but at 14% FP — and that 14% is **entirely** the
legitimate two-iteration supervisor exchanges (A→B→A→B, 4 runs), a common and
correct multi-agent pattern. Firing on those is genuinely wrong, so we take the
precision-first threshold **5**: it drops the FP to 0% and only gives up one
borderline 4-run positive (recall 93%). A loop that has only gone around twice is
genuinely ambiguous with legitimate iteration; requiring 2.5 round-trips (5 runs)
to fire is the honest boundary. This mirrors the precision-first choice made for
the SYCOPHANCY evaluator.

`CRITICAL_LOOP_RUNS = 7` escalates a sustained loop (≥7 runs, ~3.5 round-trips) to
CRITICAL — a runaway that is clearly not self-terminating.

## Known limitations (disclosed)

- **Needs `parent_run_id` along the chain.** Auto-threaded (Phase 2.1) for nested
  `dt.run()` calls on the same task or an asyncio child task. A sub-agent
  dispatched to a bare thread, or a framework that collapses a whole crew into a
  single run, produces no multi-run graph to walk and never fires. See
  `docs/multi-agent.md`.
- **Fires per run once the loop is sustained.** Each run past the threshold in an
  ongoing loop gets its own signal (deduped per run by failure type). This is
  intentional — the loop is still burning tokens — but means a 10-deep loop
  produces several signals, grouped by the alert layer.

## Ship status

**Live** — listed in `LIVE_DETECTORS`
(`services/detector/detector_svc/db.py`). Shipped shadow first, matching the
SILENT_TRUNCATION / MODEL_FALLBACK_DRIFT precedent, and promoted once real
multi-agent traffic confirmed the FP rate held outside the calibration corpus.
To reverse, remove it from `LIVE_DETECTORS`; `min_loop_runs` in `detectors.yml`
is the first knob to reach for if it is noisier than expected.
