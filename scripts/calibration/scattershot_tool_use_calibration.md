# SCATTERSHOT_TOOL_USE calibration

Date: 2026-08-19

## Goal

Choose conservative defaults for the two structural thresholds:

- `MAX_DISTINCT_TOOLS`: how many different tool names indicate a run is
  widening rather than converging.
- `MIN_TOTAL_CALLS`: how much sustained tool activity is required before the
  distinct count is meaningful.

The false-positive class is a legitimate multi-tool agent whose normal workflow
touches many tools once each. The second threshold exists specifically to keep
those one-pass pipelines below the fire line.

## Corpus

The deterministic corpus in `scripts/calibrate_scattershot_tool_use.py` has 36
labeled runs:

- 16 positives: scattershot research, coding, browser, data, and communication
  runs that keep adding unrelated approaches.
- 20 negatives: legitimate fixed pipelines, repeated use of a small tool set,
  and broad workflows that call each required tool once.

Two five-tool positives are intentionally included to measure the recall lost by
a precision-first six-tool threshold. The corpus uses the production detector
class and has no network, LLM, or third-party dependency.

## Sweep

The script sweeps distinct-tool thresholds 4–8 and minimum total calls up to 12.
Representative boundary results:

| Distinct tools | Total calls | False-positive rate | True-positive recall |
|---:|---:|---:|---:|
| 5 | 8 | 35% | 100% |
| 6 | 6 | 10% | 88% |
| 6 | 7 | 5% | 88% |
| **6** | **8** | **0%** | **88%** |
| 6 | 9 | 0% | 19% |
| 7 | 8 | 0% | 62% |
| 8 | 8 | 0% | 31% |

Run the full sweep with:

```bash
PYTHONPATH=packages/sdk-py python scripts/calibrate_scattershot_tool_use.py
```

## Recommendation

**Superseded — do not rely on the sweep above.** The "0/20 false positives,
87.5% recall" headline was an artifact of how the corpus was built, not a
measurement of the boundary:

- Only 2 of the 20 negatives reached 6+ distinct tools, and both had **zero**
  repeated calls. They were excluded solely by being one or two calls short of
  `MIN_TOTAL_CALLS`, not by any structural property.
- Every positive had exactly 8 or 9 calls, which is why recall collapsed
  88% → 19% when the total threshold moved 8 → 9.

The two classes were separated by a single integer at exactly the threshold
being recommended, so the sweep was scoring its own construction. The ETL
negative `[trigger, lookup, validate, transform, persist, notify, audit]` fires
the moment that pipeline gains one more stage or one retry.

The harness has been rebuilt: negatives now clear both floors outright (broad
8–12-tool pipelines, with and without retries), positives span 8–16 calls, and
the sweep adds a repeat-ratio axis. The `ship?` gate now requires recall ≥ 50%
as well as FP < 15%, because a gate on false positives alone will happily
recommend a configuration that never fires.

**Current result: no threshold triple clears both bars.** Repeat ratio does not
separate the classes — positives span 1.00–2.33, negatives 1.00–3.00, and the
negative median (1.40) is higher than the positive median (1.33). The best
precise configuration is 0% FP at 20% recall.

This is a conclusion about the heuristic, not the thresholds. Tool breadth plus
repetition does not identify scattershot behaviour. The promising axis is
non-convergence — a run that fanned out *and* never produced a final answer —
which this corpus cannot score, since it carries tool-name sequences only with
no `exit_reason` or per-call success data. Extending the corpus to carry those
is the next step.

`SCATTERSHOT_TOOL_USE` stays outside `LIVE_DETECTORS`.
