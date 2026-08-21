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

Ship `MAX_DISTINCT_TOOLS = 6` and `MIN_TOTAL_CALLS = 8`. This is the
highest-recall threshold pair under the `<15%` false-positive target after
preferring the lower false-positive rate among equal-recall pairs. It produced
0/20 false positives and detected 14/16 positives (87.5% recall) on this corpus.

This synthetic calibration cannot represent every specialized orchestration
pipeline. `SCATTERSHOT_TOOL_USE` therefore remains outside `LIVE_DETECTORS` and
ships shadowed until its precision is validated on real traffic.
