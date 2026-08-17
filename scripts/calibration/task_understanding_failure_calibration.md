# TASK_UNDERSTANDING_FAILURE calibration

Calibration of the `TASK_UNDERSTANDING_FAILURE` run-level evaluator before ship.
Reproduce with `python scripts/calibrate_task_understanding_failure.py`
(`--reanalyze` re-sweeps cached scores without LLM calls).

## Method

108 labeled synthetic single-run samples scored once each by the real evaluator
(`gpt-4o-mini` via DeepEval GEval):

- **50 wrong-task** (positive): the response addresses a different task than the
  user asked (asked about pricing, answered about features; asked to schedule a
  meeting, explained scheduling best practices; asked for a refund status, quoted
  the refund policy; and so on).
- **50 right-task** (negative): the response is aimed at the task the user asked.
- **8 hard negatives** (negative), the cases that separate this evaluator from
  its neighbors and must NOT fire:
  - right task but only PARTIALLY done (that is TASK_COMPLETION's concern),
  - a genuinely AMBIGUOUS request where the agent picked a reasonable reading,
  - a MULTI-PART request where the agent answered every part.

`fired = score < threshold`. We sweep the threshold offline and measure the
false-positive rate (negatives that fire) and true-positive recall.

## Results

| Class | n | min | max | mean |
|---|---|---|---|---|
| wrong task | 50 | 0.00 | 0.21 | 0.13 |
| right task (incl. hard negatives) | 58 | 0.60 | 1.00 | 0.97 |

No overlap: the highest wrong-task score is 0.21, the lowest right-task score is
0.60, a 0.39 margin around the default 0.5 threshold. Every hard negative
(partial, ambiguous, multi-part) scored on the right-task side.

Threshold sweep:

| threshold | FP rate | TP recall |
|---|---|---|
| 0.30 - 0.60 | 0% | 100% |
| 0.65 | 2% | 100% |
| 0.70 | 3% | 100% |

## Decision

Ship at **threshold 0.5** (`fired = score < 0.5`), inside the 0.21 to 0.60 gap:
0% false positives, 100% recall, and the same cutoff the other evaluators use.
Note the margin is tighter than CONFUSION_LOOP's (some legitimate right-task
answers score as low as 0.60) because "right task, done differently" is a
subtler line than "same question re-asked"; the hard negatives confirm 0.5 still
separates cleanly.

## Honest caveats

- Synthetic calibration with clean class separation. Real traces are noisier;
  re-check the 0.5 threshold and the 15% FP bar against design-partner traces
  when available. Threshold is a config knob (`threshold=...`).
- The evaluator weighs `TOOLS_CALLED` only when a run actually has tools (GEval
  rejects that param with no tools); task understanding is judged primarily from
  the user's request vs. the response.
