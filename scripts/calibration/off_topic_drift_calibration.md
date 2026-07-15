# OFF_TOPIC_DRIFT calibration

Calibration of the `OFF_TOPIC_DRIFT` run-level evaluator before ship. Reproduce
with `python scripts/calibrate_off_topic_drift.py` (`--reanalyze` re-sweeps
cached scores without LLM calls).

## Method

100 labeled synthetic single-run samples scored once each by the real evaluator
(`gpt-4o-mini` via DeepEval GEval):

- **50 drift** (positive): the response starts on the user's question and then
  wanders off it (into unrelated features, roadmap/marketing language, or a
  tangent).
- **50 on-topic** (negative), from two sub-classes cycled evenly:
  - a focused answer that stays on the question, and
  - **broad / multi-faceted** questions answered thoroughly, covering many
    RELATED aspects. This is the FP-risk class: a thorough answer touches many
    points, which can look like wandering but is not drift.

`fired = score < threshold`. We sweep the threshold and measure FP (on-topic
that fires) and TP recall (drift that fires).

## Results

| Class | n | min | max | mean |
|---|---|---|---|---|
| drift | 50 | 0.20 | 0.44 | 0.33 |
| on-topic (incl. broad multi-aspect) | 50 | 0.82 | 1.00 | 0.97 |

No overlap: the highest drift score is 0.44, the lowest on-topic score is 0.82,
a 0.38 margin around the default 0.5 threshold. Every broad / multi-faceted
answer scored on the on-topic side — covering related aspects of a broad
question was not mistaken for drift.

Threshold sweep:

| threshold | FP rate | TP recall |
|---|---|---|
| 0.40 | 0% | 90% |
| 0.45 - 0.70 | 0% | 100% |

## Decision

Ship at the package-standard **threshold 0.5** (`fired = score < 0.5`), inside
the 0.44 to 0.82 gap: 0% false positives, 100% recall, and the same cutoff the
other run-level evaluators use.

## Honest caveats

- Synthetic calibration with clean separation. Real traces are noisier — the
  drift in the synthetic positives is fairly overt. Re-check against
  design-partner traces; subtler drift may score higher and need a threshold
  revisit. Threshold is a constructor knob.
- This is a single-run evaluator; the "LOW single / MEDIUM repeated" severity
  framing is emergent from the pipeline's confidence-to-severity mapping, not
  set here.
