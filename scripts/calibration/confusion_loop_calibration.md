# CONFUSION_LOOP calibration

Calibration of the `CONFUSION_LOOP` semantic evaluator before ship. Reproduce
with `python scripts/calibrate_confusion_loop.py` (scores cache to
`confusion_loop_scores.json`; re-sweep offline with `--reanalyze`).

## Method

108 labeled synthetic conversations scored once each by the real evaluator
(`gpt-4o-mini` via DeepEval's ConversationalGEval):

- **50 confusion loops** (positive): the user asks a question, the agent gives a
  non-resolving answer, and the user re-asks the same underlying question two or
  three times ("no, I meant...", "that's not what I asked", re-phrasing).
- **50 progressing conversations** (negative): the agent resolves each request
  and the user moves on to a new, genuinely different question.
- **8 hard negatives** (negative): conversations that superficially look like a
  loop (a "no, I meant..." correction, a re-reference to the same topic) but are
  resolved and move on. These are where a lazily-worded criteria false-fires.

The evaluator returns `confidence = 1 - score`. A signal fires when
`score < threshold`. We sweep the threshold offline and measure, per threshold,
the false-positive rate (negatives that fire) and true-positive recall
(loops that fire).

## Results

Score distribution (raw GEval score, higher = healthier):

| Class | n | min | max | mean |
|---|---|---|---|---|
| confusion loop | 50 | 0.00 | 0.20 | 0.10 |
| progressing (incl. hard negatives) | 58 | 0.83 | 1.00 | 0.99 |

The classes do not overlap. The highest-scoring loop is 0.20; the lowest-scoring
negative is 0.83, a 0.63 margin straddling the default 0.5 threshold.

Threshold sweep:

| threshold | FP rate | TP recall |
|---|---|---|
| 0.30 - 0.70 | 0% | 100% |

Every hard negative scored 0.83 or higher, so the criteria does not confuse a
one-time correction with a loop.

## Decision

Ship at the package-standard **threshold 0.5** (`fired = score < 0.5`). It sits
in the middle of the 0.20 to 0.83 gap, gives 0% false positives (well under the
15% bar) and 100% recall on this set, and matches the cutoff the other
evaluators use so callers need no per-evaluator logic.

## Honest caveats

- This is **synthetic** calibration. The class separation is clean because the
  synthetic loops and non-loops are unambiguous. Real conversations are noisier;
  the 0.5 threshold and the 15% FP bar should be re-checked against
  design-partner traces when available, and the threshold is a config knob
  (`ConfusionLoopEvaluator(threshold=...)`) if real FP exceeds the bar.
- The evaluator does not gate on conversation length. "Requires a conversation"
  (does not fire on a single run) is enforced upstream by the worker's
  `MIN_CONVERSATION_RUNS` sampling floor, not by the evaluator itself.
