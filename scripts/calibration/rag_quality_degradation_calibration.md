# RAG quality degradation: synthetic sensitivity assessment

Run `PYTHONPATH=packages/sdk-py python scripts/calibrate_rag_quality_degradation.py --report scripts/calibration/rag_quality_degradation_calibration.md`.
No network, models, or infrastructure are required.

## Fixed boundary corpus

15 hand-labeled traces: TP 4, FP 2, FN 1, TN 8; recall 80.0%, false-positive rate 20.0%, precision 66.7%.

The labels describe intended query scenarios, not information available to the detector. Both a legitimate intent change and a three-call benign mixed-query trace fire. A noisy but flat three-call high-score trace does not. A gradual decline below the default threshold is missed. These fixed hand-assigned scores are boundary checks, not a representative scorer population or production precision estimate.

## Stationary benign-distribution stress test

Seed 20261001; 2000 independent traces per cell; default threshold 0.3. Each cell invokes the actual detector on IID scores with no temporal degradation. All signals here are false positives by construction. The mixed model changes query difficulty independently at each retrieval, not over time.

| Benign IID score model | n=3 | n=4 | n=6 | n=10 |
|---|---:|---:|---:|---:|
| uniform(0,1) | 29.8% | 28.0% | 24.6% | 17.2% |
| Beta(8,2) | 2.3% | 1.1% | 0.5% | 0.1% |
| Beta(2,8) | 29.0% | 27.6% | 24.1% | 19.4% |
| Beta(9,1) | 0.4% | 0.1% | 0.1% | 0.0% |
| 70/30 easy/hard query mix | 28.0% | 24.4% | 19.3% | 12.4% |

Rates are finite Monte Carlo estimates, not production rates. At 2,000 trials the maximum binomial standard error is about 1.1 percentage points per cell. Increasing the trial count reduces sampling noise, not the synthetic-population bias.

## Operational interpretation

The same relative-drop threshold behaves very differently across score distributions. Three-call traces compare one early sample with two late samples and are especially noisy. Raising min_retrievals alone does not establish an acceptable false-positive rate. The 0.3 threshold is a heuristic default, not a universally calibrated operating point. An absolute-gap floor would also depend on the scorer's scale and is not added here.

Keep this detector shadow-only. Before alerts or interventions, evaluate labeled real traffic for each index/scorer and query mix, select a threshold against the actual false-positive budget, and measure recall separately. Query intent changes cannot be distinguished from degradation using scores alone. Use consistent higher-is-better scores; raw distances and cross-index comparisons are unsupported. No LLM judge is used.
