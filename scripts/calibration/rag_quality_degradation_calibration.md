# RAG quality degradation: synthetic calibration

Run `PYTHONPATH=packages/sdk-py python scripts/calibrate_rag_quality_degradation.py`.
No network, models, or infrastructure are required.

The 13 hand-labeled synthetic traces include five decline scenarios and eight
legitimate retrieval patterns. Negatives all have enough scored calls; stable
low-score runs, oscillation, query recovery and legitimate intent changes are
included. A gradual decline below the default threshold is deliberately labeled
positive to expose missed recall. Labels describe intended query scenarios;
the detector only observes scores, so it cannot distinguish intent changes.

Default results: TP 4, FP 1, FN 1, TN 7; recall 80%, false-positive rate 12.5%,
precision 80%. The false positive is a legitimate change in query intent. These
small synthetic results are reproducible boundary checks, **not** estimates of
production precision. Scores from different indexes are separately tested and
never compared. Missing, nonfinite and negative scores are ignored.

Keep the detector in shadow mode until evaluation on labeled real traffic.
Use higher-is-better relevance values with a consistent scorer within an index;
raw distance values are unsupported. The threshold controls relative change,
not semantic answer quality. No LLM judge is used.
