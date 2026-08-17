# EXCESSIVE_RETRIEVAL calibration

The deterministic `EXCESSIVE_RETRIEVAL` detector fires when a completed run has
at least eight retrieval calls. Reproduce the boundary check with:

```bash
PYTHONPATH=packages/sdk-py python scripts/calibrate_excessive_retrieval.py
```

The corpus covers empty, ordinary, just-below-threshold, at-threshold, and well
above-threshold runs. All eight expected classifications pass. The detector ships
in shadow mode because synthetic boundary checks establish deterministic behavior,
not real-traffic precision; review its shadow fire rate before promotion.
