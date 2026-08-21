# Adding a structural detector

Most Dunetrace detectors are small, self-contained heuristics that read a
completed run's reconstructed state and return one signal. This guide walks the
full path end to end, using a real open issue —
[#53 `EXCESSIVE_RETRIEVAL`](https://github.com/dunetrace/dunetrace/issues/53) — as
the worked example.

A structural detector is a subclass of `BaseDetector` with one method,
`on_run_completion(state) -> Optional[FailureSignal]`. It runs after a run
completes, sees the whole `RunState`, and returns a signal or `None`. It never
runs in your agent's request path — detection happens server-side in the detector
worker (and in-process only for the SDK's own policy checks).

## What's in `RunState`

Your detector reads these (all reconstructed for you — no SDK or wire changes
needed to consume them). See `packages/sdk-py/dunetrace/models.py`.

| Field | What it holds |
|---|---|
| `state.tool_calls` | `list[ToolCall]` — `tool_name`, `args`, `success`, `error`, `output`, `output_length`, `step_index` |
| `state.llm_calls` | `list[LlmCall]` — `model`, `prompt_tokens`, `completion_tokens`, `finish_reason`, `output_length`, `output_text` |
| `state.retrievals` | `list[RetrievalResult]` — `index_name`, `result_count`, `top_score`, `content`, `step_index` |
| `state.memory_events` | `list[MemoryEvent]` — `op` (`written`/`read`/`cleared`), `key`, `value`, `source` |
| `state.external_signals` | `list[ExternalSignal]` — rate limits, cache misses, etc. |
| `state.input_text` / `state.system_prompt` | raw input + system prompt (when instrumented) |
| `state.exit_reason`, `state.current_step` | how/where the run ended |
| `state.baseline_p75_*` | per-agent learned baselines (may be `None`) |

## The 9 steps

### 1. Write the detector class

In `packages/sdk-py/dunetrace/detectors.py`, subclass `BaseDetector`. Tunable
knobs are **UPPERCASE class attributes** (they become YAML-configurable in step 4
and auto-validate at construction).

```python
class ExcessiveRetrievalDetector(BaseDetector):
    """
    A run fired an unusually high number of retrieval calls — usually a bad
    query loop, re-searching without converging.

    Tunable: MAX_RETRIEVALS.
    """

    name = "EXCESSIVE_RETRIEVAL"
    SEVERITY = Severity.MEDIUM
    MAX_RETRIEVALS = 8

    def on_run_completion(self, state: RunState) -> Optional[FailureSignal]:
        if len(state.retrievals) < self.MAX_RETRIEVALS:
            return None
        return FailureSignal(
            failure_type=FailureType.EXCESSIVE_RETRIEVAL,
            severity=self.SEVERITY,
            run_id=state.run_id,
            agent_id=state.agent_id,
            agent_version=state.agent_version,
            step_index=state.retrievals[-1].step_index,
            confidence=0.8,
            evidence={
                "retrieval_count": len(state.retrievals),
                "threshold": self.MAX_RETRIEVALS,
                "indexes": sorted({r.index_name for r in state.retrievals}),
            },
        )
```

Rules of the road:
- **Fire once per run** (return on the first match) — every Tier 1 detector holds
  a one-signal-per-run contract.
- The **`evidence` dict is what the root-cause explainer reads** — put the numbers
  that explain *why* it fired. If your evidence carries a step range, expose it
  through the field names `_get_step_range()` recognizes (see `explain_common.py`);
  otherwise the signal's `step_index` is used.
- Keep it cheap — detectors run against every completed run.

Then add an instance to the `TIER1_DETECTORS` list at the bottom of the file.

### 2. Add the `FailureType` — in BOTH enums

This is the step first-time contributors miss and it fails CI. Add your member to
**both**:

- `packages/sdk-py/dunetrace/models.py` → `class FailureType`
- `packages/schemas-py/dunetrace_schemas/enums.py` → `class FailureType`

```python
EXCESSIVE_RETRIEVAL = "EXCESSIVE_RETRIEVAL"
```

`packages/schemas-py/tests/test_sdk_parity.py` asserts the two are identical — if
you add to one and not the other, the `schemas-py` CI job goes red.

### 3. Register in the detector service

`services/detector/detector_svc/detectors.py`:

```python
from dunetrace.detectors import (
    ...,
    ExcessiveRetrievalDetector,
)

_DETECTOR_CLASSES = {
    ...,
    "excessive_retrieval": ExcessiveRetrievalDetector,
}
```

If your detector has tunables, map the YAML keys to the UPPERCASE attributes in
`services/detector/detector_svc/config_loader.py`'s `_PARAM_MAP`:

```python
"excessive_retrieval": {"max_retrievals": "MAX_RETRIEVALS"},
```

### 4. Add a `detectors.yml` block — and ship shadow

Add a section under `default:` in `detectors.yml` documenting the tunables and an
`alert_policy`:

```yaml
  excessive_retrieval:
    max_retrievals: 8
    alert_policy:
      mode: immediate
      threshold: 1
      window_runs: 1
```

**New detectors ship in shadow mode**: leave your detector *out* of
`LIVE_DETECTORS` in `services/detector/detector_svc/db.py`. Shadow signals are
stored and visible but never alert, so you can validate the false-positive rate
on real traffic before promoting it.

### 5. Add an explainer template — the build fails without one

`services/explainer/explainer_svc/templates.py`: write an
`explain_<your_detector>()` returning an `Explanation`, and register it in the
`TEMPLATES` dict. `test_template_coverage.py` asserts that *every* producible
`FailureType` has one, so omitting it turns the suite red.

The failure it prevents is worse than a red build, though. Without a template
`explain()` falls through to `_fallback()`, which emits a bare title and no
evidence-derived text — so every Slack alert and every dashboard explanation for
your detector is content-free while still looking like a working alert. This
step was missing from this checklist, and the first detector added after that
omission shipped exactly that way.

Read the evidence keys defensively (`ev.get(...)`) — a signal written before a
later evidence change still has to render.

### 6. Tests

Add unit tests in `packages/sdk-py/tests/test_detectors.py` (or the detector
service's `test_all_detectors.py`): at least one run that fires, one just under
the threshold that doesn't, and the tunable being respected.

### 7. Calibrate

Every recent detector ships with a calibration script + report that measures
false-positive / true-positive rate on a labeled corpus. Copy the closest
existing one — `scripts/calibrate_agent_handoff_failure.py` (structural,
deterministic) is a good template — into
`scripts/calibrate_excessive_retrieval.py`, and write
`scripts/calibration/excessive_retrieval_calibration.md` with the results. Aim for
**FP < 15%**; if you can't get there, that's a signal to rethink the heuristic (or
disclose it and keep it shadow).

**Gate on recall as well as FP.** A configuration that fires on almost nothing
scores 0% false positives trivially, so an FP-only gate will happily recommend an
inert detector. Require something like recall >= 50% too.

**Build the corpus so it can disprove you.** The failure mode is a negative set
whose members are excluded by being *just under* a threshold rather than by any
structural property — then the sweep is scoring the corpus, not the boundary.
Check that your negatives clear every floor outright and are rejected only by the
condition you actually believe in, and print the class distributions so a reader
can see the separation instead of trusting a headline number.
`scripts/calibrate_scattershot_tool_use.py` documents both mistakes concretely.

### 8. Document it

In `docs/detectors.md`: add a row to the summary table, a detail section (what it
catches, signal, severity, config, how it differs from related detectors), bump
the detector count in the intro line, and add it to the shadow-mode list. The
`validate-detector-docs` check requires every detector name mentioned in the docs
to exist in code and vice-versa.

### 9. Run the checks

```bash
make test
pre-commit run --all-files    # ruff, mypy, docs-consistency, the test suites
```

## Detector patterns to copy

| If your detector is… | Look at |
|---|---|
| A single-run tool/LLM/retrieval heuristic | `AgentHandoffFailureDetector`, `SilentTruncationDetector` |
| Content/pattern matching (injection, markers) | `MemoryPoisonedDetector`, `RetrievedContentInjectionDetector` |
| Cross-run (needs `parent_run_id`) | `HandoffContextLossDetector`, `DelegationLoopDetector` — these run at the *worker* level, not via `on_run_completion`; read their docstrings first |
| Semantic (LLM-judged, not structural) | `services/semantic/semantic_svc/evaluators/` (a different pattern — DeepEval-backed) |

## Common gotchas

- **Enum parity (step 2)** — the #1 CI failure. Both `FailureType` enums.
- **Shadow by default** — don't add to `LIVE_DETECTORS` in the same PR.
- **Docs consistency** — a detector name in `docs/detectors.md`/README must exist
  in code (and it must be reachable from `code_names()` in
  `scripts/validate_detector_docs.py`; TIER1 members are picked up automatically,
  worker-level detectors like handoff/delegation are added by name there).
- **Evidence step range** — use `_get_step_range()`'s field names or fall back to
  `signal.step_index`; never read evidence step fields directly downstream.

Reference PR: [#52 `AGENT_HANDOFF_FAILURE`](https://github.com/dunetrace/dunetrace/pull/52)
touches exactly these files and is a good end-to-end example.
