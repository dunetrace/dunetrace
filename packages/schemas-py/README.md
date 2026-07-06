# dunetrace-schemas

Canonical Pydantic v2 wire-format schemas shared across Dunetrace backend
services. This package validates data at service boundaries (currently: the
ingest API's request payloads) — it is not part of the SDK's hot instrumentation
path.

## Why this exists

Before this package, the event wire-format was hand-maintained independently in
`ingest_svc/schemas.py` (`IngestEvent`), separately from the SDK's own
`dunetrace.models.AgentEvent` dataclass. Keeping the two in sync by hand had
already caused real bugs (stale event-type validation, wire-format keys drifting
out of sync with what the SDK actually sends). `dunetrace-schemas` is the single
source of truth for the wire shape now; `ingest_svc` imports it directly instead
of re-declaring it.

## Design constraints

- **No dependency on the `dunetrace` SDK package.** The SDK guarantees zero
  required dependencies of its own — it must not gain a transitive dependency on
  `pydantic` through this package. `tests/test_sdk_parity.py` keeps the two
  enum sets in sync via an explicit test, not an import.
- **No dependency from the SDK's hot path onto this package either.** The SDK's
  `AgentEvent`/`FailureSignal`/`RunState` stay plain dataclasses — Pydantic v2
  construction, even with its Rust core, is measurably slower than raw
  dataclass field assignment, and the SDK's `_emit()` budget is <1µs.
- **<10µs construction** — see `tests/test_benchmark.py`. The original target
  was <5µs; real measurement showed 4.3-6.9µs across the two models in this
  package (run-to-run variance puts `AgentEventSchema` right at the 5µs line
  and `FailureSignalSchema` consistently just over it), so the enforced bound
  was set at <10µs with real margin rather than either loosening it silently
  or cutting validation to force a number. Both are still ~100x cheaper than
  the SDK's <500µs per-hook budget these models never run inside.

## Contents

- `dunetrace_schemas.enums` — `EventType`, `Severity`, `FailureType`
- `dunetrace_schemas.events` — `AgentEventSchema`, `VALID_EVENT_TYPES`
- `dunetrace_schemas.signals` — `FailureSignalSchema`

## Usage

```python
from dunetrace_schemas import AgentEventSchema

event = AgentEventSchema(
    event_type="tool.called",
    run_id="run-1",
    agent_id="agent-1",
    agent_version="v1",
    step_index=3,
    payload={"tool_name": "search", "args": "..."},
)
```

## Testing

```bash
cd packages/schemas-py
pip install -e .
PYTHONPATH=../sdk-py:. python -m pytest tests/ -v
```

The SDK (`packages/sdk-py`) must be on the path for `test_sdk_parity.py`; the
other test modules only need this package installed.
