# Contributing to Dunetrace

Thanks for helping make agent monitoring better. This guide gets you from a fresh
clone to a merged PR.

## Where to start

- **[Good first issues](https://github.com/dunetrace/dunetrace/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)** —
  most are "add one structural detector," which is a clean, self-contained change.
- **Adding a detector?** Follow the step-by-step
  [Adding a detector guide](docs/contributing/adding-a-detector.md) — it walks the
  full registration path (the one thing newcomers usually miss is enum parity,
  covered there).
- Comment on the issue to get it assigned before you start.

For anything larger (a new integration, an architecture change), open an issue
first so we can agree on the approach.

## Dev setup

Requires **Python 3.11+**, **Node.js 22+**, and **Docker + Docker Compose**.

```bash
git clone https://github.com/dunetrace/dunetrace
cd dunetrace

# Install the SDK editable, with dev + framework extras (gives you pytest,
# ruff, mypy, pytest-timeout, regex, langchain/langgraph for the tests).
pip install -e "packages/sdk-py[dev,langchain]"

# Optional: the full stack (only needed for end-to-end / dashboard work)
cp .env.example .env
docker compose up -d
```

## Running tests

No Docker needed — every suite mocks its dependencies.

```bash
make test              # everything
make test-detector     # one service (also: test-api / test-explainer / test-alerts / test-ingest)

# A single package directly (note the PYTHONPATH prefix — each service has its own):
env -u PYTHONPATH PYTHONPATH=packages/sdk-py python -m pytest packages/sdk-py/tests/ -q
```

The `PYTHONPATH` for each service is documented at the top of
[`CLAUDE.md`](CLAUDE.md) if you run a suite by hand.

## Before you push

Install the git hooks once — they run ruff, mypy, the docs-consistency check,
and the test suites on the files you touched:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files   # to check everything up front
```

CI runs the same checks. The most common first-PR failure is the **`FailureType`
enum parity** test — if you add a detector, add its enum member to **both**
`packages/sdk-py/dunetrace/models.py` and
`packages/schemas-py/dunetrace_schemas/enums.py`. The detector guide covers this.

## Opening the PR

1. Fork, branch off `main`.
2. Make the change, add tests, run `make test` + `pre-commit run --all-files`.
3. Push and open a PR describing what and why. Link the issue (`Fixes #NN`).
4. New detectors ship **shadow by default** (they don't alert until validated on
   real traffic) — the guide explains how.

## Code style

- Python: `ruff` handles formatting and linting (config in `pyproject.toml`).
- Match the surrounding code — comment density, naming, and idiom.
- Keep the SDK **zero-runtime-dependency** (`packages/sdk-py` has no required
  deps; anything you import at runtime must be optional and guarded).
