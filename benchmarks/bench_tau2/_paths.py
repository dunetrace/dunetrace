"""Repo paths the harness imports from.

The converter needs the SDK only. Replay also imports the ingest and detector
services, which every Makefile target puts on PYTHONPATH by hand. Installing
them here lets `python -m bench_tau2` run from any working directory with the
same layout the services' own test targets use.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Same order as the detector's test target, plus ingest for the persist path.
SERVICE_PATHS = [
    "packages/schemas-py",
    "packages/sdk-py",
    "services/ingest",
    "services/detector",
]


def install_repo_paths() -> None:
    for rel in reversed(SERVICE_PATHS):
        path = str(REPO_ROOT / rel)
        if path not in sys.path:
            sys.path.insert(0, path)
