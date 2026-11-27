"""τ²-bench measurement harness for Dunetrace's detectors.

Turns graded τ²-bench leaderboard trajectories into the events the Dunetrace
SDK emits, replays them through the production ingest and detector code
against a dedicated Postgres, and records every signal fired. See
benchmarks/README.md for the method.
"""

from bench_tau2._paths import REPO_ROOT, install_repo_paths

__all__ = ["REPO_ROOT", "install_repo_paths"]
