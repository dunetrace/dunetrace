"""
Re-export of the canonical run builder.

This module used to hold its own copy. Two copies drifted — see
``dunetrace/run_builder.py``'s docstring — so the implementation now lives
beside the models it rebuilds and both services import it. Kept as a module so
existing imports and test patches (``api_svc.run_builder.build_run_state``) keep working.
"""

from __future__ import annotations

from dunetrace.run_builder import build_run_state

__all__ = ["build_run_state"]
