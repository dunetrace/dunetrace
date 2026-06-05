"""
Module-level ContextVar tracking the active RunContext.

Used by auto-instrumentation patches and middleware to correlate
LLM/tool calls with the correct run without passing the run object
explicitly through every call frame.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from dunetrace.run_context import RunContext

_current_run: ContextVar[Optional["RunContext"]] = ContextVar("dunetrace_current_run", default=None)


def get_current_run() -> "Optional[RunContext]":
    """
    Return the active :class:`~dunetrace.run_context.RunContext` for the
    current async task or thread, or ``None`` if no run is active.

    Use this inside middleware handlers or auto-instrumented code to access
    the run without passing it through your call stack::

        from dunetrace import get_current_run

        run = get_current_run()
        if run:
            run.tool_called("db_query", {"table": "users"})
    """
    return _current_run.get()
