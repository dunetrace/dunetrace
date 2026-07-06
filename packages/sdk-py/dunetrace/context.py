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

# Set to True while a framework-level integration (LangChain, CrewAI, ...) is
# driving a call that may itself pass through an SDK-level auto-instrumentation
# patch (openai/anthropic/httpx/requests) further down the call stack. Those
# patches check this flag and skip emitting their own event when it's set —
# the framework-level integration already emits an equivalent, richer event
# for the same logical call — so a single LangChain LLM call backed by
# ChatOpenAI doesn't get counted twice.
_in_framework_call: ContextVar[bool] = ContextVar("dunetrace_in_framework_call", default=False)


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
