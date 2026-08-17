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

# Set to True for the duration of the HTTP request an LLM client patch is
# already recording. Every vendor SDK here is built on httpx (openai, anthropic,
# mistralai) or requests, so without this a single LLM call is recorded twice:
# once as llm.called by the provider patch, and once as tool.called named after
# the hostname by the httpx/requests patch. That inflates tool_call_count —
# which is both a policy trigger and what TOOL_LOOP counts — for anyone who
# calls dt.auto_instrument() with no framework list.
#
# Distinct from _in_framework_call, which suppresses the *inner LLM* patch under
# an outer framework integration. This one suppresses the *inner HTTP* patch
# under an outer LLM patch; both can be set at once for a LangChain call.
# Mirrors the TypeScript SDK's httpSuppression (packages/sdk-ts/src/context.ts).
_http_suppressed: ContextVar[bool] = ContextVar("dunetrace_http_suppressed", default=False)


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
