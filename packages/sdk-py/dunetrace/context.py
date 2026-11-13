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


def active_run() -> "Optional[RunContext]":
    """The run events in this context should attach to, or None.

    An implicit run (dunetrace.implicit) can be closed from another thread -- the
    idle sweeper -- while this context still points at it. A closed run reads as
    no run, everywhere, so nothing attaches events to it after its terminal
    event.
    """
    run = _current_run.get()
    if run is None or getattr(run, "_closed", False):
        return None
    return run


def explicit_run() -> "Optional[RunContext]":
    """The active run if the agent or a framework opened it, None if the only
    thing active is an implicit (guessed) run. Integrations that open an exact
    run of their own (CrewAI kickoff, a LangGraph invoke) use this to decide:
    an implicit run must not stop them, and client.run() closes it for them."""
    run = active_run()
    if run is None or getattr(run, "implicit", False):
        return None
    return run


class PendingEntry:
    """A run boundary that opens lazily (see dunetrace.middleware's auto-installed
    request boundary). The run is opened by the first patched LLM call inside it,
    so a request that never calls a model -- a health check, a static file --
    produces no run at all."""

    __slots__ = ("client", "agent_id", "opened_by", "user_input", "cm", "ctx")

    def __init__(self, client, agent_id: str, opened_by: str, user_input: str = "") -> None:
        self.client = client
        self.agent_id = agent_id
        self.opened_by = opened_by
        self.user_input = user_input
        self.cm = None
        self.ctx = None


_pending_entry: ContextVar[Optional[PendingEntry]] = ContextVar(
    "dunetrace_pending_entry", default=None
)


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
    return active_run()
