"""
Runs that start and end without ``dt.run()``.

Every Dunetrace event belongs to a run, and a run needs a start and an end.
``dt.run()``, the decorators and the middleware give both explicitly. This
module is what makes ``dt.init()`` alone enough: it decides where a run starts
and, the harder half, where it ends, for code that never calls Dunetrace.

How a run ends, in the order the SDK tries them
------------------------------------------------
1. **A framework says so.** Entry points with a natural end open an exact
   run and close it when they return: an HTTP request through the ASGI/WSGI
   middleware, a LangGraph ``invoke``/``stream``, a CrewAI ``kickoff``, an
   OpenAI Agents ``Runner`` trace. These are ordinary runs: nothing about them
   is guessed, and the detectors treat them as such.
2. **The agent's own ``dt.run()``** closes an implicit run that was open in the
   same context, so a script that mixes both never nests a real run under a
   guessed one.
3. **Idle.** When a patched LLM call arrives with no run active and no
   framework boundary in sight, the SDK opens an *implicit* run and attaches
   the following calls in the same thread or task to it. That run closes
   after ``idle_s`` seconds with no events (default 30, env
   ``DUNETRACE_IMPLICIT_RUN_IDLE_S``). This is the rule for scripts and
   notebooks. It is a guess, and the run says so: ``run.started`` carries
   ``implicit: true`` and ``opened_by``, and the terminal event's
   ``exit_reason`` is ``idle``. The detector holds every signal from an
   implicit run in shadow, so a guessed boundary can never page anyone.
4. **Process exit.** ``dt.shutdown()`` and the at-exit flush close whatever
   implicit runs are still open with ``exit_reason: process_exit``.
5. **A crash.** An unhandled exception on the main thread or a worker thread
   closes the implicit run in that context with ``run.errored``. With no run
   open at all, a one-event run is opened and errored so the crash is not lost.

The known weak spot is a synchronous server whose worker threads outlive
requests: without the middleware (auto-installed for FastAPI and Flask, see
``dunetrace.auto``) the calls of many requests on one thread would share an
implicit run until the thread goes idle. That is exactly why implicit runs are
shadowed, and why the entry-point rules above run first.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import weakref
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlsplit

from dunetrace.context import _pending_entry, active_run

if TYPE_CHECKING:  # pragma: no cover
    from dunetrace.client import Dunetrace
    from dunetrace.run_context import RunContext

logger = logging.getLogger("dunetrace.implicit")

DEFAULT_IDLE_S = 30.0
# Hard caps on a guessed run. In a busy thread with no request boundary an
# implicit run never goes idle, so without them it grows without bound and is
# only ever seen server-side as one endless stalled run. Hitting either closes
# it (exit_reason max_age / max_events); the next call opens a fresh one.
DEFAULT_MAX_AGE_S = 600.0
DEFAULT_MAX_EVENTS = 1000
_SWEEP_INTERVAL_S = 1.0


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# ── DSN ──────────────────────────────────────────────────────────────────────


def parse_dsn(dsn: str) -> Tuple[str, str]:
    """Split ``https://<api_key>@host[:port][/path]`` into ``(endpoint, api_key)``.

    One value to copy from the dashboard, one thing to put in a secret store,
    and no way to set the key without the endpoint. The key travels as a
    bearer token on every request exactly as before; the DSN is a convenience,
    not a security mechanism, and must never be logged whole.
    """
    parts = urlsplit(dsn.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("DUNETRACE_DSN must look like https://<api_key>@host[:port][/path]")
    key = unquote(parts.username or "")
    if not key:
        raise ValueError("DUNETRACE_DSN has no api key before the '@'")
    netloc = parts.hostname + (f":{parts.port}" if parts.port else "")
    return f"{parts.scheme}://{netloc}{parts.path.rstrip('/')}", key


def dsn_host(dsn: str) -> str:
    """The host part of a DSN, for log lines that must not carry the key."""
    try:
        return urlsplit(dsn.strip()).hostname or "?"
    except ValueError:
        return "?"


# ── Implicit runs ────────────────────────────────────────────────────────────


class _Handle:
    __slots__ = ("cm", "ctx", "opened_by", "opened_at")

    def __init__(self, cm: Any, ctx: "RunContext", opened_by: str) -> None:
        self.cm = cm
        self.ctx = ctx
        self.opened_by = opened_by
        self.opened_at = time.time()


def _script_name() -> str:
    try:
        name = os.path.basename(sys.argv[0] or "")
    except Exception:
        name = ""
    return name or "implicit"


class ImplicitRunManager:
    """Opens, tracks and closes the runs the SDK guesses at. One per client.

    All closing paths funnel through ``_finish`` so the terminal event is
    emitted exactly once, and a closed run is flagged on the ``RunContext`` so
    a stale context variable can never attach new events to it.
    """

    def __init__(
        self,
        client: "Dunetrace",
        *,
        enabled: bool,
        idle_s: float,
        max_age_s: float = DEFAULT_MAX_AGE_S,
        max_events: int = DEFAULT_MAX_EVENTS,
    ) -> None:
        self._client_ref = weakref.ref(client)
        self.enabled = enabled
        self.idle_s = max(1.0, float(idle_s))
        self.max_age_s = max(self.idle_s, float(max_age_s))
        self.max_events = max(1, int(max_events))
        self._lock = threading.Lock()
        self._open: Dict[str, _Handle] = {}
        self._sweeper: Optional[threading.Thread] = None
        self._announced = False

    # ── open / close ────────────────────────────────────────────────────────

    def open(self, opened_by: str) -> "Optional[RunContext]":
        """Open an implicit run in the current context, or None when disabled."""
        client = self._client_ref()
        if client is None or not self.enabled:
            return None
        agent_id = (
            getattr(client, "_default_agent_id", "")
            or os.environ.get("DUNETRACE_AGENT_ID", "")
            or _script_name()
        )
        try:
            cm = client.run(agent_id, _opened_by=opened_by, _implicit=True)
            ctx = cm.__enter__()
        except Exception:
            logger.debug("Dunetrace: could not open an implicit run", exc_info=True)
            return None
        with self._lock:
            self._open[ctx.run_id] = _Handle(cm, ctx, opened_by)
            self._ensure_sweeper()
        if not self._announced:
            self._announced = True
            logger.info(
                "Dunetrace: no run was open when %s was called, so one was opened for "
                "agent %r. It closes after %.0fs without events or at process exit, and its "
                "signals are held in shadow. Wrap the agent in dt.run() or a framework entry "
                "point for exact run boundaries. See docs/integrations/auto-instrumentation.md.",
                opened_by,
                agent_id,
                self.idle_s,
            )
        return ctx

    def close(
        self, run_id: str, exit_reason: str = "idle", exc: Optional[BaseException] = None
    ) -> bool:
        with self._lock:
            handle = self._open.pop(run_id, None)
        if handle is None:
            return False
        self._finish(handle, exit_reason, exc)
        return True

    def close_current(self, exit_reason: str) -> bool:
        """Close the implicit run active in *this* context, if any. Called when
        an explicit run opens, so a real run never nests under a guessed one."""
        run = active_run()
        if run is None or not getattr(run, "implicit", False):
            return False
        return self.close(run.run_id, exit_reason)

    def close_all(
        self, exit_reason: str = "process_exit", exc: Optional[BaseException] = None
    ) -> int:
        with self._lock:
            handles = list(self._open.values())
            self._open.clear()
        for handle in handles:
            self._finish(handle, exit_reason, exc)
        return len(handles)

    def open_count(self) -> int:
        with self._lock:
            return len(self._open)

    def _finish(self, handle: _Handle, exit_reason: str, exc: Optional[BaseException]) -> None:
        ctx = handle.ctx
        try:
            if exc is not None and not isinstance(exc, Exception):
                # KeyboardInterrupt / SystemExit: client.run() handles Exception,
                # so throwing one of these in would skip the terminal event. Record
                # the interruption as the exit reason and close normally.
                ctx.exit_reason = ctx.exit_reason or type(exc).__name__
                handle.cm.__exit__(None, None, None)
            elif exc is not None:
                handle.cm.__exit__(type(exc), exc, exc.__traceback__)
            else:
                if ctx.exit_reason is None:
                    ctx.exit_reason = exit_reason
                handle.cm.__exit__(None, None, None)
        except Exception:
            logger.debug("Dunetrace: closing implicit run %s failed", ctx.run_id, exc_info=True)
        finally:
            # After the terminal event, never before: _emit drops events on a
            # closed run, and run.completed itself must still go out.
            ctx._closed = True

    # ── idle sweep ──────────────────────────────────────────────────────────

    def sweep(self, now: Optional[float] = None) -> int:
        """Close every implicit run idle for longer than ``idle_s``. Returns
        how many closed. ``now`` is wall-clock seconds; events carry
        ``time.time()`` timestamps, so idleness is measured on the same clock."""
        now = time.time() if now is None else now
        due: List[Tuple[_Handle, str]] = []
        with self._lock:
            for h in list(self._open.values()):
                reason = self._due(h, now)
                if reason:
                    due.append((h, reason))
                    self._open.pop(h.ctx.run_id, None)
        for h, reason in due:
            self._finish(h, reason, None)
        return len(due)

    def _due(self, handle: _Handle, now: float) -> Optional[str]:
        if now - handle.opened_at >= self.max_age_s:
            return "max_age"
        if len(handle.ctx.state.events) >= self.max_events:
            return "max_events"
        # Idle is measured from the last event, but a call still in flight (a
        # slow reasoning-model call, a stream the caller is still iterating) is
        # not idle: closing under it attached its response to a run that had
        # already sent run.completed.
        if _in_flight(handle.ctx):
            return None
        if now - _last_activity(handle) >= self.idle_s:
            return "idle"
        return None

    def _ensure_sweeper(self) -> None:
        # Called under self._lock. One daemon thread while any run is open;
        # it exits when the last one closes and is restarted lazily.
        if self._sweeper is not None and self._sweeper.is_alive():
            return
        self._sweeper = threading.Thread(
            target=self._sweep_loop, name="dunetrace-implicit", daemon=True
        )
        self._sweeper.start()

    def _sweep_loop(self) -> None:
        while True:
            time.sleep(_SWEEP_INTERVAL_S)
            try:
                self.sweep()
            except Exception:
                logger.debug("Dunetrace: implicit run sweep failed", exc_info=True)
            with self._lock:
                if not self._open:
                    self._sweeper = None
                    return


def _in_flight(ctx: "RunContext") -> bool:
    """An LLM call has started and not responded, or a stream is still open."""
    called = responded = 0
    for e in ctx.state.events:
        kind = e.event_type.value
        if kind == "llm.called":
            called += 1
        elif kind == "llm.responded":
            responded += 1
    if called > responded:
        return True
    return any(ref() is not None for ref in getattr(ctx, "_open_streams", ()))


def _last_activity(handle: _Handle) -> float:
    events = handle.ctx.state.events
    if events:
        try:
            return float(events[-1].timestamp)
        except Exception:  # pragma: no cover - defensive
            pass
    return handle.opened_at


def resolve_run(client: "Optional[Dunetrace]", opened_by: str) -> "Optional[RunContext]":
    """The run a patched LLM call should attach to.

    The active run wins. A closed implicit run left in the context variable
    counts as none. With no run and a client that allows implicit runs, one is
    opened; a bare ``dunetrace.auto.auto_instrument()`` without a client stays
    attach-only, as before.
    """
    run = active_run()
    if run is not None:
        return run
    # A lazy request boundary (the auto-installed middleware): the first LLM
    # call inside the request opens its run; later ones reuse it, even from a
    # copied context (a sync endpoint on a thread pool).
    pending = _pending_entry.get()
    if pending is not None:
        return _open_pending(pending)
    if client is None:
        return None
    manager = getattr(client, "_implicit", None)
    if manager is None:
        return None
    return manager.open(opened_by)


def _open_pending(pending) -> "Optional[RunContext]":
    if pending.ctx is not None:
        return None if getattr(pending.ctx, "_closed", False) else pending.ctx
    try:
        cm = pending.client.run(
            pending.agent_id, user_input=pending.user_input, _opened_by=pending.opened_by
        )
        ctx = cm.__enter__()
    except Exception:
        logger.debug("Dunetrace: could not open the request run", exc_info=True)
        return None
    pending.cm, pending.ctx = cm, ctx
    return ctx


# ── Crash hooks ──────────────────────────────────────────────────────────────

_clients: "weakref.WeakSet[Dunetrace]" = weakref.WeakSet()
_hooks_lock = threading.Lock()
_hooks_installed = False
_RECORDED_ATTR = "_dunetrace_recorded"


def mark_recorded(exc: BaseException) -> None:
    """``dt.run()`` calls this after emitting run.errored, so the crash hook
    below does not record the same exception a second time when it reaches
    the top of the stack."""
    try:
        setattr(exc, _RECORDED_ATTR, True)
    except Exception:  # pragma: no cover - exotic exception types
        pass


def record_crash(exc: BaseException, where: str) -> int:
    """Turn an unhandled exception into ``run.errored``. Returns how many runs
    were closed. Never raises: this runs inside an exception hook."""
    if exc is None or getattr(exc, _RECORDED_ATTR, False):
        return 0
    closed = 0
    for client in list(_clients):
        manager = getattr(client, "_implicit", None)
        if manager is None:
            continue
        try:
            # A closed implicit run left in the context counts as no run: the
            # crash gets a one-event run instead of being lost.
            run = active_run()
            if run is not None and getattr(run, "implicit", False):
                if manager.close(run.run_id, exc=exc):
                    closed += 1
            elif run is None and manager.enabled:
                ctx = manager.open(where)
                if ctx is not None and manager.close(ctx.run_id, exc=exc):
                    closed += 1
        except Exception:
            logger.debug("Dunetrace: recording a crash failed", exc_info=True)
    if closed:
        mark_recorded(exc)
    return closed


def install_crash_hooks(client: "Dunetrace") -> None:
    """Chain onto ``sys.excepthook`` and ``threading.excepthook`` once per
    process. The previous hooks still run, so the traceback is printed exactly
    as before; Dunetrace only observes."""
    global _hooks_installed
    _clients.add(client)
    with _hooks_lock:
        if _hooks_installed:
            return
        _hooks_installed = True

    previous_sys = sys.excepthook

    def _sys_hook(exc_type, exc, tb):  # type: ignore[no-untyped-def]
        try:
            record_crash(exc, "sys.excepthook")
        except Exception:
            pass
        previous_sys(exc_type, exc, tb)

    sys.excepthook = _sys_hook

    previous_thread = getattr(threading, "excepthook", None)
    if previous_thread is not None:

        def _thread_hook(args):  # type: ignore[no-untyped-def]
            try:
                record_crash(args.exc_value, "threading.excepthook")
            except Exception:
                pass
            previous_thread(args)

        threading.excepthook = _thread_hook


def _reset_for_tests() -> None:
    global _hooks_installed
    with _hooks_lock:
        _hooks_installed = False
    _clients.clear()


__all__: List[str] = [
    "DEFAULT_IDLE_S",
    "DEFAULT_MAX_AGE_S",
    "DEFAULT_MAX_EVENTS",
    "ImplicitRunManager",
    "env_bool",
    "install_crash_hooks",
    "mark_recorded",
    "parse_dsn",
    "dsn_host",
    "record_crash",
    "resolve_run",
]
