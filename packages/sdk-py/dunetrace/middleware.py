"""
ASGI and WSGI middleware for request-level Dunetrace run tracking.

Each HTTP request automatically opens a run, sets it as the active context
so ``get_current_run()`` works inside handlers, and closes it when the
response is sent.

ASGI (FastAPI / Starlette)::

    from dunetrace.middleware import DunetraceASGIMiddleware

    app.add_middleware(
        DunetraceASGIMiddleware,
        dt=dt,
        agent_id="my-api",
        model="gpt-4o",
    )

WSGI (Flask / Django)::

    from dunetrace.middleware import DunetraceWSGIMiddleware

    app.wsgi_app = DunetraceWSGIMiddleware(app.wsgi_app, dt=dt, agent_id="my-api")

Accessing the run inside a handler::

    from dunetrace import get_current_run

    run = get_current_run()
    if run:
        run.tool_called("db_query", {"table": "users"})
        result = db.query(...)
        run.tool_responded("db_query", success=True, output_length=len(result))
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional, TYPE_CHECKING

from dunetrace.context import _current_run

if TYPE_CHECKING:
    from dunetrace.client import Dunetrace


# ── ASGI ──────────────────────────────────────────────────────────────────────

class DunetraceASGIMiddleware:
    """
    ASGI middleware that opens a Dunetrace run for every HTTP/WebSocket request.

    The run is bound to the current async task context and accessible via
    :func:`~dunetrace.get_current_run` anywhere downstream.

    ``scope["state"]["dunetrace_run"]`` is also set for frameworks that
    surface ASGI scope state (e.g. Starlette ``request.state``).
    """

    def __init__(
        self,
        app:      Callable,
        dt:       "Dunetrace",
        agent_id: str,
        *,
        model:    str              = "unknown",
        tools:    Optional[List[str]] = None,
    ) -> None:
        self.app      = app
        self.dt       = dt
        self.agent_id = agent_id
        self.model    = model
        self.tools    = tools or []

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        method     = scope.get("method", "")
        path       = scope.get("path", "/")
        user_input = f"{method} {path}".strip()

        with self.dt.run(
            self.agent_id,
            user_input=user_input,
            model=self.model,
            tools=self.tools,
        ) as run:
            token = _current_run.set(run)
            try:
                if "state" not in scope:
                    scope["state"] = {}
                scope["state"]["dunetrace_run"] = run
                await self.app(scope, receive, send)
                run.final_answer()
            except Exception:
                # RUN_ERRORED is emitted by dt.run() on exception propagation
                raise
            finally:
                _current_run.reset(token)


# ── WSGI ──────────────────────────────────────────────────────────────────────

class DunetraceWSGIMiddleware:
    """
    WSGI middleware that opens a Dunetrace run for every HTTP request.

    The run is bound to the current thread's context variable and accessible
    via :func:`~dunetrace.get_current_run` anywhere downstream.

    ``environ["dunetrace.run"]`` is also set for direct WSGI environ access.
    """

    def __init__(
        self,
        app:      Callable,
        dt:       "Dunetrace",
        agent_id: str,
        *,
        model:    str              = "unknown",
        tools:    Optional[List[str]] = None,
    ) -> None:
        self.app      = app
        self.dt       = dt
        self.agent_id = agent_id
        self.model    = model
        self.tools    = tools or []

    def __call__(self, environ: dict, start_response: Callable) -> Iterable[bytes]:
        method     = environ.get("REQUEST_METHOD", "")
        path       = environ.get("PATH_INFO", "/")
        user_input = f"{method} {path}".strip()

        with self.dt.run(
            self.agent_id,
            user_input=user_input,
            model=self.model,
            tools=self.tools,
        ) as run:
            token = _current_run.set(run)
            try:
                environ["dunetrace.run"] = run
                response = self.app(environ, start_response)
                run.final_answer()
                return response
            except Exception:
                raise
            finally:
                _current_run.reset(token)
