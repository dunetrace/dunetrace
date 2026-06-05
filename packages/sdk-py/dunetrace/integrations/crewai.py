"""
Dunetrace integration for CrewAI 1.x.

Uses CrewAI's global hook system (register_before_llm_call_hook, etc.) to track
every LLM and tool call made by any agent in the crew. Hooks emit events to
whichever dt.run() context is active in the current thread.

Usage::

    from dunetrace import Dunetrace
    from dunetrace.integrations.crewai import DunetraceCrewCallback

    dt = Dunetrace()
    cb = DunetraceCrewCallback(dt, agent_id="research-crew", model="gpt-4o-mini")
    cb.install()   # register global hooks once

    crew = Crew(agents=[...], tasks=[...])

    with dt.run("research-crew", user_input="AI trends", model="gpt-4o-mini") as run:
        result = crew.kickoff(inputs={"topic": "AI trends"})
        run.final_answer()

    cb.uninstall()  # remove hooks when done (or let them persist for the process lifetime)
    dt.shutdown()

Compatible with crewai >= 1.0.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, List, Optional

try:
    from crewai import hooks as _crewai_hooks
    from crewai.hooks.llm_hooks import LLMCallHookContext
    from crewai.hooks.tool_hooks import ToolCallHookContext

    _CREWAI_AVAILABLE = True
except ImportError:
    _CREWAI_AVAILABLE = False

if TYPE_CHECKING:
    from dunetrace.client import Dunetrace

logger = logging.getLogger("dunetrace.crewai")


class DunetraceCrewCallback:
    """
    Dunetrace observer for CrewAI 1.x.

    Registers global before/after hooks on LLM calls and tool calls so that
    all agent activity inside any ``dt.run()`` context is captured automatically.

    Install once before the crew runs; uninstall when done.  Multiple instances
    or nested contexts are safe — hooks check ``get_current_run()`` each time.
    """

    def __init__(
        self,
        client: "Dunetrace",
        agent_id: str = "crewai-crew",
        model: str = "unknown",
        tools: Optional[List[str]] = None,
    ) -> None:
        if not _CREWAI_AVAILABLE:
            raise ImportError("crewai >= 1.0 is not installed. Run: pip install crewai")
        self._client = client
        self._agent_id = agent_id
        self._model = model
        self._tools = tools or []

        # Per-call timers keyed by thread-id so concurrent crews don't collide.
        self._llm_start: dict[int, float] = {}
        self._tool_start: dict[str, float] = {}
        self._lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def install(self) -> None:
        """Register Dunetrace hooks globally. Call once before any crew kickoff.

        Idempotent: calling install() more than once does not register duplicate hooks.
        """
        existing_before = _crewai_hooks.get_before_llm_call_hooks()
        if self._before_llm in existing_before:
            logger.debug("Dunetrace CrewAI hooks already installed — skipping")
            return
        _crewai_hooks.register_before_llm_call_hook(self._before_llm)
        _crewai_hooks.register_after_llm_call_hook(self._after_llm)
        _crewai_hooks.register_before_tool_call_hook(self._before_tool)
        _crewai_hooks.register_after_tool_call_hook(self._after_tool)
        logger.debug("Dunetrace CrewAI hooks installed for agent_id=%s", self._agent_id)

    def uninstall(self) -> None:
        """Remove the Dunetrace hooks. Safe to call even if not installed."""
        for fn, unreg in [
            (self._before_llm, _crewai_hooks.unregister_before_llm_call_hook),
            (self._after_llm, _crewai_hooks.unregister_after_llm_call_hook),
            (self._before_tool, _crewai_hooks.unregister_before_tool_call_hook),
            (self._after_tool, _crewai_hooks.unregister_after_tool_call_hook),
        ]:
            try:
                unreg(fn)
            except Exception:
                pass
        logger.debug(
            "Dunetrace CrewAI hooks uninstalled for agent_id=%s", self._agent_id
        )

    # ── LLM hooks ──────────────────────────────────────────────────────────────

    def _before_llm(self, ctx: "LLMCallHookContext") -> None:
        from dunetrace.context import _current_run

        run = _current_run.get(None)
        if run is None:
            return
        model = self._model
        try:
            llm = ctx.llm
            model = getattr(llm, "model", None) or model
        except Exception:
            pass

        run.llm_called(model, prompt_tokens=_estimate_tokens(ctx.messages))
        with self._lock:
            self._llm_start[threading.get_ident()] = time.time()

    def _after_llm(self, ctx: "LLMCallHookContext") -> None:
        from dunetrace.context import _current_run
        from dunetrace.models import hash_content

        run = _current_run.get(None)
        if run is None:
            return
        with self._lock:
            t0 = self._llm_start.pop(threading.get_ident(), None)
        latency = int((time.time() - (t0 or time.time())) * 1000)
        output_text = ctx.response or ""
        run.llm_responded(
            latency_ms=latency,
            output_hash=hash_content(output_text),
            output_length=len(output_text),
            finish_reason="stop",
        )

    # ── Tool hooks ─────────────────────────────────────────────────────────────

    def _before_tool(self, ctx: "ToolCallHookContext") -> None:
        from dunetrace.context import _current_run

        run = _current_run.get(None)
        if run is None:
            return
        run.tool_called(ctx.tool_name, ctx.tool_input or {})
        key = f"{threading.get_ident()}:{ctx.tool_name}"
        with self._lock:
            self._tool_start[key] = time.time()

    def _after_tool(self, ctx: "ToolCallHookContext") -> None:
        from dunetrace.context import _current_run

        run = _current_run.get(None)
        if run is None:
            return
        key = f"{threading.get_ident()}:{ctx.tool_name}"
        with self._lock:
            t0 = self._tool_start.pop(key, None)
        latency = int((time.time() - (t0 or time.time())) * 1000)
        result = ctx.tool_result or ""
        run.tool_responded(
            ctx.tool_name,
            success=True,
            output_length=len(result),
            latency_ms=latency,
        )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _estimate_tokens(messages: Any) -> int:
    if not messages:
        return 0
    try:
        total = sum(
            len(
                str(
                    m.get("content", "")
                    if isinstance(m, dict)
                    else getattr(m, "content", m)
                )
            )
            for m in messages
        )
        return total // 4
    except Exception:
        return 0
