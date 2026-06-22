"""
Hermes Agent integration for Dunetrace.

Maps Hermes plugin hooks to Dunetrace events so every run_conversation()
call is tracked as a Dunetrace run — tool calls, LLM calls, latency, tokens.

Two usage patterns:

1. **Programmatic** (script / CI / testing):

    from dunetrace import Dunetrace
    from dunetrace.integrations.hermes import DunetraceHermesPlugin

    dt = Dunetrace()
    plugin = DunetraceHermesPlugin(dt, agent_id="my-hermes-agent")
    plugin.attach()           # registers hooks with the global PluginManager

    agent = AIAgent(model="...", ...)
    result = await agent.run_conversation("What is 12 * 8?")
    await dt.flush_async()

2. **Hermes plugin** (persistent, for CLI users):

    Copy the `register` function below into
    ~/.hermes/plugins/dunetrace/__init__.py, then run:
        hermes plugins enable dunetrace

Hook → Dunetrace event mapping:
    pre_llm_call      → run.started   (once per turn, before first LLM call)
    pre_api_request   → llm.called    (every LLM API call)
    post_api_request  → llm.responded (with actual token counts and latency)
    pre_tool_call     → tool.called   (args are hashed inside _safe_emit)
    post_tool_call    → tool.responded
    on_session_end    → run.completed / run.errored

The Hermes `turn_id` is used as the Dunetrace `run_id` (one user message =
one run). Session IDs span multiple turns; we track each turn independently.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

from dunetrace.models import AgentEvent, EventType, agent_version as calc_version, hash_content

if TYPE_CHECKING:
    from dunetrace import Dunetrace

logger = logging.getLogger("dunetrace.hermes")


@dataclass
class _TurnCtx:
    """Per-turn state: one run_conversation() call = one _TurnCtx."""

    run_id: str
    step: int = 0
    start_time: float = field(default_factory=time.time)
    # api_request_id → wall-clock start time (for latency calculation)
    llm_starts: Dict[str, float] = field(default_factory=dict)
    # tool_call_id → wall-clock start time
    tool_starts: Dict[str, float] = field(default_factory=dict)


class DunetraceHermesPlugin:
    """
    Translates Hermes plugin hooks into Dunetrace events.

    One instance is safe to reuse across multiple run_conversation() calls
    and across multiple concurrent invocations — state is keyed by turn_id.
    """

    def __init__(
        self,
        client: "Dunetrace",
        agent_id: str,
        model: str = "",
        system_prompt: str = "",
        tools: Optional[list] = None,
    ) -> None:
        self._client = client
        self._agent_id = agent_id
        self._version = calc_version(system_prompt, model, tools or [])
        self._lock = threading.Lock()
        self._turns: Dict[str, _TurnCtx] = {}  # turn_id → ctx

    # ── Public API ─────────────────────────────────────────────────────────────

    def attach(self) -> None:
        """Register all hooks with the Hermes global PluginManager.

        Call this once before running any agent. Safe to call multiple times
        (Hermes appends; duplicate handlers cause duplicate events, so avoid
        calling attach() more than once per instance).
        """
        try:
            from hermes_cli.plugins import get_plugin_manager
        except ImportError as exc:
            raise ImportError(
                "hermes-agent is not installed. Run: pip install hermes-agent"
            ) from exc

        mgr = get_plugin_manager()
        for hook, method in (
            ("pre_llm_call", self._pre_llm_call),
            ("pre_api_request", self._pre_api_request),
            ("post_api_request", self._post_api_request),
            ("pre_tool_call", self._pre_tool_call),
            ("post_tool_call", self._post_tool_call),
            ("on_session_end", self._on_session_end),
        ):
            mgr._hooks.setdefault(hook, []).append(method)
        logger.debug("DunetraceHermesPlugin attached (agent_id=%s)", self._agent_id)

    # ── Hook implementations ───────────────────────────────────────────────────

    def _pre_llm_call(
        self,
        *,
        turn_id: str = "",
        session_id: str = "",
        user_message: str = "",
        model: str = "",
        **_: Any,
    ) -> None:
        """Fires once per user message, before the first LLM API call — run boundary."""
        run_id = turn_id or session_id or str(uuid.uuid4())
        ctx = _TurnCtx(run_id=run_id)
        with self._lock:
            self._turns[turn_id] = ctx

        self._safe_emit(
            EventType.RUN_STARTED,
            ctx,
            step=0,
            payload={
                "input_hash": hash_content(user_message) if user_message else "",
                "model": model,
            },
        )

    def _pre_api_request(
        self,
        *,
        turn_id: str = "",
        api_request_id: str = "",
        api_call_count: int = 0,
        model: str = "",
        approx_input_tokens: int = 0,
        started_at: Optional[float] = None,
        **_: Any,
    ) -> None:
        ctx = self._get_ctx(turn_id)
        if not ctx:
            return
        with self._lock:
            ctx.step += 1
            step = ctx.step
            ctx.llm_starts[api_request_id] = started_at or time.time()

        self._safe_emit(
            EventType.LLM_CALLED,
            ctx,
            step=step,
            payload={"model": model, "prompt_tokens": approx_input_tokens or None},
        )

    def _post_api_request(
        self,
        *,
        turn_id: str = "",
        api_request_id: str = "",
        api_duration: float = 0.0,
        finish_reason: str = "",
        usage: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        ctx = self._get_ctx(turn_id)
        if not ctx:
            return

        u = usage or {}
        latency_ms = int(api_duration * 1000) if api_duration else None

        self._safe_emit(
            EventType.LLM_RESPONDED,
            ctx,
            payload={
                # prompt_tokens are already in llm.called (approx_input_tokens).
                # Emitting them here too would cause fetch_run_tokens to double-count.
                "completion_tokens": u.get("output_tokens"),
                "latency_ms": latency_ms,
                "finish_reason": finish_reason or "stop",
            },
        )

    def _pre_tool_call(
        self,
        *,
        turn_id: str = "",
        tool_name: str = "",
        args: Optional[Dict[str, Any]] = None,
        tool_call_id: str = "",
        **_: Any,
    ) -> None:
        ctx = self._get_ctx(turn_id)
        if not ctx:
            return
        with self._lock:
            ctx.step += 1
            step = ctx.step
            ctx.tool_starts[tool_call_id] = time.time()

        args_str = str(args) if args else ""
        self._safe_emit(
            EventType.TOOL_CALLED,
            ctx,
            step=step,
            payload={
                "tool_name": tool_name,
                "args_hash": hash_content(args_str) if args_str else "",
            },
        )

    def _post_tool_call(
        self,
        *,
        turn_id: str = "",
        tool_name: str = "",
        tool_call_id: str = "",
        duration_ms: int = 0,
        status: str = "ok",
        error_message: Optional[str] = None,
        result: Any = None,
        **_: Any,
    ) -> None:
        ctx = self._get_ctx(turn_id)
        if not ctx:
            return

        ok = status == "ok"
        output_len = len(str(result)) if result is not None else 0

        payload: dict = {
            "tool_name": tool_name,
            "success": ok,
            "output_length": output_len,
            "latency_ms": duration_ms or None,
        }
        if not ok and error_message:
            payload["error_hash"] = hash_content(error_message)

        self._safe_emit(EventType.TOOL_RESPONDED, ctx, payload=payload)

    def _on_session_end(
        self,
        *,
        turn_id: str = "",
        completed: bool = True,
        interrupted: bool = False,
        **_: Any,
    ) -> None:
        with self._lock:
            ctx = self._turns.pop(turn_id, None)
        if not ctx:
            return

        event_type = (
            EventType.RUN_ERRORED if (not completed or interrupted) else EventType.RUN_COMPLETED
        )
        duration_ms = int((time.time() - ctx.start_time) * 1000)
        self._safe_emit(
            event_type,
            ctx,
            payload={"duration_ms": duration_ms, "interrupted": interrupted},
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_ctx(self, turn_id: str) -> Optional[_TurnCtx]:
        with self._lock:
            return self._turns.get(turn_id)

    def _safe_emit(
        self,
        event_type: EventType,
        ctx: _TurnCtx,
        payload: Optional[Dict[str, Any]] = None,
        step: Optional[int] = None,
    ) -> None:
        try:
            self._client._emit(
                AgentEvent(
                    event_type=event_type,
                    run_id=ctx.run_id,
                    agent_id=self._agent_id,
                    agent_version=self._version,
                    step_index=step if step is not None else ctx.step,
                    payload={k: v for k, v in (payload or {}).items() if v is not None},
                )
            )
        except Exception as exc:
            logger.warning("Dunetrace emit failed (%s): %s", event_type, exc)


# ── Hermes plugin entry point ──────────────────────────────────────────────────


def register(ctx: Any) -> None:
    """Hermes plugin entry point.

    Drop this file (or a wrapper that calls it) into
    ~/.hermes/plugins/dunetrace/__init__.py. Hermes calls register(ctx)
    at startup with a PluginContext.

    Reads config from environment:
        DUNETRACE_API_URL    — ingest endpoint (default: http://localhost:8001)
        DUNETRACE_API_KEY    — API key (default: empty)
        DUNETRACE_AGENT_ID   — agent_id tag (default: "hermes")
    """
    import os

    from dunetrace import Dunetrace

    dt = Dunetrace(
        endpoint=os.getenv("DUNETRACE_API_URL", "http://localhost:8001"),
        api_key=os.getenv("DUNETRACE_API_KEY", ""),
    )
    agent_id = os.getenv("DUNETRACE_AGENT_ID", "hermes")
    plugin = DunetraceHermesPlugin(dt, agent_id=agent_id)

    ctx.register_hook("pre_llm_call", plugin._pre_llm_call)
    ctx.register_hook("pre_api_request", plugin._pre_api_request)
    ctx.register_hook("post_api_request", plugin._post_api_request)
    ctx.register_hook("pre_tool_call", plugin._pre_tool_call)
    ctx.register_hook("post_tool_call", plugin._post_tool_call)
    ctx.register_hook("on_session_end", plugin._on_session_end)
