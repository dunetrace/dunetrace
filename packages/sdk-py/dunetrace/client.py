"""
dunetrace/client.py

The main SDK client. Zero external dependencies.
Agent thread never blocks — all I/O happens on the drain thread.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
import urllib.error
from collections import deque
from threading import Thread, Event
from typing import Optional
from contextlib import contextmanager

from dunetrace.models import AgentEvent, EventType, RunState, hash_content, agent_version
from dunetrace.run_context import RunContext

logger = logging.getLogger("dunetrace")


class Dunetrace:
    """
    Non-blocking SDK client.

    Usage:
        vig = Dunetrace(api_key="dt_live_...", agent_id="my-agent")

        with vig.run(user_input, system_prompt=SYSTEM_PROMPT, model="gpt-4o", tools=TOOLS) as run:
            # your agent code here
            run.tool_called("web_search", {"query": "..."})
            run.tool_responded("web_search", success=True, output_length=512)
    """

    def __init__(
        self,
        api_key:     str,
        agent_id:    str,
        ingest_url:  str = "https://ingest.dunetrace.io/v1/ingest",
        buffer_size: int = 10_000,
        flush_interval_ms: int = 200,
        debug: bool = False,
    ):
        self.api_key    = api_key
        self.agent_id   = agent_id
        self.ingest_url = ingest_url

        if debug:
            logging.basicConfig(level=logging.DEBUG)

        # Ring buffer — drops oldest events under backpressure.
        # Agent is NEVER blocked by a full buffer.
        self._buffer   = deque(maxlen=buffer_size)
        self._stop_evt = Event()
        self._flush_interval = flush_interval_ms / 1000.0

        self._drain_thread = Thread(
            target=self._drain_loop,
            daemon=True,
            name="dunetrace-drain",
        )
        self._drain_thread.start()
        logger.debug("Dunetrace client started. agent_id=%s", agent_id)

    # ── Public API ────────────────────────────────────────────────────────────

    @contextmanager
    def run(
        self,
        user_input:    str,
        system_prompt: str = "",
        model:         str = "unknown",
        tools:         list = None,
        parent_run_id: Optional[str] = None,
    ):
        """
        Context manager wrapping a single agent run.
        Emits run.started on enter, run.completed / run.errored on exit.
        """
        tools = tools or []
        version = agent_version(system_prompt, model, tools)
        ctx = RunContext(
            client=self,
            agent_id=self.agent_id,
            agent_version=version,
            available_tools=tools,
            input_text_hash=hash_content(user_input),
            parent_run_id=parent_run_id,
        )

        self._emit(AgentEvent(
            event_type=EventType.RUN_STARTED,
            run_id=ctx.run_id,
            agent_id=self.agent_id,
            agent_version=version,
            step_index=0,
            parent_run_id=parent_run_id,
            payload={
                "input_hash":  hash_content(user_input),
                "model":       model,
                "tools":       tools,
            },
        ))

        try:
            yield ctx
            self._emit(AgentEvent(
                event_type=EventType.RUN_COMPLETED,
                run_id=ctx.run_id,
                agent_id=self.agent_id,
                agent_version=version,
                step_index=ctx.step,
                payload={
                    "total_steps":  ctx.step,
                    "exit_reason":  ctx.exit_reason or "completed",
                    "tool_call_count": len(ctx.state.tool_calls),
                },
            ))
        except Exception as exc:
            self._emit(AgentEvent(
                event_type=EventType.RUN_ERRORED,
                run_id=ctx.run_id,
                agent_id=self.agent_id,
                agent_version=version,
                step_index=ctx.step,
                payload={
                    "error_type":    type(exc).__name__,
                    "error_hash":    hash_content(str(exc)),
                    "step_index":    ctx.step,
                },
            ))
            raise

    def shutdown(self, timeout: float = 5.0):
        """Flush remaining events and stop drain thread."""
        self._stop_evt.set()
        self._drain_thread.join(timeout=timeout)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _emit(self, event: AgentEvent) -> None:
        """
        Hot path. Must be <100μs. No I/O. No blocking.
        O(1) deque append is thread-safe in CPython.
        """
        self._buffer.append(event)

    def _drain_loop(self) -> None:
        """Background thread. Batches and ships events to ingest API."""
        while not self._stop_evt.is_set():
            batch = []
            while self._buffer and len(batch) < 100:
                batch.append(self._buffer.popleft())
            if batch:
                self._ship(batch)
            else:
                time.sleep(self._flush_interval)

        # Final flush on shutdown
        remaining = []
        while self._buffer:
            remaining.append(self._buffer.popleft())
        if remaining:
            self._ship(remaining)

    def _ship(self, batch: list) -> None:
        """POST batch to ingest API. Failures are logged, not raised."""
        payload = json.dumps({
            "api_key":  self.api_key,
            "agent_id": self.agent_id,
            "events":   [e.to_dict() for e in batch],
        }).encode()

        req = urllib.request.Request(
            self.ingest_url,
            data=payload,
            headers={
                "Content-Type":  "application/json",
                "X-Dunetrace-Agent": self.agent_id,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                logger.debug("Shipped %d events. status=%d", len(batch), resp.status)
        except Exception as exc:
            # Never propagate — agent must not be affected by ingest failures
            logger.warning("Failed to ship %d events: %s", len(batch), exc)
