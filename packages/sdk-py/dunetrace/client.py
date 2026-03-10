"""
dunetrace/client.py

Non-blocking SDK client. Zero external dependencies.
The agent thread never blocks — all I/O happens on the background drain thread.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
import urllib.error
from contextlib import contextmanager
from threading import Event, Thread
from typing import List, Optional

from dunetrace.buffer import RingBuffer
from dunetrace.detectors import PROMPT_INJECTION_DETECTOR
from dunetrace.models import AgentEvent, EventType, hash_content, agent_version
from dunetrace.run_context import RunContext

logger = logging.getLogger("dunetrace")


class Dunetrace:
    """
    Non-blocking observability client.

    Usage::

        dt = Dunetrace()  # defaults to http://localhost:8001, no key required

        with dt.run("my-agent", user_input=user_input, model="gpt-4o", tools=TOOLS) as run:
            run.llm_called("gpt-4o", prompt_tokens=150)
            run.tool_called("web_search", {"query": "..."})
            run.tool_responded("web_search", success=True, output_length=512)
            run.final_answer()

        dt.shutdown()

    Cloud::

        dt = Dunetrace(api_key="dt_live_...", endpoint="https://ingest.dunetrace.com")
    """

    def __init__(
        self,
        endpoint:          str           = "http://localhost:8001",
        api_key:           Optional[str] = None,
        *,
        buffer_size:       int  = 10_000,
        flush_interval_ms: int  = 200,
        debug:             bool = False,
    ) -> None:
        self._ingest_url     = endpoint.rstrip("/") + "/v1/ingest"
        self._api_key        = api_key or ""
        self._buffer         = RingBuffer[AgentEvent](maxsize=buffer_size)
        self._stop_evt       = Event()
        self._flush_interval = flush_interval_ms / 1000.0

        if debug:
            logging.basicConfig(level=logging.DEBUG)

        self._drain_thread = Thread(
            target=self._drain_loop,
            daemon=True,
            name="dunetrace-drain",
        )
        self._drain_thread.start()
        logger.debug("Dunetrace started. endpoint=%s", endpoint)

    # ── Public API ────────────────────────────────────────────────────────────

    @contextmanager
    def run(
        self,
        agent_id:      str,
        *,
        user_input:    str = "",
        system_prompt: str = "",
        model:         str = "unknown",
        tools:         Optional[List[str]] = None,
        parent_run_id: Optional[str] = None,
    ):
        """
        Context manager wrapping a single agent run.

        Emits ``run.started`` on enter, ``run.completed`` on clean exit,
        and ``run.errored`` if an exception escapes the block.
        """
        tools   = tools or []
        version = agent_version(system_prompt, model, tools)
        ctx     = RunContext(
            client=self,
            agent_id=agent_id,
            agent_version=version,
            available_tools=tools,
            input_text_hash=hash_content(user_input) if user_input else "",
            parent_run_id=parent_run_id,
        )

        # Run injection check on raw input before it is hashed and discarded.
        # Evidence (matched pattern names + count) is safe to transmit — no raw text.
        _injection_evidence = None
        if user_input:
            _sig = PROMPT_INJECTION_DETECTOR.check_input(user_input, ctx.state)
            if _sig:
                _injection_evidence = _sig.evidence

        payload: dict = {
            "input_hash": hash_content(user_input) if user_input else "",
            "model":      model,
            "tools":      tools,
        }
        if _injection_evidence:
            payload["injection_signal"] = _injection_evidence

        self._emit(AgentEvent(
            event_type=EventType.RUN_STARTED,
            run_id=ctx.run_id,
            agent_id=agent_id,
            agent_version=version,
            step_index=0,
            parent_run_id=parent_run_id,
            payload=payload,
        ))

        try:
            yield ctx
            self._emit(AgentEvent(
                event_type=EventType.RUN_COMPLETED,
                run_id=ctx.run_id,
                agent_id=agent_id,
                agent_version=version,
                step_index=ctx.step,
                payload={
                    "total_steps":     ctx.step,
                    "exit_reason":     ctx.exit_reason or "completed",
                    "tool_call_count": len(ctx.state.tool_calls),
                },
            ))
        except Exception as exc:
            self._emit(AgentEvent(
                event_type=EventType.RUN_ERRORED,
                run_id=ctx.run_id,
                agent_id=agent_id,
                agent_version=version,
                step_index=ctx.step,
                payload={
                    "error_type": type(exc).__name__,
                    "error_hash": hash_content(str(exc)),
                    "step_index": ctx.step,
                },
            ))
            raise

    def shutdown(self, timeout: float = 5.0) -> None:
        """Flush remaining events and stop the drain thread."""
        self._stop_evt.set()
        self._drain_thread.join(timeout=timeout)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _emit(self, event: AgentEvent) -> None:
        self._buffer.push(event)

    def _drain_loop(self) -> None:
        while not self._stop_evt.is_set():
            batch = self._buffer.drain(100)
            if batch:
                self._ship(batch)
            else:
                time.sleep(self._flush_interval)

        remaining = self._buffer.drain_all()
        if remaining:
            self._ship(remaining)

    def _ship(self, batch: List[AgentEvent]) -> None:
        payload = json.dumps({
            "api_key":  self._api_key,
            "agent_id": batch[0].agent_id if batch else "",
            "events":   [e.to_dict() for e in batch],
        }).encode()

        req = urllib.request.Request(
            self._ingest_url,
            data=payload,
            headers={
                "Content-Type":      "application/json",
                "X-Dunetrace-Agent": batch[0].agent_id if batch else "",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                logger.debug("Shipped %d events. status=%d", len(batch), resp.status)
        except urllib.error.URLError as exc:
            if "Connection refused" in str(exc):
                logger.warning(
                    "DuneTrace backend not reachable at %s — is it running?\n"
                    "  Start it with: docker compose up -d\n"
                    "  %d events dropped.",
                    self._ingest_url, len(batch),
                )
            else:
                logger.warning("Failed to ship %d events: %s", len(batch), exc)
        except Exception as exc:
            logger.warning("Failed to ship %d events: %s", len(batch), exc)


# Backwards-compatible alias
DunetraceClient = Dunetrace
