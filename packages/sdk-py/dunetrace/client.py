"""
SDK client. No external dependencies.
All network I/O runs on a background drain thread so the agent is never blocked.
"""

from __future__ import annotations

import datetime
import functools
import inspect
import json
import logging
import sys
import time
import urllib.request
import urllib.error
from contextlib import contextmanager
from threading import Event, Lock, Thread
from typing import Callable, List, Optional, Union

from dunetrace.buffer import RingBuffer
from dunetrace.context import _current_run
from dunetrace.detectors import PROMPT_INJECTION_DETECTOR
from dunetrace.models import AgentEvent, EventType, hash_content, agent_version
from dunetrace.policies import Policy, PolicyEngine, PolicyViolation
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
        endpoint: str = "http://localhost:8001",
        api_key: Optional[str] = None,
        *,
        buffer_size: int = 10_000,
        flush_interval_ms: int = 200,
        emit_as_json: bool = False,
        otel_exporter: Optional[object] = None,
        debug: bool = False,
    ) -> None:
        self._ingest_url = endpoint.rstrip("/") + "/v1/ingest" if endpoint else None
        self._api_key = api_key or ""
        self._buffer = RingBuffer[AgentEvent](maxsize=buffer_size)
        self._stop_evt = Event()
        self._flush_interval = flush_interval_ms / 1000.0
        self._emit_json = emit_as_json
        self._otel_exporter = otel_exporter  # DunetraceOTelExporter or None
        self._stdout_lock = Lock()  # one JSON line per write, no interleaving
        self._default_agent_id = ""  # set by init()
        self._policy_engine = PolicyEngine()

        if debug:
            logging.basicConfig(level=logging.DEBUG)

        self._flush_gate = Event()  # set to wake drain thread immediately

        self._drain_thread = Thread(
            target=self._drain_loop,
            daemon=True,
            name="dunetrace-drain",
        )
        self._drain_thread.start()
        logger.debug(
            "Dunetrace started. endpoint=%s emit_as_json=%s otel=%s",
            endpoint,
            emit_as_json,
            otel_exporter is not None,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    @contextmanager
    def run(
        self,
        agent_id: str,
        *,
        user_input: str = "",
        system_prompt: str = "",
        model: str = "unknown",
        tools: Optional[List[str]] = None,
        parent_run_id: Optional[str] = None,
    ):
        """
        Context manager wrapping a single agent run.

        Emits ``run.started`` on enter, ``run.completed`` on clean exit,
        and ``run.errored`` if an exception escapes the block.
        """
        tools = tools or []
        version = agent_version(system_prompt, model, tools)
        ctx = RunContext(
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
            "model": model,
            "tools": tools,
        }
        if _injection_evidence:
            payload["injection_signal"] = _injection_evidence

        self._emit(
            AgentEvent(
                event_type=EventType.RUN_STARTED,
                run_id=ctx.run_id,
                agent_id=agent_id,
                agent_version=version,
                step_index=0,
                parent_run_id=parent_run_id,
                payload=payload,
            )
        )

        # Fetch remote policies in a background thread so run start isn't delayed.
        if self._ingest_url and self._api_key and self._policy_engine.needs_fetch(agent_id):
            Thread(target=self._fetch_policies, args=(agent_id,), daemon=True).start()

        _token = _current_run.set(ctx)
        try:
            yield ctx
            # Sync RunState fields detectors read before notifying the OTel exporter.
            ctx.state.current_step = ctx.step
            if self._otel_exporter is not None:
                self._otel_exporter.notify_run_state(ctx.run_id, ctx.state)
            self._emit(
                AgentEvent(
                    event_type=EventType.RUN_COMPLETED,
                    run_id=ctx.run_id,
                    agent_id=agent_id,
                    agent_version=version,
                    step_index=ctx.step,
                    payload={
                        "total_steps": ctx.step,
                        "exit_reason": ctx.exit_reason or "completed",
                        "tool_call_count": len(ctx.state.tool_calls),
                    },
                )
            )
        except PolicyViolation as exc:
            ctx.state.current_step = ctx.step
            ctx.state.exit_reason = "policy_violation"
            if self._otel_exporter is not None:
                self._otel_exporter.notify_run_state(ctx.run_id, ctx.state)
            self._emit(
                AgentEvent(
                    event_type=EventType.RUN_ERRORED,
                    run_id=ctx.run_id,
                    agent_id=agent_id,
                    agent_version=version,
                    step_index=ctx.step,
                    payload={
                        "error_type": "PolicyViolation",
                        "error_hash": hash_content(str(exc)),
                        "exit_reason": "policy_violation",
                        "policy_name": exc.policy_name,
                        "step_index": ctx.step,
                    },
                )
            )
            raise
        except Exception as exc:
            ctx.state.current_step = ctx.step
            ctx.state.exit_reason = "error"
            if self._otel_exporter is not None:
                self._otel_exporter.notify_run_state(ctx.run_id, ctx.state)
            self._emit(
                AgentEvent(
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
                )
            )
            raise
        finally:
            _current_run.reset(_token)

    def init(
        self,
        agent_id: str = "",
        frameworks: Optional[List[str]] = None,
    ) -> "Dunetrace":
        """
        Primary entry point. Patches supported AI/HTTP clients globally and
        sets a default agent ID used when ``@dt.agent()`` is called without
        an explicit name.

        Equivalent to ``dt.auto_instrument()`` but follows the familiar
        ``init()`` convention (cf. ``Traceloop.init()``) and returns ``self``
        for chaining.

        Usage::

            from dunetrace import Dunetrace

            dt = Dunetrace(api_key="dt_live_...")
            dt.init(agent_id="my-agent")

            # All OpenAI / Anthropic / httpx / requests calls are now tracked.
            # Use @dt.agent() or get_current_run() as normal.

        :param agent_id:   Default label for runs started by ``@dt.agent()``
                           when no explicit ``agent_id`` argument is given.
        :param frameworks: Subset of frameworks to patch. ``None`` patches all
                           installed ones (openai, anthropic, httpx, requests).
        """
        self._default_agent_id = agent_id
        from dunetrace.auto import auto_instrument as _auto_instrument

        _auto_instrument(frameworks=frameworks)
        logger.debug("Dunetrace.init() agent_id=%r frameworks=%r", agent_id, frameworks)
        return self

    def auto_instrument(self, frameworks: Optional[List[str]] = None) -> None:
        """
        Monkey-patch supported AI framework clients so that LLM calls made
        inside any ``dt.run()`` context are tracked automatically — no manual
        ``run.llm_called()`` / ``run.llm_responded()`` needed.

        Supported frameworks: ``"openai"``, ``"anthropic"``.
        Uninstalled frameworks are silently skipped.

        :param frameworks: Subset to patch. ``None`` patches all installed ones.

        Usage::

            dt = Dunetrace(api_key="dt_live_...")
            dt.auto_instrument()   # patch openai + anthropic if installed

            with dt.run("my-agent", user_input=query) as run:
                # openai/anthropic calls are now tracked automatically
                response = openai_client.chat.completions.create(...)
        """
        from dunetrace.auto import auto_instrument as _auto_instrument

        _auto_instrument(frameworks=frameworks)

    def add_policy(
        self,
        name: str,
        condition: dict,
        action: dict,
        *,
        agent_id: str = "*",
        priority: int = 100,
        enabled: bool = True,
    ) -> Policy:
        """
        Register a runtime policy that fires mid-run when the condition is met.

        condition examples::

            {"trigger": "tool_call_count", "operator": "gt",  "value": 5}
            {"trigger": "cost_usd",        "operator": "gt",  "value": 0.50}
            {"trigger": "signal",          "operator": "eq",  "value": "TOOL_LOOP"}
            {"trigger": "finish_reason",   "operator": "eq",  "value": "length"}

        action examples::

            {"type": "stop"}
            {"type": "switch_model",  "params": {"model": "gpt-4o-mini"}}
            {"type": "inject_prompt", "params": {"prompt": "Stop repeating yourself."}}
            {"type": "log"}

        :param name:      Human-readable label shown in policy.triggered events.
        :param condition: Trigger definition dict.
        :param action:    Action to execute when condition is met.
        :param agent_id:  ``"*"`` (default) applies to all agents; pass a specific
                          agent_id to scope the policy.
        :param priority:  Lower numbers fire first. Default 100.
        :param enabled:   Set to False to register but not activate.
        :returns:         The Policy object (can be stored to mutate .enabled later).
        """
        policy = Policy(
            name=name,
            condition=condition,
            action=action,
            agent_id=agent_id,
            priority=priority,
            enabled=enabled,
        )
        self._policy_engine.add(policy)
        logger.debug(
            "Policy registered: %r trigger=%s action=%s",
            name,
            condition.get("trigger"),
            action.get("type"),
        )
        return policy

    def _fetch_policies(self, agent_id: str) -> None:
        """
        Background fetch of remote policies for agent_id.

        Calls GET {ingest_url_base}/v1/policies?agent_id=...&api_key=...
        Silently ignores all errors — policies are best-effort.
        """
        if not self._ingest_url or not self._api_key:
            return
        if not self._policy_engine.needs_fetch(agent_id):
            return

        self._policy_engine.mark_fetched(agent_id)  # prevent stampede

        try:
            base = self._ingest_url.replace("/v1/ingest", "")
            url = (
                f"{base}/v1/policies"
                f"?agent_id={urllib.request.quote(agent_id, safe='')}"
                f"&api_key={urllib.request.quote(self._api_key, safe='')}"
            )
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                import json as _json

                data = _json.loads(resp.read())
                self._policy_engine.load(data.get("policies", []))
        except Exception as exc:
            logger.debug("Policy fetch skipped: %s", exc)

    def agent(
        self,
        agent_id: str = "",
        *,
        model: str = "unknown",
        tools: Optional[List[str]] = None,
        system_prompt: str = "",
        input_from: Optional[str] = None,
    ) -> Callable:
        """
        Decorator that wraps a function in a ``dt.run()`` context.

        Works with both sync and async functions. The first positional argument
        is used as ``user_input`` by default; use *input_from* to name a
        different parameter.

        :param agent_id:     Name passed to ``dt.run()``.
        :param model:        Model name recorded on the run.
        :param tools:        Tool list recorded on the run.
        :param system_prompt: Used for agent version hashing.
        :param input_from:   Name of the parameter to use as ``user_input``.
                             Defaults to the first positional argument.

        Usage::

            @dt.agent("my-agent", model="gpt-4o")
            def run_agent(query: str) -> str:
                resp = openai_client.chat.completions.create(...)  # auto-tracked
                return resp.choices[0].message.content

            # async works identically
            @dt.agent("my-agent", model="claude-3-5-sonnet")
            async def run_agent_async(query: str) -> str:
                resp = await anthropic_client.messages.create(...)
                return resp.content[0].text

            # specify which argument is the user input
            @dt.agent("rag-agent", model="gpt-4o", input_from="question")
            def rag(context: str, question: str) -> str:
                ...
        """
        _agent_id = agent_id or self._default_agent_id or "agent"
        _tools = tools or []

        def decorator(fn: Callable) -> Callable:
            if inspect.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def async_wrapper(*args, **kwargs):
                    user_input = _extract_input(fn, args, kwargs, input_from)
                    with self.run(
                        _agent_id,
                        user_input=user_input,
                        model=model,
                        tools=_tools,
                        system_prompt=system_prompt,
                    ) as run:
                        result = await fn(*args, **kwargs)
                        run.final_answer()
                        return result

                return async_wrapper
            else:

                @functools.wraps(fn)
                def sync_wrapper(*args, **kwargs):
                    user_input = _extract_input(fn, args, kwargs, input_from)
                    with self.run(
                        _agent_id,
                        user_input=user_input,
                        model=model,
                        tools=_tools,
                        system_prompt=system_prompt,
                    ) as run:
                        result = fn(*args, **kwargs)
                        run.final_answer()
                        return result

                return sync_wrapper

        return decorator

    def trace(
        self,
        agent_id_or_fn: Union[str, Callable, None] = None,
        *,
        model: str = "unknown",
        tools: Optional[List[str]] = None,
        system_prompt: str = "",
        input_from: Optional[str] = None,
    ) -> Callable:
        """
        Decorator that wraps a function in a ``dt.run()`` context.
        Identical to ``@dt.agent`` but defaults ``agent_id`` to the function name
        when omitted, and supports bare ``@dt.trace`` usage.

        Usage::

            @dt.trace
            def my_agent(query: str) -> str: ...          # agent_id = "my_agent"

            @dt.trace("research-agent", model="gpt-4o")
            def my_agent(query: str) -> str: ...

            @dt.trace(model="gpt-4o")
            async def my_agent(query: str) -> str: ...    # agent_id = "my_agent"
        """
        # @dt.trace (no parens) — agent_id_or_fn is the decorated function
        if callable(agent_id_or_fn):
            fn = agent_id_or_fn
            return self.agent(
                fn.__name__,
                model=model,
                tools=tools,
                system_prompt=system_prompt,
                input_from=input_from,
            )(fn)

        # @dt.trace("name") or @dt.trace(model="gpt-4o")
        _agent_id = agent_id_or_fn or ""

        def decorator(fn: Callable) -> Callable:
            return self.agent(
                _agent_id or fn.__name__,
                model=model,
                tools=tools,
                system_prompt=system_prompt,
                input_from=input_from,
            )(fn)

        return decorator

    def tool(
        self,
        name_or_fn: Union[str, Callable, None] = None,
    ) -> Callable:
        """
        Decorator that auto-emits ``tool.called`` / ``tool.responded`` around the
        function. No-op when called outside a ``dt.run()`` context (the function
        still runs, it just isn't tracked).

        Tool arguments are serialized and SHA-256 hashed before transmission —
        raw values never leave the process.

        Usage::

            @dt.tool
            def search(query: str) -> list: ...           # tool_name = "search"

            @dt.tool("web_search")
            def search(query: str) -> list: ...           # explicit name

            @dt.tool
            async def fetch_page(url: str) -> str: ...    # async works identically
        """

        def _wrap(fn: Callable, tool_name: str) -> Callable:
            if inspect.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def async_wrapper(*args, **kwargs):
                    run = _current_run.get(None)
                    args_dict = _bind_args(fn, args, kwargs)
                    if run:
                        run.tool_called(tool_name, args_dict)
                    t0 = time.time()
                    try:
                        result = await fn(*args, **kwargs)
                    except Exception as exc:
                        if run:
                            run.tool_responded(
                                tool_name,
                                success=False,
                                latency_ms=int((time.time() - t0) * 1000),
                                error=str(exc),
                            )
                        raise
                    if run:
                        run.tool_responded(
                            tool_name,
                            success=True,
                            output_length=len(str(result)),
                            latency_ms=int((time.time() - t0) * 1000),
                        )
                    return result

                return async_wrapper
            else:

                @functools.wraps(fn)
                def sync_wrapper(*args, **kwargs):
                    run = _current_run.get(None)
                    args_dict = _bind_args(fn, args, kwargs)
                    if run:
                        run.tool_called(tool_name, args_dict)
                    t0 = time.time()
                    try:
                        result = fn(*args, **kwargs)
                    except Exception as exc:
                        if run:
                            run.tool_responded(
                                tool_name,
                                success=False,
                                latency_ms=int((time.time() - t0) * 1000),
                                error=str(exc),
                            )
                        raise
                    if run:
                        run.tool_responded(
                            tool_name,
                            success=True,
                            output_length=len(str(result)),
                            latency_ms=int((time.time() - t0) * 1000),
                        )
                    return result

                return sync_wrapper

        # @dt.tool (no parens) — name_or_fn is the function
        if callable(name_or_fn):
            return _wrap(name_or_fn, name_or_fn.__name__)

        # @dt.tool("name") — returns a decorator
        _tool_name = name_or_fn or ""

        def decorator(fn: Callable) -> Callable:
            return _wrap(fn, _tool_name or fn.__name__)

        return decorator

    def mark_deploy(
        self,
        agent_id: str,
        version: str,
        **meta,
    ) -> None:
        """
        Record a deploy marker for ``agent_id`` at the current timestamp.

        Call this from your CI/CD pipeline or at application startup to annotate
        the detector timeline with release boundaries.

        Usage::

            dt.mark_deploy("my-agent", version="v1.4.2", env="production")
            dt.mark_deploy("my-agent", version="v1.4.2", commit="abc1234")

        The call is fire-and-forget — it runs on a background thread so it never
        blocks the caller. Errors are logged at WARNING level and silently dropped.
        """
        if not self._ingest_url:
            return
        Thread(
            target=self._ship_deploy,
            args=(agent_id, version, dict(meta)),
            daemon=True,
            name="dunetrace-deploy",
        ).start()

    def _ship_deploy(self, agent_id: str, version: str, meta: dict) -> None:
        base = self._ingest_url.replace("/v1/ingest", "")
        payload = json.dumps(
            {
                "api_key": self._api_key,
                "agent_id": agent_id,
                "version": version,
                "meta": meta,
            }
        ).encode()
        req = urllib.request.Request(
            base + "/v1/deploy",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                logger.debug(
                    "Deploy marked. agent_id=%s version=%s status=%d",
                    agent_id,
                    version,
                    resp.status,
                )
        except Exception as exc:
            logger.warning("mark_deploy failed: %s", exc)

    def flush(self, *, block: bool = False, timeout: float = 5.0) -> None:
        """Wake the drain thread to ship buffered events immediately.

        block=True waits up to `timeout` seconds for the buffer to empty.
        Use this after important checkpoints (tool response, LLM response) to
        ensure observability data reaches the backend before the next step.
        """
        self._flush_gate.set()
        if block:
            deadline = time.monotonic() + timeout
            while self._buffer and time.monotonic() < deadline:
                time.sleep(0.01)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Flush remaining events and stop the drain thread."""
        self._stop_evt.set()
        self._flush_gate.set()  # wake the drain thread so shutdown is immediate
        self._drain_thread.join(timeout=timeout)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _emit(self, event: AgentEvent) -> None:
        try:
            if self._emit_json:
                self._write_json_line(event)
            if self._otel_exporter is not None:
                self._otel_exporter.handle(event)
            self._buffer.push(event)
        except Exception as exc:
            logger.warning("Dunetrace: failed to emit %s: %s", event.event_type, exc)

    def _write_json_line(self, event: AgentEvent) -> None:
        """
        Write one Loki-compatible NDJSON line to stdout.

        Fields: ts (RFC3339), level ("info"), logger ("dunetrace"), event_type and agent_id
        as Loki stream labels, run_id/step_index/payload as structured fields.
        payload contains hashes only — never raw content.
        """
        ts = datetime.datetime.utcfromtimestamp(event.timestamp).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        line = {
            "ts": ts,
            "level": "info",
            "logger": "dunetrace",
            "event_type": event.event_type.value,
            "agent_id": event.agent_id,
            "run_id": event.run_id,
            "agent_version": event.agent_version,
            "step_index": event.step_index,
            "payload": event.payload,
        }
        if event.parent_run_id:
            line["parent_run_id"] = event.parent_run_id

        serialised = json.dumps(line, separators=(",", ":"))
        with self._stdout_lock:
            sys.stdout.write(serialised + "\n")
            sys.stdout.flush()

    def _drain_loop(self) -> None:
        while not self._stop_evt.is_set():
            batch = self._buffer.drain(100)
            if batch:
                if self._ingest_url:
                    self._ship(batch)
            else:
                # Wait until either flush() signals us, shutdown() fires, or the
                # interval expires. This lets flush() wake the thread immediately.
                self._flush_gate.wait(timeout=self._flush_interval)
                self._flush_gate.clear()

        remaining = self._buffer.drain_all()
        if remaining and self._ingest_url:
            self._ship(remaining)

    def _ship(self, batch: List[AgentEvent]) -> None:
        payload = json.dumps(
            {
                "api_key": self._api_key,
                "agent_id": batch[0].agent_id if batch else "",
                "events": [e.to_dict() for e in batch],
            }
        ).encode()

        req = urllib.request.Request(
            self._ingest_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
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
                    self._ingest_url,
                    len(batch),
                )
            else:
                logger.warning("Failed to ship %d events: %s", len(batch), exc)
        except Exception as exc:
            logger.warning("Failed to ship %d events: %s", len(batch), exc)


# Backwards-compatible alias
DunetraceClient = Dunetrace


def _bind_args(fn: Callable, args: tuple, kwargs: dict) -> dict:
    """Return a {param_name: value} dict for the call, excluding self/cls."""
    try:
        bound = inspect.signature(fn).bind(*args, **kwargs)
        bound.apply_defaults()
        return {k: v for k, v in bound.arguments.items() if k not in ("self", "cls")}
    except Exception:
        return {}


def _extract_input(fn: Callable, args: tuple, kwargs: dict, input_from: Optional[str]) -> str:
    """
    Pull the user_input string from a function call.

    Priority:
    1. ``input_from`` kwarg name if specified
    2. First positional argument
    3. Empty string (no input available)
    """
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())

        if input_from:
            # Named parameter — check kwargs first, then positional by index
            if input_from in kwargs:
                return str(kwargs[input_from])
            if input_from in params:
                idx = params.index(input_from)
                if idx < len(args):
                    return str(args[idx])
        elif args:
            return str(args[0])
        elif params and params[0] in kwargs:
            return str(kwargs[params[0]])
    except Exception:
        pass
    return ""
