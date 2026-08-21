"""
SDK client. No external dependencies.
All network I/O runs on a background drain thread so the agent is never blocked.
"""

from __future__ import annotations

import atexit
import datetime
import functools
import inspect
import json
import logging
import os
import sys
import time
import weakref
from types import FrameType
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from threading import Event, Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Union

from dunetrace.buffer import RingBuffer
from dunetrace.context import _current_run
from dunetrace.detectors import PROMPT_INJECTION_DETECTOR
from dunetrace.emitters import (
    USER_AGENT,
    BatchingEmitter,
    HttpBatchingEmitter,
)
from dunetrace.models import (
    AgentEvent,
    EventType,
    agent_version,
    Exporter,
    CallableExporter,
)
from dunetrace.policies import (
    EvaluationRateLimiter,
    Policy,
    PolicyAction,
    PolicyCondition,
    PolicyEngine,
    PolicyViolation,
)
from dunetrace.run_context import RunContext

logger = logging.getLogger("dunetrace")

_SDK_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def _capture_caller_source_file() -> Optional[str]:
    """Phase 4.3 tier-2 source mapping: best-effort local file path of
    whatever code actually called dt.run(). Walks the stack past this SDK's
    own package directory AND Python's contextlib (dt.run() is
    @contextmanager-decorated, so a naive "caller is one frame up" would
    land inside contextlib's own __enter__ machinery, not the user's code)
    to find the first frame that's genuinely outside both.

    Deliberately file-path-only, no git SHA/version capture — see
    BACKLOG.md's Phase 4.3 entry for why: git plumbing inside an SDK is
    fragile across real deployment environments (Docker images without
    .git, serverless, CI checkouts), so a half-implemented version field
    was rejected in favor of fully disclosing the gap.

    Returns None (never raises) if the stack can't be walked for any
    reason — this must never break dt.run() itself.

    Uses sys._getframe() rather than inspect.stack(): the latter reads and
    caches surrounding source-code context for every frame by default,
    measurably slow enough to fail this SDK's own per-event overhead
    benchmark (test_client.py) when called on every dt.run(). Raw frame
    objects avoid that cost entirely — this only ever reads
    frame.f_code.co_filename, never source lines.
    """
    try:
        frame: "Optional[FrameType]" = sys._getframe(1)
        while frame is not None:
            filename = os.path.abspath(frame.f_code.co_filename)
            if (
                filename.startswith(_SDK_PACKAGE_DIR)
                or os.path.basename(filename) == "contextlib.py"
            ):
                frame = frame.f_back
                continue
            return filename
    except Exception:
        pass
    return None


# How long the at-exit flush may block the interpreter shutting down. Deliberately
# shorter than shutdown()'s own default: at this point the process is trying to
# exit, and a slow or unreachable collector must not hold it open. Override with
# DUNETRACE_ATEXIT_TIMEOUT (seconds); set to 0 to disable the at-exit flush.
_ATEXIT_FLUSH_TIMEOUT = 2.0


def _atexit_timeout() -> float:
    raw = os.environ.get("DUNETRACE_ATEXIT_TIMEOUT", "")
    if not raw:
        return _ATEXIT_FLUSH_TIMEOUT
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.debug("Dunetrace: bad DUNETRACE_ATEXIT_TIMEOUT=%r, using default", raw)
        return _ATEXIT_FLUSH_TIMEOUT


def _make_atexit_flush(client: "Dunetrace"):
    """Build the at-exit flush callback for *client*, or None if disabled.

    Holds only a weak reference: a strong one would park every client ever
    constructed in the atexit registry, so none could be collected even after
    the caller dropped it.
    """
    timeout = _atexit_timeout()
    if timeout <= 0:
        return None

    ref = weakref.ref(client)

    def _flush_on_exit() -> None:
        c = ref()
        if c is None:
            return
        try:
            c.shutdown(timeout=timeout)
        except Exception:
            # Interpreter teardown is a hostile place — modules may already be
            # torn down. Never let this surface as a crash on the way out.
            pass

    return _flush_on_exit


def _coerce_text(value: Any) -> str:
    """Best-effort string form of a caller-supplied text field.

    ``user_input``/``system_prompt`` are typed ``str``, but callers pass whatever
    their agent actually has — a message dict, a list of messages, an int id,
    bytes off a socket. Those flow into a regex scan (the injection detector) and
    into the event payload, and an un-coerced non-str used to raise straight out
    of ``dt.run()`` into the caller. Coerce once, here, so every downstream
    consumer sees text. Returns "" if even ``str()`` fails (objects with a
    raising ``__str__``/``__repr__``).
    """
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return ""
    try:
        return str(value)
    except Exception:
        logger.debug("Dunetrace: could not coerce %s to text", type(value).__name__, exc_info=True)
        return ""


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
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        buffer_size: int = 10_000,
        flush_interval_ms: int = 200,
        emit_as_json: bool = False,
        otel_exporter: Optional[Any] = None,
        exporters: Optional[List[Exporter]] = None,
        policy_secret: str = "",
        emitter: Optional[BatchingEmitter] = None,
        debug: bool = False,
        api_url: Optional[str] = None,
        policy_evaluation_reporting: Optional[bool] = None,
    ) -> None:
        # is not None, not `endpoint or ...` — an explicit endpoint="" is taken
        # literally rather than silently falling back. To disable HTTP shipping,
        # pass emitter=NoopBatchingEmitter() (see dunetrace.emitters); that's the
        # one supported way to opt out, not a magic endpoint value.
        _endpoint = (
            endpoint
            if endpoint is not None
            else os.environ.get("DUNETRACE_ENDPOINT", "http://localhost:8001")
        )
        self._ingest_url = _endpoint.rstrip("/") + "/v1/ingest"
        self._api_key = api_key or os.environ.get("DUNETRACE_API_KEY", "")
        # Customer API base URL (port 8002 in local docker-compose) — a
        # different service from the ingest endpoint above (port 8001), not
        # derivable from it (a real deployment may put them on entirely
        # different hostnames). Same DUNETRACE_API_URL name the MCP server
        # already uses for this exact concept (see docs/mcp-server.md).
        # Only needed for pack activation (enable_pack/disable_pack/
        # enabled_packs) — every other SDK call goes through _ingest_url.
        self._api_url = (
            api_url
            if api_url is not None
            else os.environ.get("DUNETRACE_API_URL", "http://localhost:8002")
        ).rstrip("/")
        self._emitter: BatchingEmitter = emitter or HttpBatchingEmitter(_endpoint, self._api_key)
        # Env fallback: the secret had no plumbing at all, so the documented
        # way to turn verification on required editing code. Without it the
        # default posture for every install was "no verification".
        self._policy_secret = policy_secret or os.environ.get("DUNETRACE_POLICY_SECRET", "")
        self._buffer = RingBuffer[AgentEvent](maxsize=buffer_size)
        self._stop_evt = Event()
        self._flush_interval = flush_interval_ms / 1000.0
        self._emit_json = emit_as_json
        self._stdout_lock = Lock()  # one JSON line per write, no interleaving

        # OTel export (opt-in via DUNETRACE_OTEL_* env). When enabled and the
        # caller didn't wire an exporter explicitly, build one on the shared
        # tracer. dunetrace.otel.init() never raises and returns False when
        # unconfigured, so this is a no-op for anyone not using OTel.
        if otel_exporter is None:
            from dunetrace import otel as _otel

            if _otel.init():
                from dunetrace.integrations.otel import DunetraceOTelExporter

                try:
                    _cfg = _otel.active_config()
                    otel_exporter = DunetraceOTelExporter(
                        tracer=_otel.get_tracer(),
                        capture_content=(_cfg.capture_content if _cfg else True),
                    )
                except Exception as exc:
                    logger.warning(
                        "Dunetrace: OTel exporter init failed (%s); OTel export disabled.", exc
                    )
        self._otel_exporter = otel_exporter
        self._exporters: List[Exporter] = list(exporters or [])
        self._default_agent_id = ""  # set by init()
        self._policy_engine = PolicyEngine()

        # Policy evaluation observability (Phase 5). Opt-in dashboard reporting
        # ships one rate-limited policy.evaluated event per evaluation; default
        # off (env DUNETRACE_POLICY_EVAL_REPORTING=1 to enable) to protect the
        # SDK's per-event overhead budget. Structured DEBUG logging on the
        # "dunetrace.policies.evaluation" logger is always available, independent
        # of this flag. The rate limiter is shared across all runs (per process).
        if policy_evaluation_reporting is None:
            policy_evaluation_reporting = os.environ.get(
                "DUNETRACE_POLICY_EVAL_REPORTING", ""
            ).lower() in ("1", "true", "yes")
        self._policy_evaluation_reporting = bool(policy_evaluation_reporting)
        self._policy_eval_rate_limiter = EvaluationRateLimiter()

        if debug:
            logging.basicConfig(level=logging.DEBUG)

        self._flush_gate = Event()  # set to wake drain thread immediately

        # Weakref, not the bound method — see _drain_loop's docstring for why.
        self._drain_thread = Thread(
            target=Dunetrace._drain_loop,
            args=(weakref.ref(self),),
            daemon=True,
            name="dunetrace-drain",
        )
        self._drain_thread.start()

        # Flush whatever is still buffered when the process exits. The drain
        # thread is a daemon, so without this the interpreter kills it mid-flight
        # and a short-lived process — a script, a CLI, a one-shot job, a test —
        # ships *nothing*: the events sit in the ring buffer until the process
        # dies. atexit handlers run before daemon threads are torn down, so this
        # is the last point at which a flush can still succeed.
        #
        # Registered via a weakref so the atexit registry doesn't itself keep
        # every client alive for the life of the process. An explicit
        # shutdown() unregisters it, so the common "flush once, cleanly" path
        # doesn't run twice.
        self._atexit_hook = _make_atexit_flush(self)
        if self._atexit_hook is not None:
            atexit.register(self._atexit_hook)

        logger.debug(
            "Dunetrace started. emitter=%s emit_as_json=%s otel=%s exporters=%d",
            type(self._emitter).__name__,
            emit_as_json,
            otel_exporter is not None,
            len(self._exporters),
        )

    def _auth_headers(self) -> Dict[str, str]:
        """Authorization header for gateway-fronted deployments (e.g. the
        Dunetrace Cloud gateway's tenancy middleware, which resolves the
        calling org from ``Authorization: Bearer <api_key>`` and nothing
        else — it never inspects the request body).

        Self-hosted ingest_svc (no gateway in front) still also accepts
        ``api_key`` in the request body/query string, so callers keep
        sending both for backward compatibility; this header is additive.
        """
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

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
        trace_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ):
        """
        Context manager wrapping a single agent run.

        Emits ``run.started`` on enter, ``run.completed`` on clean exit,
        and ``run.errored`` if an exception escapes the block.

        trace_id: optional correlation key for external evaluation
        integrations (Langfuse/LangSmith/Braintrust) — pass the same trace
        id your own instrumentation of that provider's SDK uses, and
        Dunetrace can match its evaluation results back to this run. Not
        folded into agent_version's hash — it identifies this run to an
        external system, not the agent's own identity/config.

        conversation_id: optional grouping key for multi-turn conversations —
        pass the same id across every dt.run() call belonging to the same
        end-user interaction/session, and Dunetrace groups the runs into one
        conversation for cross-turn analysis. Also not folded into
        agent_version's hash, same rationale as trace_id.
        """
        tools = tools or []
        # Normalize caller-supplied text once, up front — see _coerce_text.
        user_input = _coerce_text(user_input)
        system_prompt = _coerce_text(system_prompt)
        try:
            version = agent_version(system_prompt, model, tools)
        except Exception:
            # Grouping degrades to a single bucket; starting the run matters more.
            logger.debug("Dunetrace: agent_version failed, using 'unknown'", exc_info=True)
            version = "unknown"

        # Auto-thread parent_run_id: if this run opens while another run is
        # already active on this task/thread and the caller didn't pass
        # parent_run_id explicitly, inherit the active run's id. This links a
        # nested multi-agent run (an orchestrator opening a sub-agent's own
        # dt.run()) into a parent/child run graph with no manual id threading —
        # the substrate DELEGATION_LOOP and HANDOFF_CONTEXT_LOSS consume. An
        # explicit parent_run_id always wins. Propagation follows contextvars:
        # synchronous nesting and asyncio child tasks inherit automatically;
        # a sub-agent dispatched to a bare thread does not (the thread starts
        # with a fresh context) unless the caller copies context or passes
        # parent_run_id explicitly.
        if parent_run_id is None:
            _active_run = _current_run.get()
            if _active_run is not None:
                parent_run_id = _active_run.run_id

        ctx = RunContext(
            client=self,
            agent_id=agent_id,
            agent_version=version,
            available_tools=tools,
            input_text=user_input,
            system_prompt=system_prompt,
            parent_run_id=parent_run_id,
            trace_id=trace_id,
            conversation_id=conversation_id,
        )

        # Stamp the run with its OTel correlation ids (deterministic from run_id)
        # so a customer can jump from an OTel backend to the Dunetrace run and
        # back. Set only when OTel export is active; None otherwise.
        if self._otel_exporter is not None:
            from dunetrace.integrations.otel import root_span_id_hex, trace_id_hex

            ctx.otel_trace_id = trace_id_hex(ctx.run_id)
            ctx.otel_span_id = root_span_id_hex(ctx.run_id)

        # In-process content inspection: the injection detector runs here, against
        # raw user_input, before the run.started event is built. Only the match
        # evidence (pattern names + count) needs to reach this point — the check
        # itself never needs the event pipeline to see raw text to do its job.
        _injection_evidence = None
        if user_input:
            try:
                _sig = PROMPT_INJECTION_DETECTOR.check_input(user_input, ctx.state)
                if _sig:
                    _injection_evidence = _sig.evidence
            except Exception:
                # Losing one injection signal is strictly better than failing the
                # caller's run — the scan is additive detection, not a gate.
                logger.debug("Dunetrace: injection scan failed", exc_info=True)

        payload: dict = {
            "input_text": user_input,
            "system_prompt": system_prompt,
            "model": model,
            "tools": tools,
        }
        if _injection_evidence:
            payload["injection_signal"] = _injection_evidence
        # Which SDK build, and which provider libraries it patched. Additive and
        # always present for the SDK version; `instrumented` is omitted entirely
        # when nothing was auto-patched, keeping manual callers' run.started
        # byte-identical apart from the one new key.
        try:
            from dunetrace.auto import instrumentation_fingerprint

            _fp = instrumentation_fingerprint()
            payload["sdk_version"] = _fp["sdk_version"]
            if _fp["instrumented"]:
                payload["instrumented"] = _fp["instrumented"]
        except Exception:
            logger.debug("Dunetrace: instrumentation fingerprint failed", exc_info=True)
        _source_file = _capture_caller_source_file()
        if _source_file:
            payload["source_file"] = _source_file

        # _emit() guards itself, but the AgentEvent construction around it does
        # not — and nothing on the run-start path may stop the caller's run from
        # starting. Belt and braces: an untraced run beats a broken one.
        try:
            self._emit(
                AgentEvent(
                    event_type=EventType.RUN_STARTED,
                    run_id=ctx.run_id,
                    agent_id=agent_id,
                    agent_version=version,
                    step_index=0,
                    parent_run_id=parent_run_id,
                    trace_id=trace_id,
                    conversation_id=conversation_id,
                    payload=payload,
                )
            )
        except Exception:
            logger.debug("Dunetrace: failed to record run start", exc_info=True)

        # Fetch remote policies in a background thread so run start isn't delayed.
        try:
            if self._ingest_url and self._api_key and self._policy_engine.needs_fetch(agent_id):
                Thread(target=self._fetch_policies, args=(agent_id,), daemon=True).start()
        except Exception:
            # Thread() can raise RuntimeError under thread exhaustion; policies
            # are best-effort, the run proceeds with whatever is already loaded.
            logger.debug("Dunetrace: policy prefetch could not start", exc_info=True)

        _token = _current_run.set(ctx)
        try:
            try:
                yield ctx
            finally:
                # Before any run.completed/run.errored is emitted, on every exit
                # path: a streamed call the caller broke out of without draining
                # or closing has no other moment to report itself, and an
                # llm.responded arriving after run.completed would miss the run.
                ctx._flush_open_streams()
        except PolicyViolation as exc:
            # Guarded: the caller's PolicyViolation must reach them even if
            # recording it fails. Same for the generic path below.
            try:
                ctx.state.current_step = ctx.step
                ctx.state.exit_reason = "policy_violation"
                self._emit(
                    AgentEvent(
                        event_type=EventType.RUN_ERRORED,
                        run_id=ctx.run_id,
                        agent_id=agent_id,
                        agent_version=version,
                        step_index=ctx.step,
                        trace_id=trace_id,
                        conversation_id=conversation_id,
                        payload={
                            "error_type": "PolicyViolation",
                            "error": str(exc),
                            "exit_reason": "policy_violation",
                            "policy_name": exc.policy_name,
                            "step_index": ctx.step,
                        },
                    )
                )
            except Exception:
                logger.debug("Dunetrace: failed to record policy violation", exc_info=True)
            raise
        except Exception as exc:
            try:
                ctx.state.current_step = ctx.step
                ctx.state.exit_reason = "error"
                self._emit(
                    AgentEvent(
                        event_type=EventType.RUN_ERRORED,
                        run_id=ctx.run_id,
                        agent_id=agent_id,
                        agent_version=version,
                        step_index=ctx.step,
                        trace_id=trace_id,
                        conversation_id=conversation_id,
                        payload={
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "step_index": ctx.step,
                        },
                    )
                )
            except Exception:
                logger.debug("Dunetrace: failed to record run error", exc_info=True)
            raise
        else:
            # Success path. Runs only when nothing escaped the `yield`, and an
            # exception raised in an `else` block is NOT caught by the handlers
            # above — so without this guard an SDK-internal failure here would be
            # reported as if the caller's code had errored, and then re-raised
            # into a caller whose code actually succeeded.
            try:
                ctx._warn_unread_advisory()  # audit Finding 25
                ctx.state.current_step = ctx.step
                self._emit(
                    AgentEvent(
                        event_type=EventType.RUN_COMPLETED,
                        run_id=ctx.run_id,
                        agent_id=agent_id,
                        agent_version=version,
                        step_index=ctx.step,
                        trace_id=trace_id,
                        conversation_id=conversation_id,
                        payload={
                            "total_steps": ctx.step,
                            "exit_reason": ctx.exit_reason or "completed",
                            "tool_call_count": len(ctx.state.tool_calls),
                        },
                    )
                )
            except Exception:
                logger.debug("Dunetrace: failed to record run completion", exc_info=True)
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
                           Also used as the fallback ``default_agent_id`` for
                           ``langchain``/``crewai`` auto-instrumentation — see
                           ``docs/integrations/auto-instrumentation.md`` for
                           the full agent_id resolution order.
        :param frameworks: Subset of frameworks to patch. ``None`` patches all
                           installed ones (openai, anthropic, mistral, httpx,
                           requests, langchain, crewai).
        """
        self._default_agent_id = agent_id or os.environ.get("DUNETRACE_AGENT_ID", "")
        from dunetrace.auto import auto_instrument as _auto_instrument

        _auto_instrument(
            frameworks=frameworks, client=self, default_agent_id=self._default_agent_id or None
        )
        logger.debug("Dunetrace.init() agent_id=%r frameworks=%r", agent_id, frameworks)
        return self

    def auto_instrument(self, frameworks: Optional[List[str]] = None) -> None:
        """
        Monkey-patch supported AI framework clients so that LLM calls made
        inside any ``dt.run()`` context are tracked automatically — no manual
        ``run.llm_called()`` / ``run.llm_responded()`` needed.

        Supported frameworks: ``"openai"``, ``"anthropic"``, ``"httpx"``,
        ``"requests"``, ``"langchain"`` (covers LangGraph), ``"crewai"``.
        Uninstalled frameworks are silently skipped.

        ``langchain`` specifically requires the top-level call to be wrapped
        in ``dt.run(...)`` too — unlike the other frameworks here, it can only
        attach to an already-open run, never open its own. See
        ``docs/integrations/auto-instrumentation.md``. ``crewai`` and the rest
        don't have this requirement; wrapping in ``dt.run()`` is optional for
        them (it only changes which agent_id the run gets attributed to).

        :param frameworks: Subset to patch. ``None`` patches all installed ones.

        Usage::

            dt = Dunetrace(api_key="dt_live_...")
            dt.auto_instrument()   # patch all installed supported frameworks

            with dt.run("my-agent", user_input=query) as run:
                # openai/anthropic calls are now tracked automatically
                response = openai_client.chat.completions.create(...)
                # a LangChain/LangGraph agent.invoke() here would be tracked
                # too, correlated into this same run
        """
        from dunetrace.auto import auto_instrument as _auto_instrument

        _auto_instrument(
            frameworks=frameworks, client=self, default_agent_id=self._default_agent_id or None
        )

    def add_policy(
        self,
        name: str,
        condition: PolicyCondition,
        action: PolicyAction,
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
            {"trigger": "signal",          "operator": "contains", "value": "TOOL_LOOP"}
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

        Calls GET {ingest_url_base}/v1/policies?agent_id=..., authenticating with
        ``Authorization: Bearer <key>``. The key is deliberately **not** put in
        the query string: URLs are logged verbatim by web servers and proxies, so
        a key sent that way ends up in access logs on every single request. The
        server still accepts ?api_key= from older SDK builds, but this one never
        sends it.

        Silently ignores all errors — policies are best-effort.
        """
        if not self._ingest_url or not self._api_key:
            return
        if not self._policy_engine.needs_fetch(agent_id):
            return

        self._policy_engine.mark_fetched(agent_id)  # prevent stampede

        try:
            base = self._ingest_url.replace("/v1/ingest", "")
            url = f"{base}/v1/policies?agent_id={urllib.parse.quote(agent_id, safe='')}"
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": USER_AGENT,
                    **self._auth_headers(),
                },
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                import json as _json

                data = _json.loads(resp.read())
                self._policy_engine.load(
                    data.get("policies", []),
                    secret=self._policy_secret,
                    agent_id=agent_id,
                )
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
        :param system_prompt: Sent as-is to the backend (content-aware detectors need
                             it) and folded into the agent version hash for grouping.
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

        Tool arguments are serialized and transmitted as-is.

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
                        # Suppress the sync approval gate in tool_called and
                        # await the async one instead, so a require_approval
                        # policy doesn't block the event loop while waiting on a
                        # human. Raises ApprovalDenied on deny/timeout, before
                        # the tool runs.
                        run.tool_called(tool_name, args_dict, _enforce_approval=False)
                        await run._enforce_tool_approval_async(tool_name, args_dict)
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
                        result_text = str(result)
                        run.tool_responded(
                            tool_name,
                            success=True,
                            output_length=len(result_text),
                            latency_ms=int((time.time() - t0) * 1000),
                            output=result_text,
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
                        result_text = str(result)
                        run.tool_responded(
                            tool_name,
                            success=True,
                            output_length=len(result_text),
                            latency_ms=int((time.time() - t0) * 1000),
                            output=result_text,
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

    def enable_pack(self, pack_name: str) -> None:
        """
        Activate a detector pack (e.g. "voice") for this org, so
        detector_svc includes its detectors when evaluating this org's
        runs. Built-in detectors run regardless — packs are additive.

        Unlike mark_deploy(), this call is synchronous and raises on
        failure: it's a setup-time action (typically run once from a
        script or at process startup), not a per-run hot-path call, so
        there's no reason to swallow an error the caller would want to
        know about immediately (e.g. an unknown pack name).

        Requires api_key and api_url (or DUNETRACE_API_URL) to be
        configured — this hits the Customer API, not the ingest endpoint.
        """
        self._pack_request("POST", pack_name)

    def disable_pack(self, pack_name: str) -> None:
        """Deactivate a previously-activated pack for this org. See
        enable_pack() for the sync/error-raising rationale."""
        self._pack_request("DELETE", pack_name)

    def enabled_packs(self) -> List[str]:
        """Returns this org's currently-activated pack names, fetched from
        the Customer API. See enable_pack() for the sync/error-raising
        rationale."""
        if not self._api_url or not self._api_key:
            raise RuntimeError(
                "enabled_packs() requires api_key and api_url (or DUNETRACE_API_URL) "
                "to be configured — this reads from the Customer API."
            )
        req = urllib.request.Request(
            f"{self._api_url}/v1/orgs/packs",
            headers={
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
                **self._auth_headers(),
            },
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        return [row["pack_name"] for row in data]

    def _pack_request(self, method: str, pack_name: str) -> None:
        if not self._api_url or not self._api_key:
            raise RuntimeError(
                f"{'enable_pack' if method == 'POST' else 'disable_pack'}() requires "
                "api_key and api_url (or DUNETRACE_API_URL) to be configured — this "
                "calls the Customer API, not the ingest endpoint."
            )
        req = urllib.request.Request(
            f"{self._api_url}/v1/orgs/packs/{urllib.parse.quote(pack_name, safe='')}",
            method=method,
            headers={
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
                **self._auth_headers(),
            },
        )
        with urllib.request.urlopen(req, timeout=5):
            pass

    # ── Approval flow HTTP (Capability 2) ─────────────────────────────────────
    # These hit the Customer API, not the ingest endpoint. Each call is a quick
    # synchronous request; the *waiting* between polls is done by the caller
    # (RunContext.request_approval / arequest_approval), which is what makes a
    # sync-blocking and an async-non-blocking variant possible over the same
    # HTTP helpers.

    def _require_customer_api(self, what: str) -> None:
        if not self._api_url or not self._api_key:
            raise RuntimeError(
                f"{what} requires api_key and api_url (or DUNETRACE_API_URL) to be "
                "configured — approvals use the Customer API, not the ingest endpoint."
            )

    def _create_approval_request(
        self,
        run_id: str,
        agent_id: str,
        tool_name: str,
        tool_args: Optional[str],
        timeout_seconds: int,
    ) -> dict:
        self._require_customer_api("request_approval()")
        body = json.dumps(
            {
                "run_id": run_id,
                "agent_id": agent_id,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "timeout_seconds": timeout_seconds,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self._api_url}/v1/approvals",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
                **self._auth_headers(),
            },
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())

    def _get_approval(self, approval_id: int) -> dict:
        self._require_customer_api("request_approval()")
        req = urllib.request.Request(
            f"{self._api_url}/v1/approvals/{approval_id}",
            headers={
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
                **self._auth_headers(),
            },
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())

    def _decide_approval(self, approval_id: int, decision: str) -> Optional[dict]:
        """Record a decision (the SDK uses this only to mark its own 'timeout').
        Returns the updated approval, or None on a 409 — meaning a real human
        decision won the race with our timeout, and the caller should re-read
        and honor that instead."""
        self._require_customer_api("request_approval()")
        body = json.dumps({"decision": decision, "decision_channel": "sdk"}).encode()
        req = urllib.request.Request(
            f"{self._api_url}/v1/approvals/{approval_id}/decision",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
                **self._auth_headers(),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 409:  # already decided — human won the race
                return None
            raise

    def _ship_deploy(self, agent_id: str, version: str, meta: dict) -> None:
        base = self._ingest_url.replace("/v1/ingest", "")
        payload = json.dumps(
            {
                "api_key": self._api_key,  # self-hosted ingest_svc compat; see _auth_headers
                "agent_id": agent_id,
                "version": version,
                "meta": meta,
            }
        ).encode()
        req = urllib.request.Request(
            base + "/v1/deploy",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
                **self._auth_headers(),
            },
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
        """Flush remaining events and stop the drain thread.

        Idempotent, and safe to call even though an at-exit flush is registered:
        calling it explicitly cancels that hook, so the flush happens once, at
        the point you chose, with your timeout rather than the shorter at-exit
        one. Still worth calling explicitly — it gives the flush a full timeout
        and surfaces problems while your process is still alive to log them.
        """
        hook = getattr(self, "_atexit_hook", None)
        if hook is not None:
            try:
                atexit.unregister(hook)
            except Exception:
                pass
            self._atexit_hook = None
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
            for exporter in self._exporters:
                try:
                    exporter.handle(event)
                except Exception as exc:
                    logger.warning(
                        "Dunetrace: exporter %s failed on %s: %s",
                        exporter,
                        event.event_type,
                        exc,
                    )
            self._buffer.push(event)
        except Exception as exc:
            logger.warning("Dunetrace: failed to emit %s: %s", event.event_type, exc)

    def _write_json_line(self, event: AgentEvent) -> None:
        """
        Write one Loki-compatible NDJSON line to stdout.

        Fields: ts (RFC3339), level ("info"), logger ("dunetrace"), event_type and agent_id
        as Loki stream labels, run_id/step_index/payload as structured fields.
        """
        ts = datetime.datetime.fromtimestamp(event.timestamp, datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
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

    @staticmethod
    def _drain_loop(ref: "weakref.ReferenceType") -> None:
        """Ship buffered events until shutdown, or until the client is collected.

        Takes a *weak* reference rather than running as a bound method. A bound
        method holds the client strongly, and the thread holds the bound method,
        so a client the caller merely dropped could never be collected — its
        drain thread ran for the life of the process, keeping the whole client
        (buffer, emitter, sockets) alive with it. Anything constructing clients
        dynamically — per tenant, per request, per test — leaked one thread each.

        The strong reference is re-acquired each iteration and released before
        every wait, so the client stays collectable while this thread is parked.
        """
        # These are owned by the client but don't reference it back, so holding
        # them across the wait is safe — and lets us wait without a strong ref.
        client = ref()
        if client is None:
            return
        stop_evt, flush_gate = client._stop_evt, client._flush_gate
        flush_interval = client._flush_interval
        del client

        while not stop_evt.is_set():
            client = ref()
            if client is None:
                return  # caller dropped the client — nothing left to ship for
            batch = client._buffer.drain(100)
            if batch:
                client._ship(batch)
                del client
            else:
                del client
                # Wait until either flush() signals us, shutdown() fires, or the
                # interval expires. This lets flush() wake the thread immediately.
                flush_gate.wait(timeout=flush_interval)
                flush_gate.clear()

        client = ref()
        if client is None:
            return
        remaining = client._buffer.drain_all()
        if remaining:
            client._ship(remaining)

    def _ship(self, batch: List[AgentEvent]) -> bool:
        """Delegates to the configured BatchingEmitter (see dunetrace.emitters).
        Defaults to HttpBatchingEmitter — same POST-to-ingest-API behavior as
        before this was made pluggable. Never raises; returns the emitter's
        success/failure so a future durable-retry layer can react to it.
        """
        return self._emitter.ship(batch)


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
