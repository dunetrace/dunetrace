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
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from dunetrace.buffer import RingBuffer
from dunetrace.context import _current_run
from dunetrace.detector_config import DetectorConfigStore
from dunetrace.detectors import PROMPT_INJECTION_DETECTOR
from dunetrace.emitters import (
    USER_AGENT,
    BatchingEmitter,
    HttpBatchingEmitter,
    ShipOutcome,
    as_ship_outcome,
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
from dunetrace.redaction import DEFAULT_MAX_FIELD_CHARS, cap_text, compile_denylist
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


def _resolve_max_field_chars(value: Optional[int]) -> int:
    """The per-field content cap: the constructor argument when given, else
    ``DUNETRACE_MAX_FIELD_CHARS``, else ``DEFAULT_MAX_FIELD_CHARS`` (8192).

    ``0`` disables the cap — an explicit, documented opt-out for callers who
    need full fidelity and accept the payload size. A negative or unparsable
    value is not silently honoured: it logs and falls back to the default,
    because "the cap is off" must never be an accident.
    """
    if value is None:
        raw = os.environ.get("DUNETRACE_MAX_FIELD_CHARS", "").strip()
        if not raw:
            return DEFAULT_MAX_FIELD_CHARS
        try:
            value = int(raw)
        except ValueError:
            logger.warning(
                "Dunetrace: bad DUNETRACE_MAX_FIELD_CHARS=%r, using default %d",
                raw,
                DEFAULT_MAX_FIELD_CHARS,
            )
            return DEFAULT_MAX_FIELD_CHARS
    try:
        value = int(value)
    except (TypeError, ValueError):
        logger.warning(
            "Dunetrace: bad max_field_chars=%r, using default %d", value, DEFAULT_MAX_FIELD_CHARS
        )
        return DEFAULT_MAX_FIELD_CHARS
    if value < 0:
        logger.warning(
            "Dunetrace: max_field_chars must be >= 0 (0 disables the cap), got %d; "
            "using default %d",
            value,
            DEFAULT_MAX_FIELD_CHARS,
        )
        return DEFAULT_MAX_FIELD_CHARS
    return value


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
        policy_cache_path: Optional[str] = None,
        max_field_chars: Optional[int] = None,
        redact: Optional[Callable[[dict], dict]] = None,
        redact_keys: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Content caps and redaction (see ``dunetrace.redaction``):

        :param max_field_chars: Per-field character cap on every free-text
            value the SDK ships — tool args and output, LLM output, retrieval
            query/content, memory values, ``input_text`` and ``system_prompt``.
            Default 8192 (env ``DUNETRACE_MAX_FIELD_CHARS``), the same limit the
            OTLP ingest path enforces. A capped field carries
            ``<field>_truncated: true`` and ``<field>_original_length: N`` next
            to it; length fields such as ``output_length`` always report the
            real size. ``0`` disables the cap. In-path detectors read the
            capped text too (``ToolCall.args_length`` keeps the real length for
            OVERSIZED_TOOL_ARGUMENTS).
        :param redact: Optional ``dict -> dict`` hook applied to structured tool
            args before they are serialised, ahead of the built-in denylist —
            strip or mask whatever the denylist cannot know about (account
            numbers, free-text PII). It receives a shallow copy and must return
            a new dict; do not mutate nested values in place, the agent's tool
            is about to run on them. If it raises or returns a non-dict the SDK
            logs a WARNING once per process, drops its output, and continues
            with the built-in denylist alone — the agent is never blocked by
            its own redaction code. Runs on every tool call, so keep it cheap.
        :param redact_keys: Extra key names for the built-in denylist. Matching
            is case-insensitive after normalising ``-`` to ``_``, and a key
            matches when it equals an entry or ends with ``_<entry>`` — the
            defaults (``authorization``, ``api_key``, ``apikey``, ``token``,
            ``secret``, ``password``, ``cookie``, ``set-cookie``) therefore
            also catch ``Authorization``, ``X-Api-Key``, ``access_token``,
            ``client_secret``, ``db_password``. Matched values become
            ``"[REDACTED]"``; keys are kept. Only tool args (``tool_called``
            and approval requests) are structured enough to redact by key;
            plain-text fields are capped, not redacted.
        """
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

        # Live RunContexts by run_id, so _emit() can flush a run's abandoned
        # streams on the terminal event itself — see _flush_run_streams. Weak
        # values: a caller that drops a run without ever finishing it must not
        # be kept alive by this index.
        self._run_contexts: "weakref.WeakValueDictionary[str, RunContext]" = (
            weakref.WeakValueDictionary()
        )
        self._run_contexts_lock = Lock()

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
        # Server-authoritative detector thresholds for the in-path pass
        # (dunetrace/detector_config.py), refreshed by the same background
        # thread as the policy bundle with the same TTL/backoff bookkeeping.
        self._detector_config_store = DetectorConfigStore()

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

        # Opt-in on-disk cache of the last remote policy bundle, one file per
        # agent under this directory (env DUNETRACE_POLICY_CACHE_PATH). A
        # process that starts while the policy server is unreachable primes
        # the engine from it — through PolicyEngine.load(), so signature
        # verification and the unsigned-enforcing-action downgrade apply to
        # the cached copy exactly as to a live one. Default off: nothing is
        # written anywhere unless the caller asks.
        self._policy_cache_path = (
            policy_cache_path
            if policy_cache_path is not None
            else os.environ.get("DUNETRACE_POLICY_CACHE_PATH", "")
        )
        self._policy_cache_tried: set = set()  # agent_ids primed (or attempted) from cache
        self._policy_cache_write_warned = False

        # Content caps and redaction — read by RunContext on every hook via
        # self._client, so they are plain attributes, not properties.
        self._max_field_chars: int = _resolve_max_field_chars(max_field_chars)
        if redact is not None and not callable(redact):
            raise TypeError(f"redact must be callable (dict -> dict), got {type(redact).__name__}")
        self._redact_hook = redact
        self._redact_denylist = compile_denylist(redact_keys)

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

        # Content caps (dunetrace.redaction). Applied after the version hash,
        # so grouping is stable however long the prompt is, and before RunState
        # is built, so in-path detectors and the server read the same text.
        # The injection scan below still sees the raw user_input: it runs
        # in-process, is already windowed, and the tail of an oversized prompt
        # is exactly where an injection would hide.
        # getattr, not self._max_field_chars: a client built without __init__
        # (framework tests construct one via __new__) still starts runs, on the
        # documented default cap.
        _max_chars = getattr(self, "_max_field_chars", DEFAULT_MAX_FIELD_CHARS)
        input_text, input_truncated, input_len = cap_text(user_input, _max_chars)
        sys_prompt, sys_prompt_truncated, sys_prompt_len = cap_text(system_prompt, _max_chars)

        ctx = RunContext(
            client=self,
            agent_id=agent_id,
            agent_version=version,
            available_tools=tools,
            input_text=input_text,
            system_prompt=sys_prompt,
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
            "input_text": input_text,
            "system_prompt": sys_prompt,
            "model": model,
            "tools": tools,
        }
        # Markers only when a cut happened, so an ordinary run.started is
        # byte-identical to before.
        if input_truncated:
            payload["input_text_truncated"] = True
            payload["input_text_original_length"] = input_len
        if sys_prompt_truncated:
            payload["system_prompt_truncated"] = True
            payload["system_prompt_original_length"] = sys_prompt_len
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

        # Fetch remote policies and detector config in one background thread so
        # run start isn't delayed. Each has its own TTL/backoff; the thread is
        # started when either is due and each fetch re-checks for itself.
        try:
            if (
                self._ingest_url
                and self._api_key
                and (
                    self._policy_engine.needs_fetch(agent_id)
                    or self._detector_config_store.needs_fetch(agent_id)
                )
            ):
                Thread(
                    target=self._fetch_remote_config, args=(agent_id, version), daemon=True
                ).start()
        except Exception:
            # Thread() can raise RuntimeError under thread exhaustion; policies
            # are best-effort, the run proceeds with whatever is already loaded.
            logger.debug("Dunetrace: remote config prefetch could not start", exc_info=True)

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
                        payload=self._terminal_payload(
                            ctx.run_id,
                            {
                                "error_type": "PolicyViolation",
                                "error": str(exc),
                                "exit_reason": "policy_violation",
                                "policy_name": exc.policy_name,
                                "step_index": ctx.step,
                            },
                        ),
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
                        payload=self._terminal_payload(
                            ctx.run_id,
                            {
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "step_index": ctx.step,
                            },
                        ),
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
                        payload=self._terminal_payload(
                            ctx.run_id,
                            {
                                "total_steps": ctx.step,
                                "exit_reason": ctx.exit_reason or "completed",
                                "tool_call_count": len(ctx.state.tool_calls),
                            },
                        ),
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

        Never raises — policies are best-effort and this runs on a daemon
        thread — but a failure is not silent either:

        * ``mark_fetched`` is only called on success. A failed fetch goes
          through ``mark_fetch_failed`` and is retried on the engine's
          backoff (2s, 4s, 8s, capped at 15s), not after the 60s success TTL.
        * The first failure in a streak logs at WARNING (agent, host,
          exception class and message); later failures in the same streak
          at DEBUG; the success that ends a streak at INFO.
        * Fail-open: whatever bundle was last loaded stays in memory and
          keeps being enforced while fetches fail.
        * ``begin_fetch``/``end_fetch`` are the stampede guard, so two runs
          starting together still make one request.
        """
        if not self._ingest_url or not self._api_key:
            return
        engine = self._policy_engine
        if not engine.needs_fetch(agent_id):
            return
        if not engine.begin_fetch(agent_id):
            return  # another thread already has this agent's fetch on the wire

        try:
            self._prime_policies_from_cache(agent_id)

            host = self._ingest_host()
            try:
                data = self._get_json(
                    f"/v1/policies?agent_id={urllib.parse.quote(agent_id, safe='')}"
                )
                if not isinstance(data, dict):
                    raise ValueError(f"policy response is {type(data).__name__}, expected object")
                engine.load(
                    data.get("policies", []),
                    secret=self._policy_secret,
                    agent_id=agent_id,
                )
            except Exception as exc:
                streak = engine.mark_fetch_failed(agent_id)
                retained = engine.has_remote_bundle(agent_id)
                logger.log(
                    logging.WARNING if streak == 1 else logging.DEBUG,
                    "Dunetrace: remote policy fetch failed for agent %r from %s "
                    "(%s: %s); retrying in %.0fs, failure %d in a row; %s",
                    agent_id,
                    host,
                    type(exc).__name__,
                    exc,
                    engine.fetch_backoff(streak),
                    streak,
                    "still enforcing the last loaded bundle"
                    if retained
                    else "no remote policies are loaded for this agent",
                )
                return

            streak = engine.fetch_failure_streak(agent_id)
            engine.mark_fetched(agent_id)
            if streak:
                logger.info(
                    "Dunetrace: remote policy fetch for agent %r from %s recovered "
                    "after %d failure(s)",
                    agent_id,
                    host,
                    streak,
                )
            self._write_policy_cache(agent_id, data)
        except Exception:
            # Bookkeeping above must not be able to take the thread down; the
            # engine's failure record (if any) already drives the retry.
            logger.debug("Dunetrace: policy fetch bookkeeping failed", exc_info=True)
        finally:
            engine.end_fetch(agent_id)

    def _fetch_remote_config(self, agent_id: str, agent_version: str = "") -> None:
        """Background refresh of everything this agent pulls from the server:
        the policy bundle and the detector configuration, in this one thread,
        each on its own bookkeeping. Never raises."""
        try:
            if self._policy_engine.needs_fetch(agent_id):
                self._fetch_policies(agent_id)
        except Exception:
            logger.debug("Dunetrace: policy fetch raised", exc_info=True)
        try:
            if self._detector_config_store.needs_fetch(agent_id):
                self._fetch_detector_config(agent_id, agent_version)
        except Exception:
            logger.debug("Dunetrace: detector config fetch raised", exc_info=True)

    def _ingest_base(self) -> str:
        return self._ingest_url.replace("/v1/ingest", "")

    def _ingest_host(self) -> str:
        base = self._ingest_base()
        return urllib.parse.urlsplit(base).netloc or base

    def _get_json(self, path_and_query: str, timeout: float = 3.0) -> Any:
        """GET ``{ingest base}{path_and_query}`` and return the parsed JSON
        body. Authenticates with ``Authorization: Bearer <key>`` — never a
        query parameter, which would land the key in every access log.
        Raises on any transport, HTTP or parse failure; callers decide how to
        record it."""
        req = urllib.request.Request(
            f"{self._ingest_base()}{path_and_query}",
            headers={
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
                **self._auth_headers(),
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def _fetch_detector_config(self, agent_id: str, agent_version: str = "") -> None:
        """
        Background fetch of the server's effective detector configuration for
        ``agent_id`` — ``GET /v1/detector-config`` — into the
        DetectorConfigStore, so the in-path pass runs the thresholds, packs and
        baselines the detector worker would (see dunetrace/detector_config.py).

        Same contract as ``_fetch_policies``: never raises; a failure goes
        through ``mark_fetch_failed`` and is retried on the 2s/4s/8s/15s
        backoff, the first failure in a streak logs WARNING and later ones
        DEBUG, recovery logs INFO; ``begin_fetch``/``end_fetch`` are the
        stampede guard. Fail-open: the last configuration loaded stays in use
        while fetches fail, and before any has loaded the pass runs the
        detector class defaults (``TIER1_DETECTORS``). Either way the
        policy.evaluated payload says so via ``detector_config_stale``.
        """
        if not self._ingest_url or not self._api_key:
            return
        store = self._detector_config_store
        if not store.needs_fetch(agent_id):
            return
        if not store.begin_fetch(agent_id):
            return  # another thread already has this agent's fetch on the wire

        try:
            host = self._ingest_host()
            query = f"agent_id={urllib.parse.quote(agent_id, safe='')}"
            if agent_version:
                query += f"&agent_version={urllib.parse.quote(agent_version, safe='')}"
            try:
                data = self._get_json(f"/v1/detector-config?{query}")
                if not isinstance(data, dict):
                    raise ValueError(
                        f"detector config response is {type(data).__name__}, expected object"
                    )
                store.load(agent_id, data)
            except Exception as exc:
                streak = store.mark_fetch_failed(agent_id)
                logger.log(
                    logging.WARNING if streak == 1 else logging.DEBUG,
                    "Dunetrace: detector config fetch failed for agent %r from %s "
                    "(%s: %s); retrying in %.0fs, failure %d in a row; %s",
                    agent_id,
                    host,
                    type(exc).__name__,
                    exc,
                    store.fetch_backoff(streak),
                    streak,
                    "still using the last configuration received"
                    if store.has_config(agent_id)
                    else "in-path detectors are running on class defaults",
                )
                return

            streak = store.fetch_failure_streak(agent_id)
            store.mark_fetched(agent_id)
            if streak:
                logger.info(
                    "Dunetrace: detector config fetch for agent %r from %s recovered "
                    "after %d failure(s)",
                    agent_id,
                    host,
                    streak,
                )
        except Exception:
            logger.debug("Dunetrace: detector config fetch bookkeeping failed", exc_info=True)
        finally:
            store.end_fetch(agent_id)

    # ── Optional on-disk policy cache ─────────────────────────────────────────

    def _policy_cache_file(self, agent_id: str) -> str:
        return os.path.join(
            self._policy_cache_path, urllib.parse.quote(agent_id, safe="") + ".json"
        )

    def _prime_policies_from_cache(self, agent_id: str) -> None:
        """Load the cached raw bundle for ``agent_id`` into the engine, once,
        and only while no remote bundle is in memory for it. Goes through
        PolicyEngine.load() so a tampered file gets exactly the treatment a
        tampered server response gets: signature checked when a secret is
        set, enforcing actions downgraded to log-only when it is not. The
        engine is NOT marked fetched, so the bundle reads as stale and the
        network fetch still happens."""
        if not self._policy_cache_path or agent_id in self._policy_cache_tried:
            return
        self._policy_cache_tried.add(agent_id)
        if self._policy_engine.has_remote_bundle(agent_id):
            return
        path = self._policy_cache_file(agent_id)
        try:
            if not os.path.exists(path):
                return
            with open(path, "r", encoding="utf-8") as fh:
                cached = json.load(fh)
            if not isinstance(cached, dict) or cached.get("agent_id") != agent_id:
                logger.warning(
                    "Dunetrace: ignoring policy cache %s: not a bundle for %r", path, agent_id
                )
                return
            policies = cached.get("policies")
            if not isinstance(policies, list):
                logger.warning("Dunetrace: ignoring policy cache %s: malformed", path)
                return
            self._policy_engine.load(policies, secret=self._policy_secret, agent_id=agent_id)
            logger.info(
                "Dunetrace: primed %d cached remote policy(ies) for agent %r from %s; "
                "bundle is stale until a fetch succeeds",
                len(policies),
                agent_id,
                path,
            )
        except Exception as exc:
            logger.warning(
                "Dunetrace: could not load policy cache %s: %s: %s", path, type(exc).__name__, exc
            )

    def _write_policy_cache(self, agent_id: str, data: Dict[str, Any]) -> None:
        """Atomically persist the raw server response for ``agent_id``."""
        if not self._policy_cache_path:
            return
        path = self._policy_cache_file(agent_id)
        tmp = f"{path}.tmp-{os.getpid()}"
        try:
            os.makedirs(self._policy_cache_path, exist_ok=True)
            payload = {
                "agent_id": agent_id,
                "fetched_at": time.time(),
                "policies": data.get("policies", []),
            }
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, path)
        except Exception as exc:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            # A cache that cannot be written is a degraded restart story, not
            # a failure now: say so once, then stay quiet.
            logger.log(
                logging.DEBUG if self._policy_cache_write_warned else logging.WARNING,
                "Dunetrace: could not write policy cache %s: %s: %s",
                path,
                type(exc).__name__,
                exc,
            )
            self._policy_cache_write_warned = True

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

        Tool arguments are bound to the function's parameter names, passed
        through the ``redact`` hook and built-in denylist, JSON-serialised and
        capped at ``max_field_chars`` (see ``Dunetrace.__init__``); the return
        value is ``str()``-ed and capped the same way.

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

    _TERMINAL_EVENT_TYPES = frozenset({EventType.RUN_COMPLETED, EventType.RUN_ERRORED})

    def _terminal_payload(self, run_id: str, payload: dict) -> dict:
        """Stamp ``dropped_events`` onto a run.completed / run.errored payload.

        The buffer sheds whole runs under overload (see ``dunetrace.buffer``).
        Reserving the terminal's slot *first* means any shed needed to fit it
        is already reflected in the count; the key is omitted when nothing was
        dropped so a healthy run's wire format is unchanged. The detector reads
        it into ``RunState.dropped_events`` and holds the run's signals in
        shadow. Never raises — a failure here costs the marker, not the event.
        """
        try:
            dropped = self._buffer.reserve_terminal(run_id)
        except Exception:
            logger.debug("Dunetrace: dropped_events lookup failed", exc_info=True)
            dropped = 0
        if dropped > 0:
            payload["dropped_events"] = dropped
        return payload

    def _register_run(self, ctx: "RunContext") -> None:
        """Index a live RunContext by run_id. Called from RunContext.__init__.

        Every path that creates a run goes through RunContext — dt.run() and
        each framework integration alike — so this is where _emit() can find a
        run's open streams when its terminal event arrives.
        """
        try:
            with self._run_contexts_lock:
                self._run_contexts[ctx.run_id] = ctx
        except Exception:  # pragma: no cover - defensive
            logger.debug("Dunetrace: could not index run context", exc_info=True)

    def _flush_run_streams(self, run_id: str) -> None:
        """Finalize a run's abandoned streams, just before its terminal event.

        ``dt.run()`` calls ``RunContext._flush_open_streams()`` in its own
        ``finally`` (and it is idempotent, so this is a no-op there), but a
        framework integration builds a ``RunContext`` directly and emits its
        own ``run.completed`` / ``run.errored`` — LangChain's ``on_chain_end``
        / ``on_chain_error``, OpenAI-Agents' ``_finish_run``. Neither ever
        called the flush, so a stream the caller broke out of reported itself
        from ``_StreamProxy.__del__`` at garbage-collection time, emitting
        ``llm.responded`` into a run that had already closed.

        Hanging it off the terminal event is the one place no integration can
        forget: the flush's own ``llm.responded`` events are emitted (and
        buffered) before the terminal, which is the ordering the run builders
        need. Never raises — a failure here must not cost the terminal event.
        """
        try:
            with self._run_contexts_lock:
                ctx = self._run_contexts.pop(run_id, None)
            if ctx is not None:
                ctx._flush_open_streams()
        except Exception:
            logger.debug("Dunetrace: failed to flush open streams", exc_info=True)

    def _emit(self, event: AgentEvent) -> None:
        if event.event_type in self._TERMINAL_EVENT_TYPES:
            # Before the terminal is recorded anywhere: these emit their own
            # events, which must land ahead of it.
            self._flush_run_streams(event.run_id)
        # Each side channel is guarded on its own. They used to share one try
        # with the buffer push that follows them, so an NDJSON line or an OTel
        # exporter that choked on one payload cost the event its place in the
        # buffer too — the ingest API never saw it at all.
        if self._emit_json:
            try:
                self._write_json_line(event)
            except Exception as exc:
                logger.warning("Dunetrace: NDJSON write failed for %s: %s", event.event_type, exc)
        if self._otel_exporter is not None:
            try:
                self._otel_exporter.handle(event)
            except Exception as exc:
                logger.warning("Dunetrace: OTel export failed for %s: %s", event.event_type, exc)
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
        try:
            # Terminal events are forced through: they carry the run's
            # dropped_events count, so shedding one would hide the loss.
            self._buffer.push(event, force=event.event_type in self._TERMINAL_EVENT_TYPES)
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

        # default=str: emit_as_json runs synchronously inside _emit(), and a
        # payload value json.dumps cannot represent would otherwise raise past
        # the buffer push below it and lose the event outright.
        serialised = json.dumps(line, default=str, separators=(",", ":"))
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
            try:
                batch = client._buffer.drain(100)
            except Exception:
                # The one unguarded call left in the loop. If the buffer ever
                # raises, park for an interval rather than end the thread: a
                # dead drain thread is silent, permanent loss of every later
                # event, and there is no supervisor to restart it.
                logger.warning("Dunetrace: buffer drain failed", exc_info=True)
                del client
                flush_gate.wait(timeout=flush_interval)
                flush_gate.clear()
                continue
            if batch:
                try:
                    client._ship(batch)
                except Exception:
                    # _ship already catches; this is the backstop that keeps
                    # the loop alive no matter what it is wired to.
                    logger.warning("Dunetrace: drain iteration failed", exc_info=True)
                del client
            else:
                # Nothing new to ship: give the emitter its turn at any batch
                # it is holding for retry, so a backoff that came due while
                # the agent is quiet is not stuck until the next event. The
                # emitter does not reference the client back, so it is safe
                # to hold across the call after the strong ref is released.
                emitter = client._emitter
                del client
                try:
                    emitter.retry_pending()
                except Exception:
                    logger.debug("retry_pending() raised; ignoring.", exc_info=True)
                del emitter
                # Wait until either flush() signals us, shutdown() fires, or the
                # interval expires. This lets flush() wake the thread immediately.
                flush_gate.wait(timeout=flush_interval)
                flush_gate.clear()

        client = ref()
        if client is None:
            return
        try:
            remaining = client._buffer.drain_all()
        except Exception:
            logger.warning("Dunetrace: final buffer drain failed", exc_info=True)
            remaining = []
        if remaining:
            try:
                client._ship(remaining)
            except Exception:
                logger.warning("Dunetrace: final drain failed", exc_info=True)
        # Last chance for anything already due; retries not yet due are lost
        # with the process (in-memory), which DurableRetryEmitter exists for.
        try:
            client._emitter.retry_pending()
        except Exception:
            logger.debug("retry_pending() raised at shutdown; ignoring.", exc_info=True)

    def _ship(self, batch: List[AgentEvent]) -> ShipOutcome:
        """Delegates to the configured BatchingEmitter (see dunetrace.emitters).
        Defaults to HttpBatchingEmitter — same POST-to-ingest-API behavior as
        before this was made pluggable. Never raises; returns the emitter's
        ShipOutcome (normalised, since a third-party emitter may still return a
        plain bool) so a durable-retry layer can tell a transient failure from a
        batch the destination will never accept.

        ``ship()`` is documented as non-raising and every built-in emitter
        honours it, but ``emitter=`` is a public extension point and this call
        runs on the drain thread: an exception escaping here used to kill that
        thread outright, after which every event buffered forever and the
        customer lost all observability with no log line of our own. One
        warning and a failed batch is the right price.
        """
        try:
            return as_ship_outcome(self._emitter.ship(batch))
        except Exception as exc:
            logger.warning(
                "Dunetrace: emitter %s raised from ship(); %d event(s) dropped: %s",
                type(self._emitter).__name__,
                len(batch),
                exc,
                exc_info=True,
            )
            return ShipOutcome.RETRYABLE


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
