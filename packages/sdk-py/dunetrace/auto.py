"""
Automatic instrumentation patches for popular AI frameworks.

Patches are applied at the class level so all client instances are covered.
Each patch is idempotent — calling auto_instrument() more than once is safe.

Supported frameworks:
- ``openai``    — chat.completions.create (sync + async), streamed and not
- ``anthropic`` — messages.create (sync + async), streamed and not, plus
                  messages.stream. That last one needs its own patch: it calls
                  ``self._post`` directly and never routes through ``create``.
- ``mistral``   — chat.{complete,stream} and their _async twins,
                  embeddings.create, and fim.{complete,stream} (+ _async).
                  Requires mistralai v2, which moved every module under
                  ``mistralai.client``. ``Chat.parse``/``parse_stream`` are not
                  patched because they call ``complete``/``stream`` internally
                  and would double-count.

Streamed calls are wrapped in a ``_StreamProxy`` that accumulates token counts
as chunks pass through and emits one ``llm.responded`` when the stream is
exhausted or its context manager exits. Mistral and Anthropic report real usage
on a stream by default; OpenAI only does when the caller passes
``stream_options={"include_usage": True}``, so output tokens are estimated from
the accumulated text otherwise. Injecting that option ourselves would append a
usage-only chunk with an empty ``choices`` list and break callers doing
``chunk.choices[0]``.
- ``httpx``     — Client.send + AsyncClient.send (all outbound HTTP as tool calls)
- ``requests``  — Session.send (all outbound HTTP as tool calls)
- ``langchain`` — BaseChatModel.{invoke,ainvoke,stream,astream} + BaseTool.{run,arun},
                  covers LangGraph too (its nodes call these same base-class methods).
                  Also patches LangGraph's ``BaseStore.{put,get,delete}`` (+ async) so
                  long-term memory writes/reads/clears become memory.* events —
                  ``BaseStore`` implements these concretely and delegates to an
                  abstract ``batch``, so one patch covers every store backend.
- ``crewai``    — Crew/Agent.{kickoff,kickoff_async} for the run boundary, plus
                  CrewAI's own global before/after LLM+tool hooks. Also patches the
                  CrewAI memory classes' ``save``/``search``/``reset`` so short-term,
                  long-term, and entity memory become memory.* events.

The memory patches (LangGraph ``BaseStore``, CrewAI memory) only ever attach to an
already-open ``dt.run()`` — like the openai/anthropic patches — and emit
``memory.*`` annotation events (they don't advance the step counter). Framework
memory APIs don't expose the *provenance* of a written value, so auto-captured
writes carry no ``source``; use the manual ``run.memory_written(..., source=...)``
API when the provenance is known (feeds the MEMORY_POISONING detector's
risk weighting).

``crewai`` is different in kind from ``openai``/``anthropic``/``httpx``/
``requests``: those four just react to an already-open ``dt.run()`` and never
need to know an agent_id, whereas CrewAI patches the true top-level entry
point (``Crew.kickoff``/``Agent.kickoff``) and can open its *own* run when
none is open — see ``dunetrace.integrations._agent_resolution`` for the
resolution order it uses to pick an agent_id in that case.

``langchain`` is a real exception, not just a variant: it can only ever
*attach* to an already-open ``dt.run()``; it never opens its own. This is
because the handler's run-creation logic hangs off LangChain's
``on_chain_start`` callback, which fires only when a callback is attached at
the top-level chain/agent invoke — but ``auto_instrument()`` attaches the
handler at the ``BaseChatModel``/``BaseTool`` leaf level instead (the one
patch surface shared by every provider and every agent framework built on
LangChain), so ``on_chain_start`` never fires for it. Wrap the top-level
call in ``with dt.run(agent_id=...):`` for LangChain/LangGraph auto-
instrumentation to correlate correctly — see
docs/integrations/auto-instrumentation.md for the full explanation and the
agent_id resolution order.

The ``openai``/``anthropic``/``httpx``/``requests`` patches also check a
re-entrancy flag (``dunetrace.context._in_framework_call``) and skip emitting
their own event when a framework-level integration is already emitting one
for the same logical call — e.g. a LangChain call backed by ``ChatOpenAI``
would otherwise be counted once by the LangChain integration and again by the
raw ``openai`` patch underneath it.

Usage::

    dt = Dunetrace(...)
    dt.auto_instrument()                          # patch all detected frameworks
    dt.auto_instrument(["openai", "anthropic"])   # patch only LLM clients
    dt.auto_instrument(["httpx", "requests"])     # patch only HTTP clients
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import importlib
import json
import logging
import sys
import time
from typing import TYPE_CHECKING, List, Optional

from dunetrace.context import _current_run, _http_suppressed, _in_framework_call
from dunetrace.policies import ApprovalDenied, PolicyViolation

if TYPE_CHECKING:
    from dunetrace.client import Dunetrace

logger = logging.getLogger("dunetrace.auto")

# Tracks which frameworks have already been patched (prevents double-wrapping).
_PATCHED: set[str] = set()
# Package -> installed version, for every provider actually patched this process.
# Emitted on run.started so a bad run can be correlated with the exact SDK and
# provider-library versions that produced it. The SDK version previously existed
# only in a User-Agent header that is never stored alongside events, so neither
# side could answer "which build emitted this?" after the fact.
_INSTRUMENTED_VERSIONS: dict[str, str] = {}


def _record_instrumented(pkg: str) -> None:
    try:
        from importlib.metadata import version as _pkg_version

        _INSTRUMENTED_VERSIONS[pkg] = _pkg_version(pkg)
    except Exception:
        # Version lookup must never break instrumentation. "unknown" still tells
        # a reader the package was patched, which is the load-bearing half.
        _INSTRUMENTED_VERSIONS[pkg] = "unknown"


def instrumentation_fingerprint() -> dict:
    """{sdk_version, instrumented: {pkg: version}} for the current process."""
    from dunetrace import __version__

    return {"sdk_version": __version__, "instrumented": dict(_INSTRUMENTED_VERSIONS)}


def _safe_emit(action, *, swallow_control_flow: bool = False) -> None:
    """Run one Dunetrace-side emit, swallowing any error it raises.

    Auto-instrumentation wraps calls the *host application* depends on — an
    LLM request, an HTTP request, a framework entry point. Instrumentation is
    strictly additive: a bug on our side must never stop that call from
    happening, never discard its (already paid-for) result, and never replace
    the exception it raised with one of ours. Every ``run.*`` / ``_emit_*``
    call inside a patched wrapper therefore goes through here.

    ``PolicyViolation`` and ``ApprovalDenied`` are *not* failures — they are the
    runtime-prevention feature working as designed (a ``stop`` policy, a denied
    approval), so by default they propagate to the caller. Pass
    ``swallow_control_flow=True`` on the error path of a wrapper, where the
    host's original exception is the more important signal and must win.
    """
    try:
        action()
    except (PolicyViolation, ApprovalDenied):
        if swallow_control_flow:
            logger.debug("dunetrace: policy control-flow suppressed on error path", exc_info=True)
            return
        raise
    except Exception:
        logger.debug("dunetrace instrumentation emit failed", exc_info=True)


# Response shapes already reported this process. One warning per shape, not per
# call: a broken extractor fires on every LLM call in every run, and a per-call
# warning would be throttled away as noise by any log aggregator.
_WARNED_SHAPES: set[str] = set()


def _degraded_marker(vendor: str, resp: object) -> str:
    """Name the shape that defeated extraction, and warn once per process.

    WARNING, not DEBUG, deliberately. The incident this exists for ran at
    DEBUG-equivalent silence for its entire duration: a LangGraph agent whose
    `with_raw_response.create()` returned a LegacyAPIResponse instead of a
    ChatCompletion produced a fabricated ("", "stop") pair on 100% of runs,
    which is byte-for-byte EMPTY_LLM_RESPONSE's trigger condition. Nothing
    logged. Discovery took a human noticing a HIGH-severity detector firing
    16/16 including the control run.
    """
    marker = f"{vendor}_response_shape:{type(resp).__name__}"
    if marker not in _WARNED_SHAPES:
        _WARNED_SHAPES.add(marker)
        logger.warning(
            "dunetrace: cannot read %s response of type %r — recording this call as "
            "unmeasurable rather than guessing. Detectors that key on completion "
            "text or finish_reason are suppressed for affected runs and an "
            "INSTRUMENTATION_DEGRADED signal is emitted instead. This usually means "
            "a wrapper returned a raw/streaming envelope (e.g. "
            "with_raw_response.create() -> LegacyAPIResponse) rather than the "
            "parsed model. Reported once per process per shape.",
            vendor,
            type(resp).__name__,
        )
    return marker


@contextlib.contextmanager
def _suppress_http():
    """Mark the wrapped provider call as already-instrumented HTTP.

    Every LLM SDK patched here rides on httpx or requests, so the HTTP patches
    would otherwise record the same call a second time as a tool call named
    after the hostname. Wrap only the provider call itself, never our own emits,
    so a caller's genuine tool HTTP inside the same run is unaffected.
    """
    token = _http_suppressed.set(True)
    try:
        yield
    finally:
        _http_suppressed.reset(token)


def _http_run():
    """The run an HTTP patch should attribute to, or None when something further
    up the stack is already recording this call."""
    if _in_framework_call.get() or _http_suppressed.get():
        return None
    return _current_run.get()


# ── Streaming ─────────────────────────────────────────────────────────────────


def _new_stream_acc() -> dict:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "text": [],
        "finish_reason": "",
        # Characters of streamed tool-call payload (function name + arguments).
        # Counted separately from text because a tool-call turn produces no
        # content at all — see _openai_stream_collector.
        "tool_arg_chars": 0,
        "error": None,
    }


def _emit_stream_response(run, acc: dict, t0: float, call_index: Optional[int] = None) -> None:
    text = "".join(acc["text"])
    completion = acc["completion_tokens"]
    # Tool-call arguments are output the caller pays for even though they never
    # appear as content, so they count toward the estimate. Without this a
    # streamed tool-calling turn — the dominant shape of agent traffic — reports
    # zero output cost.
    billable_chars = len(text) + acc["tool_arg_chars"]
    if not completion and billable_chars:
        # OpenAI only reports usage on a stream when the caller passed
        # stream_options={"include_usage": True}. Injecting that ourselves would
        # append a final chunk with an empty choices list and break any caller
        # doing chunk.choices[0], so estimate from what came through instead.
        # Same 4-chars-per-token heuristic llm_called uses on the prompt side.
        # Mistral and Anthropic both report real usage on a stream by default,
        # so this fallback only ever fires for OpenAI.
        completion = max(1, billable_chars // 4)

    observed_anything = bool(text) or bool(acc["tool_arg_chars"]) or bool(acc["error"])
    finish_reason = acc["finish_reason"] or ("stop" if observed_anything else None)
    degraded = (
        None if finish_reason is not None else "stream_yielded_nothing:no_chunks_no_finish_reason"
    )
    if degraded is not None and degraded not in _WARNED_SHAPES:
        _WARNED_SHAPES.add(degraded)
        logger.warning(
            "dunetrace: a streamed LLM call ended with no chunks, no finish_reason "
            "and no error — recording it as unmeasurable rather than as an empty "
            "'stop' response. Reported once per process."
        )
    run.llm_responded(
        completion_tokens=completion,
        prompt_tokens=acc["prompt_tokens"],
        latency_ms=int((time.monotonic() - t0) * 1000),
        # A stream that died mid-flight must not be reported as a clean "stop":
        # EmptyLlmResponseDetector keys on finish_reason == "stop" with zero
        # output length, so a failure before the first token would otherwise be
        # misclassified as an empty response rather than counted as an error.
        #
        # The remaining fallback is narrower than it looks. A stream that carried
        # content but no terminal finish_reason demonstrably ran to completion,
        # so "stop" is an inference from observed behaviour rather than a guess
        # about an unread object. A stream that yielded NOTHING — no text, no
        # tool-call arguments, no finish_reason, no error — was not measured at
        # all, and calling that "stop" reproduces the exact ("", "stop") pair
        # this whole change exists to stop fabricating.
        finish_reason=finish_reason,
        output=text,
        output_length=len(text),
        error=acc["error"],
        instrumentation_degraded=degraded,
        call_index=call_index,
    )


class _StreamProxy:
    """Transparent wrapper around a provider stream that measures it in flight.

    Token totals aren't knowable until a stream ends, so this accumulates counts
    as chunks pass through and emits one llm_responded when the stream is
    exhausted or its context manager exits, whichever comes first. The emit-once
    flag is what makes those two paths safe to have both.

    Only ints and text fragments are retained, never chunk objects, so a long
    stream costs the same memory as the equivalent non-streaming response.

    Every provider stream here is both an iterator and a context manager, and
    __enter__ can hand back a different object than it was called on (Anthropic's
    manager returns a MessageStream), so both protocols are implemented and the
    wrapped object is rebound on enter. Everything else falls through to the
    wrapped stream, which keeps provider-specific helpers working.

    Finalizers come in sync and async flavours because the helper they call may
    itself be a coroutine function (anthropic's AsyncMessageStream.get_final_message
    is ``async def`` while MessageStream's is not). The async paths run
    ``_afinalize``, the sync paths ``_finalize``; neither substitutes for the
    other, because calling an async finalizer synchronously yields an un-awaited
    coroutine that silently records a zero-token response.
    """

    __slots__ = (
        "_inner",
        "_run",
        "_t0",
        "_collect",
        "_finalize",
        "_afinalize",
        "_acc",
        "_emitted",
        "_call_index",
        "_it",
        "_ait",
        "__weakref__",
    )

    def __init__(self, inner, run, t0, collect, finalize=None, afinalize=None):
        self._inner = inner
        self._run = run
        self._t0 = t0
        self._collect = collect
        self._finalize = finalize
        self._afinalize = afinalize
        self._acc = _new_stream_acc()
        self._emitted = False
        self._it = None
        self._ait = None
        # Which LlmCall this stream belongs to. The patcher emits llm_called
        # immediately before constructing the proxy, so the call just appended is
        # ours. A stream is drained whenever the caller feels like it, possibly
        # after other LLM calls have started, so "the last call" is not a safe
        # target by the time the response is known — see RunContext.llm_responded.
        self._call_index = len(run.state.llm_calls) - 1 if run is not None else -1
        # So the run can report this call even if the caller abandons the stream
        # without draining, closing or context-exiting it. See
        # RunContext._flush_open_streams.
        register = getattr(run, "_register_stream", None)
        if register is not None:
            register(self)

    def __getattr__(self, name):
        # object.__getattribute__ rather than self._inner: if _inner is somehow
        # unset this raises AttributeError directly instead of re-entering here.
        return getattr(object.__getattribute__(self, "_inner"), name)

    def _feed(self, chunk) -> None:
        try:
            self._collect(self._acc, chunk)
        except Exception:
            logger.debug("dunetrace stream collector failed", exc_info=True)

    def _claim(self) -> bool:
        """True for the one caller that gets to emit. Keeps the several exit
        paths (drained, broken out of, context-managed, closed) safe to have all
        at once."""
        if self._emitted:
            return False
        self._emitted = True
        return True

    def _publish(self, *, error_path: bool) -> None:
        _safe_emit(
            lambda: _emit_stream_response(self._run, self._acc, self._t0, self._call_index),
            swallow_control_flow=error_path,
        )

    def _emit(self, *, error_path: bool = False) -> None:
        if not self._claim():
            return
        if self._finalize is not None:
            try:
                self._finalize(self._inner, self._acc)
            except Exception:
                logger.debug("dunetrace stream finalizer failed", exc_info=True)
        self._publish(error_path=error_path)

    async def _aemit(self, *, error_path: bool = False) -> None:
        if not self._claim():
            return
        if self._afinalize is not None:
            try:
                await self._afinalize(self._inner, self._acc)
            except Exception:
                logger.debug("dunetrace stream finalizer failed", exc_info=True)
        self._publish(error_path=error_path)

    def _record_error(self, exc: BaseException) -> None:
        """A stream that blew up mid-flight is an errored call, not a clean one.
        Mirrors what the non-streaming wrappers report via _mistral_call_failed."""
        self._acc["finish_reason"] = "error"
        self._acc["error"] = f"{type(exc).__name__}: {exc}"[:500]

    def __iter__(self):
        # self, not a fresh generator. iter() has to be idempotent: a caller that
        # partially consumes and then resumes (islice then list, or peek then
        # loop) would otherwise get a second generator over the same underlying
        # stream while the emit latch had already fired on the first — the caller
        # receives the whole response and the run records a fraction of it.
        # Returning self also makes the proxy a real iterator, which the wrapped
        # objects all are (openai.Stream, mistralai EventStream, anthropic
        # MessageStream), so next(stream) keeps working under instrumentation.
        return self

    def _sync_iter(self):
        it = self._it
        if it is None:
            # iter(), not the object itself: the wrapped value is guaranteed
            # iterable but not guaranteed to already be an iterator.
            it = self._it = iter(self._inner)
        return it

    def __next__(self):
        try:
            chunk = next(self._sync_iter())
        except StopIteration:
            self._emit()
            raise
        except Exception as exc:
            self._record_error(exc)
            self._emit(error_path=True)
            raise
        self._feed(chunk)
        return chunk

    def __enter__(self):
        entered = self._inner.__enter__()
        if entered is not None and entered is not self._inner:
            self._inner = entered
            self._it = None  # rebound: any iterator cached off the old object is stale
        return self

    def __exit__(self, *exc):
        try:
            return self._inner.__exit__(*exc)
        finally:
            if exc and exc[0] is not None and self._acc["error"] is None:
                self._record_error(exc[1] if len(exc) > 1 and exc[1] else exc[0](""))
            self._emit(error_path=bool(exc) and exc[0] is not None)

    def __aiter__(self):
        return self

    def _async_iter(self):
        it = self._ait
        if it is None:
            it = self._ait = self._inner.__aiter__()
        return it

    async def __anext__(self):
        try:
            chunk = await self._async_iter().__anext__()
        except StopAsyncIteration:
            await self._aemit()
            raise
        except Exception as exc:
            self._record_error(exc)
            await self._aemit(error_path=True)
            raise
        self._feed(chunk)
        return chunk

    async def __aenter__(self):
        entered = await self._inner.__aenter__()
        if entered is not None and entered is not self._inner:
            self._inner = entered
            self._ait = None
        return self

    async def __aexit__(self, *exc):
        try:
            return await self._inner.__aexit__(*exc)
        finally:
            await self._aemit(error_path=bool(exc) and exc[0] is not None)

    def close(self):
        try:
            return self._inner.close()
        finally:
            self._emit(error_path=sys.exc_info()[0] is not None)

    def __del__(self):
        """Last resort for a stream the caller abandoned — `for c in stream:
        break` with no `with` block and no close().

        Now that __iter__ returns self, breaking out of a loop gives the proxy no
        signal at all (a real iterator gets none), so without this the tokens
        already paid for would never be reported. Timing is whenever the proxy is
        released rather than at the break, so `with` or close() is still the way
        to get a deterministic event. Never raises: __del__ must not, and the
        error path suppresses policy control-flow for the same reason.
        """
        try:
            if not self._emitted:
                self._emit(error_path=True)
        except Exception:  # pragma: no cover - interpreter teardown
            pass

    async def aclose(self):
        """Async streams are closed with aclose(), so the sync close() above
        never fires for them — without this the tokens already paid for on an
        explicitly-closed async stream would go unrecorded."""
        try:
            inner_aclose = getattr(self._inner, "aclose", None)
            if inner_aclose is not None:
                return await inner_aclose()
            return None
        finally:
            await self._aemit(error_path=sys.exc_info()[0] is not None)


# ── OpenAI ────────────────────────────────────────────────────────────────────
#
# Response-shape normalisation.
#
# openai's helper clients do not return the parsed model. They return a wrapper
# holding the raw HTTP response, and the parsed object is behind .parse():
#
#   client.chat.completions.create(...)                        -> ChatCompletion
#   ....with_raw_response.create(...)                          -> LegacyAPIResponse
#   ....with_streaming_response.create(...)                    -> APIResponse
#   ....with_streaming_response.create(...)  [async client]    -> AsyncAPIResponse
#
# This matters because langchain_openai calls with_raw_response.create() — 12
# call sites in chat_models/base.py — so EVERY non-streamed LangChain call
# handed our patch a LegacyAPIResponse. It has no .choices and no .usage, so the
# extractors read nothing and the run recorded zero tokens and empty output for
# a model that had answered normally.
#
# Keyed on SHAPE, not on class name. Wrapper classes live in private modules
# (openai._legacy_response, openai._response), get renamed across versions, and
# a name check would silently stop working on an upgrade — reintroducing exactly
# this bug. "Lacks .choices but has a callable .parse" identifies all three
# families and cannot match a real ChatCompletion, which has .choices.


def _is_response_wrapper(resp: object) -> bool:
    """True for a raw/streaming envelope that hides the parsed model behind
    .parse(). Deliberately narrow: a real ChatCompletion has .choices and so can
    never match, and an object with neither attribute passes straight through."""
    try:
        return not hasattr(resp, "choices") and callable(getattr(resp, "parse", None))
    except Exception:
        # A property that raises is not a shape we understand. Leave it alone.
        return False


def _unwrap_response(resp: object) -> object:
    """Parsed model behind a wrapper, or `resp` unchanged.

    Never raises. A failure here must degrade to the previous behaviour (the
    extractors read what they can off the wrapper) rather than break the host's
    call — instrumentation is strictly additive.
    """
    if not _is_response_wrapper(resp):
        return resp
    try:
        parsed = resp.parse()  # type: ignore[attr-defined]
    except Exception:
        logger.debug("dunetrace: openai response unwrap failed", exc_info=True)
        return resp
    if inspect.isawaitable(parsed):
        # A coroutine reached the SYNC path. Only possible on an async client
        # whose parse() is async — AsyncAPIResponse today, and LegacyAPIResponse
        # on the async client from openai v2 (its own docstring says so). We
        # cannot await here, so close it to avoid leaking an un-awaited
        # coroutine warning into the host's logs, and fall back.
        closer = getattr(parsed, "close", None)
        if callable(closer):
            with contextlib.suppress(Exception):
                closer()
        logger.debug("dunetrace: awaitable parse() on the sync path — not unwrapped")
        return resp
    return parsed if parsed is not None else resp


async def _unwrap_response_async(resp: object) -> object:
    """Async counterpart. Awaits parse() when it returns an awaitable.

    Awaitability is detected at RUNTIME rather than branched on the openai
    version. LegacyAPIResponse.parse is sync on both clients today, and openai's
    own docstring (_legacy_response.py) states it becomes a coroutine on the
    async client in the next major version. A naive `resp.parse()` here is
    correct today and, on that upgrade, would silently hand a coroutine object
    to the extractors — reproducing this exact bug with an un-awaited-coroutine
    warning on top. AsyncAPIResponse.parse is ALREADY a coroutine function, so
    this path is live now, not merely forward-looking.
    """
    if not _is_response_wrapper(resp):
        return resp
    try:
        parsed = resp.parse()  # type: ignore[attr-defined]
        if inspect.isawaitable(parsed):
            parsed = await parsed
    except Exception:
        logger.debug("dunetrace: openai async response unwrap failed", exc_info=True)
        return resp
    return parsed if parsed is not None else resp


def _patch_openai(
    client: "Optional[Dunetrace]" = None, default_agent_id: Optional[str] = None
) -> None:
    if "openai" in _PATCHED:
        return
    try:
        import openai.resources.chat.completions as _mod
    except ImportError:
        logger.debug("openai not installed — skipping auto-instrument")
        return

    # Sync
    _orig_create = _mod.Completions.create

    @functools.wraps(_orig_create)
    def _patched_create(self, *, messages=None, model="unknown", **kwargs):
        run = None if _in_framework_call.get() else _current_run.get()
        t0 = time.monotonic()
        if run:
            _safe_emit(
                lambda: run.llm_called(
                    model,
                    prompt_tokens=_estimate_tokens(messages),
                    provider="openai",
                    prompt_tokens_estimated=True,
                )
            )
        try:
            with _suppress_http():
                resp = _orig_create(self, messages=messages, model=model, **kwargs)
        except Exception:
            if run:
                _safe_emit(
                    lambda: run.llm_responded(
                        finish_reason="error",
                        latency_ms=int((time.monotonic() - t0) * 1000),
                    ),
                    swallow_control_flow=True,
                )
            raise
        if run:
            if kwargs.get("stream"):
                return _StreamProxy(resp, run, t0, _openai_stream_collector)
            # Unwrap BEFORE _emit_openai_response, which is where the
            # instrumentation_degraded marker is decided — a response we
            # successfully parsed must never be reported as unreadable.
            plain = _unwrap_response(resp)
            _safe_emit(lambda: _emit_openai_response(run, plain, t0))
        # ALWAYS the object _orig_create produced, never the unwrapped one.
        # LangChain reads response headers off the wrapper (that is why it asked
        # for a raw response at all); handing back a bare ChatCompletion breaks
        # it. The caller must not be able to tell instrumentation is installed.
        return resp

    _mod.Completions.create = _patched_create

    # Async
    try:
        _orig_acreate = _mod.AsyncCompletions.create

        @functools.wraps(_orig_acreate)
        async def _patched_acreate(self, *, messages=None, model="unknown", **kwargs):
            run = None if _in_framework_call.get() else _current_run.get()
            t0 = time.monotonic()
            if run:
                _safe_emit(
                    lambda: run.llm_called(
                        model,
                        prompt_tokens=_estimate_tokens(messages),
                        provider="openai",
                        prompt_tokens_estimated=True,
                    )
                )
            try:
                with _suppress_http():
                    resp = await _orig_acreate(self, messages=messages, model=model, **kwargs)
            except Exception:
                if run:
                    _safe_emit(
                        lambda: run.llm_responded(
                            finish_reason="error",
                            latency_ms=int((time.monotonic() - t0) * 1000),
                        ),
                        swallow_control_flow=True,
                    )
                raise
            if run:
                if kwargs.get("stream"):
                    return _StreamProxy(resp, run, t0, _openai_stream_collector)
                # The unwrap has to happen HERE, not inside the _safe_emit
                # lambda: that lambda is synchronous, so an unwrap needing
                # `await` cannot live in it. Await first, then hand
                # _emit_openai_response an already-plain object — which keeps
                # that function synchronous and shape-agnostic, identical to the
                # sync path.
                plain = await _unwrap_response_async(resp)
                _safe_emit(lambda: _emit_openai_response(run, plain, t0))
            # The original wrapper, as in the sync patch. See there for why.
            return resp

        _mod.AsyncCompletions.create = _patched_acreate
    except AttributeError:
        pass  # older openai version without async client

    _PATCHED.add("openai")
    _record_instrumented("openai")
    logger.debug("openai auto-instrumented")


def _emit_openai_response(run, resp, t0: float) -> None:
    usage = getattr(resp, "usage", None)
    comp_toks = getattr(usage, "completion_tokens", 0) or 0
    # Exact count from the response, overriding llm_called's chars//4 estimate.
    # llm_responded ignores a falsy value, so a response without usage keeps the
    # estimate rather than zeroing it.
    prompt_toks = getattr(usage, "prompt_tokens", 0) or 0
    reason_toks = _completion_detail_tokens(usage, "reasoning_tokens")
    latency_ms = int((time.monotonic() - t0) * 1000)
    finish = _openai_finish_reason(resp)
    text = _openai_content(resp)
    # Both extractors failing means we did not read the response at all — a raw
    # envelope, a mock, a version skew. Either one alone is enough to make the
    # run unmeasurable for the text/finish_reason detectors.
    degraded = _degraded_marker("openai", resp) if (finish is None or text is None) else None
    run.llm_responded(
        completion_tokens=comp_toks,
        prompt_tokens=prompt_toks,
        reasoning_tokens=reason_toks,
        latency_ms=latency_ms,
        finish_reason=finish,
        output=text,
        output_length=len(text) if text else 0,
        instrumentation_degraded=degraded,
    )


def _openai_finish_reason(resp) -> Optional[str]:
    """The response's finish_reason, or None if the shape could not be read.

    Returns None rather than "stop" on failure. "stop" is a *claim about the
    model's behaviour*; we have no basis for it when the object is unreadable,
    and asserting it fabricated the exact input EMPTY_LLM_RESPONSE fires on.
    A real ChatCompletion always carries a truthy finish_reason, so a falsy one
    means the shape is not what we think it is either.
    """
    try:
        return resp.choices[0].finish_reason or None
    except (AttributeError, IndexError, TypeError):
        return None


def _openai_content(resp) -> Optional[str]:
    """Assistant text, "" for a legitimately text-free turn, None if unreadable.

    The two cases are deliberately distinct. `message.content` is None on a
    tool-call-only turn — the shape was read fine and the model genuinely
    produced no text, so that is "". An exception means the object is not a
    ChatCompletion at all, which is None.
    """
    try:
        content = resp.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return None
    return content or ""


def _openai_stream_collector(acc: dict, chunk) -> None:
    usage = getattr(chunk, "usage", None)
    if usage is not None:
        acc["prompt_tokens"] = getattr(usage, "prompt_tokens", 0) or acc["prompt_tokens"]
        acc["completion_tokens"] = (
            getattr(usage, "completion_tokens", 0) or acc["completion_tokens"]
        )
    choices = getattr(chunk, "choices", None) or ()
    if not choices:
        # The usage-only chunk that include_usage appends carries no choices.
        return
    delta = getattr(choices[0], "delta", None)
    content = getattr(delta, "content", None)
    if content:
        acc["text"].append(content)
    # A tool-call turn carries no content at all — the whole output arrives on
    # delta.tool_calls[*].function.{name,arguments}. Those characters are still
    # billed output, so count them; without this the estimate in
    # _emit_stream_response has nothing to work from and a streamed tool call
    # reports zero output tokens (and zero output cost). Only the length is kept,
    # not the payload, matching how text is accumulated.
    for call in getattr(delta, "tool_calls", None) or ():
        fn = getattr(call, "function", None)
        if fn is None:
            continue
        acc["tool_arg_chars"] += len(getattr(fn, "name", None) or "") + len(
            getattr(fn, "arguments", None) or ""
        )
    finish = getattr(choices[0], "finish_reason", None)
    if finish:
        acc["finish_reason"] = str(finish)


def _completion_detail_tokens(usage, name: str) -> int:
    details = getattr(usage, "completion_tokens_details", None)
    if details is None and isinstance(usage, dict):
        details = usage.get("completion_tokens_details")
    if details is None:
        return 0
    if isinstance(details, dict):
        return int(details.get(name, 0) or 0)
    return int(getattr(details, name, 0) or 0)


# ── Anthropic ─────────────────────────────────────────────────────────────────


def _patch_anthropic(
    client: "Optional[Dunetrace]" = None, default_agent_id: Optional[str] = None
) -> None:
    if "anthropic" in _PATCHED:
        return
    try:
        import anthropic.resources.messages as _mod
    except ImportError:
        logger.debug("anthropic not installed — skipping auto-instrument")
        return

    # Sync
    _orig_create = _mod.Messages.create

    @functools.wraps(_orig_create)
    def _patched_create(self, *, model="unknown", messages=None, max_tokens=1024, **kwargs):
        run = None if _in_framework_call.get() else _current_run.get()
        t0 = time.monotonic()
        if run:
            _safe_emit(
                lambda: run.llm_called(
                    model,
                    prompt_tokens=_estimate_tokens(messages),
                    provider="anthropic",
                    prompt_tokens_estimated=True,
                )
            )
        try:
            with _suppress_http():
                resp = _orig_create(
                    self, model=model, messages=messages, max_tokens=max_tokens, **kwargs
                )
        except Exception:
            if run:
                _safe_emit(
                    lambda: run.llm_responded(
                        finish_reason="error",
                        latency_ms=int((time.monotonic() - t0) * 1000),
                    ),
                    swallow_control_flow=True,
                )
            raise
        if run:
            if kwargs.get("stream"):
                return _StreamProxy(resp, run, t0, _anthropic_stream_collector)
            _safe_emit(lambda: _emit_anthropic_response(run, resp, t0))
        return resp

    _mod.Messages.create = _patched_create

    # Async
    try:
        _orig_acreate = _mod.AsyncMessages.create

        @functools.wraps(_orig_acreate)
        async def _patched_acreate(
            self, *, model="unknown", messages=None, max_tokens=1024, **kwargs
        ):
            run = None if _in_framework_call.get() else _current_run.get()
            t0 = time.monotonic()
            if run:
                _safe_emit(
                    lambda: run.llm_called(
                        model,
                        prompt_tokens=_estimate_tokens(messages),
                        provider="anthropic",
                        prompt_tokens_estimated=True,
                    )
                )
            try:
                with _suppress_http():
                    resp = await _orig_acreate(
                        self,
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
            except Exception:
                if run:
                    _safe_emit(
                        lambda: run.llm_responded(
                            finish_reason="error",
                            latency_ms=int((time.monotonic() - t0) * 1000),
                        ),
                        swallow_control_flow=True,
                    )
                raise
            if run:
                if kwargs.get("stream"):
                    return _StreamProxy(resp, run, t0, _anthropic_stream_collector)
                _safe_emit(lambda: _emit_anthropic_response(run, resp, t0))
            return resp

        _mod.AsyncMessages.create = _patched_acreate
    except AttributeError:
        pass  # older anthropic version without async client

    # messages.stream() never routes through create() — it calls self._post
    # directly — so without its own patch a call made that way emits nothing at
    # all, not even a wrong token count. Both variants return a manager rather
    # than a coroutine, so neither wrapper is async.
    for _cls_name, _mgr_kind in (("Messages", "sync"), ("AsyncMessages", "async")):
        try:
            _cls = getattr(_mod, _cls_name)
            _orig_stream = _cls.stream
        except AttributeError:
            continue

        def _make_stream_patch(orig_stream, is_async):
            @functools.wraps(orig_stream)
            def _patched_stream(self, *, model="unknown", messages=None, **kwargs):
                run = None if _in_framework_call.get() else _current_run.get()
                t0 = time.monotonic()
                if run:
                    _safe_emit(
                        lambda: run.llm_called(
                            model,
                            prompt_tokens=_estimate_tokens(messages),
                            provider="anthropic",
                            prompt_tokens_estimated=True,
                        )
                    )
                try:
                    with _suppress_http():
                        manager = orig_stream(self, model=model, messages=messages, **kwargs)
                except Exception:
                    if run:
                        _safe_emit(
                            lambda: run.llm_responded(
                                finish_reason="error",
                                latency_ms=int((time.monotonic() - t0) * 1000),
                            ),
                            swallow_control_flow=True,
                        )
                    raise
                if not run:
                    return manager
                # AsyncMessageStream.get_final_message is a coroutine function
                # while MessageStream's is not, so the finalizer has to match the
                # manager's kind. Handing the async one to the sync slot would
                # leave an un-awaited coroutine and emit a zero-token response.
                return _StreamProxy(
                    manager,
                    run,
                    t0,
                    _anthropic_stream_collector,
                    None if is_async else _anthropic_stream_finalize,
                    _anthropic_stream_afinalize if is_async else None,
                )

            return _patched_stream

        _cls.stream = _make_stream_patch(_orig_stream, _mgr_kind == "async")

    _PATCHED.add("anthropic")
    _record_instrumented("anthropic")
    logger.debug("anthropic auto-instrumented")


def _emit_anthropic_response(run, resp, t0: float) -> None:
    usage = getattr(resp, "usage", None)
    comp_toks = getattr(usage, "output_tokens", 0) or 0
    # Anthropic names it input_tokens, not prompt_tokens. Exact count from the
    # response, overriding llm_called's chars//4 estimate. llm_responded ignores
    # a falsy value, so a response without usage keeps the estimate.
    prompt_toks = getattr(usage, "input_tokens", 0) or 0
    latency_ms = int((time.monotonic() - t0) * 1000)
    # Same rule as OpenAI: a missing stop_reason is an unreadable shape, not an
    # "end_turn". Sentinel-free getattr so an absent attribute is distinguishable
    # from a present-but-falsy one.
    finish = getattr(resp, "stop_reason", None) or None
    text = _anthropic_content(resp)
    degraded = _degraded_marker("anthropic", resp) if (finish is None or text is None) else None
    run.llm_responded(
        completion_tokens=comp_toks,
        prompt_tokens=prompt_toks,
        latency_ms=latency_ms,
        finish_reason=finish,
        output=text,
        output_length=len(text) if text else 0,
        instrumentation_degraded=degraded,
    )


def _anthropic_content(resp) -> Optional[str]:
    """Concatenated text of ALL content blocks, "" for a text-free turn, None if
    the response shape could not be read at all. See _openai_content.

    Anthropic replies are a LIST of blocks, and the text is not necessarily in
    block 0. With extended thinking enabled block 0 is the thinking block, whose
    `text` is empty — reading only the first block reported output_length 0 for
    every response from a reasoning model. Any multi-block reply (text + tool_use,
    or several text blocks) was likewise truncated to its first block. Same shape
    as _bedrock_converse_text, which joins for exactly this reason.

    Blocks are pydantic objects, not dicts, so text is read with getattr. A block
    with no text attribute (thinking, tool_use, redacted_thinking) contributes
    nothing, which keeps a tool-only turn at "" — the shape was read fine and the
    model genuinely produced no text. Only an unreadable envelope is None: no
    `content` attribute, or a `content` that is not a list.
    """
    content = getattr(resp, "content", None)
    if not isinstance(content, list):
        return None
    return "".join(str(getattr(b, "text", "") or "") for b in content)


def _anthropic_stream_collector(acc: dict, event) -> None:
    """Anthropic splits usage across two event types: message_start carries the
    input tokens, message_delta the running output count and the stop reason."""
    etype = getattr(event, "type", "") or ""
    if etype == "message_start":
        usage = getattr(getattr(event, "message", None), "usage", None)
        if usage is not None:
            acc["prompt_tokens"] = getattr(usage, "input_tokens", 0) or acc["prompt_tokens"]
    elif etype == "content_block_delta":
        text = getattr(getattr(event, "delta", None), "text", None)
        if text:
            acc["text"].append(text)
    elif etype == "message_delta":
        usage = getattr(event, "usage", None)
        if usage is not None:
            acc["completion_tokens"] = (
                getattr(usage, "output_tokens", 0) or acc["completion_tokens"]
            )
        stop = getattr(getattr(event, "delta", None), "stop_reason", None)
        if stop:
            acc["finish_reason"] = str(stop)


def _anthropic_needs_finalize(inner, acc: dict):
    """The assembled-message getter, or None when the finalizer has nothing to
    do. Shared by the sync and async finalizers so the "only when iteration
    produced nothing" rule is stated once."""
    if acc["completion_tokens"] or acc["text"]:
        return None
    return getattr(inner, "get_final_message", None)


def _anthropic_stream_finalize(inner, acc: dict) -> None:
    """Last resort for a stream consumed through a helper rather than by
    iterating the proxy, which leaves the collector with nothing. The SDK's
    MessageStream keeps the assembled message, so read the real totals off it.
    Only runs when iteration produced no tokens, and never raises: the caller
    wraps this in its own try/except.

    Sync managers only — AsyncMessageStream.get_final_message is a coroutine
    function and is handled by _anthropic_stream_afinalize."""
    get_final = _anthropic_needs_finalize(inner, acc)
    if get_final is None:
        return
    _anthropic_apply_final(get_final(), acc)


async def _anthropic_stream_afinalize(inner, acc: dict) -> None:
    """Async twin of _anthropic_stream_finalize.

    This is the path that actually carries the data for async callers: the
    documented idiom is ``async with client.messages.stream(...) as s: async for
    text in s.text_stream:``, which reaches text_stream on the inner manager
    through __getattr__ and so never feeds the proxy's collector. Everything the
    run knows about the call therefore comes from get_final_message()."""
    get_final = _anthropic_needs_finalize(inner, acc)
    if get_final is None:
        return
    _anthropic_apply_final(await get_final(), acc)


def _anthropic_apply_final(message, acc: dict) -> None:
    usage = getattr(message, "usage", None)
    if usage is not None:
        acc["prompt_tokens"] = getattr(usage, "input_tokens", 0) or acc["prompt_tokens"]
        acc["completion_tokens"] = getattr(usage, "output_tokens", 0) or 0
    text = _anthropic_content(message)
    if text:
        acc["text"].append(text)
    stop = getattr(message, "stop_reason", None)
    if stop:
        acc["finish_reason"] = str(stop)


# ── Mistral ───────────────────────────────────────────────────────────────────

# Host fragments that identify a deployment when the client class alone can't.
_MISTRAL_DIRECT_HOST = "api.mistral.ai"
_MISTRAL_AZURE_HOSTS = (".inference.ai.azure.com", ".models.ai.azure.com", ".azure.com")
_MISTRAL_AWS_HOSTS = (".amazonaws.com", ".api.aws")


def _mistral_deployment(sub_sdk, server_url: Optional[str] = None) -> str:
    """Classify where a Mistral call is served from.

    Returns "direct", "azure", "gcp", "aws", "self_hosted", or "unknown".

    Class identity is checked before the URL on purpose. MistralAzure built
    without an explicit server_url falls back to https://api.mistral.ai, so a
    URL-first check would report those calls as direct. Verified against
    mistralai 2.9.1.

    Only used for a debug log today. The value is not put on the event, because
    LlmCall carries no deployment field and adding one would touch the wire
    schema and both run builders for something no detector reads.
    """
    module = (getattr(type(sub_sdk), "__module__", "") or "").lower()
    if ".azure." in module:
        return "azure"
    if ".gcp." in module:
        return "gcp"

    url = server_url
    if not url:
        # get_server_details() resolves the configured base URL. The raw
        # sdk_configuration.server_url attribute is None on a default client,
        # so it can't be read directly.
        try:
            url = sub_sdk.sdk_configuration.get_server_details()[0]
        except Exception:
            return "unknown"

    host = (url or "").lower()
    if _MISTRAL_DIRECT_HOST in host:
        return "direct"
    if any(fragment in host for fragment in _MISTRAL_AZURE_HOSTS):
        return "azure"
    if any(fragment in host for fragment in _MISTRAL_AWS_HOSTS):
        return "aws"
    return "self_hosted"


def _mistral_call_start(sub_sdk, model: str, kwargs: dict):
    """Common preamble for every patched Mistral method."""
    run = None if _in_framework_call.get() else _current_run.get()
    t0 = time.monotonic()
    if run:
        # Embeddings and FIM have no messages= to estimate from, so the estimate
        # is 0 there and llm_responded backfills the real count from usage.
        _safe_emit(
            lambda: run.llm_called(
                model,
                prompt_tokens=_estimate_tokens(kwargs.get("messages")),
                provider="mistral",
                prompt_tokens_estimated=True,
            )
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "mistral call served via %s",
                _mistral_deployment(sub_sdk, kwargs.get("server_url")),
            )
    return run, t0


def _mistral_call_failed(run, t0: float) -> None:
    if run:
        _safe_emit(
            lambda: run.llm_responded(
                finish_reason="error",
                latency_ms=int((time.monotonic() - t0) * 1000),
            ),
            swallow_control_flow=True,
        )


# Four wrapper shapes cover all ten patched Mistral methods. Written as
# factories rather than ten near-identical function bodies, which is a local
# choice inside this provider and not a cross-provider refactor.


def _mistral_sync_patch(orig):
    @functools.wraps(orig)
    def _wrapped(self, *, model="unknown", **kwargs):
        run, t0 = _mistral_call_start(self, model, kwargs)
        try:
            with _suppress_http():
                resp = orig(self, model=model, **kwargs)
        except Exception:
            _mistral_call_failed(run, t0)
            raise
        if run:
            _safe_emit(lambda: _emit_mistral_response(run, resp, t0))
        return resp

    return _wrapped


def _mistral_async_patch(orig):
    @functools.wraps(orig)
    async def _wrapped(self, *, model="unknown", **kwargs):
        run, t0 = _mistral_call_start(self, model, kwargs)
        try:
            with _suppress_http():
                resp = await orig(self, model=model, **kwargs)
        except Exception:
            _mistral_call_failed(run, t0)
            raise
        if run:
            _safe_emit(lambda: _emit_mistral_response(run, resp, t0))
        return resp

    return _wrapped


def _mistral_stream_patch(orig):
    @functools.wraps(orig)
    def _wrapped(self, *, model="unknown", **kwargs):
        run, t0 = _mistral_call_start(self, model, kwargs)
        try:
            with _suppress_http():
                stream = orig(self, model=model, **kwargs)
        except Exception:
            _mistral_call_failed(run, t0)
            raise
        if not run:
            return stream
        return _StreamProxy(stream, run, t0, _mistral_stream_collector)

    return _wrapped


def _mistral_astream_patch(orig):
    """stream_async is a coroutine that resolves to the stream, so the awaited
    result is what gets wrapped, not the coroutine."""

    @functools.wraps(orig)
    async def _wrapped(self, *, model="unknown", **kwargs):
        run, t0 = _mistral_call_start(self, model, kwargs)
        try:
            with _suppress_http():
                stream = await orig(self, model=model, **kwargs)
        except Exception:
            _mistral_call_failed(run, t0)
            raise
        if not run:
            return stream
        return _StreamProxy(stream, run, t0, _mistral_stream_collector)

    return _wrapped


def _mistral_stream_collector(acc: dict, event) -> None:
    """Mistral wraps each chunk in a CompletionEvent, so the payload is at
    event.data. Usage arrives on the final chunk by default, verified against
    the live API, so no opt-in flag is needed the way OpenAI needs one."""
    chunk = getattr(event, "data", event)
    usage = getattr(chunk, "usage", None)
    if usage is not None:
        acc["prompt_tokens"] = getattr(usage, "prompt_tokens", 0) or acc["prompt_tokens"]
        acc["completion_tokens"] = (
            getattr(usage, "completion_tokens", 0) or acc["completion_tokens"]
        )
    choices = getattr(chunk, "choices", None) or ()
    if not choices:
        return
    content = getattr(getattr(choices[0], "delta", None), "content", None)
    if isinstance(content, str):
        if content:
            acc["text"].append(content)
    elif isinstance(content, list):
        acc["text"].append("".join(str(getattr(c, "text", "") or "") for c in content))
    finish = getattr(choices[0], "finish_reason", None)
    if finish:
        acc["finish_reason"] = str(finish)


def _patch_mistral(
    client: "Optional[Dunetrace]" = None, default_agent_id: Optional[str] = None
) -> None:
    if "mistral" in _PATCHED:
        return
    try:
        # mistralai v2 moved every module under mistralai.client and dropped the
        # top-level __init__.py, so this import path is v2-only by construction.
        # v1 and the pre-1.0 MistralClient are not supported.
        import mistralai.client.chat as _chat
    except ImportError:
        logger.debug("mistralai not installed, skipping auto-instrument")
        return

    # Chat.parse and Chat.parse_stream are deliberately left alone: they call
    # Chat.complete / Chat.stream internally, so patching both would emit two
    # events for one API call.
    for _name, _factory in (
        ("complete", _mistral_sync_patch),
        ("complete_async", _mistral_async_patch),
        ("stream", _mistral_stream_patch),
        ("stream_async", _mistral_astream_patch),
    ):
        _orig = getattr(_chat.Chat, _name, None)
        if _orig is not None:
            setattr(_chat.Chat, _name, _factory(_orig))

    # Embeddings and FIM ship in the same distribution, but each import is
    # guarded so a trimmed install or a future reshuffle can't cost us chat
    # coverage. Embedding responses have no choices, so the shared emitter
    # records tokens with empty output and finish_reason "stop".
    #
    # MistralAzure and MistralGCP carry their own Chat/Fim classes in separate
    # modules — mistralai.azure.client.chat.Chat is NOT
    # mistralai.client.chat.Chat (verified against 2.9.1) — so patching the core
    # client alone leaves every hyperscaler-hosted call uninstrumented while
    # still reporting success. They expose the same keyword-only surface, so the
    # same factories apply. Azure ships chat only; GCP ships chat and fim;
    # neither ships embeddings, hence the per-module guard doing real work here.
    _CHAT_METHODS = (
        ("complete", _mistral_sync_patch),
        ("complete_async", _mistral_async_patch),
        ("stream", _mistral_stream_patch),
        ("stream_async", _mistral_astream_patch),
    )
    for _mod_name, _cls_name, _methods in (
        (
            "mistralai.client.embeddings",
            "Embeddings",
            (("create", _mistral_sync_patch), ("create_async", _mistral_async_patch)),
        ),
        ("mistralai.client.fim", "Fim", _CHAT_METHODS),
        ("mistralai.azure.client.chat", "Chat", _CHAT_METHODS),
        ("mistralai.gcp.client.chat", "Chat", _CHAT_METHODS),
        ("mistralai.gcp.client.fim", "Fim", _CHAT_METHODS),
    ):
        try:
            _sub = importlib.import_module(_mod_name)
            _cls = getattr(_sub, _cls_name)
        except (ImportError, AttributeError):
            # _mod_name, not _cls_name: three of these are called "Chat".
            logger.debug("mistralai %s not available, not patched", _mod_name)
            continue
        for _name, _factory in _methods:
            _orig = getattr(_cls, _name, None)
            if _orig is not None:
                setattr(_cls, _name, _factory(_orig))

    _PATCHED.add("mistral")
    _record_instrumented("mistralai")
    logger.debug("mistral auto-instrumented")


def _emit_mistral_response(run, resp, t0: float) -> None:
    usage = getattr(resp, "usage", None)
    comp_toks = getattr(usage, "completion_tokens", 0) or 0
    # Mistral returns the exact prompt token count on every response. The
    # openai/anthropic patchers leave llm_called's chars//4 estimate standing;
    # backfilling the real number costs nothing here and makes cost_usd, and so
    # the cost_usd policy trigger, correct rather than approximate.
    prompt_toks = getattr(usage, "prompt_tokens", 0) or 0
    latency_ms = int((time.monotonic() - t0) * 1000)
    finish = _mistral_finish_reason(resp)
    text = _mistral_content(resp)
    degraded = _degraded_marker("mistral", resp) if (finish is None or text is None) else None
    run.llm_responded(
        completion_tokens=comp_toks,
        prompt_tokens=prompt_toks,
        latency_ms=latency_ms,
        finish_reason=finish,
        output=text,
        output_length=len(text) if text else 0,
        instrumentation_degraded=degraded,
    )


def _mistral_finish_reason(resp) -> Optional[str]:
    """See _openai_finish_reason — None on an unreadable shape, never "stop"."""
    try:
        reason = resp.choices[0].finish_reason
    except (AttributeError, IndexError, TypeError):
        return None
    return str(reason) if reason else None


def _mistral_content(resp) -> Optional[str]:
    """Text of the first choice.

    Mistral assistant content is either a plain string or a list of content
    chunks, so both shapes are handled. Chunks without text contribute nothing,
    which keeps tool-call and image replies at length 0 rather than crashing.
    """
    try:
        content = resp.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(getattr(chunk, "text", "") or "") for chunk in content)
    return ""


# ── Bedrock (botocore) ────────────────────────────────────────────────────────

# Bedrock is reached through boto3's bedrock-runtime client, not through any
# vendor SDK, and botocore rides on urllib3 rather than httpx or requests — so
# neither the provider patches nor the HTTP patches see it. Patching
# BaseClient._make_api_call is the one interception point every boto3 call goes
# through; it covers *every* Bedrock-hosted model, not just Mistral.
#
# Signature and operation names verified against botocore 1.43.67:
#   BaseClient._make_api_call(self, operation_name, api_params)
#   bedrock-runtime ops: Converse, ConverseStream, InvokeModel,
#                        InvokeModelWithResponseStream
_BEDROCK_SERVICE = "bedrock-runtime"
_BEDROCK_LLM_OPS = frozenset(
    {"Converse", "ConverseStream", "InvokeModel", "InvokeModelWithResponseStream"}
)
# Bedrock reports token counts for a non-streaming InvokeModel in response
# headers, which is the only place they can be read without consuming the
# caller's body stream.
_BEDROCK_INPUT_TOKENS_HEADER = "x-amzn-bedrock-input-token-count"
_BEDROCK_OUTPUT_TOKENS_HEADER = "x-amzn-bedrock-output-token-count"


def _is_bedrock_llm_call(botocore_client, operation_name: str) -> bool:
    if operation_name not in _BEDROCK_LLM_OPS:
        return False
    try:
        return botocore_client.meta.service_model.service_name == _BEDROCK_SERVICE
    except AttributeError:
        return False


def _bedrock_prompt_estimate(api_params: dict) -> int:
    """Converse carries structured messages; InvokeModel carries an opaque,
    model-specific JSON body. Both are only estimated — the real counts arrive
    on the response and override this."""
    messages = api_params.get("messages")
    if messages is not None:
        return _estimate_tokens(messages)
    body = api_params.get("body")
    if isinstance(body, (str, bytes, bytearray)):
        return max(0, len(body) // 4)
    return 0


def _bedrock_header_tokens(resp: dict) -> tuple[int, int]:
    try:
        headers = resp["ResponseMetadata"]["HTTPHeaders"]
        return (
            int(headers.get(_BEDROCK_INPUT_TOKENS_HEADER, 0) or 0),
            int(headers.get(_BEDROCK_OUTPUT_TOKENS_HEADER, 0) or 0),
        )
    except (KeyError, TypeError, ValueError):
        return 0, 0


def _bedrock_converse_text(resp: dict) -> Optional[str]:
    """Concatenated text blocks of the assistant message.

    "" for a tool-only turn (the shape was read, there was simply no text);
    None when the envelope is not a Converse response at all. See
    _openai_content for why the two are kept apart.
    """
    try:
        content = resp["output"]["message"]["content"]
    except (KeyError, TypeError):
        return None
    if not isinstance(content, list):
        return None
    return "".join(str(b.get("text", "") or "") for b in content if isinstance(b, dict))


def _emit_bedrock_converse(run, resp: dict, t0: float) -> None:
    usage = resp.get("usage") or {}
    text = _bedrock_converse_text(resp)
    # stopReason absent means this isn't a Converse envelope; don't invent "stop".
    raw_stop = resp.get("stopReason") if isinstance(resp, dict) else None
    finish = str(raw_stop) if raw_stop else None
    degraded = _degraded_marker("bedrock", resp) if (finish is None or text is None) else None
    run.llm_responded(
        completion_tokens=int(usage.get("outputTokens", 0) or 0),
        prompt_tokens=int(usage.get("inputTokens", 0) or 0),
        latency_ms=int((time.monotonic() - t0) * 1000),
        finish_reason=finish,
        output=text,
        output_length=len(text) if text else 0,
        instrumentation_degraded=degraded,
    )


def _emit_bedrock_invoke(run, resp: dict, t0: float) -> None:
    """InvokeModel's payload is a streaming body in a model-specific format.
    Reading it here would consume it and hand the caller an empty stream, so the
    token counts come from the response headers and the output text is left
    unknown rather than guessed."""
    prompt_toks, completion_toks = _bedrock_header_tokens(resp)
    run.llm_responded(
        completion_tokens=completion_toks,
        prompt_tokens=prompt_toks,
        latency_ms=int((time.monotonic() - t0) * 1000),
        # Not "stop": that plus output_length 0 is EmptyLlmResponseDetector's
        # firing condition, and an unread body is not an empty response.
        finish_reason="complete",
    )


def _bedrock_converse_stream_collector(acc: dict, event) -> None:
    """ConverseStream events are plain dicts, one key each."""
    if not isinstance(event, dict):
        return
    delta = (event.get("contentBlockDelta") or {}).get("delta") or {}
    text = delta.get("text")
    if text:
        acc["text"].append(str(text))
    stop = (event.get("messageStop") or {}).get("stopReason")
    if stop:
        acc["finish_reason"] = str(stop)
    usage = (event.get("metadata") or {}).get("usage") or {}
    if usage:
        acc["prompt_tokens"] = int(usage.get("inputTokens", 0) or 0) or acc["prompt_tokens"]
        acc["completion_tokens"] = (
            int(usage.get("outputTokens", 0) or 0) or acc["completion_tokens"]
        )


def _bedrock_invoke_stream_collector(acc: dict, event) -> None:
    """InvokeModelWithResponseStream chunks carry model-specific JSON, so the
    text shape can't be assumed. What *is* model-agnostic is the
    amazon-bedrock-invocationMetrics object Bedrock appends to the final chunk —
    that's where the real token counts come from."""
    if not isinstance(event, dict):
        return
    payload = (event.get("chunk") or {}).get("bytes")
    if not payload:
        return
    try:
        body = json.loads(payload)
    except (ValueError, TypeError):
        return
    if not isinstance(body, dict):
        return
    metrics = body.get("amazon-bedrock-invocationMetrics")
    if isinstance(metrics, dict):
        acc["prompt_tokens"] = int(metrics.get("inputTokenCount", 0) or 0) or acc["prompt_tokens"]
        acc["completion_tokens"] = (
            int(metrics.get("outputTokenCount", 0) or 0) or acc["completion_tokens"]
        )


def _patch_botocore(
    client: "Optional[Dunetrace]" = None, default_agent_id: Optional[str] = None
) -> None:
    if "botocore" in _PATCHED:
        return
    try:
        from botocore.client import BaseClient
    except ImportError:
        logger.debug("botocore not installed, skipping auto-instrument")
        return

    _orig_make_api_call = BaseClient._make_api_call

    @functools.wraps(_orig_make_api_call)
    def _patched_make_api_call(self, operation_name, api_params):
        # Every boto3 call in the process reaches this wrapper, so the cheap
        # operation-name check comes first and non-Bedrock traffic (S3, SQS, ...)
        # takes one frozenset lookup and returns.
        if not _is_bedrock_llm_call(self, operation_name):
            return _orig_make_api_call(self, operation_name, api_params)

        run = None if _in_framework_call.get() else _current_run.get()
        model = api_params.get("modelId") or "unknown"
        t0 = time.monotonic()
        if run:
            _safe_emit(
                lambda: run.llm_called(
                    model,
                    prompt_tokens=_bedrock_prompt_estimate(api_params),
                    prompt_tokens_estimated=True,
                    # The vendor is in the model id (anthropic.*, mistral.*,
                    # meta.*); what served the call is Bedrock.
                    provider="bedrock",
                )
            )
        try:
            resp = _orig_make_api_call(self, operation_name, api_params)
        except Exception:
            if run:
                _safe_emit(
                    lambda: run.llm_responded(
                        finish_reason="error",
                        latency_ms=int((time.monotonic() - t0) * 1000),
                    ),
                    swallow_control_flow=True,
                )
            raise

        if not run:
            return resp

        if operation_name == "Converse":
            _safe_emit(lambda: _emit_bedrock_converse(run, resp, t0))
        elif operation_name == "InvokeModel":
            _safe_emit(lambda: _emit_bedrock_invoke(run, resp, t0))
        elif isinstance(resp, dict):
            # Both streaming operations hand back an iterable EventStream, under
            # a different key each. Swapping in the proxy keeps the caller's
            # `for event in resp[...]` working unchanged.
            key = "stream" if operation_name == "ConverseStream" else "body"
            collector = (
                _bedrock_converse_stream_collector
                if operation_name == "ConverseStream"
                else _bedrock_invoke_stream_collector
            )
            inner = resp.get(key)
            if inner is not None:
                resp[key] = _StreamProxy(inner, run, t0, collector)
        return resp

    BaseClient._make_api_call = _patched_make_api_call
    _PATCHED.add("botocore")
    _record_instrumented("botocore")
    logger.debug("botocore (bedrock-runtime) auto-instrumented")


# ── httpx ─────────────────────────────────────────────────────────────────────


def _patch_httpx(
    client: "Optional[Dunetrace]" = None, default_agent_id: Optional[str] = None
) -> None:
    if "httpx" in _PATCHED:
        return
    try:
        import httpx
    except ImportError:
        logger.debug("httpx not installed — skipping auto-instrument")
        return

    # Sync
    _orig_send = httpx.Client.send

    @functools.wraps(_orig_send)
    def _patched_send(self, request, **kwargs):
        run = _http_run()
        tool_name = _http_tool_name(request)
        t0 = time.monotonic()
        if run:
            # Not swallow_control_flow: a `stop` policy on this tool call must
            # still prevent the request from going out.
            _safe_emit(lambda: run.tool_called(tool_name, {"url": str(request.url)}))
        try:
            resp = _orig_send(self, request, **kwargs)
        except Exception:
            if run:
                _safe_emit(
                    lambda: run.tool_responded(
                        tool_name,
                        success=False,
                        latency_ms=int((time.monotonic() - t0) * 1000),
                    ),
                    swallow_control_flow=True,
                )
            raise
        if run:
            _safe_emit(lambda: _emit_http_response(run, tool_name, resp, t0))
        return resp

    httpx.Client.send = _patched_send

    # Async
    _orig_asend = httpx.AsyncClient.send

    @functools.wraps(_orig_asend)
    async def _patched_asend(self, request, **kwargs):
        run = _http_run()
        tool_name = _http_tool_name(request)
        t0 = time.monotonic()
        if run:
            _safe_emit(lambda: run.tool_called(tool_name, {"url": str(request.url)}))
        try:
            resp = await _orig_asend(self, request, **kwargs)
        except Exception:
            if run:
                _safe_emit(
                    lambda: run.tool_responded(
                        tool_name,
                        success=False,
                        latency_ms=int((time.monotonic() - t0) * 1000),
                    ),
                    swallow_control_flow=True,
                )
            raise
        if run:
            _safe_emit(lambda: _emit_http_response(run, tool_name, resp, t0))
        return resp

    httpx.AsyncClient.send = _patched_asend

    _PATCHED.add("httpx")
    _record_instrumented("httpx")
    logger.debug("httpx auto-instrumented")


# ── requests ──────────────────────────────────────────────────────────────────


def _patch_requests(
    client: "Optional[Dunetrace]" = None, default_agent_id: Optional[str] = None
) -> None:
    if "requests" in _PATCHED:
        return
    try:
        import requests
    except ImportError:
        logger.debug("requests not installed — skipping auto-instrument")
        return

    _orig_send = requests.Session.send

    @functools.wraps(_orig_send)
    def _patched_send(self, request, **kwargs):
        run = _http_run()
        t0 = time.monotonic()
        if run:
            # Not swallow_control_flow: a `stop` policy on this tool call must
            # still prevent the request from going out.
            _safe_emit(
                lambda: run.tool_called(
                    _requests_tool_name(request),
                    {"url": request.url},
                )
            )
        try:
            resp = _orig_send(self, request, **kwargs)
        except Exception:
            if run:
                _safe_emit(
                    lambda: run.tool_responded(
                        _requests_tool_name(request),
                        success=False,
                        latency_ms=int((time.monotonic() - t0) * 1000),
                    ),
                    swallow_control_flow=True,
                )
            raise
        if run:
            # Reading status_code/headers off the response is part of the emit,
            # so it belongs inside the guard too.
            def _emit_requests_response() -> None:
                success = resp.status_code < 400
                output_len = int(resp.headers.get("content-length", 0) or 0)
                run.tool_responded(
                    _requests_tool_name(request),
                    success=success,
                    output_length=output_len,
                    latency_ms=int((time.monotonic() - t0) * 1000),
                    error=str(resp.status_code) if not success else None,
                )

            _safe_emit(_emit_requests_response)
        return resp

    requests.Session.send = _patched_send

    _PATCHED.add("requests")
    _record_instrumented("requests")
    logger.debug("requests auto-instrumented")


# ── HTTP helpers ──────────────────────────────────────────────────────────────


def _http_tool_name(request) -> str:
    """Use hostname as the tool name — stable, no raw URL transmitted."""
    try:
        return request.url.host
    except AttributeError:
        try:
            from urllib.parse import urlparse

            return urlparse(str(request.url)).hostname or "http"
        except Exception:
            return "http"


def _requests_tool_name(request) -> str:
    try:
        from urllib.parse import urlparse

        return urlparse(request.url).hostname or "http"
    except Exception:
        return "http"


def _emit_http_response(run, tool_name: str, resp, t0: float) -> None:
    """Emit tool_responded for a completed httpx request."""
    latency_ms = int((time.monotonic() - t0) * 1000)
    status = getattr(resp, "status_code", 0)
    success = 200 <= status < 400
    try:
        output_len = int(resp.headers.get("content-length", 0) or 0)
    except Exception:
        output_len = 0
    run.tool_responded(
        tool_name,
        success=success,
        output_length=output_len,
        latency_ms=latency_ms,
        error=str(status) if not success else None,
    )


# ── LangChain / LangGraph ─────────────────────────────────────────────────────


def _patch_langchain(
    client: "Optional[Dunetrace]" = None, default_agent_id: Optional[str] = None
) -> None:
    """Patch BaseChatModel + BaseTool so LangChain/LangGraph agents are tracked
    without threading ``callbacks=[...]`` through every ``.invoke()`` call.

    Reuses ``DunetraceCallbackHandler`` (the existing manual integration)
    rather than re-implementing event emission: this patch's only job is to
    make sure a shared handler instance is present in the callbacks for every
    chat-model / tool call, and to set the re-entrancy flag so the
    openai/anthropic patches don't double-count calls LangChain makes through
    them (e.g. ``ChatOpenAI``).

    Requires an already-open ``dt.run(agent_id=...)`` around the top-level
    call — see the module docstring above for why. The handler still resolves
    an agent_id per the tiered scheme when on_chain_start *does* fire (e.g.
    the caller separately passed ``callbacks=[handler]`` to a chain), but
    that path isn't reachable through this leaf-level patch alone.
    """
    # LangGraph long-term memory (BaseStore) is instrumented independently of the
    # BaseChatModel/BaseTool patches below — it needs no client and no open-run
    # semantics beyond "attach if a run is active", so patch it first, before the
    # client-None guard can short-circuit this function.
    _patch_langgraph_store()

    if "langchain" in _PATCHED:
        return
    try:
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.runnables.config import ensure_config
        from langchain_core.tools import BaseTool
    except ImportError:
        logger.debug("langchain not installed — skipping auto-instrument")
        return
    if client is None:
        logger.warning(
            "Dunetrace: langchain auto-instrumentation requires a client — "
            "use dt.auto_instrument() or dt.init(), not the bare "
            "dunetrace.auto.auto_instrument() function. Skipping."
        )
        return

    from dunetrace.integrations.langchain import DunetraceCallbackHandler

    # One shared handler for the whole process. agent_id is resolved per-run
    # inside on_chain_start (ambient dt.run() -> config.metadata["agent_id"]
    # -> default_agent_id -> loud fallback), so a single instance serves every
    # invocation regardless of which agent_id ends up being used.
    _handler = DunetraceCallbackHandler(client, agent_id=default_agent_id)

    def _callback_list(cbs):
        if cbs is None:
            return []
        if hasattr(cbs, "handlers"):  # BaseCallbackManager
            return list(cbs.handlers)
        return list(cbs)

    def _inject_into_config(config):
        config = ensure_config(config)
        existing = _callback_list(config.get("callbacks"))
        if _handler in existing:
            return config
        config = dict(config)
        config["callbacks"] = existing + [_handler]
        return config

    def _inject_into_callbacks_kwarg(callbacks):
        existing = _callback_list(callbacks)
        if _handler in existing:
            return existing
        return existing + [_handler]

    # ── BaseChatModel ─────────────────────────────────────────────────────────
    _orig_invoke = BaseChatModel.invoke

    @functools.wraps(_orig_invoke)
    def _patched_invoke(self, input, config=None, *, stop=None, **kwargs):
        config = _inject_into_config(config)
        token = _in_framework_call.set(True)
        try:
            return _orig_invoke(self, input, config, stop=stop, **kwargs)
        finally:
            _in_framework_call.reset(token)

    BaseChatModel.invoke = _patched_invoke

    _orig_ainvoke = BaseChatModel.ainvoke

    @functools.wraps(_orig_ainvoke)
    async def _patched_ainvoke(self, input, config=None, *, stop=None, **kwargs):
        config = _inject_into_config(config)
        token = _in_framework_call.set(True)
        try:
            return await _orig_ainvoke(self, input, config, stop=stop, **kwargs)
        finally:
            _in_framework_call.reset(token)

    BaseChatModel.ainvoke = _patched_ainvoke

    _orig_stream = BaseChatModel.stream

    @functools.wraps(_orig_stream)
    def _patched_stream(self, input, config=None, *, stop=None, **kwargs):
        config = _inject_into_config(config)
        token = _in_framework_call.set(True)
        try:
            yield from _orig_stream(self, input, config, stop=stop, **kwargs)
        finally:
            _in_framework_call.reset(token)

    BaseChatModel.stream = _patched_stream

    _orig_astream = BaseChatModel.astream

    @functools.wraps(_orig_astream)
    async def _patched_astream(self, input, config=None, *, stop=None, **kwargs):
        config = _inject_into_config(config)
        token = _in_framework_call.set(True)
        try:
            async for chunk in _orig_astream(self, input, config, stop=stop, **kwargs):
                yield chunk
        finally:
            _in_framework_call.reset(token)

    BaseChatModel.astream = _patched_astream

    # ── BaseTool ──────────────────────────────────────────────────────────────
    _orig_run = BaseTool.run

    @functools.wraps(_orig_run)
    def _patched_run(
        self,
        tool_input,
        verbose=None,
        start_color="green",
        color="green",
        callbacks=None,
        *,
        tags=None,
        metadata=None,
        run_name=None,
        run_id=None,
        config=None,
        tool_call_id=None,
        **kwargs,
    ):
        callbacks = _inject_into_callbacks_kwarg(callbacks)
        token = _in_framework_call.set(True)
        try:
            return _orig_run(
                self,
                tool_input,
                verbose=verbose,
                start_color=start_color,
                color=color,
                callbacks=callbacks,
                tags=tags,
                metadata=metadata,
                run_name=run_name,
                run_id=run_id,
                config=config,
                tool_call_id=tool_call_id,
                **kwargs,
            )
        finally:
            _in_framework_call.reset(token)

    BaseTool.run = _patched_run

    _orig_arun = BaseTool.arun

    @functools.wraps(_orig_arun)
    async def _patched_arun(
        self,
        tool_input,
        verbose=None,
        start_color="green",
        color="green",
        callbacks=None,
        *,
        tags=None,
        metadata=None,
        run_name=None,
        run_id=None,
        config=None,
        tool_call_id=None,
        **kwargs,
    ):
        callbacks = _inject_into_callbacks_kwarg(callbacks)
        token = _in_framework_call.set(True)
        try:
            return await _orig_arun(
                self,
                tool_input,
                verbose=verbose,
                start_color=start_color,
                color=color,
                callbacks=callbacks,
                tags=tags,
                metadata=metadata,
                run_name=run_name,
                run_id=run_id,
                config=config,
                tool_call_id=tool_call_id,
                **kwargs,
            )
        finally:
            _in_framework_call.reset(token)

    BaseTool.arun = _patched_arun

    _PATCHED.add("langchain")
    _record_instrumented("langchain")
    logger.debug("langchain auto-instrumented")


# ── CrewAI ────────────────────────────────────────────────────────────────────


def _patch_crewai(
    client: "Optional[Dunetrace]" = None, default_agent_id: Optional[str] = None
) -> None:
    """Install CrewAI's global LLM/tool hooks and patch Crew/Agent kickoff so a
    run boundary exists even when the caller didn't wrap it in ``dt.run()``.

    CrewAI's hook system (``crewai.hooks``) only covers individual LLM/tool
    calls, not a run boundary — unlike LangChain's ``on_chain_start``, there's
    no "kickoff started" hook to key off of. So the run boundary itself is
    provided by patching ``Crew.kickoff``/``kickoff_async`` and
    ``Agent.kickoff``/``kickoff_async`` directly: when no ``dt.run()`` is
    already open, one is opened around the call, using an agent_id resolved
    per the tiered scheme (CrewAI's bonus tier: ``Crew.name`` if the caller
    set one, or the ``Agent.role`` for a directly-kicked-off agent).
    """
    if "crewai" in _PATCHED:
        return
    try:
        from crewai import Agent, Crew
    except ImportError:
        logger.debug("crewai not installed — skipping auto-instrument")
        return

    # CrewAI memory (short-term/long-term/entity) needs no client — patch it up
    # front so it is covered even if the client-None guard below short-circuits.
    _patch_crewai_memory()
    if client is None:
        logger.warning(
            "Dunetrace: crewai auto-instrumentation requires a client — "
            "use dt.auto_instrument() or dt.init(), not the bare "
            "dunetrace.auto.auto_instrument() function. Skipping."
        )
        return

    from dunetrace.integrations._agent_resolution import resolve_agent_id
    from dunetrace.integrations.crewai import DunetraceCrewCallback

    DunetraceCrewCallback(client).install()

    # ── Crew.kickoff / kickoff_async ─────────────────────────────────────────
    _orig_crew_kickoff = Crew.kickoff

    def _crew_agent_id(crew, inputs) -> str:
        """Resolve the agent_id for this kickoff. Reads framework internals we
        don't control, so it falls back to a usable default rather than letting
        a resolution error escape into the caller's kickoff."""
        try:
            native = crew.name if crew.name and crew.name != "crew" else None
            return resolve_agent_id(
                per_call_agent_id=(inputs or {}).get("agent_id"),
                framework_native_agent_id=native,
                default_agent_id=default_agent_id,
                integration="crewai",
            )
        except Exception:
            logger.debug("dunetrace: crewai agent_id resolution failed", exc_info=True)
            return default_agent_id or "crewai-agent"

    @functools.wraps(_orig_crew_kickoff)
    def _patched_crew_kickoff(self, inputs=None, **kwargs):
        if _current_run.get() is not None:
            return _orig_crew_kickoff(self, inputs, **kwargs)
        agent_id = _crew_agent_id(self, inputs)
        with client.run(agent_id, user_input=str((inputs or {}).get("topic", ""))) as run:
            token = _in_framework_call.set(True)
            try:
                result = _orig_crew_kickoff(self, inputs, **kwargs)
            finally:
                _in_framework_call.reset(token)
            _safe_emit(run.final_answer)
            return result

    Crew.kickoff = _patched_crew_kickoff

    if hasattr(Crew, "kickoff_async"):
        _orig_crew_kickoff_async = Crew.kickoff_async

        @functools.wraps(_orig_crew_kickoff_async)
        async def _patched_crew_kickoff_async(self, inputs=None, **kwargs):
            if _current_run.get() is not None:
                return await _orig_crew_kickoff_async(self, inputs, **kwargs)
            agent_id = _crew_agent_id(self, inputs)
            with client.run(agent_id, user_input=str((inputs or {}).get("topic", ""))) as run:
                token = _in_framework_call.set(True)
                try:
                    result = await _orig_crew_kickoff_async(self, inputs, **kwargs)
                finally:
                    _in_framework_call.reset(token)
                _safe_emit(run.final_answer)
                return result

        Crew.kickoff_async = _patched_crew_kickoff_async

    # ── Agent.kickoff / kickoff_async (standalone agent, no Crew) ────────────
    if hasattr(Agent, "kickoff"):
        _orig_agent_kickoff = Agent.kickoff

        @functools.wraps(_orig_agent_kickoff)
        def _patched_agent_kickoff(self, messages, *args, **kwargs):
            if _current_run.get() is not None:
                return _orig_agent_kickoff(self, messages, *args, **kwargs)
            agent_id = resolve_agent_id(
                framework_native_agent_id=getattr(self, "role", None),
                default_agent_id=default_agent_id,
                integration="crewai",
            )
            with client.run(agent_id, user_input=str(messages)) as run:
                token = _in_framework_call.set(True)
                try:
                    result = _orig_agent_kickoff(self, messages, *args, **kwargs)
                finally:
                    _in_framework_call.reset(token)
                _safe_emit(run.final_answer)
                return result

        Agent.kickoff = _patched_agent_kickoff

    if hasattr(Agent, "kickoff_async"):
        _orig_agent_kickoff_async = Agent.kickoff_async

        @functools.wraps(_orig_agent_kickoff_async)
        async def _patched_agent_kickoff_async(self, messages, *args, **kwargs):
            if _current_run.get() is not None:
                return await _orig_agent_kickoff_async(self, messages, *args, **kwargs)
            agent_id = resolve_agent_id(
                framework_native_agent_id=getattr(self, "role", None),
                default_agent_id=default_agent_id,
                integration="crewai",
            )
            with client.run(agent_id, user_input=str(messages)) as run:
                token = _in_framework_call.set(True)
                try:
                    result = await _orig_agent_kickoff_async(self, messages, *args, **kwargs)
                finally:
                    _in_framework_call.reset(token)
                _safe_emit(run.final_answer)
                return result

        Agent.kickoff_async = _patched_agent_kickoff_async

    _PATCHED.add("crewai")
    _record_instrumented("crewai")
    logger.debug("crewai auto-instrumented")


# ── Framework memory channels ─────────────────────────────────────────────────

# Cap auto-captured memory values so a large store payload never bloats an event.
# The manual run.memory_written() API does not truncate; this cap is specific to
# values pulled off framework internals we don't control the size of.
_MEMORY_VALUE_LIMIT = 4000


def _safe_memory(action) -> None:
    """Run a memory-event emit, swallowing any error. Memory instrumentation is
    strictly additive observability — a bug here must never break the framework
    call it is wrapping. Thin alias for _safe_emit, kept for readability at the
    memory call sites; no memory.* event currently reaches policy evaluation,
    but routing through _safe_emit means it behaves correctly if one ever does."""
    _safe_emit(action)


def _mem_text(value) -> str:
    """Best-effort string form of a written memory value, capped in length."""
    if isinstance(value, str):
        s = value
    else:
        s = None
        for attr in ("task", "data", "value", "content", "text"):
            v = getattr(value, attr, None)
            if isinstance(v, str) and v:
                s = v
                break
        if s is None:
            try:
                s = json.dumps(value, default=str, ensure_ascii=False)
            except Exception:
                s = str(value)
    return s if len(s) <= _MEMORY_VALUE_LIMIT else s[:_MEMORY_VALUE_LIMIT]


def _store_key(namespace, key) -> str:
    """Flatten a LangGraph (namespace_tuple, key) pair into one memory key."""
    try:
        ns = "/".join(str(p) for p in namespace) if namespace else ""
    except TypeError:
        ns = str(namespace)
    return f"{ns}:{key}" if ns else str(key)


def _patch_langgraph_store() -> None:
    """Patch LangGraph's ``BaseStore.{put,get,delete}`` (+ async) so long-term
    memory operations become ``memory.*`` events. ``BaseStore`` defines these
    concretely and delegates to an abstract ``batch``/``abatch``, so patching
    the base intercepts every backend (InMemoryStore, PostgresStore, …)."""
    if "langgraph_store" in _PATCHED:
        return
    try:
        from langgraph.store.base import BaseStore
    except ImportError:
        logger.debug("langgraph not installed — skipping memory auto-instrument")
        return

    _orig_put = BaseStore.put

    @functools.wraps(_orig_put)
    def _patched_put(self, namespace, key, value, *args, **kwargs):
        run = _current_run.get()
        if run is not None:
            _safe_memory(lambda: run.memory_written(_store_key(namespace, key), _mem_text(value)))
        return _orig_put(self, namespace, key, value, *args, **kwargs)

    BaseStore.put = _patched_put

    _orig_get = BaseStore.get

    @functools.wraps(_orig_get)
    def _patched_get(self, namespace, key, *args, **kwargs):
        run = _current_run.get()
        if run is not None:
            _safe_memory(lambda: run.memory_read(_store_key(namespace, key)))
        return _orig_get(self, namespace, key, *args, **kwargs)

    BaseStore.get = _patched_get

    _orig_delete = BaseStore.delete

    @functools.wraps(_orig_delete)
    def _patched_delete(self, namespace, key, *args, **kwargs):
        run = _current_run.get()
        if run is not None:
            _safe_memory(lambda: run.memory_cleared(_store_key(namespace, key)))
        return _orig_delete(self, namespace, key, *args, **kwargs)

    BaseStore.delete = _patched_delete

    _orig_aput = BaseStore.aput

    @functools.wraps(_orig_aput)
    async def _patched_aput(self, namespace, key, value, *args, **kwargs):
        run = _current_run.get()
        if run is not None:
            _safe_memory(lambda: run.memory_written(_store_key(namespace, key), _mem_text(value)))
        return await _orig_aput(self, namespace, key, value, *args, **kwargs)

    BaseStore.aput = _patched_aput

    _orig_aget = BaseStore.aget

    @functools.wraps(_orig_aget)
    async def _patched_aget(self, namespace, key, *args, **kwargs):
        run = _current_run.get()
        if run is not None:
            _safe_memory(lambda: run.memory_read(_store_key(namespace, key)))
        return await _orig_aget(self, namespace, key, *args, **kwargs)

    BaseStore.aget = _patched_aget

    _orig_adelete = BaseStore.adelete

    @functools.wraps(_orig_adelete)
    async def _patched_adelete(self, namespace, key, *args, **kwargs):
        run = _current_run.get()
        if run is not None:
            _safe_memory(lambda: run.memory_cleared(_store_key(namespace, key)))
        return await _orig_adelete(self, namespace, key, *args, **kwargs)

    BaseStore.adelete = _patched_adelete

    _PATCHED.add("langgraph_store")
    logger.debug("langgraph store memory auto-instrumented")


# CrewAI memory classes, keyed by a short memory-kind label. Ordered so the
# concrete subclasses that *override* save() are wrapped in their own right,
# while ShortTermMemory (which inherits Memory.save) is covered by the base.
_CREWAI_MEMORY_CLASSES = [
    ("crewai.memory.short_term.short_term_memory", "ShortTermMemory", "short_term"),
    ("crewai.memory.long_term.long_term_memory", "LongTermMemory", "long_term"),
    ("crewai.memory.entity.entity_memory", "EntityMemory", "entity"),
    ("crewai.memory.memory", "Memory", "memory"),
]


def _wrap_crewai_memory_class(cls, mem_key: str) -> bool:
    """Wrap a CrewAI memory class's save/search/reset — but only methods it
    defines *itself* (``in cls.__dict__``), so a method inherited from the base
    ``Memory`` isn't double-wrapped once the base is patched too."""
    did = False

    if "save" in cls.__dict__:
        _orig_save = cls.save

        @functools.wraps(_orig_save)
        def _patched_save(self, *args, _orig=_orig_save, _k=mem_key, **kwargs):
            run = _current_run.get()
            if run is not None:
                val = args[0] if args else kwargs.get("value", kwargs.get("item"))
                _safe_memory(lambda: run.memory_written(_k, _mem_text(val)))
            return _orig(self, *args, **kwargs)

        cls.save = _patched_save
        did = True

    if "search" in cls.__dict__:
        _orig_search = cls.search

        @functools.wraps(_orig_search)
        def _patched_search(self, *args, _orig=_orig_search, **kwargs):
            run = _current_run.get()
            if run is not None:
                q = args[0] if args else kwargs.get("query", kwargs.get("task"))
                if q:
                    _safe_memory(lambda: run.memory_read(str(q)[:200]))
            return _orig(self, *args, **kwargs)

        cls.search = _patched_search
        did = True

    if "reset" in cls.__dict__:
        _orig_reset = cls.reset

        @functools.wraps(_orig_reset)
        def _patched_reset(self, *args, _orig=_orig_reset, **kwargs):
            run = _current_run.get()
            if run is not None:
                _safe_memory(lambda: run.memory_cleared(None))
            return _orig(self, *args, **kwargs)

        cls.reset = _patched_reset
        did = True

    return did


def _patch_crewai_memory() -> None:
    """Patch CrewAI's short-term / long-term / entity memory so save/search/reset
    become ``memory.*`` events. No client needed — these attach to an already-open
    run (CrewAI's kickoff patch opens one)."""
    if "crewai_memory" in _PATCHED:
        return
    patched_any = False
    for modpath, clsname, mem_key in _CREWAI_MEMORY_CLASSES:
        try:
            mod = importlib.import_module(modpath)
            cls = getattr(mod, clsname)
        except Exception:
            continue
        patched_any = _wrap_crewai_memory_class(cls, mem_key) or patched_any
    if patched_any:
        _PATCHED.add("crewai_memory")
        logger.debug("crewai memory auto-instrumented")


# ── Shared helpers ────────────────────────────────────────────────────────────


def _estimate_tokens(messages) -> int:
    """Rough estimate: 4 chars ≈ 1 token. Never sends raw content."""
    if not messages:
        return 0
    chars = sum(
        len(str(m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")))
        for m in messages
    )
    return max(1, chars // 4)


# ── Dispatch ──────────────────────────────────────────────────────────────────

_PATCHERS = {
    "openai": _patch_openai,
    "anthropic": _patch_anthropic,
    "mistral": _patch_mistral,
    "botocore": _patch_botocore,
    "httpx": _patch_httpx,
    "requests": _patch_requests,
    "langchain": _patch_langchain,
    "crewai": _patch_crewai,
}


def auto_instrument(
    frameworks: Optional[List[str]] = None,
    client: "Optional[Dunetrace]" = None,
    default_agent_id: Optional[str] = None,
) -> None:
    """
    Monkey-patch supported AI framework clients to emit Dunetrace events
    automatically whenever they are called inside a ``dt.run()`` context.

    :param frameworks: List of framework names to patch. ``None`` patches all
                       detected installed frameworks.
    :param client: Required for ``langchain``/``crewai`` — those integrations
                   can open their own run when no ``dt.run()`` is active, and
                   need a client to do it with. Unused by the other four
                   frameworks, which only ever attach to an already-open run.
    :param default_agent_id: Fallback agent_id for runs ``langchain``/``crewai``
                       open themselves. Normally set via ``dt.init(agent_id=...)``
                       or the ``DUNETRACE_AGENT_ID`` environment variable rather
                       than passed here directly.
    """
    targets = frameworks if frameworks is not None else list(_PATCHERS)
    for name in targets:
        patcher = _PATCHERS.get(name)
        if patcher is None:
            logger.warning(
                "Unknown framework %r for auto_instrument — supported: %s",
                name,
                list(_PATCHERS),
            )
            continue
        patcher(client=client, default_agent_id=default_agent_id)
