"""
Automatic instrumentation patches for popular AI frameworks.

Patches are applied at the class level so all client instances are covered.
Each patch is idempotent — calling auto_instrument() more than once is safe.

Supported frameworks:
- ``openai``    — chat.completions.create (sync + async)
- ``anthropic`` — messages.create (sync + async)
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

import functools
import importlib
import json
import logging
import time
from typing import TYPE_CHECKING, List, Optional

from dunetrace.context import _current_run, _in_framework_call

if TYPE_CHECKING:
    from dunetrace.client import Dunetrace

logger = logging.getLogger("dunetrace.auto")

# Tracks which frameworks have already been patched (prevents double-wrapping).
_PATCHED: set[str] = set()


# ── OpenAI ────────────────────────────────────────────────────────────────────


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
            run.llm_called(model, prompt_tokens=_estimate_tokens(messages))
        try:
            resp = _orig_create(self, messages=messages, model=model, **kwargs)
        except Exception:
            if run:
                run.llm_responded(
                    finish_reason="error",
                    latency_ms=int((time.monotonic() - t0) * 1000),
                )
            raise
        if run:
            _emit_openai_response(run, resp, t0)
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
                run.llm_called(model, prompt_tokens=_estimate_tokens(messages))
            try:
                resp = await _orig_acreate(self, messages=messages, model=model, **kwargs)
            except Exception:
                if run:
                    run.llm_responded(
                        finish_reason="error",
                        latency_ms=int((time.monotonic() - t0) * 1000),
                    )
                raise
            if run:
                _emit_openai_response(run, resp, t0)
            return resp

        _mod.AsyncCompletions.create = _patched_acreate
    except AttributeError:
        pass  # older openai version without async client

    _PATCHED.add("openai")
    logger.debug("openai auto-instrumented")


def _emit_openai_response(run, resp, t0: float) -> None:
    usage = getattr(resp, "usage", None)
    comp_toks = getattr(usage, "completion_tokens", 0) or 0
    reason_toks = _completion_detail_tokens(usage, "reasoning_tokens")
    latency_ms = int((time.monotonic() - t0) * 1000)
    finish = _openai_finish_reason(resp)
    text = _openai_content(resp)
    run.llm_responded(
        completion_tokens=comp_toks,
        reasoning_tokens=reason_toks,
        latency_ms=latency_ms,
        finish_reason=finish,
        output=text,
        output_length=len(text) if text else 0,
    )


def _openai_finish_reason(resp) -> str:
    try:
        return resp.choices[0].finish_reason or "stop"
    except (AttributeError, IndexError):
        return "stop"


def _openai_content(resp) -> str:
    try:
        return resp.choices[0].message.content or ""
    except (AttributeError, IndexError):
        return ""


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
            run.llm_called(model, prompt_tokens=_estimate_tokens(messages))
        try:
            resp = _orig_create(
                self, model=model, messages=messages, max_tokens=max_tokens, **kwargs
            )
        except Exception:
            if run:
                run.llm_responded(
                    finish_reason="error",
                    latency_ms=int((time.monotonic() - t0) * 1000),
                )
            raise
        if run:
            _emit_anthropic_response(run, resp, t0)
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
                run.llm_called(model, prompt_tokens=_estimate_tokens(messages))
            try:
                resp = await _orig_acreate(
                    self,
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    **kwargs,
                )
            except Exception:
                if run:
                    run.llm_responded(
                        finish_reason="error",
                        latency_ms=int((time.monotonic() - t0) * 1000),
                    )
                raise
            if run:
                _emit_anthropic_response(run, resp, t0)
            return resp

        _mod.AsyncMessages.create = _patched_acreate
    except AttributeError:
        pass  # older anthropic version without async client

    _PATCHED.add("anthropic")
    logger.debug("anthropic auto-instrumented")


def _emit_anthropic_response(run, resp, t0: float) -> None:
    usage = getattr(resp, "usage", None)
    comp_toks = getattr(usage, "output_tokens", 0) or 0
    latency_ms = int((time.monotonic() - t0) * 1000)
    finish = getattr(resp, "stop_reason", "end_turn") or "end_turn"
    text = _anthropic_content(resp)
    run.llm_responded(
        completion_tokens=comp_toks,
        latency_ms=latency_ms,
        finish_reason=finish,
        output=text,
        output_length=len(text) if text else 0,
    )


def _anthropic_content(resp) -> str:
    try:
        block = resp.content[0]
        return getattr(block, "text", "") or ""
    except (AttributeError, IndexError):
        return ""


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
        run = None if _in_framework_call.get() else _current_run.get()
        tool_name = _http_tool_name(request)
        t0 = time.monotonic()
        if run:
            run.tool_called(tool_name, {"url": str(request.url)})
        try:
            resp = _orig_send(self, request, **kwargs)
        except Exception:
            if run:
                run.tool_responded(
                    tool_name,
                    success=False,
                    latency_ms=int((time.monotonic() - t0) * 1000),
                )
            raise
        if run:
            _emit_http_response(run, tool_name, resp, t0)
        return resp

    httpx.Client.send = _patched_send

    # Async
    _orig_asend = httpx.AsyncClient.send

    @functools.wraps(_orig_asend)
    async def _patched_asend(self, request, **kwargs):
        run = None if _in_framework_call.get() else _current_run.get()
        tool_name = _http_tool_name(request)
        t0 = time.monotonic()
        if run:
            run.tool_called(tool_name, {"url": str(request.url)})
        try:
            resp = await _orig_asend(self, request, **kwargs)
        except Exception:
            if run:
                run.tool_responded(
                    tool_name,
                    success=False,
                    latency_ms=int((time.monotonic() - t0) * 1000),
                )
            raise
        if run:
            _emit_http_response(run, tool_name, resp, t0)
        return resp

    httpx.AsyncClient.send = _patched_asend

    _PATCHED.add("httpx")
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
        run = None if _in_framework_call.get() else _current_run.get()
        t0 = time.monotonic()
        if run:
            run.tool_called(
                _requests_tool_name(request),
                {"url": request.url},
            )
        try:
            resp = _orig_send(self, request, **kwargs)
        except Exception:
            if run:
                run.tool_responded(
                    _requests_tool_name(request),
                    success=False,
                    latency_ms=int((time.monotonic() - t0) * 1000),
                )
            raise
        if run:
            tool_name = _requests_tool_name(request)
            success = resp.status_code < 400
            output_len = int(resp.headers.get("content-length", 0) or 0)
            run.tool_responded(
                tool_name,
                success=success,
                output_length=output_len,
                latency_ms=int((time.monotonic() - t0) * 1000),
                error=str(resp.status_code) if not success else None,
            )
        return resp

    requests.Session.send = _patched_send

    _PATCHED.add("requests")
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
        native = crew.name if crew.name and crew.name != "crew" else None
        return resolve_agent_id(
            per_call_agent_id=(inputs or {}).get("agent_id"),
            framework_native_agent_id=native,
            default_agent_id=default_agent_id,
            integration="crewai",
        )

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
            run.final_answer()
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
                run.final_answer()
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
                run.final_answer()
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
                run.final_answer()
                return result

        Agent.kickoff_async = _patched_agent_kickoff_async

    _PATCHED.add("crewai")
    logger.debug("crewai auto-instrumented")


# ── Framework memory channels ─────────────────────────────────────────────────

# Cap auto-captured memory values so a large store payload never bloats an event.
# The manual run.memory_written() API does not truncate; this cap is specific to
# values pulled off framework internals we don't control the size of.
_MEMORY_VALUE_LIMIT = 4000


def _safe_memory(action) -> None:
    """Run a memory-event emit, swallowing any error. Memory instrumentation is
    strictly additive observability — a bug here must never break the framework
    call it is wrapping."""
    try:
        action()
    except Exception:
        logger.debug("dunetrace memory instrumentation emit failed", exc_info=True)


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
