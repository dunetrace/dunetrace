"""
Automatic instrumentation patches for popular AI frameworks.

Patches are applied at the class level so all client instances are covered.
Each patch is idempotent — calling auto_instrument() more than once is safe.

Supported frameworks:
- ``openai``    — chat.completions.create (sync + async)
- ``anthropic`` — messages.create (sync + async)
- ``httpx``     — Client.send + AsyncClient.send (all outbound HTTP as tool calls)
- ``requests``  — Session.send (all outbound HTTP as tool calls)

Usage::

    dt = Dunetrace(...)
    dt.auto_instrument()                          # patch all detected frameworks
    dt.auto_instrument(["openai", "anthropic"])   # patch only LLM clients
    dt.auto_instrument(["httpx", "requests"])     # patch only HTTP clients
"""

from __future__ import annotations

import functools
import logging
import time
from typing import List, Optional

from dunetrace.context import _current_run
from dunetrace.models import hash_content

logger = logging.getLogger("dunetrace.auto")

# Tracks which frameworks have already been patched (prevents double-wrapping).
_PATCHED: set[str] = set()


# ── OpenAI ────────────────────────────────────────────────────────────────────


def _patch_openai() -> None:
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
        run = _current_run.get()
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
            run = _current_run.get()
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
    latency_ms = int((time.monotonic() - t0) * 1000)
    finish = _openai_finish_reason(resp)
    text = _openai_content(resp)
    run.llm_responded(
        completion_tokens=comp_toks,
        latency_ms=latency_ms,
        finish_reason=finish,
        output_hash=hash_content(text) if text else "",
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


# ── Anthropic ─────────────────────────────────────────────────────────────────


def _patch_anthropic() -> None:
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
        run = _current_run.get()
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
            run = _current_run.get()
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
        output_hash=hash_content(text) if text else "",
        output_length=len(text) if text else 0,
    )


def _anthropic_content(resp) -> str:
    try:
        block = resp.content[0]
        return getattr(block, "text", "") or ""
    except (AttributeError, IndexError):
        return ""


# ── httpx ─────────────────────────────────────────────────────────────────────


def _patch_httpx() -> None:
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
        run = _current_run.get()
        tool_name = _http_tool_name(request)
        t0 = time.monotonic()
        if run:
            run.tool_called(tool_name, {"url_hash": hash_content(str(request.url))})
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
        run = _current_run.get()
        tool_name = _http_tool_name(request)
        t0 = time.monotonic()
        if run:
            run.tool_called(tool_name, {"url_hash": hash_content(str(request.url))})
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


def _patch_requests() -> None:
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
        run = _current_run.get()
        t0 = time.monotonic()
        if run:
            run.tool_called(
                _requests_tool_name(request),
                {"url_hash": hash_content(request.url)},
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
}


def auto_instrument(frameworks: Optional[List[str]] = None) -> None:
    """
    Monkey-patch supported AI framework clients to emit Dunetrace events
    automatically whenever they are called inside a ``dt.run()`` context.

    :param frameworks: List of framework names to patch. ``None`` patches all
                       detected installed frameworks.
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
        patcher()
