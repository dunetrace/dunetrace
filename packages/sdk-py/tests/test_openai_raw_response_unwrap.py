"""
Normalisation of openai raw-response wrappers in auto-instrumentation.

The bug: langchain_openai/chat_models/base.py calls
`self.client.with_raw_response.create(**payload)` (12 call sites), so the
patched `Completions.create` returned an `openai._legacy_response.LegacyAPIResponse`
rather than a `ChatCompletion`. That object has no `.choices` and no `.usage`, so
every non-streamed LangChain LLM call recorded
`{"output": "", "output_length": 0, "completion_tokens": 0}` even when the model
had answered normally.

Both halves of the patch were working. The object they received was wrong.

Tests construct the wrappers directly — no network, no API key.
"""

from __future__ import annotations

import asyncio
import unittest

from dunetrace import Dunetrace
from dunetrace.auto import (
    _emit_openai_response,
    _is_response_wrapper,
    _unwrap_response,
    _unwrap_response_async,
)
from dunetrace.models import EventType


# ── Shapes ───────────────────────────────────────────────────────────────────


def _chat_completion(text="The capital of France is Paris.", completion_tokens=8):
    """A ChatCompletion-shaped object: has .choices and .usage."""
    msg = type("Msg", (), {"content": text})()
    choice = type("Choice", (), {"finish_reason": "stop", "message": msg})()
    usage = type("Usage", (), {"prompt_tokens": 24, "completion_tokens": completion_tokens})()
    return type("ChatCompletion", (), {"choices": [choice], "usage": usage})()


class _SyncWrapper:
    """LegacyAPIResponse / APIResponse shape: no .choices, sync .parse()."""

    def __init__(self, parsed):
        self._parsed = parsed
        self.parse_calls = 0
        self.headers = {"x-request-id": "req_123"}

    def parse(self):
        self.parse_calls += 1
        return self._parsed


class _AwaitableWrapper:
    """AsyncAPIResponse shape, and LegacyAPIResponse on the async client from
    openai v2 — parse() returns an awaitable. This is the forward-compat trap:
    a naive resp.parse() hands the extractors a coroutine object."""

    def __init__(self, parsed):
        self._parsed = parsed
        self.parse_calls = 0
        self.headers = {"x-request-id": "req_456"}

    def parse(self):
        self.parse_calls += 1

        async def _coro():
            return self._parsed

        return _coro()


class _MemoisingWrapper(_SyncWrapper):
    """Mirrors openai's real `_parsed_by_type` memoisation: parse() returns the
    SAME object every time and never re-reads the body."""

    def __init__(self, parsed):
        super().__init__(parsed)
        self._cache = None
        self.body_reads = 0

    def parse(self):
        self.parse_calls += 1
        if self._cache is None:
            self.body_reads += 1
            self._cache = self._parsed
        return self._cache


def _client():
    c = Dunetrace(api_key="k")
    c._ship = lambda batch: None
    return c


def _llm_responded(run):
    return [e for e in run.state.events if e.event_type == EventType.LLM_RESPONDED][-1]


# ── 1 & 2: raw-response wrappers record real values ─────────────────────────


class TestWrapperRecordsRealValues(unittest.TestCase):
    """Criteria 1 and 2. The recorded payload must match what the plain
    `create` path records for the same underlying completion."""

    def _baseline(self):
        c = _client()
        with c.run("a") as run:
            _emit_openai_response(run, _chat_completion(), 0.0)
            payload = dict(_llm_responded(run).payload)
        c.shutdown(timeout=1)
        return payload

    def test_sync_with_raw_response_matches_the_plain_path(self):
        wrapper = _SyncWrapper(_chat_completion())
        c = _client()
        with c.run("a") as run:
            _emit_openai_response(run, _unwrap_response(wrapper), 0.0)
            payload = _llm_responded(run).payload
        c.shutdown(timeout=1)

        base = self._baseline()
        self.assertEqual(payload["output"], "The capital of France is Paris.")
        self.assertEqual(payload["output_length"], 31)
        self.assertEqual(payload["completion_tokens"], 8)
        self.assertEqual(payload["prompt_tokens"], 24)
        self.assertEqual(payload["finish_reason"], "stop")
        for k in ("output", "output_length", "completion_tokens", "finish_reason"):
            self.assertEqual(payload[k], base[k], f"{k} diverges from the plain path")

    def test_async_with_raw_response_matches_the_plain_path(self):
        wrapper = _SyncWrapper(_chat_completion())

        async def go():
            return await _unwrap_response_async(wrapper)

        c = _client()
        with c.run("a") as run:
            _emit_openai_response(run, asyncio.run(go()), 0.0)
            payload = _llm_responded(run).payload
        c.shutdown(timeout=1)
        self.assertEqual(payload["output"], "The capital of France is Paris.")
        self.assertEqual(payload["completion_tokens"], 8)
        self.assertEqual(payload["prompt_tokens"], 24)


# ── 3: coroutine parse() ────────────────────────────────────────────────────


class TestCoroutineParse(unittest.TestCase):
    """Criterion 3 — the openai v2 trap, plus AsyncAPIResponse today.

    Coverage here is against a stub by necessity: the real coroutine-parse
    LegacyAPIResponse arrives with openai v2. The stub is kept precisely
    because it is the only thing that will catch that upgrade regressing this.
    """

    def test_async_path_awaits_an_awaitable_parse(self):
        wrapper = _AwaitableWrapper(_chat_completion(text="Awaited fine.", completion_tokens=3))

        async def go():
            return await _unwrap_response_async(wrapper)

        plain = asyncio.run(go())
        self.assertFalse(asyncio.iscoroutine(plain), "handed the extractors a coroutine")
        c = _client()
        with c.run("a") as run:
            _emit_openai_response(run, plain, 0.0)
            payload = _llm_responded(run).payload
        c.shutdown(timeout=1)
        self.assertEqual(payload["output"], "Awaited fine.")
        self.assertEqual(payload["completion_tokens"], 3)

    def test_sync_path_declines_an_awaitable_rather_than_leaking_it(self):
        """A coroutine must never reach the sync extractors, and must not leak
        an un-awaited-coroutine warning into the host's logs."""
        import warnings

        wrapper = _AwaitableWrapper(_chat_completion())
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            out = _unwrap_response(wrapper)
        self.assertIs(out, wrapper, "sync path must fall back, not return a coroutine")

    def test_real_async_api_response_parse_is_already_a_coroutine(self):
        """Not merely forward-compat: this is true in the installed openai."""
        import inspect

        from openai._response import AsyncAPIResponse

        self.assertTrue(inspect.iscoroutinefunction(AsyncAPIResponse.parse))


# ── 4: caller transparency ──────────────────────────────────────────────────


class TestCallerTransparency(unittest.TestCase):
    """Criterion 4. Identity, not equality.

    LangChain asked for a raw response because it reads headers off the wrapper.
    Returning the unwrapped ChatCompletion breaks it, and an equality assertion
    would not catch that.
    """

    def test_sync_patch_returns_the_original_object(self):
        import openai.resources.chat.completions as _mod

        from dunetrace.auto import _patch_openai

        completion = _chat_completion()
        sentinel = _SyncWrapper(completion)
        orig = _mod.Completions.create
        _mod.Completions.create = lambda self, **kw: sentinel
        try:
            from dunetrace import auto

            auto._PATCHED.discard("openai")
            _patch_openai()
            c = _client()
            with c.run("a") as run:
                returned = _mod.Completions.create(None, messages=[], model="gpt-4o")
                payload = _llm_responded(run).payload
            c.shutdown(timeout=1)
        finally:
            _mod.Completions.create = orig
            from dunetrace import auto

            auto._PATCHED.discard("openai")

        self.assertIs(returned, sentinel, "caller got the unwrapped object")
        self.assertEqual(returned.headers["x-request-id"], "req_123")
        # ...and instrumentation still saw the real values.
        self.assertEqual(payload["output"], "The capital of France is Paris.")
        self.assertEqual(payload["completion_tokens"], 8)

    def test_async_patch_returns_the_original_object(self):
        import openai.resources.chat.completions as _mod

        from dunetrace import auto
        from dunetrace.auto import _patch_openai

        sentinel = _AwaitableWrapper(_chat_completion())

        async def _fake(self, **kw):
            return sentinel

        orig = _mod.AsyncCompletions.create
        _mod.AsyncCompletions.create = _fake
        try:
            auto._PATCHED.discard("openai")
            _patch_openai()
            c = _client()
            with c.run("a") as run:
                returned = asyncio.run(
                    _mod.AsyncCompletions.create(None, messages=[], model="gpt-4o")
                )
                payload = _llm_responded(run).payload
            c.shutdown(timeout=1)
        finally:
            _mod.AsyncCompletions.create = orig
            auto._PATCHED.discard("openai")

        self.assertIs(returned, sentinel)
        self.assertEqual(payload["output"], "The capital of France is Paris.")


# ── 5 & 6: plain path and pass-through ──────────────────────────────────────


class TestNonWrapperShapes(unittest.TestCase):
    def test_plain_chat_completion_is_returned_untouched(self):
        """Criterion 5 — identical events to before this change."""
        completion = _chat_completion()
        self.assertIs(_unwrap_response(completion), completion)
        self.assertFalse(_is_response_wrapper(completion))

    def test_object_with_neither_choices_nor_parse_passes_through(self):
        """Criterion 6 — pass through without raising."""
        opaque = object()
        self.assertIs(_unwrap_response(opaque), opaque)

        c = _client()
        with c.run("a") as run:
            _emit_openai_response(run, _unwrap_response(opaque), 0.0)
            payload = _llm_responded(run).payload
        c.shutdown(timeout=1)
        # Unchanged from today: nothing readable, so the call is recorded as
        # unmeasurable rather than as an empty response.
        self.assertEqual(payload["output_length"], 0)
        self.assertIn("instrumentation_degraded", payload)

    def test_a_raising_parse_never_reaches_the_host(self):
        class _Exploding:
            def parse(self):
                raise RuntimeError("body already consumed")

        boom = _Exploding()
        self.assertIs(_unwrap_response(boom), boom)

        async def go():
            return await _unwrap_response_async(boom)

        self.assertIs(asyncio.run(go()), boom)

    def test_a_property_that_raises_does_not_break_shape_detection(self):
        class _Hostile:
            @property
            def choices(self):
                raise ValueError("nope")

            def parse(self):
                return _chat_completion()

        h = _Hostile()
        self.assertFalse(_is_response_wrapper(h))
        self.assertIs(_unwrap_response(h), h)


# ── 7: no double-read ───────────────────────────────────────────────────────


class TestNoDoubleRead(unittest.TestCase):
    """Criterion 7. openai memoises parse() via _parsed_by_type, so our call
    cannot consume the body or disturb LangChain's own .parse(). Confirmed
    here rather than trusted."""

    def test_our_parse_then_the_callers_parse_agree_and_do_not_re_read(self):
        completion = _chat_completion()
        wrapper = _MemoisingWrapper(completion)

        ours = _unwrap_response(wrapper)
        theirs = wrapper.parse()  # what LangChain does next

        self.assertIs(ours, completion)
        self.assertIs(theirs, ours, "caller got a different object than we did")
        self.assertEqual(wrapper.parse_calls, 2)
        self.assertEqual(wrapper.body_reads, 1, "body was read twice")

    def test_real_openai_memoisation_attribute_still_exists(self):
        """If openai drops _parsed_by_type, our extra parse() could start
        costing a second body read — fail loudly rather than silently."""
        import inspect

        from openai._legacy_response import LegacyAPIResponse

        self.assertIn("_parsed_by_type", inspect.getsource(LegacyAPIResponse.parse))


# ── 8: end-to-end regression ────────────────────────────────────────────────


class TestEndToEndRegression(unittest.TestCase):
    """Criterion 8 — the regression that started all of it, made permanent.

    Drives the patched Completions.create the way langchain_openai does
    (via with_raw_response) and asserts both that the values are real and that
    EMPTY_LLM_RESPONSE does not fire.
    """

    def test_langchain_shaped_call_records_real_output_and_does_not_fire_empty(self):
        import openai.resources.chat.completions as _mod

        from dunetrace import auto
        from dunetrace.auto import _patch_openai
        from dunetrace.detectors import EmptyLlmResponseDetector
        from dunetrace.models import FailureType
        from dunetrace.detectors import run_detectors

        # What langchain_openai receives from with_raw_response.create()
        wrapper = _SyncWrapper(_chat_completion(text="Paris is the capital.", completion_tokens=6))
        orig = _mod.Completions.create
        _mod.Completions.create = lambda self, **kw: wrapper
        try:
            auto._PATCHED.discard("openai")
            _patch_openai()
            c = _client()
            with c.run("langgraph-agent") as run:
                for _ in range(3):
                    _mod.Completions.create(None, messages=[{"content": "hi"}], model="gpt-4o")
                payload = _llm_responded(run).payload
                state = run.state
            c.shutdown(timeout=1)
        finally:
            _mod.Completions.create = orig
            auto._PATCHED.discard("openai")

        self.assertEqual(payload["output"], "Paris is the capital.")
        self.assertGreater(payload["output_length"], 0)
        self.assertEqual(payload["completion_tokens"], 6)
        self.assertEqual(payload["finish_reason"], "stop")
        self.assertNotIn("instrumentation_degraded", payload)

        self.assertIsNone(
            EmptyLlmResponseDetector().on_run_completion(state),
            "EMPTY_LLM_RESPONSE fired on a run where the model answered normally",
        )
        fired = {s.failure_type for s in run_detectors(state)}
        self.assertNotIn(FailureType.EMPTY_LLM_RESPONSE, fired)
        self.assertNotIn(FailureType.INSTRUMENTATION_DEGRADED, fired)


def _openai_httpx():
    """The httpx module object *openai* holds a reference to.

    tests/test_auto_instrument.py installs a fake httpx in sys.modules and its
    cleanup then DELETES the entry rather than restoring the original. A later
    `import httpx` therefore builds a SECOND real httpx module — same file,
    different module object, different `Client` class. openai captured the
    first one at import time, so its isinstance check rejects a client built
    from the second, with the memorable message
    "Expected an instance of `httpx.Client` but got <class 'httpx.Client'>".

    Sourcing httpx from openai itself makes the two identical by construction,
    which is what lets this test run in the full suite instead of skipping.
    """
    from openai import _base_client

    return _base_client.httpx


def _langchain_available():
    try:
        import langchain_openai  # noqa: F401

        return True
    except Exception:
        return False


@unittest.skipUnless(_langchain_available(), "langchain_openai not installed")
class TestRealLangChainEndToEnd(unittest.TestCase):
    """Criterion 8, driven through the REAL langchain_openai code path.

    The stubbed version above proves the mechanism; this proves the integration.
    langchain_openai calls client.with_raw_response.create() itself — nothing
    here stubs that out — so this exercises the actual 12-call-site path that
    produced the incident. Only the HTTP transport is faked, so no network and
    no API key are needed.
    """

    def test_chat_openai_invoke_records_real_output_and_fires_nothing(self):
        import sys

        httpx = _openai_httpx()
        sys.modules["httpx"] = httpx

        from langchain_openai import ChatOpenAI

        from dunetrace import auto
        from dunetrace.detectors import run_detectors
        from dunetrace.models import FailureType

        body = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "The capital of France is Paris."},
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "usage": {"prompt_tokens": 24, "completion_tokens": 8, "total_tokens": 32},
        }
        auto._PATCHED.discard("openai")
        try:
            auto.auto_instrument(["openai"])
            transport = httpx.MockTransport(
                lambda request: httpx.Response(200, json=body, headers={"x-request-id": "req_e2e"})
            )
            try:
                llm = ChatOpenAI(
                    model="gpt-4o",
                    api_key="sk-test",
                    http_client=httpx.Client(transport=transport),
                )
            except TypeError as exc:  # pragma: no cover - environment-dependent
                # openai validates http_client with isinstance against its own
                # httpx reference. When the whole suite runs, an earlier test
                # (tests/test_auto_instrument.py) swaps sys.modules["httpx"] for
                # a fake and its cleanup deletes the entry rather than restoring
                # it, leaving this process with httpx state openai's check
                # rejects — the memorable "Expected an instance of
                # `httpx.Client` but got <class 'httpx.Client'>".
                #
                # That is pre-existing pollution, not a property of the code
                # under test, and the stubbed end-to-end above already asserts
                # the same regression without needing a live client. Skipping
                # here keeps this test honest rather than forcing past a
                # condition that would make it lie about what it exercised.
                # Run this file on its own and it executes fully.
                self.skipTest(f"httpx/openai module state polluted by an earlier test: {exc}")
            c = _client()
            with c.run("langgraph-agent") as run:
                result = llm.invoke("What is the capital of France?")
                payload = dict(_llm_responded(run).payload)
                state = run.state
            c.shutdown(timeout=1)
        finally:
            auto._PATCHED.discard("openai")
        # LangChain still gets its answer — instrumentation is invisible to it.
        self.assertEqual(result.content, "The capital of France is Paris.")

        # ...and we now record what actually happened.
        self.assertEqual(payload["output"], "The capital of France is Paris.")
        self.assertEqual(payload["output_length"], 31)
        self.assertEqual(payload["completion_tokens"], 8)
        self.assertEqual(payload["prompt_tokens"], 24)
        self.assertEqual(payload["finish_reason"], "stop")
        self.assertNotIn("instrumentation_degraded", payload)

        # The regression: this run used to fire EMPTY_LLM_RESPONSE at HIGH on
        # every turn. It must now fire nothing at all.
        fired = {s.failure_type for s in run_detectors(state)}
        self.assertNotIn(FailureType.EMPTY_LLM_RESPONSE, fired)
        self.assertNotIn(FailureType.INSTRUMENTATION_DEGRADED, fired)


if __name__ == "__main__":
    unittest.main(verbosity=2)
