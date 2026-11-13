"""Regression tests for the ContextVar handling in DunetraceCallbackHandler.

These drive the REAL langchain_core callback managers. That matters: every
other LangChain test in this tree stubs LangChain out and calls the hooks
directly, in the test's own context, so the dispatcher — the only thing that
introduces ``copy_context()`` — is never exercised. A whole class of bug was
therefore invisible to the suite.

What went wrong, and what each test pins:

On the async path LangChain dispatches a *synchronous* handler through
``run_in_executor(None, partial(copy_context().run, event, ...))``, giving every
callback a fresh context copy. So ``_current_run.set()`` in ``on_chain_start``
landed in a copy that was discarded when the callback returned, and the matching
reset in ``_cleanup`` raised ``ValueError: Token was created in a different
Context``. Two silent failures followed: ``get_current_run()`` returned ``None``
inside tools for the entire run, and the stale sweep could abort an incoming
invocation before it was ever registered.

The fix is ``run_inline = True`` on the handler plus a defensive reset.
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import MagicMock

from dunetrace.context import _current_run, get_current_run
from dunetrace.integrations.langchain import DunetraceCallbackHandler

try:
    from langchain_core.callbacks.manager import AsyncCallbackManager, CallbackManager

    HAVE_LANGCHAIN = True
except Exception:  # pragma: no cover - exercised only where langchain is absent
    HAVE_LANGCHAIN = False


def _handler(agent_id: str = "ctx-test") -> DunetraceCallbackHandler:
    client = MagicMock()
    client.run.return_value.__enter__ = MagicMock()
    return DunetraceCallbackHandler(client, agent_id=agent_id, model="fake")


@unittest.skipUnless(HAVE_LANGCHAIN, "langchain_core not installed")
class TestRunInline(unittest.TestCase):
    """run_inline is what keeps set and reset in one context. Do not remove it."""

    def test_handler_declares_run_inline(self):
        self.assertIs(
            DunetraceCallbackHandler.run_inline,
            True,
            "run_inline=False sends this handler through copy_context() per callback "
            "on the async path, which silently breaks get_current_run() in tools",
        )

    def test_langchain_still_honours_run_inline(self):
        """Pins the upstream contract the fix depends on."""
        from langchain_core.callbacks.base import BaseCallbackHandler

        self.assertIs(BaseCallbackHandler.run_inline, False)
        import inspect

        from langchain_core.callbacks import manager

        src = inspect.getsource(manager._ahandle_event_for_handler)
        self.assertIn("run_inline", src)
        self.assertIn("copy_context", src)


@unittest.skipUnless(HAVE_LANGCHAIN, "langchain_core not installed")
class TestCurrentRunReachableFromCallbacks(unittest.TestCase):
    """get_current_run() must work on BOTH paths, not just sync."""

    def _drive_sync(self, handler):
        mgr = CallbackManager(handlers=[handler])
        rm = mgr.on_chain_start({"name": "root"}, {"input": "hi"})
        seen = get_current_run()
        rm.on_chain_end({"output": "bye"})
        return seen

    async def _drive_async(self, handler):
        mgr = AsyncCallbackManager(handlers=[handler])
        rm = await mgr.on_chain_start({"name": "root"}, {"input": "hi"})
        seen = get_current_run()
        await rm.on_chain_end({"output": "bye"})
        return seen

    def test_sync_path_exposes_the_run(self):
        self.assertIsNotNone(self._drive_sync(_handler()))

    def test_async_path_exposes_the_run(self):
        """The regression. Returned None before run_inline was set."""
        seen = asyncio.run(self._drive_async(_handler()))
        self.assertIsNotNone(
            seen,
            "get_current_run() is None inside an async LangChain run — the pattern "
            "documented in docs/integrate-langchain-agent.md raises AttributeError "
            "in the user's own tool",
        )

    def test_context_var_does_not_leak_after_the_run(self):
        for label, drive in (
            ("sync", lambda h: self._drive_sync(h)),
            ("async", lambda h: asyncio.run(self._drive_async(h))),
        ):
            with self.subTest(path=label):
                token = _current_run.set(None)
                try:
                    drive(_handler())
                    self.assertIsNone(get_current_run(), f"{label} path leaked the run")
                finally:
                    _current_run.reset(token)


@unittest.skipUnless(HAVE_LANGCHAIN, "langchain_core not installed")
class TestCleanupNeverBreaksTheCaller(unittest.TestCase):
    """A cross-context token must not cost an invocation or the flush barrier."""

    def test_reset_tolerates_a_foreign_token(self):
        handler = _handler()
        mgr = CallbackManager(handlers=[handler])
        rm = mgr.on_chain_start({"name": "root"}, {"input": "hi"})

        root = next(iter(handler._runs))
        # A token minted in another context is exactly what the stale sweep and
        # a concurrently-shared handler produce.
        import contextvars

        handler._runs[root].ctx_token = contextvars.copy_context().run(
            lambda: _current_run.set(None)
        )

        rm.on_chain_end({"output": "bye"})  # must not raise
        self.assertEqual(len(handler._runs), 0)

    def test_flush_runs_even_when_cleanup_fails(self):
        """The barrier is in a finally, so a short-lived process still ships."""
        handler = _handler()
        mgr = CallbackManager(handlers=[handler])
        rm = mgr.on_chain_start({"name": "root"}, {"input": "hi"})
        handler._cleanup = MagicMock(side_effect=RuntimeError("boom"))

        rm.on_chain_end({"output": "bye"})

        handler._client.flush.assert_called()

    def test_stale_sweep_failure_does_not_drop_the_new_invocation(self):
        """One abandoned run used to silently erase the next invocation."""
        handler = _handler()
        mgr = CallbackManager(handlers=[handler])
        rm_a = mgr.on_chain_start({"name": "a"}, {"input": "first"})
        del rm_a

        root_a = next(iter(handler._runs))
        handler._runs[root_a].start_time = 0.0  # older than _STALE_RUN_SECS
        handler._runs[root_a].ctx_token = "not-a-token"  # reset will raise

        rm_b = mgr.on_chain_start({"name": "b"}, {"input": "second"})

        self.assertEqual(
            len(handler._runs),
            1,
            "the sweep aborted on_chain_start, so invocation B was never registered",
        )
        rm_b.on_chain_end({"output": "done"})
        self.assertIsNotNone(handler.last_run_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
