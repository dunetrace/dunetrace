"""
Tests for DunetraceCallbackHandler.

These tests mock LangChain's callback interface so they run without
installing langchain.
"""
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Stub out langchain so we can import the handler without the real package
# ---------------------------------------------------------------------------
def _stub_langchain() -> None:
    if "langchain" in sys.modules:
        return

    lc = types.ModuleType("langchain")
    lc_cb = types.ModuleType("langchain.callbacks")
    lc_cb_base = types.ModuleType("langchain.callbacks.base")

    class BaseCallbackHandler:
        def __init__(self): pass

    lc_cb_base.BaseCallbackHandler = BaseCallbackHandler
    lc.callbacks = lc_cb
    lc_cb.base = lc_cb_base

    sys.modules["langchain"] = lc
    sys.modules["langchain.callbacks"] = lc_cb
    sys.modules["langchain.callbacks.base"] = lc_cb_base

    # Patch the availability flag so handler.__init__ doesn't raise
    import dunetrace.integrations.langchain as _lc_mod
    _lc_mod._LANGCHAIN_AVAILABLE = True


_stub_langchain()

from dunetrace.integrations.langchain import DunetraceCallbackHandler
from dunetrace.models import EventType


def _make_handler(tools=None):
    client = MagicMock()
    emitted = []
    client._emit.side_effect = emitted.append
    handler = DunetraceCallbackHandler(client, agent_id="test-agent", model="gpt-4o", tools=tools or ["search"])
    return handler, emitted


class TestDunetraceCallbackHandler(unittest.TestCase):

    def test_run_start_emits_run_started(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "hello"}, run_id="lc-1")
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].event_type, EventType.RUN_STARTED)

    def test_run_end_emits_run_completed(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "hello"}, run_id="lc-1")
        handler.on_chain_end({}, run_id="lc-1")
        types = [e.event_type for e in emitted]
        self.assertIn(EventType.RUN_COMPLETED, types)

    def test_chain_error_emits_run_errored(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "hello"}, run_id="lc-1")
        handler.on_chain_error(ValueError("oops"), run_id="lc-1")
        types = [e.event_type for e in emitted]
        self.assertIn(EventType.RUN_ERRORED, types)

    def test_sub_chain_start_is_ignored(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "root"}, run_id="lc-root")
        start_count = len(emitted)
        handler.on_chain_start({}, {"input": "sub"}, run_id="lc-sub")
        self.assertEqual(len(emitted), start_count)  # no extra event

    def test_handler_resets_after_completion(self):
        handler, emitted = _make_handler()
        handler.on_chain_start({}, {"input": "run1"}, run_id="lc-1")
        handler.on_chain_end({}, run_id="lc-1")
        # Start a second run
        handler.on_chain_start({}, {"input": "run2"}, run_id="lc-2")
        handler.on_chain_end({}, run_id="lc-2")
        started = [e for e in emitted if e.event_type == EventType.RUN_STARTED]
        self.assertEqual(len(started), 2)

    def test_no_raw_input_in_payload(self):
        handler, emitted = _make_handler()
        secret = "my secret query"
        handler.on_chain_start({}, {"input": secret}, run_id="lc-1")
        import json
        for event in emitted:
            self.assertNotIn(secret, json.dumps(event.payload))


if __name__ == "__main__":
    unittest.main()
