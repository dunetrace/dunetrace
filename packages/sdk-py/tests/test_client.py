"""Tests for DunetraceClient — no network required."""
import json
import threading
import unittest
import urllib.request
from unittest.mock import MagicMock, patch

from dunetrace.client import DunetraceClient
from dunetrace.models import EventType


def _make_client(**kwargs) -> DunetraceClient:
    defaults = dict(api_key="dt_test", debug=False)
    defaults.update(kwargs)
    return DunetraceClient(**defaults)


class TestDunetraceClientRun(unittest.TestCase):

    def test_run_emits_started_and_completed(self):
        emitted = []
        client = _make_client()
        client._ship = lambda batch: emitted.extend(batch)

        with client.run("hello", model="gpt-4o", tools=["search"]):
            pass

        client.shutdown(timeout=2)
        types = [e.event_type for e in emitted]
        self.assertIn(EventType.RUN_STARTED, types)
        self.assertIn(EventType.RUN_COMPLETED, types)

    def test_run_emits_errored_on_exception(self):
        emitted = []
        client = _make_client()
        client._ship = lambda batch: emitted.extend(batch)

        try:
            with client.run("hello"):
                raise ValueError("boom")
        except ValueError:
            pass

        client.shutdown(timeout=2)
        types = [e.event_type for e in emitted]
        self.assertIn(EventType.RUN_ERRORED, types)
        self.assertNotIn(EventType.RUN_COMPLETED, types)

    def test_run_context_helpers(self):
        emitted = []
        client = _make_client()
        client._ship = lambda batch: emitted.extend(batch)

        with client.run("test input", model="gpt-4o", tools=["calc"]) as run:
            run.llm_called("gpt-4o", prompt_tokens=100)
            run.llm_responded(finish_reason="tool_calls", output_length=30)
            run.tool_called("calc", {"expr": "1+1"})
            run.tool_responded("calc", success=True, output_length=1)
            run.final_answer()

        client.shutdown(timeout=2)
        types = [e.event_type for e in emitted]
        self.assertIn(EventType.LLM_CALLED, types)
        self.assertIn(EventType.TOOL_CALLED, types)
        self.assertIn(EventType.TOOL_RESPONDED, types)

    def test_no_raw_content_in_events(self):
        """Verify that no payload field ever contains the raw user input."""
        secret = "my secret prompt"
        emitted = []
        client = _make_client()
        client._ship = lambda batch: emitted.extend(batch)

        with client.run(secret, model="gpt-4o"):
            pass

        client.shutdown(timeout=2)
        for event in emitted:
            payload_str = json.dumps(event.payload)
            self.assertNotIn(secret, payload_str)

    def test_shutdown_flushes_remaining(self):
        # flush_interval_ms=500 — won't auto-flush within test, but short enough
        # that the drain thread wakes and exits well within shutdown(timeout=3).
        client = _make_client(flush_interval_ms=500)

        with client.run("hello"):
            pass

        client.shutdown(timeout=3)
        self.assertFalse(client._drain_thread.is_alive())


class TestDunetraceClientShip(unittest.TestCase):

    def test_ship_failure_does_not_raise(self):
        client = _make_client()

        from dunetrace.models import AgentEvent, EventType
        dummy_event = AgentEvent(
            event_type=EventType.RUN_STARTED,
            run_id="r1",
            agent_id="a1",
            agent_version="v1",
            step_index=0,
        )
        # Patch urlopen to raise — should not propagate
        with patch("urllib.request.urlopen", side_effect=OSError("network down")):
            client._ship([dummy_event])  # must not raise


if __name__ == "__main__":
    unittest.main()
