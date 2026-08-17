"""
Tests for the BatchingEmitter ABC and its four concrete implementations, plus
Dunetrace's wiring of a custom emitter and the endpoint-resolution fix.

No network required — urllib.request.urlopen is mocked throughout.
Run: python -m unittest tests.test_emitters -v
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from dunetrace.client import Dunetrace
from dunetrace.emitters import (
    USER_AGENT,
    BatchingEmitter,
    ConsoleBatchingEmitter,
    FileBatchingEmitter,
    HttpBatchingEmitter,
    NoopBatchingEmitter,
)
from dunetrace.models import AgentEvent, EventType


def _event(agent_id: str = "agent-1", run_id: str = "run-1") -> AgentEvent:
    return AgentEvent(
        event_type=EventType.RUN_STARTED,
        run_id=run_id,
        agent_id=agent_id,
        agent_version="v1",
        step_index=0,
        payload={"key": "value"},
    )


# ── BatchingEmitter ABC ─────────────────────────────────────────────────────────


class TestBatchingEmitterABC(unittest.TestCase):
    def test_cannot_instantiate_directly(self):
        with self.assertRaises(TypeError):
            BatchingEmitter()  # type: ignore[abstract]

    def test_subclass_without_ship_cannot_instantiate(self):
        class Incomplete(BatchingEmitter):
            pass

        with self.assertRaises(TypeError):
            Incomplete()  # type: ignore[abstract]


# ── HttpBatchingEmitter ──────────────────────────────────────────────────────────


class TestHttpBatchingEmitter(unittest.TestCase):
    def test_ship_success_returns_true(self):
        emitter = HttpBatchingEmitter("http://localhost:8001", api_key="dt_test")
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value.status = 200
            result = emitter.ship([_event()])
        self.assertTrue(result)

    def test_ship_failure_returns_false_not_raise(self):
        emitter = HttpBatchingEmitter("http://localhost:8001")
        with patch("urllib.request.urlopen", side_effect=OSError("network down")):
            result = emitter.ship([_event()])
        self.assertFalse(result)

    def test_ship_sends_expected_headers_and_body(self):
        emitter = HttpBatchingEmitter("http://localhost:8001", api_key="dt_live_abc")
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value.status = 200
            emitter.ship([_event(agent_id="agent-x")])

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.full_url, "http://localhost:8001/v1/ingest")
        self.assertEqual(req.get_header("User-agent"), USER_AGENT)
        self.assertEqual(req.get_header("Authorization"), "Bearer dt_live_abc")
        self.assertEqual(req.get_header("X-dunetrace-agent"), "agent-x")
        body = json.loads(req.data)
        self.assertEqual(body["api_key"], "dt_live_abc")
        self.assertEqual(body["agent_id"], "agent-x")
        self.assertEqual(len(body["events"]), 1)

    def test_endpoint_trailing_slash_normalised(self):
        emitter = HttpBatchingEmitter("http://localhost:8001/")
        self.assertEqual(emitter._ingest_url, "http://localhost:8001/v1/ingest")

    def test_no_api_key_sends_no_authorization_header(self):
        emitter = HttpBatchingEmitter("http://localhost:8001")
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value.status = 200
            emitter.ship([_event()])
        req = mock_urlopen.call_args[0][0]
        self.assertIsNone(req.get_header("Authorization"))


# ── NoopBatchingEmitter ──────────────────────────────────────────────────────────


class TestNoopBatchingEmitter(unittest.TestCase):
    def test_ship_always_returns_true(self):
        emitter = NoopBatchingEmitter()
        self.assertTrue(emitter.ship([_event(), _event()]))

    def test_ship_empty_batch_returns_true(self):
        self.assertTrue(NoopBatchingEmitter().ship([]))

    def test_ship_performs_no_network_call(self):
        emitter = NoopBatchingEmitter()
        with patch("urllib.request.urlopen") as mock_urlopen:
            emitter.ship([_event()])
        mock_urlopen.assert_not_called()


# ── ConsoleBatchingEmitter ───────────────────────────────────────────────────────


class TestConsoleBatchingEmitter(unittest.TestCase):
    def test_ship_prints_one_json_line_per_event(self):
        emitter = ConsoleBatchingEmitter()
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = emitter.ship([_event(run_id="r1"), _event(run_id="r2")])
        self.assertTrue(result)
        lines = [l for l in buf.getvalue().splitlines() if l]
        self.assertEqual(len(lines), 2)
        parsed = [json.loads(l) for l in lines]
        self.assertEqual(parsed[0]["run_id"], "r1")
        self.assertEqual(parsed[1]["run_id"], "r2")

    def test_ship_empty_batch_prints_nothing(self):
        emitter = ConsoleBatchingEmitter()
        buf = io.StringIO()
        with redirect_stdout(buf):
            emitter.ship([])
        self.assertEqual(buf.getvalue(), "")


# ── FileBatchingEmitter ──────────────────────────────────────────────────────────


class TestFileBatchingEmitter(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)

    def test_ship_appends_json_lines(self):
        emitter = FileBatchingEmitter(self.path)
        result = emitter.ship([_event(run_id="r1"), _event(run_id="r2")])
        self.assertTrue(result)
        with open(self.path) as f:
            lines = [l for l in f.read().splitlines() if l]
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["run_id"], "r1")

    def test_ship_appends_across_multiple_calls(self):
        emitter = FileBatchingEmitter(self.path)
        emitter.ship([_event(run_id="r1")])
        emitter.ship([_event(run_id="r2")])
        with open(self.path) as f:
            lines = [l for l in f.read().splitlines() if l]
        self.assertEqual(len(lines), 2)

    def test_ship_to_unwritable_path_returns_false_not_raise(self):
        emitter = FileBatchingEmitter("/nonexistent-dir-xyz/events.jsonl")
        result = emitter.ship([_event()])
        self.assertFalse(result)


# ── Dunetrace(emitter=...) wiring ────────────────────────────────────────────────


class TestDunetraceEmitterWiring(unittest.TestCase):
    def test_default_emitter_is_http(self):
        client = Dunetrace(api_key="dt_test")
        self.assertIsInstance(client._emitter, HttpBatchingEmitter)
        client.shutdown(timeout=1)

    def test_custom_emitter_is_used_instead_of_default(self):
        emitter = NoopBatchingEmitter()
        client = Dunetrace(emitter=emitter)
        self.assertIs(client._emitter, emitter)
        client.shutdown(timeout=1)

    def test_ship_delegates_to_custom_emitter(self):
        shipped = []

        class RecordingEmitter(BatchingEmitter):
            def ship(self, batch):
                shipped.extend(batch)
                return True

        client = Dunetrace(emitter=RecordingEmitter())
        result = client._ship([_event()])
        self.assertTrue(result)
        self.assertEqual(len(shipped), 1)
        client.shutdown(timeout=1)

    def test_noop_emitter_disables_http_shipping_end_to_end(self):
        """The documented way to disable HTTP shipping — no network call is ever made."""
        client = Dunetrace(emitter=NoopBatchingEmitter())
        with client.run("agent-1") as run:
            run.tool_called("search", {"q": "x"})
        with patch("urllib.request.urlopen") as mock_urlopen:
            client.flush(block=True, timeout=2)
        mock_urlopen.assert_not_called()
        client.shutdown(timeout=1)


# ── endpoint resolution fix (is not None, not `endpoint or ...`) ────────────────


class TestEndpointResolution(unittest.TestCase):
    def test_omitted_endpoint_defaults_to_localhost(self):
        client = Dunetrace(api_key="dt_test")
        self.assertEqual(client._emitter._ingest_url, "http://localhost:8001/v1/ingest")
        client.shutdown(timeout=1)

    def test_explicit_empty_string_endpoint_is_taken_literally(self):
        """Post-fix: endpoint="" is no longer silently replaced by the localhost
        default — `is not None`, not `endpoint or ...`. Disabling shipping is done
        via emitter=NoopBatchingEmitter(), not a magic endpoint value."""
        client = Dunetrace(endpoint="", api_key="dt_test")
        self.assertEqual(client._emitter._ingest_url, "/v1/ingest")
        client.shutdown(timeout=1)

    def test_explicit_endpoint_used_verbatim(self):
        client = Dunetrace(endpoint="http://example.com:9000", api_key="dt_test")
        self.assertEqual(client._emitter._ingest_url, "http://example.com:9000/v1/ingest")
        client.shutdown(timeout=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
