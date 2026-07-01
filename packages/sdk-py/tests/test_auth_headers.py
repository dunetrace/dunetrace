"""
Tests proving the SDK sends Authorization: Bearer <api_key> on every network
call — not just the api_key body/query-string field. The Dunetrace Cloud
gateway's tenancy middleware (dunetrace-cloud's app/middleware/tenancy.py)
only ever reads the Authorization header; before this fix, every cloud-backed
request was silently unauthenticated (401) regardless of key validity,
because the SDK only ever put api_key in the body/query string.

No network required — urllib.request.urlopen is mocked throughout.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from dunetrace.client import DunetraceClient
from dunetrace.models import AgentEvent, EventType


def _make_client(**kwargs) -> DunetraceClient:
    defaults = dict(api_key="dt_test_key", debug=False)
    defaults.update(kwargs)
    return DunetraceClient(**defaults)


def _dummy_event() -> AgentEvent:
    return AgentEvent(
        event_type=EventType.RUN_STARTED,
        run_id="r1",
        agent_id="a1",
        agent_version="v1",
        step_index=0,
    )


class TestAuthHeadersHelper(unittest.TestCase):
    def test_returns_bearer_header_when_api_key_set(self):
        client = _make_client(api_key="dt_live_abc123")
        self.assertEqual(client._auth_headers(), {"Authorization": "Bearer dt_live_abc123"})

    def test_empty_when_no_api_key(self):
        client = _make_client(api_key="")
        self.assertEqual(client._auth_headers(), {})


class TestShipSendsAuthHeader(unittest.TestCase):
    def test_ship_sends_authorization_header(self):
        client = _make_client(api_key="dt_live_abc123")
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value.status = 200
            client._ship([_dummy_event()])

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.get_header("Authorization"), "Bearer dt_live_abc123")

    def test_ship_still_sends_api_key_in_body_for_self_host_compat(self):
        """Self-hosted ingest_svc (no gateway) reads api_key from the body —
        must not regress even though the header is now the primary path."""
        client = _make_client(api_key="dt_live_abc123")
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value.status = 200
            client._ship([_dummy_event()])

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        self.assertEqual(body["api_key"], "dt_live_abc123")

    def test_ship_without_api_key_sends_no_authorization_header(self):
        client = _make_client(api_key="")
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value.status = 200
            client._ship([_dummy_event()])

        req = mock_urlopen.call_args[0][0]
        self.assertIsNone(req.get_header("Authorization"))


class TestFetchPoliciesSendsAuthHeader(unittest.TestCase):
    def test_fetch_policies_sends_authorization_header(self):
        client = _make_client(api_key="dt_live_abc123")
        response = MagicMock()
        response.read.return_value = b'{"policies": []}'
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = response
            client._fetch_policies("agent-1")

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.get_header("Authorization"), "Bearer dt_live_abc123")


class TestShipDeploySendsAuthHeader(unittest.TestCase):
    def test_ship_deploy_sends_authorization_header(self):
        client = _make_client(api_key="dt_live_abc123")
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value.status = 200
            client._ship_deploy("agent-1", "v1.0.0", {})

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.get_header("Authorization"), "Bearer dt_live_abc123")


if __name__ == "__main__":
    unittest.main(verbosity=2)
