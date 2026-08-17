"""
Tests proving the SDK identifies itself with a real User-Agent on every
network call, instead of urllib's default "Python-urllib/x.y".

Confirmed against the live Dunetrace Cloud gateway: a POST to
https://app.dunetrace.com/v1/ingest with User-Agent left at urllib's default
gets blocked outright with HTTP 403 (Cloudflare error 1010, bot protection)
before the request ever reaches the app. The identical request with a normal
User-Agent (e.g. curl's) gets past Cloudflare fine.

No network required — urllib.request.urlopen is mocked throughout.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from dunetrace.client import USER_AGENT, DunetraceClient
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


class TestUserAgentConstant(unittest.TestCase):
    def test_user_agent_is_not_the_urllib_default(self):
        self.assertNotIn("python-urllib", USER_AGENT.lower())

    def test_user_agent_identifies_the_sdk(self):
        self.assertTrue(USER_AGENT.startswith("dunetrace-sdk-py/"))


class TestShipSendsUserAgent(unittest.TestCase):
    def test_ship_sends_user_agent_header(self):
        client = _make_client()
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value.status = 200
            client._ship([_dummy_event()])

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.get_header("User-agent"), USER_AGENT)


class TestFetchPoliciesSendsUserAgent(unittest.TestCase):
    def test_fetch_policies_sends_user_agent_header(self):
        client = _make_client()
        response = MagicMock()
        response.read.return_value = b'{"policies": []}'
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = response
            client._fetch_policies("agent-1")

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.get_header("User-agent"), USER_AGENT)


class TestShipDeploySendsUserAgent(unittest.TestCase):
    def test_ship_deploy_sends_user_agent_header(self):
        client = _make_client()
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value.status = 200
            client._ship_deploy("agent-1", "v1.0.0", {})

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.get_header("User-agent"), USER_AGENT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
