"""
Tests for Phase 1.0's SDK-side pack activation convenience calls
(enable_pack/disable_pack/enabled_packs on Dunetrace). These hit the
Customer API (api_url, default http://localhost:8002 or DUNETRACE_API_URL)
— a different service from the ingest endpoint every other SDK call uses.
No network — urllib.request.urlopen is mocked throughout.

Run: python -m unittest tests.test_pack_activation -v
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from dunetrace.client import USER_AGENT, Dunetrace


def _make_client(**kwargs) -> Dunetrace:
    defaults = dict(api_key="dt_test_key", api_url="http://localhost:8002", debug=False)
    defaults.update(kwargs)
    return Dunetrace(**defaults)


class TestEnablePack(unittest.TestCase):
    def test_sends_post_to_the_correct_url(self):
        client = _make_client()
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value.status = 201
            client.enable_pack("voice")

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.full_url, "http://localhost:8002/v1/orgs/packs/voice")
        self.assertEqual(req.get_method(), "POST")

    def test_sends_user_agent_and_auth_headers(self):
        client = _make_client()
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value.status = 201
            client.enable_pack("voice")

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.get_header("User-agent"), USER_AGENT)
        self.assertEqual(req.get_header("Authorization"), "Bearer dt_test_key")

    def test_raises_when_api_url_not_configured(self):
        client = _make_client(api_url="")
        with self.assertRaises(RuntimeError):
            client.enable_pack("voice")

    def test_raises_when_api_key_not_configured(self):
        client = _make_client(api_key="")
        with self.assertRaises(RuntimeError):
            client.enable_pack("voice")

    def test_pack_name_is_url_encoded(self):
        client = _make_client()
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value.status = 201
            client.enable_pack("a pack/with slash")

        req = mock_urlopen.call_args[0][0]
        self.assertNotIn(" ", req.full_url)
        self.assertNotIn("/with slash", req.full_url)


class TestDisablePack(unittest.TestCase):
    def test_sends_delete_to_the_correct_url(self):
        client = _make_client()
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value.status = 204
            client.disable_pack("voice")

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.full_url, "http://localhost:8002/v1/orgs/packs/voice")
        self.assertEqual(req.get_method(), "DELETE")


class TestEnabledPacks(unittest.TestCase):
    def test_returns_pack_names_from_response(self):
        client = _make_client()
        response = MagicMock()
        response.read.return_value = (
            b'[{"pack_name": "voice", "enabled_at": "2026-01-01T00:00:00", "enabled_by": null}]'
        )
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = response
            result = client.enabled_packs()
        self.assertEqual(result, ["voice"])

    def test_empty_list_when_no_packs_activated(self):
        client = _make_client()
        response = MagicMock()
        response.read.return_value = b"[]"
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = response
            result = client.enabled_packs()
        self.assertEqual(result, [])

    def test_sends_user_agent_header(self):
        client = _make_client()
        response = MagicMock()
        response.read.return_value = b"[]"
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = response
            client.enabled_packs()

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.get_header("User-agent"), USER_AGENT)

    def test_raises_when_not_configured(self):
        client = _make_client(api_key="")
        with self.assertRaises(RuntimeError):
            client.enabled_packs()


class TestApiUrlConfiguration(unittest.TestCase):
    def test_defaults_to_localhost_8002(self):
        client = Dunetrace(api_key="k")
        self.assertEqual(client._api_url, "http://localhost:8002")

    def test_env_var_used_when_no_explicit_arg(self):
        with patch.dict("os.environ", {"DUNETRACE_API_URL": "https://api.example.com"}):
            client = Dunetrace(api_key="k")
        self.assertEqual(client._api_url, "https://api.example.com")

    def test_explicit_arg_wins_over_env_var(self):
        with patch.dict("os.environ", {"DUNETRACE_API_URL": "https://api.example.com"}):
            client = Dunetrace(api_key="k", api_url="https://override.example.com")
        self.assertEqual(client._api_url, "https://override.example.com")

    def test_trailing_slash_stripped(self):
        client = Dunetrace(api_key="k", api_url="http://localhost:8002/")
        self.assertEqual(client._api_url, "http://localhost:8002")


if __name__ == "__main__":
    unittest.main(verbosity=2)
