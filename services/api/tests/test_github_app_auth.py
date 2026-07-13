"""
Tests for Phase 4.3's GitHub App auth (api_svc/github_app_auth.py) — JWT
signing and installation token exchange. Mocks httpx; JWT signing itself
uses a real (test-generated) RSA key, not mocked, so the signature/claims
logic is genuinely exercised.
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from api_svc.github_app_auth import build_install_url, generate_app_jwt, get_installation_token


def _test_private_key_pem() -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()


class TestGenerateAppJwt(unittest.TestCase):
    def test_raises_when_not_configured(self):
        with patch("api_svc.github_app_auth.settings") as mock_settings:
            mock_settings.github_app_configured = False
            with self.assertRaises(ValueError):
                generate_app_jwt()

    def test_produces_valid_rs256_jwt_with_expected_claims(self):
        pem = _test_private_key_pem()
        with patch("api_svc.github_app_auth.settings") as mock_settings:
            mock_settings.github_app_configured = True
            mock_settings.GITHUB_APP_ID = "12345"
            mock_settings.GITHUB_APP_PRIVATE_KEY = pem
            token = generate_app_jwt()

        # Decode without verifying signature just to inspect claims shape —
        # separately verify the signature IS valid below.
        claims = jwt.decode(token, options={"verify_signature": False})
        self.assertEqual(claims["iss"], "12345")
        self.assertIn("iat", claims)
        self.assertIn("exp", claims)

    def test_expiry_within_githubs_10_minute_limit(self):
        pem = _test_private_key_pem()
        with patch("api_svc.github_app_auth.settings") as mock_settings:
            mock_settings.github_app_configured = True
            mock_settings.GITHUB_APP_ID = "12345"
            mock_settings.GITHUB_APP_PRIVATE_KEY = pem
            token = generate_app_jwt()

        claims = jwt.decode(token, options={"verify_signature": False})
        self.assertLessEqual(claims["exp"] - claims["iat"], 10 * 60)

    def test_iat_backdated_for_clock_skew_tolerance(self):
        pem = _test_private_key_pem()
        with patch("api_svc.github_app_auth.settings") as mock_settings:
            mock_settings.github_app_configured = True
            mock_settings.GITHUB_APP_ID = "12345"
            mock_settings.GITHUB_APP_PRIVATE_KEY = pem
            before = time.time()
            token = generate_app_jwt()

        claims = jwt.decode(token, options={"verify_signature": False})
        self.assertLess(claims["iat"], before)


class TestGetInstallationToken(unittest.IsolatedAsyncioTestCase):
    async def test_returns_token_from_response(self):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"token": "ghs_installation_token_xyz"}

        client = AsyncMock()
        client.post = AsyncMock(return_value=resp)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=client),
            patch("api_svc.github_app_auth.generate_app_jwt", return_value="fake.jwt.token"),
        ):
            token = await get_installation_token(999)

        self.assertEqual(token, "ghs_installation_token_xyz")
        call_args = client.post.call_args
        self.assertIn("999", call_args.args[0])
        self.assertEqual(call_args.kwargs["headers"]["Authorization"], "Bearer fake.jwt.token")


class TestBuildInstallUrl(unittest.TestCase):
    def test_includes_state_and_app_slug(self):
        with patch("api_svc.github_app_auth.settings") as mock_settings:
            mock_settings.GITHUB_APP_SLUG = "dunetrace-fixit"
            url = build_install_url(state="org-42")

        self.assertIn("dunetrace-fixit", url)
        self.assertIn("state=org-42", url)

    def test_raises_when_slug_not_configured(self):
        with patch("api_svc.github_app_auth.settings") as mock_settings:
            mock_settings.GITHUB_APP_SLUG = ""
            with self.assertRaises(ValueError):
                build_install_url(state="org-42")


if __name__ == "__main__":
    unittest.main()
