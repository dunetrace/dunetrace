"""Endpoint-level tests for Phase 4.1's ElevenLabs integration config
(api_svc/routers/elevenlabs.py), its key-validation client
(api_svc/elevenlabs_client.py), and the encrypt-at-rest path. Calls route
functions directly (this codebase's established pattern, see
test_integrations.py), with mocked DB calls and a mocked ElevenLabs API. No
network, no DB.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException

from api_svc.elevenlabs_client import (
    ElevenLabsAuthError,
    ElevenLabsUnreachable,
    validate_api_key,
)
from api_svc.routers.elevenlabs import (
    ElevenLabsIntegrationRequest,
    get_elevenlabs_integration,
    remove_elevenlabs_integration,
    set_elevenlabs_integration,
)

_STATUS = {
    "poll_interval_secs": 300,
    "enabled": True,
    "last_polled_at": None,
    "last_success_at": None,
    "consecutive_failures": 0,
}


def _body(**overrides):
    fields = {"api_key": "sk-eleven-secret"}
    fields.update(overrides)
    return ElevenLabsIntegrationRequest(**fields)


# ── Request model validation ───────────────────────────────────────────────────


class TestRequestModel(unittest.TestCase):
    def test_defaults_to_conservative_five_minute_poll(self):
        self.assertEqual(_body().poll_interval_secs, 300)

    def test_rejects_sub_minute_poll_interval(self):
        with self.assertRaises(Exception):
            ElevenLabsIntegrationRequest(api_key="k", poll_interval_secs=5)

    def test_rejects_empty_api_key(self):
        with self.assertRaises(Exception):
            ElevenLabsIntegrationRequest(api_key="")


# ── validate_api_key (the on-save live check) ──────────────────────────────────


def _mock_httpx_client(response=None, raise_exc=None):
    """Build a patch target for httpx.AsyncClient whose .get() returns `response`
    or raises `raise_exc`, usable as an async context manager."""
    client = AsyncMock()
    if raise_exc is not None:
        client.get = AsyncMock(side_effect=raise_exc)
    else:
        client.get = AsyncMock(return_value=response)
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


class TestValidateApiKey(unittest.IsolatedAsyncioTestCase):
    async def test_accepts_2xx(self):
        resp = httpx.Response(200, json={"history": [], "has_more": False})
        with patch(
            "api_svc.elevenlabs_client.httpx.AsyncClient", return_value=_mock_httpx_client(resp)
        ):
            await validate_api_key("good-key")  # must not raise

    async def test_401_raises_auth_error(self):
        resp = httpx.Response(401, json={"detail": "unauthorized"})
        with patch(
            "api_svc.elevenlabs_client.httpx.AsyncClient", return_value=_mock_httpx_client(resp)
        ):
            with self.assertRaises(ElevenLabsAuthError):
                await validate_api_key("bad-key")

    async def test_422_raises_auth_error(self):
        resp = httpx.Response(422, json={"detail": "bad request"})
        with patch(
            "api_svc.elevenlabs_client.httpx.AsyncClient", return_value=_mock_httpx_client(resp)
        ):
            with self.assertRaises(ElevenLabsAuthError):
                await validate_api_key("weird-key")

    async def test_500_raises_unreachable(self):
        resp = httpx.Response(503, text="upstream down")
        with patch(
            "api_svc.elevenlabs_client.httpx.AsyncClient", return_value=_mock_httpx_client(resp)
        ):
            with self.assertRaises(ElevenLabsUnreachable):
                await validate_api_key("any-key")

    async def test_network_error_raises_unreachable(self):
        with patch(
            "api_svc.elevenlabs_client.httpx.AsyncClient",
            return_value=_mock_httpx_client(raise_exc=httpx.ConnectError("no route")),
        ):
            with self.assertRaises(ElevenLabsUnreachable):
                await validate_api_key("any-key")


# ── POST /v1/orgs/integrations/elevenlabs ──────────────────────────────────────


class TestSetElevenLabsIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_validates_then_encrypts_then_stores(self):
        with (
            patch(
                "api_svc.routers.elevenlabs.encrypt_credentials",
                return_value="encrypted-token",
            ) as encrypt_mock,
            patch("api_svc.routers.elevenlabs.validate_api_key", AsyncMock()) as validate_mock,
            patch(
                "api_svc.routers.elevenlabs.upsert_elevenlabs_integration", AsyncMock()
            ) as upsert_mock,
            patch(
                "api_svc.routers.elevenlabs.get_elevenlabs_integration_status",
                AsyncMock(return_value=_STATUS),
            ),
        ):
            result = await set_elevenlabs_integration(_body(), org_id="org-1")

        encrypt_mock.assert_called_once_with({"api_key": "sk-eleven-secret"})
        validate_mock.assert_awaited_once_with("sk-eleven-secret")
        upsert_mock.assert_awaited_once_with("org-1", "encrypted-token", 300)
        self.assertTrue(result.configured)

    async def test_invalid_key_returns_400_and_does_not_store(self):
        with (
            patch("api_svc.routers.elevenlabs.encrypt_credentials", return_value="tok"),
            patch(
                "api_svc.routers.elevenlabs.validate_api_key",
                AsyncMock(side_effect=ElevenLabsAuthError("rejected")),
            ),
            patch(
                "api_svc.routers.elevenlabs.upsert_elevenlabs_integration", AsyncMock()
            ) as upsert_mock,
        ):
            with self.assertRaises(HTTPException) as ctx:
                await set_elevenlabs_integration(_body(), org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 400)
        upsert_mock.assert_not_awaited()  # a bad key must never be persisted

    async def test_unreachable_returns_502_and_does_not_store(self):
        with (
            patch("api_svc.routers.elevenlabs.encrypt_credentials", return_value="tok"),
            patch(
                "api_svc.routers.elevenlabs.validate_api_key",
                AsyncMock(side_effect=ElevenLabsUnreachable("no route")),
            ),
            patch(
                "api_svc.routers.elevenlabs.upsert_elevenlabs_integration", AsyncMock()
            ) as upsert_mock,
        ):
            with self.assertRaises(HTTPException) as ctx:
                await set_elevenlabs_integration(_body(), org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 502)
        upsert_mock.assert_not_awaited()

    async def test_missing_master_key_returns_503_before_calling_elevenlabs(self):
        with (
            patch(
                "api_svc.routers.elevenlabs.encrypt_credentials",
                side_effect=ValueError("DUNETRACE_MASTER_KEY is not configured"),
            ),
            patch("api_svc.routers.elevenlabs.validate_api_key", AsyncMock()) as validate_mock,
        ):
            with self.assertRaises(HTTPException) as ctx:
                await set_elevenlabs_integration(_body(), org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 503)
        validate_mock.assert_not_awaited()  # no outbound call when server misconfigured

    async def test_response_never_contains_the_credential(self):
        with (
            patch("api_svc.routers.elevenlabs.encrypt_credentials", return_value="tok"),
            patch("api_svc.routers.elevenlabs.validate_api_key", AsyncMock()),
            patch("api_svc.routers.elevenlabs.upsert_elevenlabs_integration", AsyncMock()),
            patch(
                "api_svc.routers.elevenlabs.get_elevenlabs_integration_status",
                AsyncMock(return_value=_STATUS),
            ),
        ):
            result = await set_elevenlabs_integration(_body(), org_id="org-1")
        dumped = result.model_dump()
        self.assertNotIn("api_key", dumped)
        self.assertNotIn("sk-eleven-secret", str(dumped.values()))


class TestGetElevenLabsIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_not_configured_returns_configured_false(self):
        with patch(
            "api_svc.routers.elevenlabs.get_elevenlabs_integration_status",
            AsyncMock(return_value=None),
        ):
            result = await get_elevenlabs_integration(org_id="org-1")
        self.assertFalse(result.configured)

    async def test_configured_reports_health_fields(self):
        with patch(
            "api_svc.routers.elevenlabs.get_elevenlabs_integration_status",
            AsyncMock(
                return_value={**_STATUS, "poll_interval_secs": 600, "consecutive_failures": 3}
            ),
        ):
            result = await get_elevenlabs_integration(org_id="org-1")
        self.assertTrue(result.configured)
        self.assertEqual(result.poll_interval_secs, 600)
        self.assertEqual(result.consecutive_failures, 3)


class TestRemoveElevenLabsIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_not_found_returns_404(self):
        with patch(
            "api_svc.routers.elevenlabs.delete_elevenlabs_integration",
            AsyncMock(return_value=False),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await remove_elevenlabs_integration(org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_found_deletes_without_raising(self):
        with patch(
            "api_svc.routers.elevenlabs.delete_elevenlabs_integration",
            AsyncMock(return_value=True),
        ):
            await remove_elevenlabs_integration(org_id="org-1")  # must not raise


if __name__ == "__main__":
    unittest.main()
