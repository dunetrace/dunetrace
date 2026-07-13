"""
Endpoint-level tests for Phase 2.1's Langfuse integration config, Phase 2.2's
LangSmith config, and Phase 2.3's Braintrust config (all in
api_svc/routers/integrations.py) plus the encryption helper
(api_svc/crypto.py). Calls route functions directly (this codebase's
established pattern), mocked DB calls. No network, no DB.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from api_svc.crypto import encrypt_credentials
from api_svc.routers.integrations import (
    BraintrustIntegrationRequest,
    LangfuseIntegrationRequest,
    LangSmithIntegrationRequest,
    get_braintrust_integration,
    get_langfuse_integration,
    get_langsmith_integration,
    remove_braintrust_integration,
    remove_langfuse_integration,
    remove_langsmith_integration,
    set_braintrust_integration,
    set_langfuse_integration,
    set_langsmith_integration,
)


class TestEncryptCredentials(unittest.TestCase):
    def test_raises_without_master_key(self):
        with patch("api_svc.crypto.settings") as mock_settings:
            mock_settings.MASTER_KEY = ""
            with self.assertRaises(ValueError):
                encrypt_credentials({"public_key": "pk", "secret_key": "sk"})

    def test_produces_a_token_not_the_raw_secret(self):
        from cryptography.fernet import Fernet

        key = Fernet.generate_key().decode()
        with patch("api_svc.crypto.settings") as mock_settings:
            mock_settings.MASTER_KEY = key
            token = encrypt_credentials({"public_key": "pk", "secret_key": "sk-super-secret"})
        self.assertNotIn("sk-super-secret", token)

    def test_roundtrips_with_the_same_key(self):
        import json

        from cryptography.fernet import Fernet

        key = Fernet.generate_key().decode()
        with patch("api_svc.crypto.settings") as mock_settings:
            mock_settings.MASTER_KEY = key
            token = encrypt_credentials({"public_key": "pk", "secret_key": "sk"})

        decrypted = json.loads(Fernet(key.encode()).decrypt(token.encode()).decode())
        self.assertEqual(decrypted, {"public_key": "pk", "secret_key": "sk"})


def _body(**overrides):
    fields = {"endpoint_url": "https://cloud.langfuse.com", "public_key": "pk", "secret_key": "sk"}
    fields.update(overrides)
    return LangfuseIntegrationRequest(**fields)


class TestSetLangfuseIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_encrypts_before_storing(self):
        with (
            patch(
                "api_svc.routers.integrations.encrypt_credentials",
                return_value="encrypted-token",
            ) as encrypt_mock,
            patch(
                "api_svc.routers.integrations.upsert_external_integration", AsyncMock()
            ) as upsert_mock,
            patch(
                "api_svc.routers.integrations.get_external_integration_status",
                AsyncMock(
                    return_value={
                        "endpoint_url": "https://cloud.langfuse.com",
                        "poll_interval_secs": 60,
                        "enabled": True,
                        "last_polled_at": None,
                        "last_success_at": None,
                        "consecutive_failures": 0,
                    }
                ),
            ),
        ):
            result = await set_langfuse_integration(_body(), org_id="org-1")

        encrypt_mock.assert_called_once_with({"public_key": "pk", "secret_key": "sk"})
        upsert_mock.assert_awaited_once_with(
            "org-1", "langfuse", "https://cloud.langfuse.com", "encrypted-token", 60
        )
        self.assertTrue(result.configured)

    async def test_missing_master_key_returns_503(self):
        with patch(
            "api_svc.routers.integrations.encrypt_credentials",
            side_effect=ValueError("DUNETRACE_MASTER_KEY is not configured"),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await set_langfuse_integration(_body(), org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 503)

    async def test_response_never_contains_the_credential(self):
        with (
            patch("api_svc.routers.integrations.encrypt_credentials", return_value="tok"),
            patch("api_svc.routers.integrations.upsert_external_integration", AsyncMock()),
            patch(
                "api_svc.routers.integrations.get_external_integration_status",
                AsyncMock(
                    return_value={
                        "endpoint_url": "https://cloud.langfuse.com",
                        "poll_interval_secs": 60,
                        "enabled": True,
                        "last_polled_at": None,
                        "last_success_at": None,
                        "consecutive_failures": 0,
                    }
                ),
            ),
        ):
            result = await set_langfuse_integration(_body(), org_id="org-1")

        dumped = result.model_dump()
        self.assertNotIn("secret_key", dumped)
        self.assertNotIn("public_key", dumped)
        self.assertNotIn("sk", str(dumped.values()))


class TestGetLangfuseIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_not_configured_returns_configured_false(self):
        with patch(
            "api_svc.routers.integrations.get_external_integration_status",
            AsyncMock(return_value=None),
        ):
            result = await get_langfuse_integration(org_id="org-1")
        self.assertFalse(result.configured)

    async def test_configured_reports_health_fields(self):
        with patch(
            "api_svc.routers.integrations.get_external_integration_status",
            AsyncMock(
                return_value={
                    "endpoint_url": "https://cloud.langfuse.com",
                    "poll_interval_secs": 120,
                    "enabled": True,
                    "last_polled_at": None,
                    "last_success_at": None,
                    "consecutive_failures": 4,
                }
            ),
        ):
            result = await get_langfuse_integration(org_id="org-1")
        self.assertTrue(result.configured)
        self.assertEqual(result.poll_interval_secs, 120)
        self.assertEqual(result.consecutive_failures, 4)


class TestRemoveLangfuseIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_not_found_returns_404(self):
        with patch(
            "api_svc.routers.integrations.delete_external_integration",
            AsyncMock(return_value=False),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await remove_langfuse_integration(org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_found_deletes_without_raising(self):
        with patch(
            "api_svc.routers.integrations.delete_external_integration",
            AsyncMock(return_value=True),
        ):
            await remove_langfuse_integration(org_id="org-1")  # must not raise


def _langsmith_body(**overrides):
    fields = {
        "endpoint_url": "https://api.smith.langchain.com",
        "api_key": "ls-key",
        "project_name": "Dunetrace",
    }
    fields.update(overrides)
    return LangSmithIntegrationRequest(**fields)


class TestSetLangsmithIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_encrypts_before_storing(self):
        with (
            patch(
                "api_svc.routers.integrations.encrypt_credentials",
                return_value="encrypted-token",
            ) as encrypt_mock,
            patch(
                "api_svc.routers.integrations.upsert_external_integration", AsyncMock()
            ) as upsert_mock,
            patch(
                "api_svc.routers.integrations.get_external_integration_status",
                AsyncMock(
                    return_value={
                        "endpoint_url": "https://api.smith.langchain.com",
                        "poll_interval_secs": 60,
                        "enabled": True,
                        "last_polled_at": None,
                        "last_success_at": None,
                        "consecutive_failures": 0,
                    }
                ),
            ),
        ):
            result = await set_langsmith_integration(_langsmith_body(), org_id="org-1")

        encrypt_mock.assert_called_once_with({"api_key": "ls-key", "project_name": "Dunetrace"})
        upsert_mock.assert_awaited_once_with(
            "org-1", "langsmith", "https://api.smith.langchain.com", "encrypted-token", 60
        )
        self.assertTrue(result.configured)

    async def test_missing_master_key_returns_503(self):
        with patch(
            "api_svc.routers.integrations.encrypt_credentials",
            side_effect=ValueError("DUNETRACE_MASTER_KEY is not configured"),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await set_langsmith_integration(_langsmith_body(), org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 503)


class TestGetLangsmithIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_not_configured_returns_configured_false(self):
        with patch(
            "api_svc.routers.integrations.get_external_integration_status",
            AsyncMock(return_value=None),
        ):
            result = await get_langsmith_integration(org_id="org-1")
        self.assertFalse(result.configured)


class TestRemoveLangsmithIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_not_found_returns_404(self):
        with patch(
            "api_svc.routers.integrations.delete_external_integration",
            AsyncMock(return_value=False),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await remove_langsmith_integration(org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_found_deletes_without_raising(self):
        with patch(
            "api_svc.routers.integrations.delete_external_integration",
            AsyncMock(return_value=True),
        ):
            await remove_langsmith_integration(org_id="org-1")  # must not raise


def _braintrust_body(**overrides):
    fields = {
        "endpoint_url": "https://api-eu.braintrust.dev",
        "api_key": "bt-key",
        "project_id": "proj-1",
    }
    fields.update(overrides)
    return BraintrustIntegrationRequest(**fields)


class TestSetBraintrustIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_encrypts_before_storing(self):
        with (
            patch(
                "api_svc.routers.integrations.encrypt_credentials",
                return_value="encrypted-token",
            ) as encrypt_mock,
            patch(
                "api_svc.routers.integrations.upsert_external_integration", AsyncMock()
            ) as upsert_mock,
            patch(
                "api_svc.routers.integrations.get_external_integration_status",
                AsyncMock(
                    return_value={
                        "endpoint_url": "https://api-eu.braintrust.dev",
                        "poll_interval_secs": 60,
                        "enabled": True,
                        "last_polled_at": None,
                        "last_success_at": None,
                        "consecutive_failures": 0,
                    }
                ),
            ),
        ):
            result = await set_braintrust_integration(_braintrust_body(), org_id="org-1")

        encrypt_mock.assert_called_once_with({"api_key": "bt-key", "project_id": "proj-1"})
        upsert_mock.assert_awaited_once_with(
            "org-1", "braintrust", "https://api-eu.braintrust.dev", "encrypted-token", 60
        )
        self.assertTrue(result.configured)

    async def test_missing_master_key_returns_503(self):
        with patch(
            "api_svc.routers.integrations.encrypt_credentials",
            side_effect=ValueError("DUNETRACE_MASTER_KEY is not configured"),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await set_braintrust_integration(_braintrust_body(), org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 503)


class TestGetBraintrustIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_not_configured_returns_configured_false(self):
        with patch(
            "api_svc.routers.integrations.get_external_integration_status",
            AsyncMock(return_value=None),
        ):
            result = await get_braintrust_integration(org_id="org-1")
        self.assertFalse(result.configured)


class TestRemoveBraintrustIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_not_found_returns_404(self):
        with patch(
            "api_svc.routers.integrations.delete_external_integration",
            AsyncMock(return_value=False),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await remove_braintrust_integration(org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_found_deletes_without_raising(self):
        with patch(
            "api_svc.routers.integrations.delete_external_integration",
            AsyncMock(return_value=True),
        ):
            await remove_braintrust_integration(org_id="org-1")  # must not raise


if __name__ == "__main__":
    unittest.main()
