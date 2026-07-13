"""
Endpoint-level tests for Phase 1.0's pack activation API
(api_svc/routers/packs.py). Calls route functions directly (this
codebase's established pattern), mocked DB calls. No network, no DB.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from api_svc.routers.packs import (
    delete_deactivate_pack,
    get_all_packs,
    get_org_packs,
    post_activate_pack,
)


class TestGetAllPacks(unittest.IsolatedAsyncioTestCase):
    async def test_returns_catalog_with_no_org_context(self):
        with patch(
            "api_svc.routers.packs.list_all_packs",
            AsyncMock(
                return_value=[
                    {
                        "name": "voice",
                        "description": "Voice agent detectors",
                        "detector_names": ["TranscriptionConfidenceDropDetector"],
                    }
                ]
            ),
        ):
            result = await get_all_packs()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "voice")

    async def test_empty_catalog_returns_empty_list(self):
        with patch("api_svc.routers.packs.list_all_packs", AsyncMock(return_value=[])):
            result = await get_all_packs()
        self.assertEqual(result, [])


class TestGetOrgPacks(unittest.IsolatedAsyncioTestCase):
    async def test_returns_this_orgs_activated_packs(self):
        with patch(
            "api_svc.routers.packs.list_org_enabled_packs",
            AsyncMock(
                return_value=[
                    {
                        "pack_name": "voice",
                        "enabled_at": __import__("datetime").datetime(2026, 1, 1),
                        "enabled_by": None,
                    }
                ]
            ),
        ) as mock_fn:
            result = await get_org_packs(org_id="org-1")
        mock_fn.assert_awaited_once_with("org-1")
        self.assertEqual(result[0].pack_name, "voice")

    async def test_no_activated_packs_returns_empty_list(self):
        with patch("api_svc.routers.packs.list_org_enabled_packs", AsyncMock(return_value=[])):
            result = await get_org_packs(org_id="org-1")
        self.assertEqual(result, [])


class TestPostActivatePack(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_pack_returns_404(self):
        with patch("api_svc.routers.packs.pack_exists", AsyncMock(return_value=False)):
            with self.assertRaises(HTTPException) as ctx:
                await post_activate_pack("nonexistent", org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_known_pack_activates_and_never_stores_a_secret_in_enabled_by(self):
        with (
            patch("api_svc.routers.packs.pack_exists", AsyncMock(return_value=True)),
            patch("api_svc.routers.packs.activate_pack", AsyncMock()) as mock_activate,
        ):
            result = await post_activate_pack("voice", org_id="org-1")
        mock_activate.assert_awaited_once_with("org-1", "voice", enabled_by=None)
        self.assertEqual(result, {"pack_name": "voice", "enabled": True})

    async def test_activation_is_scoped_to_the_calling_orgs_id_only(self):
        """org_id passed to activate_pack must be exactly what require_org
        resolved — never something else derivable from the request, since
        that's the whole point of not taking org_id as a path param."""
        with (
            patch("api_svc.routers.packs.pack_exists", AsyncMock(return_value=True)),
            patch("api_svc.routers.packs.activate_pack", AsyncMock()) as mock_activate,
        ):
            await post_activate_pack("voice", org_id="org-A")
            await post_activate_pack("voice", org_id="org-B")
        calls = [c.args for c in mock_activate.await_args_list]
        self.assertEqual(calls, [("org-A", "voice"), ("org-B", "voice")])


class TestDeleteDeactivatePack(unittest.IsolatedAsyncioTestCase):
    async def test_not_activated_returns_404(self):
        with patch("api_svc.routers.packs.deactivate_pack", AsyncMock(return_value=False)):
            with self.assertRaises(HTTPException) as ctx:
                await delete_deactivate_pack("voice", org_id="org-1")
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_activated_pack_deactivates_without_raising(self):
        with patch("api_svc.routers.packs.deactivate_pack", AsyncMock(return_value=True)):
            await delete_deactivate_pack("voice", org_id="org-1")  # must not raise


if __name__ == "__main__":
    unittest.main()
