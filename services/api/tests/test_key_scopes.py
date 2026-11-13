"""
An agent's own credential must not be able to grant the approval it is
blocked on.

Approval exists to gate "sending a customer email, deleting data, wiring
money". The agent process being gated sends its org API key on every call, so
authorising the decision endpoint with the same ambient key meant the thing
being gated could open its own gate.

Run:
    PYTHONPATH=packages/schemas-py:packages/sdk-py:services/explainer:services/api \
      python -m pytest services/api/tests/test_key_scopes.py -v
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from api_svc.auth import require_scope
from dunetrace_schemas.scopes import ADMIN, APPROVE, DEFAULT_SCOPES, INGEST, has_scope, normalise


class TestScopeSemantics(unittest.TestCase):
    def test_agent_key_cannot_approve(self):
        self.assertFalse(has_scope([INGEST], APPROVE))

    def test_admin_implies_everything(self):
        self.assertTrue(has_scope([ADMIN], APPROVE))
        self.assertTrue(has_scope([ADMIN], INGEST))

    def test_missing_scopes_fail_closed_to_ingest_only(self):
        """A key predating scopes is ingest-only, not unrestricted."""
        self.assertTrue(has_scope(None, INGEST))
        self.assertFalse(has_scope(None, APPROVE))

    def test_default_is_ingest_only(self):
        self.assertEqual(DEFAULT_SCOPES, (INGEST,))
        self.assertEqual(normalise(None), (INGEST,))

    def test_unknown_scopes_are_dropped_not_trusted(self):
        self.assertEqual(normalise(["wildcard", "approve"]), (APPROVE,))

    def test_normalise_is_case_insensitive(self):
        self.assertEqual(normalise(["ADMIN"]), (ADMIN,))


class TestRequireScope(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_a_key_without_the_scope(self):
        dependency = require_scope(APPROVE)
        with self.assertRaises(HTTPException) as ctx:
            await dependency(resolved=("org-1", (INGEST,)))
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("approve", str(ctx.exception.detail))

    async def test_accepts_a_key_with_the_scope(self):
        dependency = require_scope(APPROVE)
        self.assertEqual(await dependency(resolved=("org-1", (APPROVE,))), "org-1")

    async def test_accepts_admin(self):
        dependency = require_scope(APPROVE)
        self.assertEqual(await dependency(resolved=("org-1", (ADMIN,))), "org-1")


class TestApprovalDecisionIsScoped(unittest.IsolatedAsyncioTestCase):
    async def test_endpoint_depends_on_the_approve_scope(self):
        """Guards the wiring itself: swapping this back to require_org would
        silently restore the inverted trust boundary."""
        import inspect

        from api_svc.routers import approvals

        source = inspect.getsource(approvals.post_approval_decision)
        self.assertIn('require_scope("approve")', source)
        self.assertNotIn("Depends(require_org)", source)


class TestIssuedKeysAreIngestOnlyByDefault(unittest.IsolatedAsyncioTestCase):
    async def test_create_key_defaults_to_ingest(self):
        from api_svc.db import queries

        captured = {}

        class _Conn:
            async def execute(self, *a):
                return None

            async def fetchrow(self, sql, *args):
                if "INSERT INTO api_keys" in sql:
                    captured["scopes"] = args[-1]
                import datetime

                return {"id": 1, "created_at": datetime.datetime.now(datetime.timezone.utc)}

            def transaction(self):
                class _Tx:
                    async def __aenter__(s):
                        return None

                    async def __aexit__(s, *e):
                        return False

                return _Tx()

        class _Pool:
            def acquire(self):
                class _Ctx:
                    async def __aenter__(s):
                        return _Conn()

                    async def __aexit__(s, *e):
                        return False

                return _Ctx()

        with patch.object(queries, "_pool", _Pool()):
            result = await queries.create_api_key("org-1")

        self.assertEqual(captured["scopes"], [INGEST])
        self.assertEqual(result["scopes"], [INGEST])
        # The plaintext is returned to the caller once and never stored.
        self.assertTrue(result["key"].startswith("dt_"))


if __name__ == "__main__":
    unittest.main()
