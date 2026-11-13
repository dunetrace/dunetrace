"""
An agent's own credential must not be able to mint keys, write policies,
rewire alerting or change org-wide detection.

Every SDK/agent key is `ingest`-only. Before the `admin` gate, that key could
call `POST /v1/keys` and issue itself an `admin` key, create a `stop` policy
that terminates any agent's runs, or point Slack alerts at an attacker's
webhook — so the `approve` scope on the approval decision endpoint was the
only scope check that meant anything.

These tests go through the real app (`create_app()`) in `AUTH_MODE=prod` with
the key store stubbed, so they prove the *wiring*: router-level `require_org`
plus route-level `require_scope("admin")` composing into a 403 for an ingest
key and a pass for an admin key, with one key lookup per request.

Run:
    PYTHONPATH=packages/schemas-py:packages/sdk-py:services/explainer:services/api \
      python -m pytest services/api/tests/test_scope_enforcement.py -v
"""

from __future__ import annotations

import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from api_svc.config import settings
from api_svc.main import create_app
from dunetrace_schemas.scopes import ADMIN, APPROVE, INGEST

_HEADERS = {"Authorization": "Bearer dt_live_test"}

_POLICY_BODY = {
    "name": "stop-loops",
    "agent_id": "*",
    "condition": {"trigger": "tool_call_count", "operator": "gt", "value": 20},
    "action": {"type": "stop"},
}

_POLICY_ROW = {
    "id": 1,
    "org_id": "org-1",
    "name": "stop-loops",
    "agent_id": "*",
    "condition": _POLICY_BODY["condition"],
    "action": _POLICY_BODY["action"],
    "priority": 100,
    "enabled": True,
}

_KEY_ROW = {
    "id": 7,
    "key": "dt_live_new",
    "key_prefix": "dt_live_ne…_new",
    "org_id": "org-1",
    "org_name": "org-1",
    "rate_limit_rpm": 600,
    "scopes": ["approve"],
    "created_at": "2026-01-01T00:00:00+00:00",
}

_CD_CONFIG = {
    "detector_name": "CUSTOM_MANY_TOOLS",
    "conditions": [{"metric": "tool_call_count", "operator": ">=", "threshold": 5}],
    "severity": "MEDIUM",
    "evidence_template": "x",
    "fix_template": "y",
    "requires_content": False,
}

#: Every mutating route this change gates. (method, path, json body or None).
_ADMIN_WRITES = [
    ("GET", "/v1/keys", None),
    ("POST", "/v1/keys", {}),
    ("DELETE", "/v1/keys/1", None),
    ("POST", "/v1/policies", _POLICY_BODY),
    ("PUT", "/v1/policies/1", {"enabled": False}),
    ("DELETE", "/v1/policies/1", None),
    ("PATCH", "/v1/policies/1/toggle", None),
    ("PATCH", "/v1/orgs/semantic-feedback", {"enabled": True, "auto_suppress": False}),
    ("POST", "/v1/orgs/integrations/slack", {"webhook_url": "https://hooks.slack.com/x"}),
    ("DELETE", "/v1/orgs/integrations/slack", None),
    (
        "POST",
        "/v1/orgs/integrations/linear",
        {"api_key": "k", "webhook_secret": "s", "team_id": "t"},
    ),
    ("DELETE", "/v1/orgs/integrations/linear", None),
    (
        "POST",
        "/v1/orgs/integrations/langfuse",
        {"endpoint_url": "https://lf", "public_key": "p", "secret_key": "s"},
    ),
    ("DELETE", "/v1/orgs/integrations/langfuse", None),
    (
        "POST",
        "/v1/orgs/integrations/langsmith",
        {"endpoint_url": "https://ls", "api_key": "k", "project_name": "p"},
    ),
    ("DELETE", "/v1/orgs/integrations/langsmith", None),
    (
        "POST",
        "/v1/orgs/integrations/braintrust",
        {"endpoint_url": "https://bt", "api_key": "k", "project_id": "p"},
    ),
    ("DELETE", "/v1/orgs/integrations/braintrust", None),
    ("POST", "/v1/orgs/integrations/github", {"repos": [{"repo": "o/r"}]}),
    ("DELETE", "/v1/orgs/integrations/github", None),
    ("POST", "/v1/orgs/integrations/elevenlabs", {"api_key": "k"}),
    ("DELETE", "/v1/orgs/integrations/elevenlabs", None),
    ("PUT", "/v1/orgs/otel-receiver/enabled", {"enabled": False}),
    ("POST", "/v1/orgs/packs/voice", None),
    ("DELETE", "/v1/orgs/packs/voice", None),
    ("POST", "/v1/agents/my-agent/source-config", {"repo": "o/r"}),
    ("DELETE", "/v1/agents/my-agent/source-config", None),
    ("POST", "/v1/custom-detectors", {"description": "d", "config": _CD_CONFIG}),
    ("PATCH", "/v1/custom-detectors/1", {"status": "active"}),
    ("DELETE", "/v1/custom-detectors/1", None),
]


class _ScopedClient(unittest.TestCase):
    """The real app in prod auth mode, with the key store stubbed to `scopes`."""

    scopes: tuple = (INGEST,)

    def setUp(self):
        self._stack = ExitStack()
        self._stack.enter_context(patch.object(settings, "AUTH_MODE", "prod"))
        self._stack.enter_context(patch.object(settings, "INTERNAL_TOKEN", ""))
        self.verify = AsyncMock(return_value=("org-1", self.scopes))
        self._stack.enter_context(patch("api_svc.auth.verify_api_key_with_scopes", self.verify))
        # No `with`: the lifespan (DB pool) must not run.
        self.client = TestClient(create_app())

    def tearDown(self):
        self._stack.close()

    def _call(self, method: str, path: str, body=None):
        return self.client.request(method, path, json=body, headers=_HEADERS)


class TestIngestKeyIsRefusedOnWrites(_ScopedClient):
    scopes = (INGEST,)

    def test_every_admin_write_is_403(self):
        for method, path, body in _ADMIN_WRITES:
            with self.subTest(route=f"{method} {path}"):
                r = self._call(method, path, body)
                self.assertEqual(r.status_code, 403, r.text)
                self.assertIn("'admin' scope", r.json()["detail"])

    def test_key_is_still_looked_up_not_bypassed(self):
        """403 came from the scope check, not from a missing/invalid key."""
        self._call("POST", "/v1/policies", _POLICY_BODY)
        self.verify.assert_awaited()

    def test_reads_stay_open_to_an_ingest_key(self):
        cases = [
            ("api_svc.routers.policies.list_policies", "/v1/policies", []),
            ("api_svc.routers.packs.list_org_enabled_packs", "/v1/orgs/packs", []),
            ("api_svc.routers.custom_detectors.list_custom_detectors", "/v1/custom-detectors", []),
            (
                "api_svc.routers.alert_integrations.get_org_alert_integration_status",
                "/v1/orgs/integrations/slack",
                None,
            ),
            (
                "api_svc.routers.orgs.get_organization_semantic_feedback",
                "/v1/orgs/semantic-feedback",
                {"enabled": False, "auto_suppress": False},
            ),
            (
                "api_svc.routers.otel_receiver.get_org_otel_ingestion_enabled",
                "/v1/orgs/otel-receiver/enabled",
                True,
            ),
        ]
        for target, path, value in cases:
            with self.subTest(path=path):
                with patch(target, AsyncMock(return_value=value)):
                    r = self.client.get(path, headers=_HEADERS)
                self.assertEqual(r.status_code, 200, r.text)

    def test_preview_endpoints_stay_open(self):
        """Previews persist nothing; the LLM translation is still ingest-callable."""
        with patch(
            "api_svc.routers.custom_detectors.translate_description",
            AsyncMock(return_value=_CD_CONFIG),
        ):
            r = self._call("POST", "/v1/custom-detectors/preview", {"description": "many tools"})
        self.assertEqual(r.status_code, 200, r.text)

    def test_missing_key_is_401_not_403(self):
        """The two failures stay distinguishable: no credential vs. wrong credential."""
        r = self.client.post("/v1/policies", json=_POLICY_BODY)
        self.assertEqual(r.status_code, 401)


class TestApproveKeyIsRefusedOnWrites(_ScopedClient):
    """`approve` is for deciding approvals — it implies nothing else."""

    scopes = (APPROVE,)

    def test_cannot_mint_keys(self):
        r = self._call("POST", "/v1/keys", {"scopes": ["approve"]})
        self.assertEqual(r.status_code, 403)

    def test_cannot_create_policy(self):
        r = self._call("POST", "/v1/policies", _POLICY_BODY)
        self.assertEqual(r.status_code, 403)


class TestAdminKeySucceeds(_ScopedClient):
    scopes = (ADMIN,)

    def test_mints_a_key(self):
        with patch("api_svc.db.queries.create_api_key", AsyncMock(return_value=_KEY_ROW)) as create:
            r = self._call("POST", "/v1/keys", {"scopes": ["approve"]})
        self.assertEqual(r.status_code, 201, r.text)
        self.assertEqual(create.call_args.kwargs["scopes"], ["approve"])
        self.assertEqual(create.call_args.kwargs["org_id"], "org-1")

    def test_mints_an_admin_key(self):
        """An admin can delegate everything it holds."""
        with patch("api_svc.db.queries.create_api_key", AsyncMock(return_value=_KEY_ROW)) as create:
            r = self._call("POST", "/v1/keys", {"scopes": ["admin"]})
        self.assertEqual(r.status_code, 201, r.text)
        self.assertEqual(create.call_args.kwargs["scopes"], ["admin"])

    def test_lists_and_revokes_keys(self):
        with patch("api_svc.db.queries.list_api_keys", AsyncMock(return_value=[])):
            self.assertEqual(self._call("GET", "/v1/keys").status_code, 200)
        with patch("api_svc.db.queries.revoke_api_key", AsyncMock(return_value=True)):
            self.assertEqual(self._call("DELETE", "/v1/keys/1").status_code, 204)

    def test_creates_a_policy(self):
        with (
            patch("api_svc.routers.policies.create_policy", AsyncMock(return_value=_POLICY_ROW)),
            patch("api_svc.routers.policies.log_policy_audit", AsyncMock()),
        ):
            r = self._call("POST", "/v1/policies", _POLICY_BODY)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["id"], 1)

    def test_configures_slack_alerts(self):
        with (
            patch("api_svc.routers.alert_integrations.encrypt_credentials", return_value="enc"),
            patch("api_svc.routers.alert_integrations.upsert_org_alert_integration", AsyncMock()),
            patch(
                "api_svc.routers.alert_integrations.get_org_alert_integration_status",
                AsyncMock(return_value={"enabled": True, "config": {"channel": ""}}),
            ),
        ):
            r = self._call(
                "POST", "/v1/orgs/integrations/slack", {"webhook_url": "https://hooks.slack.com/x"}
            )
        self.assertEqual(r.status_code, 201, r.text)

    def test_changes_org_settings(self):
        with patch(
            "api_svc.routers.orgs.update_organization_semantic_feedback", AsyncMock()
        ) as update:
            r = self._call(
                "PATCH", "/v1/orgs/semantic-feedback", {"enabled": True, "auto_suppress": False}
            )
        self.assertEqual(r.status_code, 200, r.text)
        update.assert_awaited_once_with("org-1", True, False)

    def test_activates_a_pack(self):
        with (
            patch("api_svc.routers.packs.pack_exists", AsyncMock(return_value=True)),
            patch("api_svc.routers.packs.activate_pack", AsyncMock()) as activate,
        ):
            r = self._call("POST", "/v1/orgs/packs/voice")
        self.assertEqual(r.status_code, 201, r.text)
        self.assertEqual(activate.call_args.args[:2], ("org-1", "voice"))

    def test_key_is_looked_up_once_per_request(self):
        """Router-level require_org + route-level require_scope("admin") is ONE
        api_keys lookup, not two — the resolution is memoised on request.state."""
        with (
            patch("api_svc.routers.policies.create_policy", AsyncMock(return_value=_POLICY_ROW)),
            patch("api_svc.routers.policies.log_policy_audit", AsyncMock()),
        ):
            self._call("POST", "/v1/policies", _POLICY_BODY)
        self.assertEqual(self.verify.await_count, 1)

    def test_memo_does_not_leak_across_requests(self):
        """The second request resolves its own key — it never inherits the first's admin."""
        self.verify.side_effect = [("org-1", (ADMIN,)), ("org-2", (INGEST,))]
        with (
            patch("api_svc.routers.policies.create_policy", AsyncMock(return_value=_POLICY_ROW)),
            patch("api_svc.routers.policies.log_policy_audit", AsyncMock()),
        ):
            first = self._call("POST", "/v1/policies", _POLICY_BODY)
            second = self._call("POST", "/v1/policies", _POLICY_BODY)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 403)


class TestTrustedGatewayScopes(unittest.TestCase):
    """The x-internal-token path: scopes come from x-scopes, default ingest, never admin."""

    def setUp(self):
        self._stack = ExitStack()
        self._stack.enter_context(patch.object(settings, "AUTH_MODE", "prod"))
        self._stack.enter_context(patch.object(settings, "INTERNAL_TOKEN", "gw-secret"))
        self.verify = AsyncMock(side_effect=AssertionError("trusted path must not hit api_keys"))
        self._stack.enter_context(patch("api_svc.auth.verify_api_key_with_scopes", self.verify))
        self.client = TestClient(create_app())

    def tearDown(self):
        self._stack.close()

    @staticmethod
    def _headers(**extra):
        return {"x-internal-token": "gw-secret", "x-org-id": "org-gw", **extra}

    def _create_policy(self, headers):
        with (
            patch("api_svc.routers.policies.create_policy", AsyncMock(return_value=_POLICY_ROW)),
            patch("api_svc.routers.policies.log_policy_audit", AsyncMock()),
        ):
            return self.client.post("/v1/policies", json=_POLICY_BODY, headers=headers)

    def test_no_x_scopes_is_ingest_only(self):
        self.assertEqual(self._create_policy(self._headers()).status_code, 403)

    def test_empty_x_scopes_is_ingest_only(self):
        self.assertEqual(self._create_policy(self._headers(**{"x-scopes": ""})).status_code, 403)

    def test_unknown_x_scopes_is_ingest_only(self):
        r = self._create_policy(self._headers(**{"x-scopes": "bogus,superuser"}))
        self.assertEqual(r.status_code, 403)

    def test_x_scopes_is_normalised(self):
        """Case-insensitive, unknown values dropped: 'ADMIN,bogus' is admin."""
        r = self._create_policy(self._headers(**{"x-scopes": "ADMIN,bogus"}))
        self.assertEqual(r.status_code, 200, r.text)

    def test_reads_need_no_x_scopes(self):
        with patch("api_svc.routers.policies.list_policies", AsyncMock(return_value=[])):
            r = self.client.get("/v1/policies", headers=self._headers())
        self.assertEqual(r.status_code, 200)

    def test_approve_gateway_caller_cannot_mint_admin(self):
        r = self.client.post(
            "/v1/keys", json={"scopes": ["admin"]}, headers=self._headers(**{"x-scopes": "approve"})
        )
        self.assertEqual(r.status_code, 403)

    def test_admin_gateway_caller_can_mint(self):
        with patch("api_svc.db.queries.create_api_key", AsyncMock(return_value=_KEY_ROW)):
            r = self.client.post(
                "/v1/keys",
                json={"scopes": ["approve"]},
                headers=self._headers(**{"x-scopes": "admin"}),
            )
        self.assertEqual(r.status_code, 201, r.text)


def _required_scopes(dependant) -> set:
    """Every scope named by a require_scope() dependency anywhere under `dependant`."""
    found = set()
    scope = getattr(dependant.call, "required_scope", None)
    if scope:
        found.add(scope)
    for sub in dependant.dependencies:
        found |= _required_scopes(sub)
    return found


class TestRouteTableHasNoUnscopedWrites(unittest.TestCase):
    """Walk the live route table rather than the source: a new POST/PUT/PATCH/DELETE
    added under a managed prefix without `require_scope("admin")` fails here,
    whatever it is called."""

    _PREFIXES = ("/v1/keys", "/v1/policies", "/v1/orgs", "/v1/custom-detectors")
    # Deliberately ingest-callable: they persist nothing.
    _ALLOWED_UNSCOPED = {
        ("POST", "/v1/orgs/integrations/linear/preview-teams"),
        ("POST", "/v1/custom-detectors/preview"),
    }

    def test_every_managed_write_route_requires_admin(self):
        app = create_app()
        checked = 0
        for route in app.routes:
            methods = getattr(route, "methods", None) or set()
            path = getattr(route, "path", "")
            managed = path.startswith(self._PREFIXES) or path.endswith("/source-config")
            if not managed:
                continue
            for method in sorted(methods & {"POST", "PUT", "PATCH", "DELETE"}):
                if (method, path) in self._ALLOWED_UNSCOPED:
                    continue
                checked += 1
                with self.subTest(route=f"{method} {path}"):
                    self.assertEqual(_required_scopes(route.dependant), {"admin"})
        self.assertEqual(checked, len(_ADMIN_WRITES) - 1)  # GET /v1/keys is in the list too


if __name__ == "__main__":
    unittest.main()
