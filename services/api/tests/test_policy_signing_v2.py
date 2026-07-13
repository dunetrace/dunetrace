"""
Policy HMAC signing — versioned canonical form (Phase 4).

The real cross-package contract: api_svc's _sign_policy produces a signature that
the SDK's _verify_policy_signature accepts. These tests exercise that round-trip
directly, plus the migration guarantee (existing v1-signed policies keep
verifying) and the downgrade-attack defense (a v2 policy can't be relabeled v1).

Run from services/api/ with:
  PYTHONPATH=../../packages/sdk-py:../explainer:. \
    python -m unittest discover -s tests -p "test_policy_signing_v2.py"
"""

import hashlib
import hmac
import json
import unittest

from api_svc.db.queries import _policy_canonical, _sign_policy, _sig_version_for
from dunetrace.policies import _verify_policy_signature

SECRET = "test-signing-secret-abc123"  # gitleaks:allow  (fake HMAC key for tests)

LEGACY_COND = {"trigger": "tool_call_count", "operator": "gt", "value": 5}
EXPR_COND = {
    "trigger": "before_tool_call",
    "operator": "eq",
    "value": "refund_customer",
    "match": {"args.amount": {"gt": 10000}},
}
ACTION = {"type": "stop"}


def _policy_dict(policy_id, cond, action, sig, sig_version, **over):
    d = {
        "id": policy_id,
        "agent_id": "billing",
        "name": "p",
        "condition": cond,
        "action": action,
        "enabled": True,
        "priority": 100,
        "signature": sig,
        "sig_version": sig_version,
    }
    d.update(over)
    return d


def _sign(policy_id, cond, action, agent_id="billing", name="p", enabled=True, priority=100):
    return _sign_policy(policy_id, agent_id, name, cond, action, enabled, priority, SECRET)


class TestVersionSelection(unittest.TestCase):
    def test_legacy_condition_is_v1(self):
        self.assertEqual(_sig_version_for(LEGACY_COND), 1)

    def test_expression_condition_is_v2(self):
        self.assertEqual(_sig_version_for(EXPR_COND), 2)

    def test_sign_returns_version(self):
        sig, ver = _sign(1, LEGACY_COND, ACTION)
        self.assertEqual(ver, 1)
        sig2, ver2 = _sign(2, EXPR_COND, ACTION)
        self.assertEqual(ver2, 2)


class TestRoundTrip(unittest.TestCase):
    def test_v1_legacy_round_trip(self):
        sig, ver = _sign(10, LEGACY_COND, ACTION)
        policy = _policy_dict(10, LEGACY_COND, ACTION, sig, ver)
        self.assertTrue(_verify_policy_signature(policy, SECRET))

    def test_v2_expression_round_trip(self):
        sig, ver = _sign(11, EXPR_COND, ACTION)
        self.assertEqual(ver, 2)
        policy = _policy_dict(11, EXPR_COND, ACTION, sig, ver)
        self.assertTrue(_verify_policy_signature(policy, SECRET))

    def test_empty_secret_signs_empty_but_reports_version(self):
        sig, ver = _sign_policy(1, "a", "n", EXPR_COND, ACTION, True, 100, "")
        self.assertEqual(sig, "")
        self.assertEqual(ver, 2)


class TestTamperDetection(unittest.TestCase):
    def test_tampered_legacy_value_fails(self):
        sig, ver = _sign(20, LEGACY_COND, ACTION)
        tampered = dict(LEGACY_COND, value=1)  # lower the threshold
        policy = _policy_dict(20, tampered, ACTION, sig, ver)
        self.assertFalse(_verify_policy_signature(policy, SECRET))

    def test_tampered_match_value_fails(self):
        sig, ver = _sign(21, EXPR_COND, ACTION)
        tampered = json.loads(json.dumps(EXPR_COND))
        tampered["match"]["args.amount"]["gt"] = 1  # gut the threshold
        policy = _policy_dict(21, tampered, ACTION, sig, ver)
        self.assertFalse(_verify_policy_signature(policy, SECRET))

    def test_stripping_match_block_fails(self):
        # Attacker removes the expensive expression guard entirely.
        sig, ver = _sign(22, EXPR_COND, ACTION)
        stripped = {k: v for k, v in EXPR_COND.items() if k != "match"}
        policy = _policy_dict(22, stripped, ACTION, sig, ver)
        self.assertFalse(_verify_policy_signature(policy, SECRET))

    def test_tampered_action_fails(self):
        sig, ver = _sign(23, EXPR_COND, ACTION)
        policy = _policy_dict(23, EXPR_COND, {"type": "log"}, sig, ver)
        self.assertFalse(_verify_policy_signature(policy, SECRET))


class TestDowngradeAttack(unittest.TestCase):
    def test_v2_policy_relabeled_v1_fails(self):
        # Sign as v2, then claim sig_version=1 (and strip match) — must not verify.
        sig, ver = _sign(30, EXPR_COND, ACTION)
        self.assertEqual(ver, 2)
        stripped = {k: v for k, v in EXPR_COND.items() if k != "match"}
        forged = _policy_dict(30, stripped, ACTION, sig, 1)  # claims v1
        self.assertFalse(_verify_policy_signature(forged, SECRET))

    def test_v2_policy_relabeled_v1_keeping_match_fails(self):
        sig, ver = _sign(31, EXPR_COND, ACTION)
        forged = _policy_dict(31, EXPR_COND, ACTION, sig, 1)  # v2 sig, claims v1
        self.assertFalse(_verify_policy_signature(forged, SECRET))

    def test_v1_policy_relabeled_v2_fails(self):
        sig, ver = _sign(32, LEGACY_COND, ACTION)
        forged = _policy_dict(32, LEGACY_COND, ACTION, sig, 2)  # v1 sig, claims v2
        self.assertFalse(_verify_policy_signature(forged, SECRET))


class TestMigrationCompat(unittest.TestCase):
    def test_v1_canonical_is_byte_identical_to_original_scheme(self):
        # A legacy policy signed by the NEW code must produce exactly the signature
        # the ORIGINAL (pre-Phase-4) scheme produced — proving already-signed
        # policies in the DB keep verifying.
        original_canonical = "\x00".join(
            [
                "40",
                "billing",
                "p",
                json.dumps(LEGACY_COND, sort_keys=True),
                json.dumps(ACTION, sort_keys=True),
                str(True),
                str(100),
            ]
        )
        original_sig = hmac.new(
            SECRET.encode(), original_canonical.encode(), hashlib.sha256
        ).hexdigest()
        new_sig, ver = _sign(40, LEGACY_COND, ACTION)
        self.assertEqual(ver, 1)
        self.assertEqual(new_sig, original_sig)

    def test_pre_phase4_policy_without_sig_version_verifies(self):
        # An existing DB policy signed before this change has no sig_version (or a
        # default of 1) and a v1 signature — it must still verify under new code.
        original_canonical = "\x00".join(
            [
                "41",
                "billing",
                "p",
                json.dumps(LEGACY_COND, sort_keys=True),
                json.dumps(ACTION, sort_keys=True),
                str(True),
                str(100),
            ]
        )
        original_sig = hmac.new(
            SECRET.encode(), original_canonical.encode(), hashlib.sha256
        ).hexdigest()
        policy = _policy_dict(41, LEGACY_COND, ACTION, original_sig, sig_version=None)
        del policy["sig_version"]  # simulate a row/serialization with no version at all
        self.assertTrue(_verify_policy_signature(policy, SECRET))

    def test_sdk_and_api_canonical_agree(self):
        # The two _policy_canonical implementations (api + sdk) must match exactly.
        from dunetrace.policies import _policy_canonical as sdk_canonical

        for ver in (1, 2):
            a = _policy_canonical(ver, 5, "billing", "p", EXPR_COND, ACTION, True, 100)
            b = sdk_canonical(ver, 5, "billing", "p", EXPR_COND, ACTION, True, 100)
            self.assertEqual(a, b, f"canonical mismatch at v{ver}")


if __name__ == "__main__":
    unittest.main()
