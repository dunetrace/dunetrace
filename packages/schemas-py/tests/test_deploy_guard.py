"""
The startup guard refuses exactly one pair — ENV=prod with AUTH_MODE=dev — and
nothing else. Every service's config.py calls it at import, so a false
positive here would take down the quickstart and a false negative would let a
production deployment come up with auth off.

Run: PYTHONPATH=packages/schemas-py python -m pytest packages/schemas-py/tests/test_deploy_guard.py -v
"""

from __future__ import annotations

import unittest

from dunetrace_schemas.deploy_guard import (
    DEV_AUTH_MODES,
    PROD_ENVS,
    assert_safe_deployment,
    format_refusal,
    is_unsafe_deployment,
)


class TestRefusedPairs(unittest.TestCase):
    def test_prod_with_dev_auth_is_refused(self):
        with self.assertRaises(SystemExit):
            assert_safe_deployment("ingest", "prod", "dev")

    def test_every_prod_spelling_with_every_dev_auth_mode_is_refused(self):
        for env in PROD_ENVS:
            for auth_mode in DEV_AUTH_MODES:
                with self.subTest(env=env, auth_mode=auth_mode):
                    self.assertTrue(is_unsafe_deployment(env, auth_mode))
                    with self.assertRaises(SystemExit):
                        assert_safe_deployment("api", env, auth_mode)

    def test_matching_is_case_and_whitespace_insensitive(self):
        # An operator typing `ENV=Production` or `AUTH_MODE=Dev ` (trailing
        # space from a .env editor) has still deployed prod with auth off.
        for env, auth_mode in (
            ("PROD", "DEV"),
            ("Production", "Test"),
            (" prod ", " local "),
            ("prod\n", "dev\n"),
        ):
            with self.subTest(env=env, auth_mode=auth_mode):
                self.assertTrue(is_unsafe_deployment(env, auth_mode))

    def test_exit_carries_a_message_and_a_nonzero_status(self):
        # SystemExit with a str payload: Python prints it to stderr and exits 1.
        # An int payload of 0 (or None) would exit *successfully* — a refusal
        # that looks like a clean shutdown to a supervisor is worse than none.
        with self.assertRaises(SystemExit) as cm:
            assert_safe_deployment("api", "prod", "dev")
        self.assertIsInstance(cm.exception.code, str)
        self.assertTrue(cm.exception.code.strip())

    def test_message_names_the_service_and_both_values(self):
        message = format_refusal("detector", "prod", "dev")
        self.assertIn("dunetrace/detector", message)
        self.assertIn("ENV=prod", message)
        self.assertIn("AUTH_MODE=dev", message)
        # The fix is spelled out, not left to the reader.
        self.assertIn("AUTH_MODE=prod", message)
        self.assertIn("docker-compose.prod.yml", message)
        self.assertIn("docs/operations.md", message)
        # Multi-line by design: this lands in a container log next to a
        # traceback-shaped wall of text and must be readable at a glance.
        self.assertGreater(message.count("\n"), 5)

    def test_message_shows_unset_values_explicitly(self):
        message = format_refusal("api", None, "dev")
        self.assertIn("ENV=<unset>", message)


class TestPermittedPairs(unittest.TestCase):
    def test_quickstart_pair_passes(self):
        # docker-compose.yml's defaults: ENV=dev, AUTH_MODE=dev.
        assert_safe_deployment("ingest", "dev", "dev")
        self.assertFalse(is_unsafe_deployment("dev", "dev"))

    def test_production_pair_passes(self):
        # docker-compose.prod.yml: ENV=prod, AUTH_MODE=prod.
        assert_safe_deployment("api", "prod", "prod")
        self.assertFalse(is_unsafe_deployment("production", "prod"))

    def test_dev_env_with_prod_auth_passes(self):
        # Locked-down local instance — stricter than needed, never refused.
        assert_safe_deployment("api", "dev", "prod")

    def test_unset_values_pass(self):
        # Unset ENV is the workers' default (they read "dev"); an unset
        # AUTH_MODE means fail-closed prod in both HTTP services. Neither is
        # the known-bad pair, and the guard must not invent a stricter rule.
        for env, auth_mode in ((None, None), ("", ""), (None, "dev"), ("prod", None), ("prod", "")):
            with self.subTest(env=env, auth_mode=auth_mode):
                self.assertFalse(is_unsafe_deployment(env, auth_mode))
                assert_safe_deployment("alerts", env, auth_mode)

    def test_unrecognised_env_names_are_not_treated_as_prod(self):
        # "staging" with auth off is a choice the guard does not police — it
        # refuses the one pair it is sure about and validates no vocabulary.
        for env in ("staging", "preprod", "qa", "live"):
            with self.subTest(env=env):
                self.assertFalse(is_unsafe_deployment(env, "dev"))
                assert_safe_deployment("api", env, "dev")

    def test_unrecognised_auth_modes_are_not_treated_as_dev(self):
        # An unknown AUTH_MODE fails closed in the services (is_dev is False),
        # so it is not an open deployment and must not be refused here.
        for auth_mode in ("strict", "sso", "0", "false"):
            with self.subTest(auth_mode=auth_mode):
                self.assertFalse(is_unsafe_deployment("prod", auth_mode))
                assert_safe_deployment("api", "prod", auth_mode)


class TestVocabularyMatchesTheServices(unittest.TestCase):
    """The guard's idea of "auth disabled" must be the services' idea.

    api_svc.config.Settings.is_dev and ingest_svc.config.Settings.is_dev both
    test membership in {"dev", "local", "test"}; if that set ever grows, the
    guard has to grow with it or a new spelling of "auth off" slips past.
    """

    def test_dev_auth_modes_match_is_dev(self):
        self.assertEqual(DEV_AUTH_MODES, frozenset({"dev", "local", "test"}))

    def test_prod_envs(self):
        self.assertEqual(PROD_ENVS, frozenset({"prod", "production"}))


if __name__ == "__main__":
    unittest.main()
