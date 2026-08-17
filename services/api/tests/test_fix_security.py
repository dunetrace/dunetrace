"""
Tests for Phase 4.3's security guardrails (api_svc/fix_security.py) — the
deterministic, pre-write checks that gate real diff application. No DB, no
network, no LLM — these are pure functions by design, so every case is
directly testable.
"""

from __future__ import annotations

import unittest

from api_svc.fix_security import is_sensitive_path, validate_target_path


class TestIsSensitivePath(unittest.TestCase):
    def test_github_workflows_blocked(self):
        self.assertTrue(is_sensitive_path(".github/workflows/ci.yml"))

    def test_github_actions_blocked(self):
        self.assertTrue(is_sensitive_path(".github/actions/deploy/action.yml"))

    def test_gitlab_ci_blocked(self):
        self.assertTrue(is_sensitive_path(".gitlab-ci.yml"))

    def test_circleci_blocked(self):
        self.assertTrue(is_sensitive_path(".circleci/config.yml"))

    def test_jenkinsfile_blocked(self):
        self.assertTrue(is_sensitive_path("Jenkinsfile"))

    def test_dotenv_blocked(self):
        self.assertTrue(is_sensitive_path(".env"))

    def test_dotenv_variant_blocked(self):
        self.assertTrue(is_sensitive_path(".env.production"))

    def test_dotenv_nested_blocked(self):
        self.assertTrue(is_sensitive_path("config/.env"))

    def test_pem_file_blocked(self):
        self.assertTrue(is_sensitive_path("certs/server.pem"))

    def test_key_file_blocked(self):
        self.assertTrue(is_sensitive_path("keys/private.key"))

    def test_id_rsa_blocked(self):
        self.assertTrue(is_sensitive_path("id_rsa"))
        self.assertTrue(is_sensitive_path(".ssh/id_rsa_deploy"))

    def test_secret_in_name_blocked(self):
        self.assertTrue(is_sensitive_path("config/secrets.py"))

    def test_credential_in_name_blocked(self):
        self.assertTrue(is_sensitive_path("aws_credentials.json"))

    def test_password_in_name_blocked(self):
        self.assertTrue(is_sensitive_path("db_password_reset.py"))

    def test_dockerfile_blocked(self):
        self.assertTrue(is_sensitive_path("Dockerfile"))

    def test_docker_compose_blocked(self):
        self.assertTrue(is_sensitive_path("docker-compose.prod.yml"))

    def test_case_insensitive(self):
        self.assertTrue(is_sensitive_path("SECRETS.PY"))
        self.assertTrue(is_sensitive_path("DOCKERFILE"))

    def test_ordinary_source_file_not_sensitive(self):
        self.assertFalse(is_sensitive_path("services/agents/support_bot.py"))

    def test_leading_slash_normalized(self):
        self.assertTrue(is_sensitive_path("/.env"))

    def test_ordinary_python_file_with_agent_in_name_not_blocked(self):
        """Sanity check the substring rules aren't so broad they block normal code."""
        self.assertFalse(is_sensitive_path("src/agents/my_agent.py"))


class TestValidateTargetPath(unittest.TestCase):
    def test_matching_paths_ok(self):
        ok, reason = validate_target_path("src/agents/bot.py", "src/agents/bot.py")
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_mismatched_paths_rejected(self):
        ok, reason = validate_target_path("src/agents/bot.py", "src/other/file.py")
        self.assertFalse(ok)
        self.assertIn("src/other/file.py", reason)
        self.assertIn("src/agents/bot.py", reason)

    def test_none_diff_target_skips_match_check(self):
        """When there's no LLM-authored diff to compare against (Phase 4.3's
        real-diff-application path — we generate the diff ourselves), only
        the sensitive-path check applies."""
        ok, reason = validate_target_path("src/agents/bot.py", None)
        self.assertTrue(ok)

    def test_resolved_path_itself_sensitive_rejected(self):
        ok, reason = validate_target_path(".github/workflows/ci.yml", None)
        self.assertFalse(ok)
        self.assertIn("sensitive", reason)

    def test_leading_slash_normalized_for_comparison(self):
        ok, _ = validate_target_path("/src/agents/bot.py", "src/agents/bot.py")
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
