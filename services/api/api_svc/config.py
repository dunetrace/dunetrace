"""
services/api/api_svc/config.py
"""

from __future__ import annotations

import os


def _load_dotenv(path: str = ".env") -> None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                val = val.strip()
                if " #" in val:
                    val = val[: val.index(" #")].strip()
                os.environ.setdefault(key.strip(), val)
    except FileNotFoundError:
        pass


_load_dotenv()


class Settings:
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://dunetrace:dunetrace@localhost:5432/dunetrace",
    )
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    # audit Finding 34: real build version reported by /health (was hardcoded
    # 0.1.0). Set APP_VERSION (or GIT_COMMIT) in deploy to identify the build.
    APP_VERSION: str = os.getenv("APP_VERSION") or os.getenv("GIT_COMMIT") or "0.5.0"
    # Fails CLOSED: an unset AUTH_MODE means full auth, not skipped auth. Dev
    # mode disables authentication entirely, so it must be an explicit opt-in —
    # a deployment that forgets to set this gets a locked-down API rather than
    # an open one. Both compose files set `dev` explicitly for the local
    # quickstart; anything else (k8s, systemd, bare uvicorn) inherits `prod`.
    AUTH_MODE: str = os.getenv("AUTH_MODE", "prod")

    # Trusted-upstream bypass — mirrors ingest_svc's INTERNAL_TOKEN. When set, requests
    # carrying a matching x-internal-token header skip this service's own api_keys
    # lookup and take customer identity from x-customer-id instead. Empty by default
    # so self-hosted/dev deployments are unaffected.
    INTERNAL_TOKEN: str = os.getenv("INTERNAL_TOKEN", "")

    # Rate limit stamped on keys created through POST /v1/keys. Operator-set,
    # not caller-set: key creation is self-service, so letting the request body
    # choose this made the quota self-granting.
    DEFAULT_KEY_RATE_LIMIT_RPM: int = int(os.getenv("DEFAULT_KEY_RATE_LIMIT_RPM", "600"))
    PAGE_SIZE_DEFAULT: int = int(os.getenv("PAGE_SIZE_DEFAULT", "50"))
    PAGE_SIZE_MAX: int = int(os.getenv("PAGE_SIZE_MAX", "500"))

    # LLM for this service's own features: native explain, fix diff generation,
    # custom-detector translation, issue summarisation. With API_LLM_PROVIDER
    # unset the first configured key wins in the order below (Anthropic first,
    # as before). Set it to anthropic|openai|mistral to pin one — a deployment
    # that must keep this text inside a European provider sets `mistral`.
    # See api_svc/llm_provider.py; distinct from the semantic worker's
    # SEMANTIC_LLM_PROVIDER, which selects a different service's evaluators.
    API_LLM_PROVIDER: str = os.getenv("API_LLM_PROVIDER", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")

    # GitHub integration — for opening fix PRs from code-change signals.
    # GITHUB_TOKEN/GITHUB_REPO is the original single-tenant PAT-based path
    # (self-hosted installs, one repo) — kept as a fallback when an org has
    # no per-org GitHub App installation configured (Phase 4.3), same
    # "org-config-first-then-global-fallback" precedent Phase 4.1 established
    # for Slack.
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    GITHUB_REPO: str = os.getenv("GITHUB_REPO", "")  # e.g. "owner/repo"
    GITHUB_BASE_BRANCH: str = os.getenv("GITHUB_BASE_BRANCH", "main")

    # GitHub App (Phase 4.3) — one Dunetrace-owned App, installed per-org by
    # each customer onto their own chosen repos. GITHUB_APP_PRIVATE_KEY is
    # the PEM the App's GitHub settings page generates (paste the whole
    # thing, including the BEGIN/END lines, as a single .env value with
    # literal \n for newlines). Neither value is per-org — the App itself
    # is one operator-level registration; installation_id (stored per org
    # in org_github_integrations) is what distinguishes one customer's
    # installation from another's.
    GITHUB_APP_ID: str = os.getenv("GITHUB_APP_ID", "")
    GITHUB_APP_PRIVATE_KEY: str = os.getenv("GITHUB_APP_PRIVATE_KEY", "").replace("\\n", "\n")
    GITHUB_APP_SLUG: str = os.getenv("GITHUB_APP_SLUG", "")  # for building the install URL

    # Slack interactive callbacks — from Settings > Basic Information > Signing Secret
    # Required to verify that button-click payloads actually come from Slack.
    SLACK_SIGNING_SECRET: str = os.getenv("SLACK_SIGNING_SECRET", "")

    # Policy payload signing — HMAC-SHA256 key shared between the API and SDK clients.
    # Set POLICY_SIGNING_SECRET to the same value in both environments.
    # If empty, signatures are not verified (dev/test only).
    POLICY_SIGNING_SECRET: str = os.getenv("POLICY_SIGNING_SECRET", "")

    # Encrypts external-integration credentials at rest (Langfuse/LangSmith/
    # Braintrust API keys — Phase 2.1). Must be a valid Fernet key (32
    # url-safe base64-encoded bytes) and identical to integrations_svc's own
    # DUNETRACE_MASTER_KEY, since that service is the one that decrypts.
    # Generate one with:
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    MASTER_KEY: str = os.getenv("DUNETRACE_MASTER_KEY", "")

    @property
    def github_configured(self) -> bool:
        return bool(self.GITHUB_TOKEN and self.GITHUB_REPO)

    @property
    def github_app_configured(self) -> bool:
        return bool(self.GITHUB_APP_ID and self.GITHUB_APP_PRIVATE_KEY)

    @property
    def is_dev(self) -> bool:
        return self.AUTH_MODE.lower() in {"dev", "local", "test"}


settings = Settings()
