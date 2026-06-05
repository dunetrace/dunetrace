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
    AUTH_MODE: str = os.getenv("AUTH_MODE", "dev")
    PAGE_SIZE_DEFAULT: int = int(os.getenv("PAGE_SIZE_DEFAULT", "50"))
    PAGE_SIZE_MAX: int = int(os.getenv("PAGE_SIZE_MAX", "500"))

    # Langfuse integration
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    # LLM for explain endpoint (Anthropic preferred; falls back to OpenAI)
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # GitHub integration — for opening fix PRs from code-change signals
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    GITHUB_REPO: str = os.getenv("GITHUB_REPO", "")  # e.g. "owner/repo"
    GITHUB_BASE_BRANCH: str = os.getenv("GITHUB_BASE_BRANCH", "main")

    # Slack interactive callbacks — from Settings > Basic Information > Signing Secret
    # Required to verify that button-click payloads actually come from Slack.
    SLACK_SIGNING_SECRET: str = os.getenv("SLACK_SIGNING_SECRET", "")

    # Policy payload signing — HMAC-SHA256 key shared between the API and SDK clients.
    # Set POLICY_SIGNING_SECRET to the same value in both environments.
    # If empty, signatures are not verified (dev/test only).
    POLICY_SIGNING_SECRET: str = os.getenv("POLICY_SIGNING_SECRET", "")

    @property
    def langfuse_configured(self) -> bool:
        return bool(self.LANGFUSE_PUBLIC_KEY and self.LANGFUSE_SECRET_KEY)

    @property
    def github_configured(self) -> bool:
        return bool(self.GITHUB_TOKEN and self.GITHUB_REPO)

    @property
    def is_dev(self) -> bool:
        return self.AUTH_MODE.lower() in {"dev", "local", "test"}


settings = Settings()
