"""
services/alerts/app/config.py
"""
from __future__ import annotations
import os


def _load_dotenv(path: str = ".env") -> None:
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())
    except FileNotFoundError:
        pass


_load_dotenv()


class Settings:
    DATABASE_URL:      str   = os.getenv("DATABASE_URL",
                                "postgresql://dunetrace:dunetrace@localhost:5432/dunetrace")

    # ── Slack ──────────────────────────────────────────────────────────────────
    # Set SLACK_WEBHOOK_URL to enable Slack alerts.
    # Get one at: https://api.slack.com/messaging/webhooks
    SLACK_WEBHOOK_URL: str   = os.getenv("SLACK_WEBHOOK_URL", "")
    SLACK_CHANNEL:     str   = os.getenv("SLACK_CHANNEL", "#agent-alerts")

    # Minimum severity to alert on. One of: LOW, MEDIUM, HIGH, CRITICAL
    SLACK_MIN_SEVERITY: str  = os.getenv("SLACK_MIN_SEVERITY", "HIGH")

    # ── Generic webhook ────────────────────────────────────────────────────────
    # A JSON POST will be sent to this URL for every alert.
    # Useful for PagerDuty, Linear, custom webhooks, etc.
    WEBHOOK_URL:       str   = os.getenv("WEBHOOK_URL", "")
    WEBHOOK_SECRET:    str   = os.getenv("WEBHOOK_SECRET", "")   # HMAC-SHA256 signing key

    # ── Worker ─────────────────────────────────────────────────────────────────
    POLL_INTERVAL:     float = float(os.getenv("POLL_INTERVAL", "10"))
    BATCH_SIZE:        int   = int(os.getenv("BATCH_SIZE", "50"))

    # Retry behaviour for failed HTTP calls
    MAX_RETRIES:       int   = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_BACKOFF:     float = float(os.getenv("RETRY_BACKOFF", "2.0"))  # seconds, doubled each retry

    LOG_LEVEL:         str   = os.getenv("LOG_LEVEL", "INFO")

    @property
    def slack_enabled(self) -> bool:
        return bool(self.SLACK_WEBHOOK_URL)

    @property
    def webhook_enabled(self) -> bool:
        return bool(self.WEBHOOK_URL)


settings = Settings()

# Severity order for threshold comparisons
SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
