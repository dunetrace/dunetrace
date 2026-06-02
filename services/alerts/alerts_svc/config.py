"""
services/alerts/alerts_svc/config.py
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
                val = val.strip()
                if " #" in val:
                    val = val[: val.index(" #")].strip()
                os.environ.setdefault(key.strip(), val)
    except FileNotFoundError:
        pass


_load_dotenv()


class Settings:
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "postgresql://dunetrace:dunetrace@localhost:5432/dunetrace"
    )

    # Slack
    # Set SLACK_WEBHOOK_URL to enable Slack alerts.
    # Get one at: https://api.slack.com/messaging/webhooks
    SLACK_WEBHOOK_URL: str = os.getenv("SLACK_WEBHOOK_URL", "")
    SLACK_CHANNEL: str = os.getenv("SLACK_CHANNEL", "#agent-alerts")

    # Minimum severity to alert on. One of: LOW, MEDIUM, HIGH, CRITICAL
    SLACK_MIN_SEVERITY: str = os.getenv("SLACK_MIN_SEVERITY", "LOW")

    # Generic webhook
    # A JSON POST will be sent to this URL for every alert.
    # Useful for PagerDuty, Linear, custom webhooks, etc.
    WEBHOOK_URL: str = os.getenv("WEBHOOK_URL", "")
    WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET", "")  # HMAC-SHA256 signing key

    # Worker
    POLL_INTERVAL: float = float(os.getenv("POLL_INTERVAL", "10"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "50"))

    # Alert deduplication — same (agent_id, failure_type) silenced for this window after first alert.
    # Set to 0 to disable. Suppressed count is reported when the window re-opens.
    ALERT_DEDUP_WINDOW: int = int(os.getenv("ALERT_DEDUP_WINDOW", "3600"))  # seconds

    # Retry behaviour for failed HTTP calls
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_BACKOFF: float = float(os.getenv("RETRY_BACKOFF", "2.0"))  # seconds, doubled each retry

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Weekly digest
    DIGEST_ENABLED: bool = os.getenv("DIGEST_ENABLED", "true").lower() == "true"
    DIGEST_DAY: int = int(os.getenv("DIGEST_DAY", "0"))  # 0=Monday … 6=Sunday
    DIGEST_HOUR: int = int(os.getenv("DIGEST_HOUR", "9"))  # UTC hour to send
    DASHBOARD_URL: str = os.getenv("DASHBOARD_URL", "https://app.dunetrace.io")

    @property
    def slack_enabled(self) -> bool:
        return bool(self.SLACK_WEBHOOK_URL)

    @property
    def webhook_enabled(self) -> bool:
        return bool(self.WEBHOOK_URL)

    @property
    def digest_enabled(self) -> bool:
        return self.DIGEST_ENABLED and bool(self.SLACK_WEBHOOK_URL)


settings = Settings()

# Severity order for threshold comparisons
SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

# ── Alert policies (loaded from detectors.yml) ────────────────────────────────

_DEFAULT_POLICY = {"mode": "immediate", "threshold": 1, "window_runs": 1}


def load_alert_policies(yml_path: str | None = None) -> dict[str, dict]:
    """Parse detectors.yml and return {failure_type_upper: policy_dict}.
    Falls back to _DEFAULT_POLICY (immediate) for any detector not listed.
    Safe to call at import time — returns {} on any error."""
    path = yml_path or os.getenv("DETECTORS_YML", "/app/detectors.yml")
    try:
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        default_section = raw.get("default") or {}
        policies: dict[str, dict] = {}
        for detector_key, cfg in default_section.items():
            if not isinstance(cfg, dict):
                continue
            ap = cfg.get("alert_policy")
            if isinstance(ap, dict):
                policies[detector_key.upper()] = {
                    "mode": ap.get("mode", "immediate"),
                    "threshold": int(ap.get("threshold", 1)),
                    "window_runs": int(ap.get("window_runs", 1)),
                }
        return policies
    except FileNotFoundError:
        return {}
    except Exception as exc:
        import logging

        logging.getLogger("dunetrace.alerts.config").warning(
            "Failed to load alert policies from %s: %s — using immediate for all", path, exc
        )
        return {}


def get_alert_policy(policies: dict[str, dict], failure_type: str) -> dict:
    """Return policy for failure_type, falling back to immediate."""
    return policies.get(failure_type.upper(), _DEFAULT_POLICY)
