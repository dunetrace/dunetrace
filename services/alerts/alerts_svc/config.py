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

    # Phase 4.1 — per-org Slack/Linear integration credentials are encrypted
    # at rest by api_svc; this must match api_svc's own DUNETRACE_MASTER_KEY
    # exactly, same convention as integrations_svc's Phase 2.1 split.
    MASTER_KEY: str = os.getenv("DUNETRACE_MASTER_KEY", "")

    # Linear (Phase 4.1). Base URL only — auth/team/project are per-org,
    # stored in org_alert_integrations, not global config.
    LINEAR_API_URL: str = os.getenv("LINEAR_API_URL", "https://api.linear.app/graphql")

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
            "Failed to load alert policies from %s: %s — using immediate for all",
            path,
            exc,
        )
        return {}


def get_alert_policy(policies: dict[str, dict], failure_type: str) -> dict:
    """Return policy for failure_type, falling back to immediate."""
    return policies.get(failure_type.upper(), _DEFAULT_POLICY)


# ── Per-detector destination routing (Phase 4.1) ──────────────────────────────
#
# "Configurable per detector: which signals go to Slack, which go to Linear."
# Recognized destination names — "linear" is accepted here even though no
# sender exists for it yet (Phase 4.1's Linear side isn't built); a detector
# routed only to "linear" today silently delivers nowhere, same as the
# pre-existing "no destinations globally configured" case deliver() already
# tolerates. Not a bug — a sequencing consequence, see BACKLOG.md.
_VALID_DESTINATIONS = {"slack", "webhook", "linear"}


def load_detector_destinations(yml_path: str | None = None) -> dict[str, list[str]]:
    """Parse detectors.yml's per-detector `destinations` key and return
    {failure_type_upper: [dest, ...]}. A detector absent from the returned
    dict means "no override" — deliver() falls back to sending to every
    globally-enabled destination, exactly today's pre-4.1 behavior. Safe to
    call at import time — returns {} on any error."""
    path = yml_path or os.getenv("DETECTORS_YML", "/app/detectors.yml")
    try:
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        default_section = raw.get("default") or {}
        destinations: dict[str, list[str]] = {}
        for detector_key, cfg in default_section.items():
            if not isinstance(cfg, dict):
                continue
            dests = cfg.get("destinations")
            if not isinstance(dests, list):
                continue
            valid = [d for d in dests if d in _VALID_DESTINATIONS]
            if valid:
                destinations[detector_key.upper()] = valid
        return destinations
    except FileNotFoundError:
        return {}
    except Exception as exc:
        import logging

        logging.getLogger("dunetrace.alerts.config").warning(
            "Failed to load detector destinations from %s: %s — using default routing for all",
            path,
            exc,
        )
        return {}


def get_detector_destinations(
    destinations: dict[str, list[str]], failure_type: str
) -> list[str] | None:
    """Return the destination override for failure_type, or None meaning
    "no override — send to every globally-enabled destination"."""
    return destinations.get(failure_type.upper())


# ── Semantic evaluator confidence floors (Phase 1.4.1) ────────────────────────
#
# Below-threshold semantic signals are still written and stored (visible in
# the dashboard) but never alerted — see worker.py's poll_once(). Structural
# signals are unaffected; this only gates rows with source == "semantic".

_DEFAULT_SEMANTIC_CONFIDENCE_FLOOR = 0.6


def load_semantic_confidence_floors(yml_path: str | None = None) -> dict[str, float]:
    """Parse docs/config/semantic-evaluators.yml into
    {evaluator_name_upper: alert_confidence_floor}. Falls back to {} (meaning
    every evaluator uses _DEFAULT_SEMANTIC_CONFIDENCE_FLOOR) for a missing
    file, missing PyYAML, or any parse error. Safe to call at import time."""
    path = yml_path or os.getenv(
        "SEMANTIC_EVALUATORS_YML", "/app/docs/config/semantic-evaluators.yml"
    )
    try:
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        floors: dict[str, float] = {}
        for evaluator_key, cfg in raw.items():
            if not isinstance(cfg, dict):
                continue
            floor = cfg.get("alert_confidence_floor")
            if floor is None:
                continue
            try:
                value = float(floor)
            except (TypeError, ValueError):
                continue
            if 0.0 <= value <= 1.0:
                floors[evaluator_key.upper()] = value
        return floors
    except FileNotFoundError:
        return {}
    except Exception as exc:
        import logging

        logging.getLogger("dunetrace.alerts.config").warning(
            "Failed to load semantic confidence floors from %s: %s — using default %.1f for all",
            path,
            exc,
            _DEFAULT_SEMANTIC_CONFIDENCE_FLOOR,
        )
        return {}


def get_semantic_confidence_floor(floors: dict[str, float], evaluator: str) -> float:
    """Return the alert confidence floor for a semantic evaluator, falling
    back to _DEFAULT_SEMANTIC_CONFIDENCE_FLOOR (0.6)."""
    return floors.get(evaluator.upper(), _DEFAULT_SEMANTIC_CONFIDENCE_FLOOR)
