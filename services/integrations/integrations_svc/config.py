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

    # How often this worker wakes to check which orgs are due for a poll —
    # NOT the same as an individual org's own poll_interval_secs (stored per
    # integration, default 60s). A short wake cadence lets orgs with short
    # poll intervals be served promptly without every org needing the same one.
    WAKE_INTERVAL: float = float(os.getenv("WAKE_INTERVAL", "15"))

    # Disabled by default, same convention as semantic_svc's
    # SEMANTIC_WORKER_ENABLED — an OSS install that never sets this never
    # opens a DB pool for this service. See run_worker().
    INTEGRATIONS_WORKER_ENABLED: bool = os.getenv(
        "INTEGRATIONS_WORKER_ENABLED", "false"
    ).lower() in ("1", "true", "yes")

    # Must match api_svc's own DUNETRACE_MASTER_KEY exactly — that service
    # encrypts customer credentials, this one is the only thing that ever
    # decrypts them (see crypto.py).
    MASTER_KEY: str = os.getenv("DUNETRACE_MASTER_KEY", "")


settings = Settings()
