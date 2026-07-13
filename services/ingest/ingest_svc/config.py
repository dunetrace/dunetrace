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
    ENV: str = os.getenv("ENV", "dev")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    # audit Finding 34: real build version reported by /health (was hardcoded 0.1.0).
    APP_VERSION: str = os.getenv("APP_VERSION") or os.getenv("GIT_COMMIT") or "0.5.0"
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://dunetrace:dunetrace@localhost:5432/dunetrace",
    )
    AUTH_MODE: str = os.getenv("AUTH_MODE", "dev")
    INTERNAL_TOKEN: str = os.getenv("INTERNAL_TOKEN", "")
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "500"))
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))  # per IP per minute
    EVENT_RETENTION_DAYS: int = int(os.getenv("EVENT_RETENTION_DAYS", "90"))

    @property
    def is_dev(self) -> bool:
        return self.ENV.lower() in {"dev", "local", "test"}


settings = Settings()
