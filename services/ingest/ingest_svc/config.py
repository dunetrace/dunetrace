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
    # Fails CLOSED — see api_svc/config.py for the full rationale. Unset means
    # ingest requires a valid API key; `dev` (explicit) accepts anonymous writes.
    AUTH_MODE: str = os.getenv("AUTH_MODE", "prod")
    INTERNAL_TOKEN: str = os.getenv("INTERNAL_TOKEN", "")
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "500"))
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))  # per IP per minute
    EVENT_RETENTION_DAYS: int = int(os.getenv("EVENT_RETENTION_DAYS", "90"))

    # ── OTLP receiver limits (Phase 3) ────────────────────────────────────────
    # Max compressed request body accepted at /v1/otlp/traces (413 above this).
    OTLP_MAX_BODY_BYTES: int = int(os.getenv("OTLP_MAX_BODY_BYTES", str(10 * 1024 * 1024)))
    # Max size a gzip body may expand to (guards against gzip bombs).
    OTLP_MAX_DECOMPRESSED_BYTES: int = int(
        os.getenv("OTLP_MAX_DECOMPRESSED_BYTES", str(50 * 1024 * 1024))
    )
    # Longest attribute string kept per event field; longer values are truncated
    # to keep one giant span from bloating storage.
    OTLP_MAX_ATTR_CHARS: int = int(os.getenv("OTLP_MAX_ATTR_CHARS", "8192"))
    # Per-org span ingestion rate. One org's burst can't starve another's.
    OTLP_MAX_SPANS_PER_SEC: int = int(os.getenv("OTLP_MAX_SPANS_PER_SEC", "1000"))
    # Backpressure: max OTLP batches being translated/persisted concurrently.
    OTLP_MAX_INFLIGHT: int = int(os.getenv("OTLP_MAX_INFLIGHT", "100"))
    # Failed-persist retry buffer depth (batches). Drop-oldest when full.
    OTLP_RETRY_BUFFER_BATCHES: int = int(os.getenv("OTLP_RETRY_BUFFER_BATCHES", "500"))

    @property
    def is_dev(self) -> bool:
        return self.ENV.lower() in {"dev", "local", "test"}


settings = Settings()
