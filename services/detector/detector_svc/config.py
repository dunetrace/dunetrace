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
    POLL_INTERVAL: float = float(os.getenv("POLL_INTERVAL", "5"))
    STALL_TIMEOUT_SECS: float = float(os.getenv("STALL_TIMEOUT_SECS", "90"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "100"))
    DETECTOR_CONCURRENCY: int = int(os.getenv("DETECTOR_CONCURRENCY", "8"))
    # Horizontal sharding by agent_id.  Run N replicas with SHARD_COUNT=N and
    # SHARD_INDEX=0..N-1.  Each worker only polls runs whose agent_id hashes
    # to its bucket.  SHARD_COUNT=1 (default) disables filtering entirely.
    SHARD_COUNT: int = int(os.getenv("SHARD_COUNT", "1"))
    SHARD_INDEX: int = int(os.getenv("SHARD_INDEX", "0"))


settings = Settings()

if settings.SHARD_COUNT < 1:
    raise ValueError(f"SHARD_COUNT must be >= 1, got {settings.SHARD_COUNT}")
if not (0 <= settings.SHARD_INDEX < settings.SHARD_COUNT):
    raise ValueError(
        f"SHARD_INDEX must be in [0, SHARD_COUNT), "
        f"got SHARD_INDEX={settings.SHARD_INDEX} SHARD_COUNT={settings.SHARD_COUNT}"
    )
