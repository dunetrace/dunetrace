from __future__ import annotations

import os

try:
    from dunetrace_schemas.deploy_guard import assert_safe_deployment
except ImportError:
    # dunetrace_schemas' package __init__ imports pydantic, which this image
    # does not install — nothing here touches the wire-format models, and it
    # is the one service image without it. Keep the refusal regardless: this
    # is dunetrace_schemas.deploy_guard's rule reduced to its minimum. Remove
    # once the image installs pydantic (or the package init stops importing
    # it eagerly), so the shared implementation is the only one.
    def assert_safe_deployment(service: str, env: str, auth_mode: str) -> None:
        prod = env.strip().lower() in {"prod", "production"}
        auth_off = auth_mode.strip().lower() in {"dev", "local", "test"}
        if prod and auth_off:
            raise SystemExit(
                f"dunetrace/{service}: refusing to start.\n"
                f"    ENV={env}  (a production deployment)\n"
                f"    AUTH_MODE={auth_mode}  (authentication disabled)\n"
                "Set AUTH_MODE=prod (docker-compose.prod.yml does) or ENV=dev for a "
                'local instance. See docs/operations.md, section "Deploying".\n'
            )


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
    # Deployment identity — read only by the startup guard at the bottom of
    # this file. This worker authenticates nothing, so neither value changes
    # its behaviour; they exist so ENV=prod with AUTH_MODE=dev is refused on
    # every container, not just the two HTTP services. Same defaults as
    # ingest_svc/api_svc: ENV is dev unless told otherwise, AUTH_MODE fails
    # closed to prod.
    ENV: str = os.getenv("ENV", "dev")
    AUTH_MODE: str = os.getenv("AUTH_MODE", "prod")

    # How often this worker wakes to check which orgs are due for a poll —
    # NOT the same as an individual org's own poll_interval_secs (stored per
    # integration, default 60s). A short wake cadence lets orgs with short
    # poll intervals be served promptly without every org needing the same one.
    WAKE_INTERVAL: float = float(os.getenv("WAKE_INTERVAL", "15"))

    # Build identity for dunetrace_build_info{service=...,version=...}. Same
    # resolution as ingest_svc/api_svc: APP_VERSION, else GIT_COMMIT, else the
    # package version. Set one of them in deploy to identify the build.
    APP_VERSION: str = os.getenv("APP_VERSION") or os.getenv("GIT_COMMIT") or "0.5.0"
    # Ports for each worker's own /metrics, /ready and /health (stdlib HTTP
    # server from dunetrace_schemas.metrics, bound on every interface so the
    # compose healthcheck and an in-network scraper can reach it; compose does
    # not publish them). Two env vars, not one METRICS_PORT: both workers run
    # from this one image and read this one Settings class, and a shared
    # variable would make a single-container deployment (both processes in one
    # network namespace) collide on the bind. 0 disables the server. Only
    # started when the worker is enabled — a disabled worker exits 0 and has
    # nothing to scrape.
    INTEGRATIONS_METRICS_PORT: int = int(os.getenv("INTEGRATIONS_METRICS_PORT", "9104"))
    ELEVENLABS_METRICS_PORT: int = int(os.getenv("ELEVENLABS_METRICS_PORT", "9105"))

    # Disabled by default, same convention as semantic_svc's
    # SEMANTIC_WORKER_ENABLED — an OSS install that never sets this never
    # opens a DB pool for this service. See run_worker().
    INTEGRATIONS_WORKER_ENABLED: bool = os.getenv(
        "INTEGRATIONS_WORKER_ENABLED", "false"
    ).lower() in ("1", "true", "yes")

    # The ElevenLabs poller (elevenlabs_worker) is a separate process/flag from
    # the evaluation-provider worker above, so an org can run one without the
    # other, and a failure in one never touches the other. Same OSS-friendly
    # default: an install that never sets this never opens a DB pool for it.
    ELEVENLABS_WORKER_ENABLED: bool = os.getenv("ELEVENLABS_WORKER_ENABLED", "false").lower() in (
        "1",
        "true",
        "yes",
    )

    # ── Correlation tuning (Phase 4.4) ──────────────────────────────────────────
    # Half-width of the timestamp window (seconds) a Dunetrace tts.generated event
    # may sit from an ElevenLabs generation's create time. Generous by default to
    # absorb the clock-domain gap: the event is emitted after the audio returns,
    # so it lags the generation by network + synthesis latency.
    CORRELATION_WINDOW_SECS: float = float(os.getenv("CORRELATION_WINDOW_SECS", "60"))
    # Relative character-count tolerance (0.10 = within 10%) for the fallback
    # match when neither generation id nor exact text is available.
    CORRELATION_CHAR_TOLERANCE: float = float(os.getenv("CORRELATION_CHAR_TOLERANCE", "0.10"))
    # A generation still uncorrelated this long after it was generated is declared
    # unmatched (recorded as drift) rather than retried forever. Events ingest in
    # near real time, so an hour with no match means there genuinely is none.
    CORRELATION_GIVEUP_SECS: float = float(os.getenv("CORRELATION_GIVEUP_SECS", "3600"))

    # Must match api_svc's own DUNETRACE_MASTER_KEY exactly — that service
    # encrypts customer credentials, this one is the only thing that ever
    # decrypts them (see crypto.py).
    MASTER_KEY: str = os.getenv("DUNETRACE_MASTER_KEY", "")


settings = Settings()

# ENV=prod with AUTH_MODE=dev never comes up — see dunetrace_schemas.deploy_guard.
assert_safe_deployment("integrations", settings.ENV, settings.AUTH_MODE)
