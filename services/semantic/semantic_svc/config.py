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
    POLL_INTERVAL: float = float(os.getenv("POLL_INTERVAL", "15"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "100"))

    # Disabled by default — an OSS install that never sets this never opens a DB
    # pool for this service, let alone calls an LLM. See run_worker().
    SEMANTIC_WORKER_ENABLED: bool = os.getenv("SEMANTIC_WORKER_ENABLED", "false").lower() in (
        "1",
        "true",
        "yes",
    )

    # "openai" | "anthropic" — which DeepEval-native model class evaluators
    # build by default. Explicit instantiation, not DeepEval's own env/keyfile
    # provider auto-detection (see evaluators/models.py's docstring for why).
    SEMANTIC_LLM_PROVIDER: str = os.getenv("SEMANTIC_LLM_PROVIDER", "openai")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Per-evaluator model override. Empty string means "use the provider's
    # cost-conscious default" — see evaluators/models.py::_DEFAULT_MODEL_BY_PROVIDER.
    HALLUCINATION_MODEL: str = os.getenv("HALLUCINATION_MODEL", "")
    TASK_COMPLETION_MODEL: str = os.getenv("TASK_COMPLETION_MODEL", "")
    USER_FRUSTRATION_MODEL: str = os.getenv("USER_FRUSTRATION_MODEL", "")
    CONFUSION_LOOP_MODEL: str = os.getenv("CONFUSION_LOOP_MODEL", "")
    TASK_UNDERSTANDING_FAILURE_MODEL: str = os.getenv("TASK_UNDERSTANDING_FAILURE_MODEL", "")
    SYCOPHANCY_SIGNAL_MODEL: str = os.getenv("SYCOPHANCY_SIGNAL_MODEL", "")
    OFF_TOPIC_DRIFT_MODEL: str = os.getenv("OFF_TOPIC_DRIFT_MODEL", "")


settings = Settings()
