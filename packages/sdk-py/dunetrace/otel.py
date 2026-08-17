"""
OTel export bootstrap for Dunetrace.

Reads config from DUNETRACE_OTEL_* env vars, builds a TracerProvider with a
bounded async export pipeline, and hands back a tracer the SDK's span emitters
use (wired in Phase 2). Opt-in: does nothing unless DUNETRACE_OTEL_ENABLED is
set, so the SDK behaves exactly as before for anyone who doesn't configure it.

Failure isolation is the whole point of this module. A bad endpoint, a slow
collector, or a missing opentelemetry install must never raise into agent code
or block the agent thread:
  - export runs on the BatchSpanProcessor background thread (async)
  - the queue is bounded and drops spans on overflow
  - a circuit breaker stops export attempts for a cooldown after repeated
    failures, so a dead collector isn't retried on every batch
  - every entry point swallows exceptions and degrades to "OTel disabled"

Config (all optional except when enabling):
  DUNETRACE_OTEL_ENABLED         "1"/"true" to turn export on. Default off.
  DUNETRACE_OTEL_ENDPOINT        OTLP endpoint URL. Required when enabled.
  DUNETRACE_OTEL_HEADERS         "k1=v1,k2=v2" auth headers (e.g. "DD-API-KEY=xxx").
  DUNETRACE_OTEL_PROTOCOL        "grpc" (default) or "http/protobuf".
  DUNETRACE_OTEL_SERVICE_NAME    service.name resource attr. Default "dunetrace".
  DUNETRACE_OTEL_SAMPLING_RATIO  0.0-1.0 head sampling ratio. Default 1.0.
  DUNETRACE_ORG_ID               optional dunetrace.org_id resource attr. The
                                 authoritative org is resolved server-side from
                                 the API key; this is only a convenience label.

Standard OTEL_* env vars the exporter reads natively (e.g. OTEL_EXPORTER_OTLP_
INSECURE for plaintext gRPC to a local collector) still apply on top of this.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Callable, Deque, Dict, Optional

logger = logging.getLogger("dunetrace.otel")

try:
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SpanExporter,
        SpanExportResult,
    )
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

    _OTEL_AVAILABLE = True
except ImportError:
    # Core SDK stays zero-dependency; OTel ships as the `dunetrace[otel]` extra.
    # When it's not installed every public function here degrades to disabled.
    _OTEL_AVAILABLE = False
    SpanExporter = object  # type: ignore[assignment,misc]

_DEFAULT_SERVICE_NAME = "dunetrace"
_VALID_PROTOCOLS = ("grpc", "http/protobuf")

# Bounded export queue. BatchSpanProcessor drops spans once the queue is full,
# which is the backpressure behavior we want: never grow memory unbounded, never
# block the agent to wait on a slow collector.
_MAX_QUEUE_SIZE = 2048
_MAX_EXPORT_BATCH_SIZE = 512


# ── Env parsing helpers ─────────────────────────────────────────────────────────


def _sdk_version() -> str:
    try:
        return version("dunetrace")
    except PackageNotFoundError:
        return "0.0.0"  # running from source without installing


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _parse_headers(raw: str) -> Dict[str, str]:
    """Parse "k1=v1,k2=v2" into a dict. Tolerant: whitespace is stripped,
    entries without a '=' or an empty key are skipped rather than raising, so a
    malformed header string can't take down export init."""
    out: Dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        if key:
            out[key] = value.strip()
    return out


def _parse_ratio(raw: Optional[str], default: float = 1.0) -> float:
    """Parse a sampling ratio, clamped to [0.0, 1.0]. A non-numeric value falls
    back to the default (full sampling) rather than raising."""
    if raw is None:
        return default
    try:
        ratio = float(raw)
    except (TypeError, ValueError):
        logger.warning("DUNETRACE_OTEL_SAMPLING_RATIO=%r is not a number, using %s", raw, default)
        return default
    return min(1.0, max(0.0, ratio))


# ── Config ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OTelConfig:
    """Resolved OTel export configuration. Build from env with from_env()."""

    enabled: bool = False
    endpoint: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    protocol: str = "grpc"
    service_name: str = _DEFAULT_SERVICE_NAME
    service_version: str = ""
    sampling_ratio: float = 1.0
    org_id: str = ""
    # PII control for span content. Dunetrace is raw-by-default, so this is True
    # unless DUNETRACE_OTEL_CAPTURE_CONTENT is turned off. When False the exporter
    # drops content-bearing span attributes (tool args, request URL, retrieval
    # query) and keeps only metadata (names, counts, statuses, latencies). Voice
    # spans never carry raw transcript/TTS text either way, only counts.
    capture_content: bool = True

    @classmethod
    def from_env(cls, **overrides: object) -> "OTelConfig":
        """Read config from DUNETRACE_OTEL_* env vars. Keyword overrides win over
        the environment, so callers (and tests) can force any field."""
        protocol = os.environ.get("DUNETRACE_OTEL_PROTOCOL", "grpc").strip().lower()
        if protocol not in _VALID_PROTOCOLS:
            logger.warning(
                "DUNETRACE_OTEL_PROTOCOL=%r not one of %s, using 'grpc'",
                protocol,
                _VALID_PROTOCOLS,
            )
            protocol = "grpc"

        service_name = (
            os.environ.get("DUNETRACE_OTEL_SERVICE_NAME", _DEFAULT_SERVICE_NAME).strip()
            or _DEFAULT_SERVICE_NAME
        )

        resolved: Dict[str, object] = dict(
            enabled=_env_bool("DUNETRACE_OTEL_ENABLED", False),
            endpoint=os.environ.get("DUNETRACE_OTEL_ENDPOINT", "").strip(),
            headers=_parse_headers(os.environ.get("DUNETRACE_OTEL_HEADERS", "")),
            protocol=protocol,
            service_name=service_name,
            service_version=_sdk_version(),
            sampling_ratio=_parse_ratio(os.environ.get("DUNETRACE_OTEL_SAMPLING_RATIO")),
            org_id=os.environ.get("DUNETRACE_ORG_ID", "").strip(),
            capture_content=_env_bool("DUNETRACE_OTEL_CAPTURE_CONTENT", True),
        )
        resolved.update(overrides)
        return cls(**resolved)  # type: ignore[arg-type]


# ── Circuit breaker ─────────────────────────────────────────────────────────────


class _CircuitBreakerExporter(SpanExporter):
    """
    Wraps a SpanExporter so a dead or slow collector can't turn into an endless
    stream of failing export attempts.

    After FAILURE_THRESHOLD failed exports inside WINDOW seconds the circuit
    opens: exports are dropped without touching the wrapped exporter for
    COOLDOWN seconds, then it tries again. A single successful export closes the
    circuit and clears the failure count.

    Runs on the BatchSpanProcessor worker thread; the lock guards the failure
    state against force_flush()/shutdown() racing from another thread. `now` is
    injectable so tests can drive the cooldown clock deterministically.
    """

    FAILURE_THRESHOLD = 5
    WINDOW = 60.0
    COOLDOWN = 60.0

    def __init__(self, wrapped: object, now: Callable[[], float] = time.monotonic) -> None:
        self._wrapped = wrapped
        self._now = now
        self._failures: Deque[float] = deque()
        self._open_until = 0.0
        self._last_warn = 0.0
        self._lock = threading.Lock()

    def export(self, spans: object) -> "SpanExportResult":
        now = self._now()
        with self._lock:
            if now < self._open_until:
                return SpanExportResult.FAILURE  # circuit open, drop this batch

        try:
            result = self._wrapped.export(spans)  # type: ignore[attr-defined]
        except Exception as exc:  # a raising exporter must not reach agent code
            self._record_failure(now, str(exc))
            return SpanExportResult.FAILURE

        if result == SpanExportResult.SUCCESS:
            with self._lock:
                self._failures.clear()
        else:
            self._record_failure(now, "exporter returned FAILURE")
        return result

    def _record_failure(self, now: float, reason: str) -> None:
        with self._lock:
            self._failures.append(now)
            while self._failures and now - self._failures[0] > self.WINDOW:
                self._failures.popleft()
            tripped = len(self._failures) >= self.FAILURE_THRESHOLD
            if tripped:
                self._open_until = now + self.COOLDOWN
                self._failures.clear()
            should_warn = tripped or (now - self._last_warn > self.WINDOW)
            if should_warn:
                self._last_warn = now
        # Log outside the lock. Throttled to at most once per WINDOW so a
        # persistently-down collector doesn't flood the log.
        if should_warn:
            if tripped:
                logger.warning(
                    "Dunetrace OTel: export failing (%s). Circuit open for %.0fs; "
                    "spans dropped meanwhile. Agent unaffected.",
                    reason,
                    self.COOLDOWN,
                )
            else:
                logger.warning(
                    "Dunetrace OTel: span export failed (%s). Continuing; agent unaffected.",
                    reason,
                )

    def shutdown(self) -> None:
        try:
            self._wrapped.shutdown()  # type: ignore[attr-defined]
        except Exception as exc:
            logger.debug("Dunetrace OTel: exporter shutdown error: %s", exc)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        try:
            return bool(self._wrapped.force_flush(timeout_millis))  # type: ignore[attr-defined]
        except Exception:
            return False


# ── Provider construction ───────────────────────────────────────────────────────


def _build_span_exporter(config: OTelConfig) -> object:
    """Construct the OTLP exporter for the configured protocol. Raises on a
    genuinely broken setup (unknown protocol, missing exporter package) — the
    caller turns that into "OTel disabled", never a crash in agent code."""
    if config.protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as HTTPSpanExporter,
        )

        return HTTPSpanExporter(endpoint=config.endpoint, headers=config.headers or None)

    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as GRPCSpanExporter,
    )

    return GRPCSpanExporter(endpoint=config.endpoint, headers=config.headers or None)


def _build_resource(config: OTelConfig) -> "Resource":
    attrs: Dict[str, str] = {
        "service.name": config.service_name,
        "service.version": config.service_version,
    }
    if config.org_id:
        attrs["dunetrace.org_id"] = config.org_id
    return Resource.create(attrs)


def build_tracer_provider(config: OTelConfig) -> Optional[object]:
    """
    Build a TracerProvider from config, or return None when export should stay
    off (disabled, opentelemetry not installed, no endpoint, or init failed).

    Never raises: a broken export pipeline degrades to None so the SDK keeps
    running with OTel simply disabled.
    """
    if not config.enabled:
        return None
    if not _OTEL_AVAILABLE:
        _warn_once(
            "DUNETRACE_OTEL_ENABLED is set but opentelemetry is not installed; "
            "OTel export disabled. Install with: pip install 'dunetrace[otel]'"
        )
        return None
    if not config.endpoint:
        logger.warning(
            "DUNETRACE_OTEL_ENABLED is set but DUNETRACE_OTEL_ENDPOINT is empty; "
            "OTel export disabled."
        )
        return None

    try:
        exporter = _CircuitBreakerExporter(_build_span_exporter(config))
        processor = BatchSpanProcessor(
            exporter,
            max_queue_size=_MAX_QUEUE_SIZE,
            max_export_batch_size=_MAX_EXPORT_BATCH_SIZE,
        )
        provider = TracerProvider(
            resource=_build_resource(config),
            sampler=ParentBased(root=TraceIdRatioBased(config.sampling_ratio)),
        )
        provider.add_span_processor(processor)
        logger.debug(
            "Dunetrace OTel provider built: endpoint=%s protocol=%s sampling_ratio=%s",
            config.endpoint,
            config.protocol,
            config.sampling_ratio,
        )
        return provider
    except Exception as exc:
        logger.warning(
            "Dunetrace OTel: failed to initialize export pipeline (%s); OTel export disabled.",
            exc,
        )
        return None


# ── Module-global tracer ─────────────────────────────────────────────────────────
#
# One provider/tracer per process. The SDK's span emitters (Phase 2+) read the
# tracer through get_tracer(); a None tracer means "OTel off", so emitting a
# span is a cheap None-check on the hot path.

_state_lock = threading.Lock()
_provider: Optional[object] = None
_tracer: Optional[object] = None
_config: Optional[OTelConfig] = None
_warned: set = set()


def _warn_once(message: str) -> None:
    """Log a warning at most once per process, so a persistently misconfigured
    deployment doesn't repeat the same line on every client construction."""
    with _state_lock:
        if message in _warned:
            return
        _warned.add(message)
    logger.warning(message)


def init(config: Optional[OTelConfig] = None, **overrides: object) -> bool:
    """
    Idempotent bootstrap of the process-global tracer. Returns True when OTel
    export is active after the call, False otherwise.

    Reads config from env when none is passed. Never raises: any failure
    degrades to disabled so agent code is unaffected. Calling it again after a
    successful init is a no-op (the first provider wins).
    """
    global _provider, _tracer, _config
    cfg = config or OTelConfig.from_env(**overrides)
    with _state_lock:
        if _provider is not None:
            return True
    provider = build_tracer_provider(cfg)
    if provider is None:
        return False
    with _state_lock:
        if _provider is not None:  # lost an init race; keep the first provider
            try:
                provider.shutdown()  # type: ignore[attr-defined]
            except Exception:
                pass
            return True
        _provider = provider
        _tracer = provider.get_tracer(_DEFAULT_SERVICE_NAME)  # type: ignore[attr-defined]
        _config = cfg
    return True


def active_config() -> Optional[OTelConfig]:
    """The config OTel export was initialized with, or None when disabled.
    Lets the client read settings (e.g. capture_content) the exporter needs."""
    return _config


def get_tracer(name: str = _DEFAULT_SERVICE_NAME) -> Optional[object]:
    """The active Dunetrace tracer, or None when OTel export is disabled.
    Callers treat None as "do nothing" so span emission stays a cheap branch."""
    return _tracer


def is_enabled() -> bool:
    """True when OTel export is active (a provider was successfully built)."""
    return _tracer is not None


def get_tracer_provider() -> Optional[object]:
    """The active TracerProvider, or None when disabled. Mostly for tests."""
    return _provider


def shutdown() -> None:
    """Flush and tear down the export pipeline. Safe to call when disabled."""
    global _provider, _tracer, _config
    with _state_lock:
        provider = _provider
        _provider = None
        _tracer = None
        _config = None
    if provider is not None:
        try:
            provider.shutdown()  # type: ignore[attr-defined]
        except Exception as exc:
            logger.debug("Dunetrace OTel: provider shutdown error: %s", exc)


def _reset_for_tests() -> None:
    """Drop the global provider/tracer and warning state without flushing the
    pipeline. Test hook only."""
    global _provider, _tracer, _config
    with _state_lock:
        _provider = None
        _tracer = None
        _config = None
        _warned.clear()
