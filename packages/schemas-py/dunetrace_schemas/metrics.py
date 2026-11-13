"""
Prometheus metrics and readiness helpers shared by every Dunetrace service.

Six processes need a scrape endpoint and a readiness check: two FastAPI
services (ingest, customer API) and four asyncio workers with no HTTP server of
their own (detector, alerts, semantic, integrations/elevenlabs). This module is
the one implementation they share; it lives here because ``dunetrace_schemas``
is on every service's PYTHONPATH and in every image. The SDK never imports it —
``prometheus_client`` is a service dependency, not an SDK one.

``prometheus_client`` is OPTIONAL: when it is missing every factory returns a
no-op object with the same ``labels`` / ``inc`` / ``dec`` / ``set`` / ``observe``
surface, ``AVAILABLE`` is ``False``, ``render()`` returns an explanatory text
body, and the service boots normally (one WARNING). Observability must never be
the reason a service does not start.

Public API
----------
``counter(name, doc, labelnames=())`` / ``gauge(...)`` / ``histogram(..., buckets=None)``
    Metric factories. Names are validated to start with ``dunetrace_``.
    Registering the same name twice returns the existing metric (tests build
    the app more than once per process; prometheus itself would raise).
``register_standard(service, build_version)``
    Registers ``dunetrace_build_info{service,version}`` (= 1) and
    ``dunetrace_schema_version{service}`` (set later via ``set_schema_version``).
``set_schema_version(version)``
    Updates the schema-version gauge for the registered service.
``render() -> (body: bytes, content_type: str)``
    The exposition payload for ``GET /metrics``.
``db_ready(pool, minimum_schema_version) -> (ok, info)``
    ``SELECT 1`` + the applied schema version (read, never created) + pool sizes.
``start_metrics_server(port, ready_check, loop=None, host="0.0.0.0")``
    A stdlib HTTP server on a daemon thread serving ``/metrics``, ``/ready`` and
    ``/health`` for the workers. ``port=0`` disables it. Never raises.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
import weakref
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Awaitable, Callable, Dict, Optional, Sequence, Tuple, Union

logger = logging.getLogger("dunetrace.metrics")

try:  # pragma: no cover - exercised through AVAILABLE in tests
    from prometheus_client import (  # type: ignore[import-not-found]
        CONTENT_TYPE_LATEST,
        REGISTRY,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    AVAILABLE = True
except ImportError:  # pragma: no cover
    AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    REGISTRY = None  # type: ignore[assignment]
    CollectorRegistry = None  # type: ignore[assignment,misc]
    Counter = Gauge = Histogram = None  # type: ignore[assignment,misc]

    def generate_latest(registry: Any = None) -> bytes:  # type: ignore[misc]
        return b""


METRIC_PREFIX = "dunetrace_"

ReadyCheck = Union[
    Callable[[], Tuple[bool, dict]],
    Callable[[], Awaitable[Tuple[bool, dict]]],
]

_warned_unavailable = False


class NoopMetric:
    """Stand-in for a prometheus metric when the client library is absent.

    Every method a caller might chain returns ``self`` or ``None`` so call sites
    never branch on availability.
    """

    def labels(self, *args: Any, **kwargs: Any) -> "NoopMetric":
        return self

    def inc(self, amount: float = 1) -> None:
        return None

    def dec(self, amount: float = 1) -> None:
        return None

    def set(self, value: float) -> None:
        return None

    def observe(self, value: float) -> None:
        return None

    def set_to_current_time(self) -> None:
        return None

    def time(self) -> "NoopMetric":
        return self

    def __enter__(self) -> "NoopMetric":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


_NOOP = NoopMetric()

# registry -> {name: metric}, so a second registration returns the first.
# Keyed on the registry OBJECT via a weak map, not id(): a dict keyed on
# id(reg) never held a reference to the registry, so once one was collected
# CPython could hand its address to the next and _make would return a metric
# bound to the DEAD registry — registered against nothing, and absent from the
# new registry's output. Measured at 3 stale hits in 200 iterations. Production
# never hit it (every service passes registry=None and shares the module-level
# REGISTRY singleton, which never dies); the exposure was tests, which build a
# fresh CollectorRegistry per case and would have started failing on a runner
# change or under xdist with no code change at all.
_registered: "weakref.WeakKeyDictionary[Any, Dict[str, Any]]" = weakref.WeakKeyDictionary()
_lock = threading.Lock()

# Standard metrics for this process (set by register_standard).
_service_name: Optional[str] = None
_schema_gauge: Any = None


def _check_name(name: str) -> None:
    if not name.startswith(METRIC_PREFIX):
        raise ValueError(f"metric name {name!r} must start with {METRIC_PREFIX!r}")


def _warn_unavailable() -> None:
    global _warned_unavailable
    if not _warned_unavailable:
        _warned_unavailable = True
        logger.warning(
            "prometheus_client is not installed; /metrics will report that and every "
            "metric is a no-op. pip install prometheus-client (or the dunetrace-schemas "
            "[metrics] extra) to enable scraping."
        )


def _make(
    kind: Any, name: str, doc: str, labelnames: Sequence[str], registry: Any, **kw: Any
) -> Any:
    _check_name(name)
    if not AVAILABLE:
        _warn_unavailable()
        return _NOOP
    reg = registry if registry is not None else REGISTRY
    with _lock:
        by_name = _registered.get(reg)
        if by_name is None:
            by_name = {}
            _registered[reg] = by_name
        existing = by_name.get(name)
        if existing is not None:
            return existing
        metric = kind(name, doc, tuple(labelnames), registry=reg, **kw)
        by_name[name] = metric
        return metric


def counter(name: str, doc: str, labelnames: Sequence[str] = (), *, registry: Any = None) -> Any:
    """A monotonically increasing counter. Name it ``dunetrace_<thing>_total``."""
    return _make(Counter, name, doc, labelnames, registry)


def gauge(name: str, doc: str, labelnames: Sequence[str] = (), *, registry: Any = None) -> Any:
    """A value that goes up and down (backlog, version, saturation flag)."""
    return _make(Gauge, name, doc, labelnames, registry)


def histogram(
    name: str,
    doc: str,
    labelnames: Sequence[str] = (),
    *,
    buckets: Optional[Sequence[float]] = None,
    registry: Any = None,
) -> Any:
    """A distribution (latencies in seconds: ``dunetrace_<thing>_seconds``)."""
    kw: Dict[str, Any] = {}
    if buckets is not None:
        kw["buckets"] = tuple(buckets)
    return _make(Histogram, name, doc, labelnames, registry, **kw)


# Latency buckets in seconds that suit both a ~20ms DB write and a multi-second
# external call; services pass their own when they need something else.
DEFAULT_LATENCY_BUCKETS: Tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)


def register_standard(service: str, build_version: str, *, registry: Any = None) -> None:
    """Register the two metrics every service exposes.

    ``dunetrace_build_info{service,version} = 1`` identifies the running build
    (the ``info``-style gauge Prometheus conventions use for that) and
    ``dunetrace_schema_version{service}`` reports the migration version the
    service found at startup — call :func:`set_schema_version` once known.
    """
    global _service_name, _schema_gauge
    _service_name = service
    build = gauge(
        "dunetrace_build_info",
        "Build identity of the running service (always 1; see labels).",
        ("service", "version"),
        registry=registry,
    )
    build.labels(service=service, version=build_version or "unknown").set(1)
    _schema_gauge = gauge(
        "dunetrace_schema_version",
        "Shared-schema migration version the service observed at startup.",
        ("service",),
        registry=registry,
    )
    _schema_gauge.labels(service=service).set(0)


def set_schema_version(version: int) -> None:
    """Update the schema-version gauge; a no-op before ``register_standard``."""
    if _schema_gauge is None or _service_name is None:
        return
    try:
        _schema_gauge.labels(service=_service_name).set(int(version))
    except Exception:  # never let telemetry raise into a startup path
        logger.debug("set_schema_version failed", exc_info=True)


def render(*, registry: Any = None) -> Tuple[bytes, str]:
    """The ``/metrics`` payload. Text explaining the situation when the client
    library is missing, so a scrape still returns 200 with a readable reason."""
    if not AVAILABLE:
        return (
            b"# prometheus_client is not installed in this image; no metrics are collected.\n",
            "text/plain; charset=utf-8",
        )
    reg = registry if registry is not None else REGISTRY
    return generate_latest(reg), CONTENT_TYPE_LATEST


# ── Readiness ────────────────────────────────────────────────────────────────


async def db_ready(pool: Any, minimum_schema_version: int) -> Tuple[bool, dict]:
    """Readiness of the database this service depends on.

    Acquires a connection, runs ``SELECT 1`` and reads the highest applied
    migration from ``schema_version`` — a read, never a ``CREATE`` (a readiness
    probe must not do DDL; a missing table simply means "not migrated" and
    reports not-ready). ``ok`` is False when the connection fails or the
    version is below ``minimum_schema_version``. Works with any pool exposing
    ``acquire()`` as an async context manager yielding a connection with
    ``fetchval`` (asyncpg's does).
    """
    info: Dict[str, Any] = {
        "db": "no_pool",
        "schema_version": None,
        "required": int(minimum_schema_version),
    }
    if pool is None:
        return False, info
    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
            version = await conn.fetchval("SELECT COALESCE(MAX(version), 0) FROM schema_version")
    except Exception as exc:
        info["db"] = type(exc).__name__
        return False, info
    info["db"] = "ok"
    info["schema_version"] = int(version or 0)
    sizes: Dict[str, Any] = {}
    for label, attr in (
        ("size", "get_size"),
        ("min", "get_min_size"),
        ("max", "get_max_size"),
        ("idle", "get_idle_size"),
    ):
        getter = getattr(pool, attr, None)
        if callable(getter):
            try:
                sizes[label] = getter()
            except Exception:
                pass
    if sizes:
        info["pool"] = sizes
    return info["schema_version"] >= info["required"], info


def _run_ready_check(
    check: ReadyCheck, loop: Optional[asyncio.AbstractEventLoop]
) -> Tuple[bool, dict]:
    """Call a sync or async readiness check from the HTTP server thread."""
    try:
        result: Any = check()
        if inspect.isawaitable(result):
            if loop is None or loop.is_closed():
                # Nothing to run the coroutine on: close it and say so.
                result.close()  # type: ignore[union-attr]
                return False, {"error": "no event loop for the readiness check"}
            future = asyncio.run_coroutine_threadsafe(result, loop)
            try:
                result = future.result(timeout=5.0)
            except Exception:
                # concurrent.futures.Future.result(timeout) raises but leaves the
                # future PENDING and the coroutine scheduled. Without this cancel
                # every timed-out probe left a db_ready() call queued on a blocked
                # loop; when the loop resumed they all ran at once and each took
                # one of the worker's five pool connections, contending with real
                # work, with nothing bounding how many had piled up.
                future.cancel()
                raise
        ok, info = result
        return bool(ok), dict(info)
    except Exception as exc:
        return False, {"error": type(exc).__name__, "detail": str(exc)[:200]}


class _Handler(BaseHTTPRequestHandler):
    # Set per server instance via the server object (see start_metrics_server).
    server: "MetricsServer"

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        path = self.path.split("?", 1)[0]
        if path == "/metrics":
            body, content_type = render(registry=self.server.registry)
            self._send(200, body, content_type)
        elif path == "/ready":
            ok, info = _run_ready_check(self.server.ready_check, self.server.loop)
            payload = {"status": "ok" if ok else "not_ready", **info}
            self._send(200 if ok else 503, json.dumps(payload).encode(), "application/json")
        elif path == "/health":
            self._send(200, b'{"status":"ok"}', "application/json")
        else:
            self._send(404, b"not found\n", "text/plain; charset=utf-8")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - http.server API
        logger.debug("metrics server: " + format, *args)


class MetricsServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    # ThreadingMixIn.block_on_close defaults to True, so server_close() joins
    # every in-flight request thread. Both shutdown() and server_close() are
    # called from an async def's finally — i.e. ON the event loop thread — while
    # an in-flight /ready handler is blocked waiting for that same loop to run
    # its coroutine. Neither side could proceed until the 5s probe timeout
    # expired, so SIGTERM stalled the worker's exit by that long against a
    # typical 10s container stop grace, delaying the pool close with it.
    # daemon_threads=True alone does not prevent the join; this does.
    block_on_close = False

    def __init__(
        self,
        address: Tuple[str, int],
        ready_check: ReadyCheck,
        loop: Optional[asyncio.AbstractEventLoop],
        registry: Any,
    ) -> None:
        super().__init__(address, _Handler)
        self.ready_check = ready_check
        self.loop = loop
        self.registry = registry

    @property
    def port(self) -> int:
        return int(self.server_address[1])


def start_metrics_server(
    port: int,
    ready_check: ReadyCheck,
    loop: Optional[asyncio.AbstractEventLoop] = None,
    *,
    host: str = "0.0.0.0",
    registry: Any = None,
) -> Optional[MetricsServer]:
    """Serve ``/metrics``, ``/ready`` and ``/health`` from a daemon thread.

    For the asyncio workers, which have no HTTP server of their own. ``loop``
    is the worker's event loop when ``ready_check`` is a coroutine function
    (the check is scheduled onto it and awaited with a short timeout, so a
    wedged loop reads as not-ready rather than hanging the probe). ``port=0``
    disables the server. Binds every interface by default so compose
    healthchecks and an in-network scraper can reach it — the ports are not
    published by compose. Returns the server (``.port``, ``.shutdown()``) or
    ``None``; never raises, because metrics are observability, not a
    dependency.
    """
    if not port:
        logger.info("Metrics server disabled (port 0).")
        return None
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
    try:
        server = MetricsServer((host, int(port)), ready_check, loop, registry)
    except Exception as exc:
        logger.warning("Metrics server could not bind %s:%s: %s", host, port, exc)
        return None
    thread = threading.Thread(target=server.serve_forever, name="dunetrace-metrics", daemon=True)
    thread.start()
    logger.info("Metrics server listening on %s:%d (/metrics, /ready, /health)", host, server.port)
    return server
