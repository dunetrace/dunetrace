"""
Pluggable batch-shipping strategies for the SDK's drain thread.

BatchingEmitter is the seam between "the drain thread pulled a batch of
events off the ring buffer" and "deliver them somewhere." HttpBatchingEmitter
(the default) posts to the ingest API; Noop/Console/File exist for local
debugging, offline/audit trails, and explicitly disabling network shipping;
DurableRetryEmitter wraps any of them to survive a backend outage across
process restarts.

No external dependencies — matches the rest of the core SDK. sqlite3 is
stdlib, not a dependency.
"""

from __future__ import annotations

import email.utils
import http.client
import json
import logging
import os
import random
import sqlite3
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from threading import Lock
from typing import Callable, Deque, Dict, Iterator, List, NamedTuple, Optional, TypeVar

from dunetrace.models import AgentEvent, EventType

logger = logging.getLogger("dunetrace")

try:
    from importlib.metadata import PackageNotFoundError as _PkgNotFoundError
    from importlib.metadata import version as _pkg_version

    _SDK_VERSION = _pkg_version("dunetrace")
except _PkgNotFoundError:
    _SDK_VERSION = "0.0.0"  # running from source without installing

# Every outbound request identifies itself. Without this, urllib's default
# "Python-urllib/x.y" User-Agent gets fingerprinted and blocked (HTTP 403) by
# Cloudflare's bot protection in front of app.dunetrace.com — confirmed via a
# direct request with that exact UA string returning Cloudflare error 1010.
USER_AGENT = f"dunetrace-sdk-py/{_SDK_VERSION}"


class ShipOutcome(Enum):
    """What one ship() attempt concluded about a batch.

    A bare bool cannot carry the one distinction a retry layer needs: whether
    a later attempt could succeed. A 503 and a 401 both used to return False,
    so DurableRetryEmitter persisted the 401 batch to disk and re-sent it every
    30s forever — head-of-line blocking everything behind it, because
    _maybe_retry_backlog stops at the first failure to preserve order — while
    HttpBatchingEmitter had already logged that same batch as "events dropped".

    - DELIVERED: accepted by the destination, or deliberately discarded
      (NoopBatchingEmitter), or durably queued by a layer that owns the retry.
      Either way the caller's buffer is free to let go of it.
    - RETRYABLE: failed for a reason another attempt could fix — 5xx, 429, 408,
      a connection error, a timeout, a full disk.
    - REJECTED: the destination refused the batch itself — 400, 401, 403, 413,
      422. A retry sends the identical bytes to the identical endpoint and
      fails identically, so the only honest outcome is to drop it.

    Truthy for DELIVERED only, so the historical ``if emitter.ship(batch):``
    reads the same and a third-party emitter that still returns a plain bool
    keeps working (see ``as_ship_outcome``).
    """

    DELIVERED = "delivered"
    RETRYABLE = "retryable"
    REJECTED = "rejected"

    def __bool__(self) -> bool:
        return self is ShipOutcome.DELIVERED


def as_ship_outcome(value: object) -> ShipOutcome:
    """Normalise a ship() return value.

    ``emitter=`` is a public extension point and the contract was a bool, so a
    third-party emitter may still return one. False becomes RETRYABLE, not
    REJECTED: an emitter that cannot tell us a batch is permanently refused
    must not have it silently dropped on its behalf.
    """
    if isinstance(value, ShipOutcome):
        return value
    return ShipOutcome.DELIVERED if value else ShipOutcome.RETRYABLE


class BatchingEmitter(ABC):
    """Strategy for delivering one drained batch of events.

    ship() must never raise — catch and log internally, returning a
    ShipOutcome so a caller (e.g. a durable-retry wrapper) can react without
    needing to parse exceptions. DELIVERED means "handled" — for
    NoopBatchingEmitter that means "intentionally discarded," not "delivered."
    RETRYABLE and REJECTED are both failures; only RETRYABLE is worth queueing.

    A bare ``True``/``False`` is still accepted from third-party emitters and
    normalised by ``as_ship_outcome``, but every built-in returns the enum.

    This was documented but not enforced anywhere, and an emitter is a public
    extension point (``Dunetrace(emitter=...)``). Every built-in implementation
    below now catches; ``Dunetrace._ship`` and the drain loop catch again, so a
    third-party emitter that breaks the contract costs one warning line rather
    than the ``dunetrace-drain`` thread — whose death silently ends all
    observability for the process, with nothing in the log but a raw
    ``threading.excepthook`` traceback.
    """

    @abstractmethod
    def ship(self, batch: List[AgentEvent]) -> ShipOutcome:
        raise NotImplementedError

    def retry_pending(self) -> int:
        """Attempt any batches this emitter is holding for retry, without a
        new batch to ship. The drain thread calls this when the buffer is
        empty, so a retry that came due while the agent is quiet is not stuck
        waiting for the next event. Returns the number of batches delivered.
        Emitters that never hold anything back keep this default.
        """
        return 0


# ── HTTP failure classification ─────────────────────────────────────────────────

_RETRYABLE_4XX = frozenset({408, 429})


# ── Serialisation that cannot take a batch down with it ─────────────────────────

# Marker payload for an event whose own payload could not be encoded. The
# server sees the event (its type, run_id and step_index are what a run is
# reconstructed from) and a note saying what was lost.
SERIALIZATION_ERROR_KEY = "_dunetrace_serialization_error"


def _event_dicts(batch: List[AgentEvent]) -> List[dict]:
    """``to_dict()`` for every event in *batch*, skipping any that raises."""
    out: List[dict] = []
    for event in batch:
        try:
            out.append(event.to_dict())
        except Exception as exc:  # pragma: no cover - AgentEvent.to_dict is total
            logger.warning("Dunetrace: dropping an event that could not be read: %s", exc)
    return out


def _repair_event_dicts(dicts: List[dict]) -> List[dict]:
    """Replace the payload of any event dict ``json.dumps`` cannot encode.

    Called only after a whole-batch encode has already failed, so the happy
    path pays nothing. One unserialisable value in one payload — a circular
    reference, a non-string dict key — must cost that payload, not the 99
    unrelated events (other runs' terminals among them) that shared its batch.
    """
    repaired: List[dict] = []
    for d in dicts:
        try:
            json.dumps(d, default=str)
            repaired.append(d)
            continue
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"[:500]
        shell = dict(d)
        shell["payload"] = {SERIALIZATION_ERROR_KEY: detail}
        try:
            json.dumps(shell, default=str)
        except Exception:  # pragma: no cover - the shell is plain scalars
            logger.warning("Dunetrace: dropping an unserialisable event (%s)", detail)
            continue
        logger.warning(
            "Dunetrace: event payload could not be serialised (%s) — shipping the event "
            "without it. Check what is being passed to external_signal()/tool_called().",
            detail,
        )
        repaired.append(shell)
    return repaired


def _encode_events(dicts: List[dict]) -> str:
    """JSON for a list of event dicts, repairing individual events if needed."""
    try:
        return json.dumps(dicts, default=str)
    except Exception:
        return json.dumps(_repair_event_dicts(dicts), default=str)


def _is_retryable_status(status: int) -> bool:
    """5xx, 429 (rate limited) and 408 (request timeout) are transient. Every
    other non-2xx is the server saying the request itself is wrong — bad key,
    oversized body, malformed events — and a retry would fail identically."""
    return status >= 500 or status in _RETRYABLE_4XX


def _parse_retry_after(value: object) -> Optional[float]:
    """Seconds from a Retry-After header: delta-seconds or an HTTP-date.
    None when absent or unparseable. The str guard matters for tests: a
    MagicMock response supports __float__, so without it a mocked header
    would silently read as 1.0."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


class _Attempt(NamedTuple):
    """Outcome of one POST. `ok` is a 2xx. Otherwise `retryable` classifies
    the failure, `status` is the HTTP status (None for connection errors),
    `retry_after` the server's Retry-After in seconds when it sent one, and
    `unreachable` marks a refused connection so the log can say how to start
    the backend."""

    ok: bool
    retryable: bool = False
    status: Optional[int] = None
    retry_after: Optional[float] = None
    error: str = ""
    unreachable: bool = False


@dataclass(slots=True)
class _PendingBatch:
    batch: List[AgentEvent]
    attempts: int = 0  # failed attempts so far
    due_at: float = 0.0  # clock() time of the next attempt
    last_error: str = ""


_T = TypeVar("_T", int, float)


def _setting(explicit: Optional[_T], env_name: str, default: _T, cast: Callable[[str], _T]) -> _T:
    """Constructor kwarg wins; else the env var; else the default. A
    non-numeric env value is ignored with a warning rather than crashing the
    client at construction — a typo in deployment config must not take the
    agent down."""
    if explicit is not None:
        return explicit
    raw = os.environ.get(env_name)
    if raw is None or not raw.strip():
        return default
    try:
        return cast(raw.strip())
    except ValueError:
        logger.warning("Ignoring %s=%r (not a number); using %s", env_name, raw, default)
        return default


class HttpBatchingEmitter(BatchingEmitter):
    """POSTs each batch to the ingest API. The default emitter.

    Usable standalone (not just via Dunetrace):
    HttpBatchingEmitter("http://localhost:8001", api_key="...").

    **Failure handling.** A 2xx is success. Anything else — a non-2xx status,
    a connection error, a timeout — makes the attempt a failure. Failures are
    classified, and the classification is what ship() returns:

    - *retryable*: 5xx, 429, 408, and connection/timeout errors. The batch is
      queued in memory and re-sent with exponential backoff (`backoff_base_s`
      doubling per attempt, jittered, capped at `backoff_cap_s`), up to
      `max_retries` times. A `Retry-After` on 503/429 is the minimum delay
      before the next attempt (still capped). When retries are exhausted the
      batch is dropped with one WARNING carrying the event count and the
      last status/error.
    - *permanent*: any other 4xx (400, 401, 403, 413, 422 ...). A retry would
      fail identically, so the batch is dropped immediately with a WARNING and
      ship() returns ``ShipOutcome.REJECTED`` — the signal a wrapper needs to
      not persist it either.

    The queue is bounded by `max_queued_events`; over that, the oldest queued
    batches are dropped with a rate-limited WARNING (once per minute, with the
    count dropped since the last one). Nothing here ever sleeps: backoff is
    scheduled through each batch's due time and attempted on the next ship()
    or retry_pending() call, so the drain thread is never blocked for longer
    than one HTTP timeout and the agent thread is never involved.

    Each knob can also be set by env var when the emitter is constructed for
    you by Dunetrace(): DUNETRACE_SHIP_MAX_RETRIES, DUNETRACE_SHIP_BACKOFF_BASE_S,
    DUNETRACE_SHIP_BACKOFF_CAP_S, DUNETRACE_SHIP_MAX_QUEUED_EVENTS. An explicit
    kwarg wins over the env var.

    A ``RETRYABLE`` return from ship() therefore does **not** mean the batch is
    lost — it is queued unless retry is disabled (`max_retries=0`, or
    disable_retry() called by a wrapper that owns retry, as DurableRetryEmitter
    does). A wrapper that re-sends on a failure must call disable_retry() or the
    batch is delivered twice.

    `clock` is injected so tests can drive backoff deterministically; it must
    be monotonic in production (the default is time.monotonic).
    """

    DEFAULT_MAX_RETRIES = 5
    DEFAULT_BACKOFF_BASE_S = 1.0
    DEFAULT_BACKOFF_CAP_S = 30.0
    DEFAULT_MAX_QUEUED_EVENTS = 10_000
    _WARNING_INTERVAL_S = 60.0

    def __init__(
        self,
        endpoint: str,
        api_key: str = "",
        *,
        max_retries: Optional[int] = None,
        backoff_base_s: Optional[float] = None,
        backoff_cap_s: Optional[float] = None,
        max_queued_events: Optional[int] = None,
        timeout_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ingest_url = endpoint.rstrip("/") + "/v1/ingest"
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._clock = clock

        self._max_retries = max(
            0, _setting(max_retries, "DUNETRACE_SHIP_MAX_RETRIES", self.DEFAULT_MAX_RETRIES, int)
        )
        self._backoff_base_s = max(
            0.0,
            _setting(
                backoff_base_s, "DUNETRACE_SHIP_BACKOFF_BASE_S", self.DEFAULT_BACKOFF_BASE_S, float
            ),
        )
        self._backoff_cap_s = max(
            0.0,
            _setting(
                backoff_cap_s, "DUNETRACE_SHIP_BACKOFF_CAP_S", self.DEFAULT_BACKOFF_CAP_S, float
            ),
        )
        self._max_queued_events = max(
            0,
            _setting(
                max_queued_events,
                "DUNETRACE_SHIP_MAX_QUEUED_EVENTS",
                self.DEFAULT_MAX_QUEUED_EVENTS,
                int,
            ),
        )

        self._lock = Lock()
        self._queue: Deque[_PendingBatch] = deque()
        self._queued_events = 0
        self._retry_delegated = False

        # Rate-limited WARNING state — same once-per-minute pattern as
        # DurableRetryEmitter's eviction warning. None, not 0.0, for "never
        # warned": the clock's epoch is unspecified and may start under 60.
        self._failures_since_warning = 0
        self._last_failure_warning: Optional[float] = None
        self._dropped_since_warning = 0
        self._dropped_batches_since_warning = 0
        self._last_drop_warning: Optional[float] = None

    # ── Public surface ────────────────────────────────────────────────────────

    @property
    def retry_enabled(self) -> bool:
        return self._max_retries > 0 and not self._retry_delegated

    @property
    def pending_events(self) -> int:
        """Events currently held for retry."""
        with self._lock:
            return self._queued_events

    def disable_retry(self) -> None:
        """Hand retry to a wrapper. DurableRetryEmitter calls this when it
        wraps an HttpBatchingEmitter: exactly one layer may re-send a failed
        batch, or it is delivered twice — the wrapper sees ship() return
        False and queues the batch; this emitter must then not also queue it.
        Meant to be called before the first ship(); anything already queued
        here keeps being retried by this emitter, since the wrapper never saw
        it."""
        self._retry_delegated = True

    def ship(self, batch: List[AgentEvent]) -> ShipOutcome:
        try:
            self._retry_due()
            result = self._post(batch)
            if result.ok:
                return ShipOutcome.DELIVERED
            self._on_failure(_PendingBatch(batch), result, from_queue=False)
            # _on_failure has already logged a non-retryable rejection as
            # "events dropped". Saying so here is what lets a wrapper stop
            # persisting a batch the ingest API will refuse identically forever.
            return ShipOutcome.RETRYABLE if result.retryable else ShipOutcome.REJECTED
        except Exception as exc:
            logger.warning("HttpBatchingEmitter: ship failed unexpectedly: %s", exc, exc_info=True)
            # An unexpected exception here says nothing about the batch, so it
            # is classified as retryable — the outcome that loses no data.
            return ShipOutcome.RETRYABLE

    def retry_pending(self) -> int:
        return self._retry_due()

    # ── Transport ─────────────────────────────────────────────────────────────

    def _auth_headers(self) -> Dict[str, str]:
        """Authorization header for gateway-fronted deployments (e.g. the
        Dunetrace Cloud gateway's tenancy middleware, which resolves the
        calling org from ``Authorization: Bearer <api_key>`` and nothing
        else). Self-hosted ingest_svc (no gateway) still also accepts
        ``api_key`` in the request body, so this header is additive.
        """
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _request_body(self, batch: List[AgentEvent]) -> bytes:
        """The POST body for *batch*.

        A payload ``json.dumps`` cannot encode used to fail the whole request
        as non-retryable, discarding up to 100 events — other runs' terminals
        included — because of one bad value in one of them. The repair pass
        keeps every other event and strips only the payload that broke.
        """
        dicts = _event_dicts(batch)
        envelope = {
            "api_key": self._api_key,  # self-hosted ingest_svc compat; see _auth_headers
            "agent_id": batch[0].agent_id if batch else "",
            "events": dicts,
        }
        try:
            return json.dumps(envelope, default=str).encode()
        except Exception:
            envelope["events"] = _repair_event_dicts(dicts)
            return json.dumps(envelope, default=str).encode()

    def _post(self, batch: List[AgentEvent]) -> _Attempt:
        """One POST. Never raises; every outcome is an _Attempt."""
        try:
            payload = self._request_body(batch)
            req = urllib.request.Request(
                self._ingest_url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Dunetrace-Agent": batch[0].agent_id if batch else "",
                    "User-Agent": USER_AGENT,
                    **self._auth_headers(),
                },
                method="POST",
            )
        except Exception as exc:
            # Unserialisable payload, malformed URL: no retry can fix this.
            return _Attempt(ok=False, retryable=False, error=f"{type(exc).__name__}: {exc}")

        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                status = getattr(resp, "status", None)
                if isinstance(status, int) and 200 <= status < 300:
                    logger.debug("Shipped %d events. status=%d", len(batch), status)
                    return _Attempt(ok=True, status=status)
                if not isinstance(status, int):
                    return _Attempt(ok=False, retryable=False, error="response carried no status")
                headers = getattr(resp, "headers", None)
                retry_after = _parse_retry_after(
                    headers.get("Retry-After") if headers is not None else None
                )
                return _Attempt(
                    ok=False,
                    retryable=_is_retryable_status(status),
                    status=status,
                    retry_after=retry_after,
                    error=f"HTTP {status}",
                )
        except urllib.error.HTTPError as exc:
            # Subclass of URLError/OSError — must be caught before them.
            status = exc.code
            retry_after = _parse_retry_after(
                exc.headers.get("Retry-After") if exc.headers is not None else None
            )
            error = f"HTTP {status}"
            excerpt = _read_excerpt(exc)
            if excerpt:
                error = f"{error}: {excerpt}"
            return _Attempt(
                ok=False,
                retryable=_is_retryable_status(status),
                status=status,
                retry_after=retry_after,
                error=error,
            )
        except (OSError, http.client.HTTPException) as exc:
            # URLError (DNS, refused, connect timeout), socket read timeouts,
            # resets, SSL errors, RemoteDisconnected/IncompleteRead: transient.
            reason = getattr(exc, "reason", None)
            unreachable = isinstance(exc, ConnectionRefusedError) or isinstance(
                reason, ConnectionRefusedError
            )
            if not unreachable and "Connection refused" in str(exc):
                unreachable = True
            return _Attempt(
                ok=False,
                retryable=True,
                error=f"{type(exc).__name__}: {exc}",
                unreachable=unreachable,
            )
        except Exception as exc:
            return _Attempt(ok=False, retryable=False, error=f"{type(exc).__name__}: {exc}")

    # ── Retry queue ───────────────────────────────────────────────────────────

    def _retry_due(self) -> int:
        """Attempt queued batches that are due, oldest first. Stops at the
        first one that is not yet due or fails again — if the backend is still
        down every later batch will fail for the same reason, and order matters
        more than attempting all of them (same rule as DurableRetryEmitter).
        The HTTP call happens outside the lock; a failed batch goes back to the
        front so it stays oldest."""
        delivered = 0
        while True:
            with self._lock:
                if not self._queue:
                    return delivered
                pending = self._queue[0]
                if pending.due_at > self._clock():
                    return delivered
                self._queue.popleft()
                self._queued_events -= len(pending.batch)
            result = self._post(pending.batch)
            if result.ok:
                delivered += 1
                logger.debug(
                    "Re-sent %d events on attempt %d.", len(pending.batch), pending.attempts + 1
                )
                continue
            self._on_failure(pending, result, from_queue=True)
            return delivered

    def _on_failure(self, pending: _PendingBatch, result: _Attempt, *, from_queue: bool) -> None:
        pending.attempts += 1
        pending.last_error = result.error
        count = len(pending.batch)

        if not result.retryable:
            logger.warning(
                "Ingest rejected %d events (%s) — not retried, events dropped.%s",
                count,
                result.error,
                self._hint(result),
            )
            return

        # A wrapper owns retry for batches it handed us; ship() returns False
        # and it queues them. Batches already in our queue predate the handoff
        # and are ours to finish.
        if self._retry_delegated and not from_queue:
            self._note_transient_failure(result, count)
            return

        if pending.attempts > self._max_retries:
            logger.warning(
                "Dropping %d events after %d failed attempt(s); last error: %s.%s",
                count,
                pending.attempts,
                result.error,
                self._hint(result),
            )
            return

        delay = self._backoff_delay(pending.attempts, result.retry_after)
        pending.due_at = self._clock() + delay
        with self._lock:
            if from_queue:
                self._queue.appendleft(pending)
            else:
                self._queue.append(pending)
            self._queued_events += count
            self._evict_if_over_bound()
        self._note_transient_failure(result, count)
        logger.debug(
            "Failed to ship %d events (%s); retry %d/%d in %.1fs.",
            count,
            result.error,
            pending.attempts,
            self._max_retries,
            delay,
        )

    def _backoff_delay(self, attempts: int, retry_after: Optional[float]) -> float:
        """Exponential in the number of failed attempts, jittered into
        [delay, 2*delay) so a fleet that lost the backend together does not
        retry together, floored at the server's Retry-After, capped."""
        delay = self._backoff_base_s * (2 ** (attempts - 1))
        delay += random.uniform(0.0, delay)
        if retry_after is not None:
            delay = max(delay, retry_after)
        return min(delay, self._backoff_cap_s)

    def _evict_if_over_bound(self) -> None:
        """Caller holds self._lock."""
        dropped_batches = 0
        dropped_events = 0
        while self._queued_events > self._max_queued_events and self._queue:
            oldest = self._queue.popleft()
            self._queued_events -= len(oldest.batch)
            dropped_batches += 1
            dropped_events += len(oldest.batch)
        if not dropped_batches:
            return
        self._dropped_batches_since_warning += dropped_batches
        self._dropped_since_warning += dropped_events
        now = self._clock()
        if (
            self._last_drop_warning is None
            or now - self._last_drop_warning >= self._WARNING_INTERVAL_S
        ):
            logger.warning(
                "HttpBatchingEmitter: dropped %d oldest queued batch(es) (%d events) waiting "
                "for retry — over the %d-event retry-queue cap. This is data loss during an "
                "outage; wrap the emitter in DurableRetryEmitter to queue to disk instead.",
                self._dropped_batches_since_warning,
                self._dropped_since_warning,
                self._max_queued_events,
            )
            self._dropped_batches_since_warning = 0
            self._dropped_since_warning = 0
            self._last_drop_warning = now

    def _note_transient_failure(self, result: _Attempt, count: int) -> None:
        """One WARNING per minute summarising transient failures, not one per
        attempt: an outage with a chatty agent would otherwise log every
        drain cycle. Detail per attempt is at DEBUG."""
        self._failures_since_warning += 1
        now = self._clock()
        if (
            self._last_failure_warning is not None
            and now - self._last_failure_warning < self._WARNING_INTERVAL_S
        ):
            return
        with self._lock:
            queued = self._queued_events
        logger.warning(
            "Failed to ship %d events (%s); %d failed attempt(s) since last warning, "
            "%d events queued for retry.%s",
            count,
            result.error,
            self._failures_since_warning,
            queued,
            self._hint(result),
        )
        self._failures_since_warning = 0
        self._last_failure_warning = now

    def _hint(self, result: _Attempt) -> str:
        if not result.unreachable:
            return ""
        return (
            f"\n  DuneTrace backend not reachable at {self._ingest_url} — is it running?"
            "\n  Start it with: docker compose up -d"
        )


def _read_excerpt(exc: urllib.error.HTTPError, limit: int = 200) -> str:
    """First bytes of an error body, for the log line — a 401 or 422 usually
    says why. Best-effort: a body that cannot be read is simply omitted."""
    try:
        raw = exc.read(limit)
    except Exception:
        return ""
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace").replace("\n", " ").strip()


class NoopBatchingEmitter(BatchingEmitter):
    """Discards every batch. The explicit way to disable HTTP shipping —
    e.g. for SDK-only OTel or NDJSON deployments that don't run the ingest API.

    Usage::

        dt = Dunetrace(emitter=NoopBatchingEmitter())
    """

    def ship(self, batch: List[AgentEvent]) -> ShipOutcome:
        return ShipOutcome.DELIVERED


class ConsoleBatchingEmitter(BatchingEmitter):
    """Prints each event in a batch as a JSON line to stdout.

    Distinct from Dunetrace(emit_as_json=True): that writes one NDJSON line
    per event, synchronously, the instant each event is created (for
    Loki/Promtail). This writes once per drain cycle, batched — meant for ad
    hoc local debugging of what the drain thread is actually shipping.
    """

    def __init__(self) -> None:
        self._lock = Lock()

    def ship(self, batch: List[AgentEvent]) -> ShipOutcome:
        try:
            with self._lock:
                for d in _event_dicts(batch):
                    try:
                        line = json.dumps(d, default=str, separators=(",", ":"))
                    except Exception:
                        line = json.dumps(
                            _repair_event_dicts([d])[0], default=str, separators=(",", ":")
                        )
                    print(line)
            return ShipOutcome.DELIVERED
        except Exception as exc:
            logger.warning("ConsoleBatchingEmitter: failed to print a batch: %s", exc)
            return ShipOutcome.RETRYABLE


class FileBatchingEmitter(BatchingEmitter):
    """Appends each event in a batch as a JSON line to a local file.

    For offline/audit trails, or environments where neither the ingest API
    nor stdout capture is available. Opens and closes the file per batch
    rather than holding a handle open — the simplest way to stay safe across
    process forks and concurrent instances without extra coordination.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = Lock()

    def ship(self, batch: List[AgentEvent]) -> ShipOutcome:
        try:
            dicts = _event_dicts(batch)
            try:
                lines = [json.dumps(d, default=str, separators=(",", ":")) for d in dicts]
            except Exception:
                lines = [
                    json.dumps(d, default=str, separators=(",", ":"))
                    for d in _repair_event_dicts(dicts)
                ]
            with self._lock:
                with open(self._path, "a") as f:
                    for line in lines:
                        f.write(line + "\n")
            return ShipOutcome.DELIVERED
        except Exception as exc:
            logger.warning("FileBatchingEmitter: failed to write to %s: %s", self._path, exc)
            # A full or briefly unavailable disk is worth another attempt; the
            # wrapper, not this emitter, decides whether one happens.
            return ShipOutcome.RETRYABLE


def _batch_to_json(batch: List[AgentEvent]) -> str:
    """The batch as a JSON array. Falls back to per-event repair, never raises
    for a payload reason — see ``_encode_events``."""
    return _encode_events(_event_dicts(batch))


def _batch_from_json(payload: str) -> List[AgentEvent]:
    events = []
    for d in json.loads(payload):
        d = dict(d)
        d["event_type"] = EventType(d["event_type"])
        events.append(AgentEvent(**d))
    return events


DEFAULT_QUEUE_PATH = os.path.expanduser("~/.dunetrace/queue.db")


class DurableRetryEmitter(BatchingEmitter):
    """Wraps any BatchingEmitter to survive a backend outage across process
    restarts. On ship() failure, the batch is persisted to a local SQLite
    queue instead of dropped; on every ship() call, any due backlog is
    retried first (oldest batch first), before the current new batch.

    Composable with any inner emitter — most commonly HttpBatchingEmitter::

        dt = Dunetrace(emitter=DurableRetryEmitter(HttpBatchingEmitter(endpoint, api_key)))

    Queue location: `queue_path` constructor arg, else `DUNETRACE_QUEUE_PATH`
    env var, else `~/.dunetrace/queue.db`. The env var override matters for
    servers running as a non-root user with no HOME set.

    Bounded: evicts the oldest queued batch(es) once either
    `max_queue_events` or `max_queue_bytes` is exceeded — same "never grow
    unboundedly, drop oldest first" philosophy as the in-memory RingBuffer.
    Eviction is logged at WARNING, rate-limited to once per minute (with the
    count of batches evicted since the last warning) so a long outage doesn't
    spam logs while still surfacing that data is being lost silently.

    Only *retryable* failures are queued. A batch the destination refuses
    outright (``ShipOutcome.REJECTED`` — 400/401/403/413/422) is dropped here
    too, matching the inner emitter's own "events dropped" log line: a retry
    re-sends identical bytes to the identical endpoint, and because the backlog
    is drained strictly oldest-first and stops at the first failure, one such
    batch at the head blocks every deliverable batch behind it until the cap
    evicts them. A stale DUNETRACE_API_KEY used to fill 100 MB of the
    customer's disk this way and deliver nothing.

    Retry cadence is decoupled from the normal flush interval — draining the
    backlog on every drain cycle (default every 200ms) would hammer a
    still-down backend with a full-timeout connection attempt every cycle.
    Default: every 30s ± 5s jitter (prevents a thundering herd if many SDK
    instances come back online together after a shared outage). A retry
    attempt stops at the first failure rather than skipping ahead — if the
    backend is still down, later batches will fail for the same reason, and
    preserving order matters more than attempting all of them.
    """

    def __init__(
        self,
        inner: BatchingEmitter,
        *,
        queue_path: Optional[str] = None,
        max_queue_events: int = 100_000,
        max_queue_bytes: int = 100 * 1024 * 1024,  # 100MB
        retry_interval_s: float = 30.0,
        retry_jitter_s: float = 5.0,
        max_batches_per_retry: int = 50,
    ) -> None:
        self._inner = inner
        # Exactly one layer may re-send a failed batch. HttpBatchingEmitter
        # queues retryable failures in memory *and* returns False; left on,
        # both it and this disk queue would re-send the same batch. The disk
        # queue is strictly the better owner (survives a restart, larger
        # bound), so the inner one is switched off here.
        if isinstance(inner, HttpBatchingEmitter):
            inner.disable_retry()
        self._path = queue_path or os.environ.get("DUNETRACE_QUEUE_PATH") or DEFAULT_QUEUE_PATH
        self._max_queue_events = max_queue_events
        self._max_queue_bytes = max_queue_bytes
        self._retry_interval_s = retry_interval_s
        self._retry_jitter_s = retry_jitter_s
        self._max_batches_per_retry = max_batches_per_retry

        self._lock = Lock()
        self._next_retry_at = 0.0  # monotonic time; 0 == due immediately
        self._evicted_since_warning = 0
        self._last_eviction_warning: Optional[float] = None

        self._db_ok = self._init_db()

    def _init_db(self) -> bool:
        try:
            directory = os.path.dirname(self._path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with self._connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS queue (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        enqueued_at REAL NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        payload TEXT NOT NULL
                    )
                    """
                )
            return True
        except Exception as exc:
            # Deliberately broad. The failure this handler exists to absorb is
            # `sqlite3.OperationalError: unable to open database file`, and
            # sqlite3.Error does NOT derive from OSError — catching OSError
            # alone let the constructor raise straight into the customer's
            # agent process, so the agent failed to boot because Dunetrace
            # could not create an *optional* retry cache. Real triggers: a
            # read-only root filesystem, a non-root container user, $HOME
            # unset or unwritable (AWS Lambda, distroless, Kubernetes
            # readOnlyRootFilesystem) — the default path is ~/.dunetrace.
            # os.makedirs contributes OSError, sqlite3.connect/execute
            # contributes sqlite3.Error, and neither is a superset.
            logger.warning(
                "DurableRetryEmitter: could not initialize queue at %s: %s. "
                "Failed batches will be dropped instead of queued. Set "
                "DUNETRACE_QUEUE_PATH to a writable location to re-enable "
                "durable retry.",
                self._path,
                exc,
            )
            return False

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, timeout=5)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def ship(self, batch: List[AgentEvent]) -> ShipOutcome:
        try:
            self._maybe_retry_backlog()
            outcome = as_ship_outcome(self._inner.ship(batch))
            if outcome is ShipOutcome.DELIVERED:
                return ShipOutcome.DELIVERED
            if outcome is ShipOutcome.REJECTED:
                # The destination refused the batch itself — a stale API key,
                # an oversized body, malformed events. The inner emitter has
                # already logged it as dropped; persisting it would make that
                # log line a lie and, because the backlog is retried strictly
                # oldest-first and stops at the first failure, would wedge the
                # queue behind a batch that can never be accepted.
                return ShipOutcome.REJECTED
            # Queued to disk is as good as delivered from the ring buffer's
            # point of view: this layer now owns getting it there.
            return ShipOutcome.DELIVERED if self._enqueue(batch) else ShipOutcome.RETRYABLE
        except Exception as exc:
            logger.warning("DurableRetryEmitter: ship failed unexpectedly: %s", exc, exc_info=True)
            return ShipOutcome.RETRYABLE

    def retry_pending(self) -> int:
        """Idle hook from the drain thread: drain due backlog (on the same
        30s ± 5s cadence as ship()) even when no new batch arrives, so an
        agent that goes quiet after an outage still flushes its disk queue.
        Also gives the inner emitter its turn, for anything it holds itself."""
        return self._maybe_retry_backlog() + self._inner.retry_pending()

    def _enqueue(self, batch: List[AgentEvent]) -> bool:
        if not self._db_ok:
            return False
        try:
            # Inside the try: this is where the batch is serialised, and a
            # payload json.dumps cannot encode used to raise straight out of
            # ship() and take the drain thread with it.
            payload = _batch_to_json(batch)
            size = len(payload.encode())
            with self._lock, self._connection() as conn:
                conn.execute(
                    "INSERT INTO queue (enqueued_at, size_bytes, payload) VALUES (?, ?, ?)",
                    (time.time(), size, payload),
                )
                self._evict_if_over_bounds(conn)
            return True
        except Exception as exc:
            logger.warning("DurableRetryEmitter: failed to queue batch: %s", exc)
            return False

    def _evict_if_over_bounds(self, conn: sqlite3.Connection) -> None:
        count, total_bytes = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM queue"
        ).fetchone()
        evicted = 0
        while count > self._max_queue_events or total_bytes > self._max_queue_bytes:
            row = conn.execute(
                "SELECT id, size_bytes FROM queue ORDER BY id ASC LIMIT 1"
            ).fetchone()
            if row is None:
                break
            oldest_id, oldest_size = row
            conn.execute("DELETE FROM queue WHERE id = ?", (oldest_id,))
            count -= 1
            total_bytes -= oldest_size
            evicted += 1

        if evicted:
            self._evicted_since_warning += evicted
            now = time.monotonic()
            # time.monotonic()'s epoch is unspecified (often time-since-boot on
            # Linux) — comparing against a 0.0 sentinel for "never warned yet"
            # silently skips the first warning on a freshly booted machine
            # where monotonic() itself is still under 60. None is unambiguous
            # regardless of the clock's starting point.
            if self._last_eviction_warning is None or now - self._last_eviction_warning >= 60:
                logger.warning(
                    "DurableRetryEmitter: evicted %d oldest batch(es) from the disk queue "
                    "at %s (over the %d-event / %d-byte cap) — this is silent data loss "
                    "during a long outage.",
                    self._evicted_since_warning,
                    self._path,
                    self._max_queue_events,
                    self._max_queue_bytes,
                )
                self._evicted_since_warning = 0
                self._last_eviction_warning = now

    def _maybe_retry_backlog(self) -> int:
        """Returns the number of backlog batches delivered (0 when not due)."""
        now = time.monotonic()
        if now < self._next_retry_at:
            return 0
        self._next_retry_at = now + random.uniform(
            self._retry_interval_s - self._retry_jitter_s,
            self._retry_interval_s + self._retry_jitter_s,
        )
        if not self._db_ok:
            return 0

        delivered = 0
        discarded = 0
        try:
            with self._lock, self._connection() as conn:
                rows = conn.execute(
                    "SELECT id, payload FROM queue ORDER BY id ASC LIMIT ?",
                    (self._max_batches_per_retry,),
                ).fetchall()
                for row_id, payload in rows:
                    try:
                        batch = _batch_from_json(payload)
                    except Exception as exc:
                        # A row this build cannot decode (truncated write, a
                        # payload from a future schema) can never be delivered.
                        # Leaving it would block every batch behind it forever,
                        # since the loop below stops at the first failure.
                        logger.warning(
                            "DurableRetryEmitter: discarding an undecodable queued batch "
                            "(row %s) from %s: %s",
                            row_id,
                            self._path,
                            exc,
                        )
                        conn.execute("DELETE FROM queue WHERE id = ?", (row_id,))
                        discarded += 1
                        continue
                    outcome = as_ship_outcome(self._inner.ship(batch))
                    if outcome is ShipOutcome.DELIVERED:
                        conn.execute("DELETE FROM queue WHERE id = ?", (row_id,))
                        delivered += 1
                    elif outcome is ShipOutcome.REJECTED:
                        # Permanently refused — already logged as dropped by the
                        # inner emitter. Deleting it is what makes that true and
                        # unblocks the rest of the queue; keeping it would mean
                        # retrying a doomed batch every 30s until the 100k-event
                        # cap silently evicted the deliverable batches behind it.
                        conn.execute("DELETE FROM queue WHERE id = ?", (row_id,))
                        discarded += 1
                    else:
                        break  # still down — stop rather than skip ahead out of order
        except Exception as exc:
            # Broad, like _init_db: sqlite3.Error is not an OSError, the inner
            # emitter is a public extension point, and this runs on the drain
            # thread where an escape is silently swallowed at DEBUG.
            logger.warning("DurableRetryEmitter: retry attempt failed: %s", exc)
        if discarded:
            logger.debug(
                "DurableRetryEmitter: dropped %d undeliverable batch(es) from the disk queue.",
                discarded,
            )
        return delivered
