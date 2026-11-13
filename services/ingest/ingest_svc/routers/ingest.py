"""POST /v1/ingest — accepts event batches from the SDK.
GET  /v1/policies — returns runtime policies for the SDK to enforce.
GET  /v1/detector-config — returns the effective detector thresholds for the SDK's client-side pass.
"""

from __future__ import annotations

import logging
import os
import secrets
import time
import uuid

from fastapi import APIRouter, HTTPException, Query, Request, status

from dunetrace_schemas import metrics as _metrics
from dunetrace_schemas.scopes import ADMIN
from ingest_svc.auth import is_trusted
from ingest_svc.db import (
    get_event_store,
    get_pool,
    insert_deploy_event,
    scrub_old_signal_evidence,
    verify_api_key,
    create_api_key,
    fetch_policies,
    get_agent_quota,
    set_agent_quota,
)
from ingest_svc.detector_config import build_detector_config
from ingest_svc.schemas import (
    IngestRequest,
    IngestResponse,
    DeployRequest,
    DeployResponse,
    KeyCreateRequest,
    KeyCreateResponse,
    PruneEventsRequest,
    PruneEventsResponse,
    QuotaSetRequest,
    QuotaResponse,
)

logger = logging.getLogger("dunetrace.ingest")
router = APIRouter()

# Seconds the SDK is told to wait before re-sending a batch that could not be
# persisted. Small on purpose: the common cause is a transient DB blip, and the
# SDK's durable queue retries on its own ~30s cadence anyway.
_PERSIST_RETRY_AFTER_S = 5

# Persist-path metrics. They live here rather than in main.py because this
# module cannot import main (main imports the routers); the shared factories
# hand back the existing metric on a repeat registration, so tests that build
# the app more than once per process are fine. Request-level counters (every
# status, 429s, 413s) are main.py's — they must count middleware short-circuits
# that never reach a route.
_persist_failures = _metrics.counter(
    "dunetrace_ingest_persist_failures_total",
    "Batches rejected with 503 because they were not durably written.",
    ("reason",),  # "exception": the store raised; "shortfall": fewer rows than events
)
_persist_seconds = _metrics.histogram(
    "dunetrace_ingest_persist_seconds",
    "Wall-clock seconds of the _persist() call per /v1/ingest batch, success or failure.",
    buckets=_metrics.DEFAULT_LATENCY_BUCKETS,
)
_events_accepted = _metrics.counter(
    "dunetrace_ingest_events_accepted_total",
    "Events acknowledged with 202 (committed before the response was sent).",
)


class PersistError(Exception):
    """A batch was not durably written.

    Raised by _persist() — for a store exception or an insert shortfall — and
    translated by the /v1/ingest route into 503 + Retry-After, so the SDK
    re-sends the batch instead of treating a 202 as "safe to discard".

    ``reason`` is the ``dunetrace_ingest_persist_failures_total`` label:
    ``"exception"`` when the store raised, ``"shortfall"`` when it reported
    fewer rows written than events sent.
    """

    def __init__(self, message: str, *, reason: str = "exception") -> None:
        super().__init__(message)
        self.reason = reason


async def _resolve_org_id(request: Request, api_key: str) -> str:
    """Resolve the org_id for this request. Raises 401 if it can't be resolved.

    Trusted path: dunetrace-cloud's gateway has already authenticated the caller
    and forwards its org identity via a header — no api_keys lookup here.
    x-org-id is the current header name; x-customer-id is accepted as a fallback
    for callers running an older cloud gateway build (pre-v0.5.0 naming).

    Untrusted path (self-hosted): resolves org_id from the OSS api_keys table.
    Keys are org-scoped, not agent-scoped — a valid key may submit events for
    any agent_id under its org, discovered on first ingest.
    """
    if is_trusted(request):
        org_id = request.headers.get("x-org-id") or request.headers.get("x-customer-id", "")
        if not org_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Trusted request missing x-org-id",
            )
        return org_id

    # Deliberately NOT reusing set_org_context's earlier verify_api_key result,
    # even though it is the same function called with the same key one layer
    # up. Caching it on request.state would make this route's AUTHENTICATION
    # decision depend on middleware state: the middleware only parses the body
    # for two paths and swallows its own exceptions, so any future divergence
    # turns into a silent authorisation, and a caller rejected here could be
    # admitted by a stale cache. One extra indexed lookup on a hashed key is a
    # fair price for keeping the auth decision self-contained.
    org_id = await verify_api_key(api_key)
    if org_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key",
        )
    return org_id


@router.post(
    "/v1/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a batch of agent events",
)
async def ingest(request: Request, body: IngestRequest) -> IngestResponse:
    """202 is a promise: the batch is committed before the response is sent.

    Persistence used to run in a BackgroundTask after the 202, and _persist
    swallowed every failure — so a DB outage looked like success to the SDK,
    which then discarded its copy of the events. Now a batch that cannot be
    written comes back as 503 + Retry-After, and the SDK keeps (and re-sends)
    it. The extra request latency is one DB round-trip (~20ms), inside the
    SDK's background drain thread, never in the agent's own path.
    """
    org_id = await _resolve_org_id(request, body.api_key)

    batch_id = str(uuid.uuid4())
    n = len(body.events)

    t0 = time.perf_counter()
    try:
        await _persist(body.events, batch_id, org_id)
    except PersistError as exc:
        _persist_failures.labels(reason=exc.reason).inc()
        # The batch is NOT acknowledged: the SDK re-sends it whole. Log every
        # identifier an operator needs to trace it; tell the client nothing
        # about the cause (it may echo SQL, bound params, or infra hostnames).
        logger.error(
            "Persist failed; batch rejected with 503. batch_id=%s org_id=%s agent_id=%s "
            "events=%d error=%s",
            batch_id,
            org_id,
            body.agent_id,
            n,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Event persistence failed; retry the batch.",
            headers={"Retry-After": str(_PERSIST_RETRY_AFTER_S)},
        )
    finally:
        _persist_seconds.observe(time.perf_counter() - t0)

    _events_accepted.inc(n)
    logger.info("Accepted. batch_id=%s agent_id=%s events=%d", batch_id, body.agent_id, n)
    return IngestResponse(accepted=n, batch_id=batch_id)


def _header_api_key(request: Request) -> str:
    """API key from ``Authorization: Bearer <key>``, or the ``X-Dunetrace-API-Key``
    header. Returns "" when neither is present. Mirrors the OTLP receiver's helper
    of the same name (ingest_svc/routers/otlp.py)."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return request.headers.get("X-Dunetrace-API-Key", "").strip()


@router.get(
    "/v1/policies",
    summary="Fetch runtime policies for an agent (SDK-facing)",
    include_in_schema=False,
)
async def get_policies(
    request: Request,
    agent_id: str = Query(...),
    api_key: str = Query(""),
) -> dict:
    """
    Called by the SDK at run start to retrieve active policies.

    Authenticates via ``Authorization: Bearer <key>`` (preferred) or the
    ``X-Dunetrace-API-Key`` header. The ``api_key`` query parameter is still
    accepted so an older SDK build keeps working, but it is **deprecated**:
    query strings are recorded verbatim in web-server and proxy access logs, so
    a key sent that way leaks into logs on every request. The header takes
    precedence when both are present. POST /v1/ingest is unaffected — it has
    always carried the key in the request body.
    """
    key = _header_api_key(request) or api_key
    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide it as 'Authorization: Bearer <key>'.",
        )
    if not _header_api_key(request) and api_key:
        logger.warning(
            "Deprecated: /v1/policies authenticated via api_key query param for "
            "agent_id=%s. Query strings leak into access logs — send "
            "'Authorization: Bearer <key>' instead.",
            agent_id,
        )
    org_id = await _resolve_org_id(request, key)
    policies = await fetch_policies(agent_id, org_id)
    return {"policies": policies}


@router.get(
    "/v1/detector-config",
    summary="Fetch the server's effective detector configuration for an agent (SDK-facing)",
    include_in_schema=False,
)
async def get_detector_config(
    request: Request,
    agent_id: str = Query(...),
    agent_version: str = Query(""),
    api_key: str = Query(""),
) -> dict:
    """
    Called by the SDK at run start, alongside /v1/policies, so its client-side
    detector pass applies the thresholds the detector worker would: the
    detectors.yml overrides for this agent's category (already merged over
    `default` and mapped to the UPPERCASE constructor kwargs), the packs the
    org has enabled, and the agent's P75 baselines. See
    ingest_svc/detector_config.py for the shape and caching; the SDK overlays
    the result on its class defaults and keeps them for anything missing.

    `agent_version` selects which version's baselines to report (they are
    per-version); omitted, the most recently recorded version is used.

    Authentication is identical to /v1/policies: ``Authorization: Bearer
    <key>`` (preferred) or ``X-Dunetrace-API-Key``; the ``api_key`` query
    parameter is accepted but deprecated for the same access-log reason.
    Never fails for a missing detectors.yml, missing PyYAML or an unreadable
    baseline table — each degrades to its empty value so the SDK can fall
    back to defaults.
    """
    key = _header_api_key(request) or api_key
    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide it as 'Authorization: Bearer <key>'.",
        )
    if not _header_api_key(request) and api_key:
        logger.warning(
            "Deprecated: /v1/detector-config authenticated via api_key query param for "
            "agent_id=%s. Query strings leak into access logs — send "
            "'Authorization: Bearer <key>' instead.",
            agent_id,
        )
    org_id = await _resolve_org_id(request, key)
    return await build_detector_config(org_id, agent_id, agent_version.strip() or None)


@router.post(
    "/v1/deploy",
    response_model=DeployResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Record a deploy marker for an agent",
)
async def mark_deploy(request: Request, body: DeployRequest) -> DeployResponse:
    org_id = await _resolve_org_id(request, body.api_key)
    row_id = await insert_deploy_event(body.agent_id, body.version, body.meta, org_id)
    logger.info(
        "Deploy marked. agent_id=%s version=%s id=%d",
        body.agent_id,
        body.version,
        row_id,
    )
    return DeployResponse(
        id=row_id,
        agent_id=body.agent_id,
        version=body.version,
        deployed_at=time.time(),
    )


def _check_admin_key(supplied: str) -> None:
    """Raises 403 unless `supplied` matches the ADMIN_API_KEY env var (which must
    itself be set — an unset/empty env var never matches, closed by default)."""
    admin_key = os.getenv("ADMIN_API_KEY", "")
    if not admin_key or not secrets.compare_digest(supplied, admin_key):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid admin key")


@router.post(
    "/v1/keys",
    response_model=KeyCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate a new org-scoped API key",
    include_in_schema=False,
)
async def create_key(body: KeyCreateRequest) -> KeyCreateResponse:
    """Admin-only endpoint. Requires ADMIN_API_KEY env var to match body.admin_key.

    Mints an ``admin`` key when ``scopes`` is omitted. That is the opposite
    default from the Customer API's ``POST /v1/keys`` (``ingest``-only unless
    asked), and deliberately so — the two endpoints answer to different callers:

    * The Customer API's is self-service, authenticated by an existing key, and
      refuses to mint any scope the caller does not already hold. Its common
      case is "a key for an agent", so ``ingest`` is both the safe and the
      usual answer there.
    * This one is gated on ``ADMIN_API_KEY`` — the operator's own deployment
      secret, not a tenant credential — and exists to bootstrap a fresh
      self-hosted install, where no key exists yet. Every key-management,
      policy-write, integration and org-settings route on the Customer API
      needs ``admin``, and that API cannot mint a scope its caller lacks, so
      if the very first key is not ``admin`` the install can never obtain one.

    An explicit ``scopes`` list is honoured as given (after normalisation), so
    an operator can still hand out an ingest-only key from here.
    """
    _check_admin_key(body.admin_key)
    name = body.org_name or body.org_id
    scopes = body.scopes if body.scopes is not None else [ADMIN]
    created = await create_api_key(
        body.org_id, org_name=name, rate_limit_rpm=body.rate_limit_rpm, scopes=scopes
    )
    # key_prefix is the non-secret identifier GET /v1/keys lists; the key
    # itself is never logged.
    logger.info(
        "API key created. org_id=%s org_name=%s rpm=%d key_prefix=%s scopes=%s",
        body.org_id,
        name,
        body.rate_limit_rpm,
        created["key_prefix"],
        ",".join(created["scopes"]),
    )
    return KeyCreateResponse(
        key=created["key"],
        key_prefix=created["key_prefix"],
        org_id=body.org_id,
        org_name=name,
        scopes=created["scopes"],
    )


@router.post(
    "/admin/prune-events",
    response_model=PruneEventsResponse,
    summary=(
        "Manually run a retention pass (drop event partitions and scrub signal "
        "evidence older than retention_days)"
    ),
    include_in_schema=False,
)
async def prune_events(body: PruneEventsRequest) -> PruneEventsResponse:
    """Admin-only endpoint. Requires ADMIN_API_KEY env var to match body.admin_key.

    Runs the same retention pass as the daily background loop (see main.py's
    _prune_loop) — for confirming a fix after the startup staleness warning, or
    reclaiming space immediately rather than waiting for the next scheduled tick.
    """
    _check_admin_key(body.admin_key)
    from ingest_svc.config import settings

    retention_days = (
        body.retention_days if body.retention_days is not None else settings.EVENT_RETENTION_DAYS
    )
    dropped = await get_event_store().prune_old_events(retention_days)
    # Not routed through EventStore: that interface covers this service's own
    # `events` write path, and signal evidence is neither an event nor written
    # here. See scrub_old_signal_evidence's docstring.
    scrubbed = await scrub_old_signal_evidence(retention_days)
    logger.info(
        "Manual retention pass (via /admin/prune-events): %d partition(s) dropped, "
        "%d signal(s) scrubbed",
        dropped,
        scrubbed,
    )
    return PruneEventsResponse(
        partitions_dropped=dropped,
        signals_scrubbed=scrubbed,
        retention_days=retention_days,
    )


@router.get(
    "/admin/keys/{key_id}/agents/{agent_id}/quota",
    response_model=QuotaResponse,
    summary="View this agent's rate-limit quota within its key's budget",
    include_in_schema=False,
)
async def get_quota(key_id: int, agent_id: str, admin_key: str = Query(...)) -> QuotaResponse:
    """Admin-only. key_id (not the raw key string — see /v1/keys's response) scopes
    this to a single key, matching how the rate limiter itself enforces per-agent
    sub-limits (see rate_limiter.py's module docstring)."""
    _check_admin_key(admin_key)
    from ingest_svc.rate_limiter import _DEFAULT_AGENT_QUOTA_PCT

    override = await get_agent_quota(key_id, agent_id)
    return QuotaResponse(
        key_id=key_id,
        agent_id=agent_id,
        quota_pct=override if override is not None else _DEFAULT_AGENT_QUOTA_PCT,
        is_override=override is not None,
    )


@router.put(
    "/admin/keys/{key_id}/agents/{agent_id}/quota",
    response_model=QuotaResponse,
    summary="Set this agent's rate-limit quota within its key's budget",
    include_in_schema=False,
)
async def set_quota(key_id: int, agent_id: str, body: QuotaSetRequest) -> QuotaResponse:
    """Admin-only. Overrides the default 20% share (see rate_limiter.py's
    _DEFAULT_AGENT_QUOTA_PCT) for this one (key_id, agent_id) pair. Takes effect
    within _CACHE_TTL seconds — the rate limiter caches quota lookups the same
    way it caches rate_limit_rpm, not read fresh on every request."""
    _check_admin_key(body.admin_key)
    try:
        await set_agent_quota(key_id, agent_id, body.quota_pct)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    logger.info(
        "Agent quota set. key_id=%d agent_id=%s quota_pct=%s", key_id, agent_id, body.quota_pct
    )
    return QuotaResponse(
        key_id=key_id, agent_id=agent_id, quota_pct=body.quota_pct, is_override=True
    )


def _is_policy_evaluation(e) -> bool:
    """True for policy.evaluated observability records, which are routed to the
    policy_evaluations table instead of being stored as run events."""
    et = getattr(e, "event_type", None)
    return getattr(et, "value", et) == "policy.evaluated"


async def _persist(events: list, batch_id: str, org_id: str) -> None:
    """Write one batch, or raise PersistError. Never swallows a failure.

    Raises PersistError when the store raises, or when it reports fewer rows
    written than trace events sent and the batch cannot be shown to already be
    on disk (see _batch_already_durable). The policy_evaluations sink stays
    best-effort: a row shortfall there is logged, not fatal — it is an
    observability side-table, and its store default is a documented no-op.

    A PersistError may leave rows behind (policy evaluations, or an earlier
    successful attempt): the SDK re-sends the whole batch, and insert_events
    drops events whose event_id is already stored, so a re-send is safe.
    """
    store = get_event_store()
    # Split observability records out of the run-event stream — they belong in
    # policy_evaluations, not events (keeps run traces clean).
    evals = [e for e in events if _is_policy_evaluation(e)]
    trace_events = [e for e in events if not _is_policy_evaluation(e)]

    try:
        if evals:
            written = await store.insert_policy_evaluations(evals, batch_id, org_id)
            if written < len(evals):
                logger.warning(
                    "policy_evaluations shortfall. batch_id=%s written=%d expected=%d",
                    batch_id,
                    written,
                    len(evals),
                )

        if not trace_events:
            logger.debug("Persisted. batch_id=%s policy_evals=%d", batch_id, len(evals))
            return

        inserted = await store.insert_events(trace_events, batch_id, org_id)
    except Exception as exc:
        raise PersistError(f"{type(exc).__name__}: {exc}", reason="exception") from exc

    expected = len(trace_events)
    if inserted < expected:
        if await _batch_already_durable(trace_events, org_id):
            logger.info(
                "Persist shortfall is a re-send of stored events; accepted. "
                "batch_id=%s inserted=%d expected=%d",
                batch_id,
                inserted,
                expected,
            )
            return
        raise PersistError(
            f"insert shortfall: inserted={inserted} expected={expected}", reason="shortfall"
        )

    logger.debug(
        "Persisted. batch_id=%s inserted=%d policy_evals=%d", batch_id, inserted, len(evals)
    )


async def _batch_already_durable(trace_events: list, org_id: str) -> bool:
    """True when every event in the batch is already stored under this org.

    Why this exists: PostgresEventStore.insert_events returns the rows it
    wrote *after* dropping events whose event_id is already in `events` (an
    at-least-once SDK re-send), and it returns 0 on failure — so a short count
    alone cannot tell "the DB is down" from "this batch was committed on the
    previous attempt and the response was lost". Both SDKs stamp event_id on
    every event and the durable retry queue stops at its first failing batch,
    so mistaking the second case for the first would 503 that batch forever and
    wedge everything queued behind it.

    Only reached on the shortfall path — never a cost on a normal request.
    Fails closed: any event without an event_id, no pool, or a query error
    means "cannot prove it", and the caller raises PersistError. Reads
    `events` directly rather than through EventStore because the store does
    not expose this; the right fix is for insert_events to report rows
    accounted for (written + deduplicated) and raise on failure.
    """
    ids = [getattr(e, "event_id", None) for e in trace_events]
    if not ids or not all(ids):
        return False
    pool = get_pool()
    if not pool:
        return False
    try:
        async with pool.acquire() as conn:
            present = await conn.fetchval(
                "SELECT count(DISTINCT event_id) FROM events "
                "WHERE org_id = $1 AND event_id = ANY($2::text[])",
                org_id,
                ids,
            )
    except Exception as exc:
        logger.warning("Could not verify batch presence: %s", type(exc).__name__)
        return False
    return int(present or 0) == len(set(ids))
