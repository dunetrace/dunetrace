# Operations

Deploying the stack, retention, storage growth, and the manual controls available for the last two. The retention material covers `services/ingest` — the only service that writes to or prunes the `events` table.

---

## Deploying

A Dunetrace stack is in one of two states, and there is one compose command for each:

| | Local quickstart | Production |
|---|---|---|
| Command | `docker compose up -d` (or `-f docker-compose.ghcr.yml`) | `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d` |
| `ENV` / `AUTH_MODE` | `dev` / `dev` — authentication **off** | `prod` / `prod` on every service — every endpoint needs an API key |
| Published ports | 5432, 8001, 8002, 3000 — all bound to `127.0.0.1` | dashboard on `0.0.0.0:3000`; everything else stays on loopback |
| Postgres password | the default (`dunetrace`) | `POSTGRES_PASSWORD`, required, no default |
| Restart policy | `unless-stopped` | `always` for the core services |

### The quickstart is loopback-only by design

`docker compose up -d` binds every published port — Postgres `5432`, ingest `8001`, customer API `8002`, dashboard `3000` — to `127.0.0.1`, so nothing is reachable from another machine. That is what makes dev mode acceptable there: with `AUTH_MODE=dev` (and `ENV=dev`) authentication is disabled on every endpoint of both HTTP services, and every request resolves to the default org **with the `admin` scope** — anyone who can reach a port can mint keys, write `stop` policies that terminate live agent runs, and spend the LLM budget. Both HTTP services log a `WARNING` at startup while it is active. The loopback binding is the control; do not widen it without also switching to the production override.

`ENV` and `AUTH_MODE` are `${VAR:-dev}` defaults in both quickstart files, so they can be overridden from the shell or `.env` (`ENV=prod AUTH_MODE=prod docker compose up -d`) — always set **both**. The production override is still the supported route, because it also handles the password, the ports and the restart policy.

### The production override

`docker-compose.prod.yml` layers on either quickstart file — the service names are the same:

```bash
DASHBOARD_BIND= POSTGRES_PASSWORD=<strong secret> ADMIN_API_KEY=<strong secret> \
  docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# with the published GHCR images instead of a local build:
DASHBOARD_BIND= POSTGRES_PASSWORD=<strong secret> ADMIN_API_KEY=<strong secret> \
  docker compose -f docker-compose.ghcr.yml -f docker-compose.prod.yml up -d
```

`DASHBOARD_BIND` is the host-interface prefix on the dashboard's published port (`"${DASHBOARD_BIND-127.0.0.1:}3000:80"` in the base file): the quickstart default keeps it on loopback, and an empty value publishes it on all interfaces. It is the only port the production stack publishes; everything else stays on loopback behind your reverse proxy.

Both secrets can live in `.env` instead of the shell — compose reads it for interpolation. Never commit that file. What the override changes:

- **`ENV=prod` and `AUTH_MODE=prod` on every app service**, as literals rather than `${...:-dev}` defaults, so a quickstart `AUTH_MODE=dev` left behind in `.env` cannot reopen the stack. Every ingest and customer-API endpoint then requires an API key, and the customer API's CORS allow-list narrows from `*` to `https://app.dunetrace.io` (see the reverse-proxy note below for what that means for a self-hosted dashboard).
- **`POSTGRES_PASSWORD` is required** and threaded into every service's `DATABASE_URL`. `docker compose config` fails with `set POSTGRES_PASSWORD` while it is unset, instead of starting Postgres on the default password.
- **Only the dashboard is published**, on `0.0.0.0:3000` (an `!override` on the `ports` key, so it replaces the loopback binding rather than adding a second one). Postgres, ingest (`8001`) and the customer API (`8002`) keep the base file's `127.0.0.1` binding.
- **`ADMIN_API_KEY`** is passed to ingest — it gates the bootstrap endpoint below. **`INTERNAL_TOKEN`** is passed to ingest and the API, and is only for a gateway in front of the services that vouches for callers with an `x-internal-token` header (identity in `x-org-id`, scopes in `x-scopes`, defaulting to ingest-only). Leave it unset if you run no such gateway.
- **`restart: always`** for the core services. The three opt-in workers keep `on-failure` from the base file: with their flag off they exit 0, and `always` would restart-loop them forever.

### Minting the first API key

With authentication on, nothing can authenticate until a key exists — and the customer API's own `POST /v1/keys` requires an `admin`-scoped key, so it cannot be the first one. The bootstrap path is the **ingest** service's `POST /v1/keys`, gated on `ADMIN_API_KEY` rather than on a key, called from the host (the port is loopback-only):

```bash
curl -s -X POST http://127.0.0.1:8001/v1/keys \
  -H 'Content-Type: application/json' \
  -d '{"org_id": "my-org", "admin_key": "<ADMIN_API_KEY>"}'

# {"key": "dt_...", "key_prefix": "dt_...", "org_id": "my-org", "org_name": "my-org",
#  "scopes": ["admin"], "created_at": 1757500000.0}
```

The plaintext key is returned once and never stored — only its SHA-256 hash is. With `scopes` omitted this endpoint mints an **`admin`** key: it is the operator's bootstrap, and an admin key is the one thing a fresh install cannot obtain any other way (pass `"scopes": ["ingest"]` to mint something narrower). Keep that key for operators and the dashboard, and mint the narrower keys your agents and approvers use from it, through the customer API:

```bash
# ingest-only — what an SDK/agent process gets (the default)
curl -s -X POST https://dunetrace.example.com/api/v1/keys \
  -H 'Authorization: Bearer <admin key>' \
  -H 'Content-Type: application/json' \
  -d '{"org_id": "my-org"}'

# a human who decides approvals
curl -s -X POST https://dunetrace.example.com/api/v1/keys \
  -H 'Authorization: Bearer <admin key>' \
  -H 'Content-Type: application/json' \
  -d '{"org_id": "my-org", "scopes": ["approve"]}'
```

A key can never mint a scope it does not itself hold (403 otherwise), so no chain of mints escalates. The three scopes — `ingest` (submit events, read the org's own data), `approve` (decide approvals) and `admin` (everything: keys, policies, integrations, packs, org settings) — are described under [Human-in-the-loop approvals](approvals.md) and in `packages/schemas-py/dunetrace_schemas/scopes.py`. Unset `ADMIN_API_KEY` after bootstrapping to close the ingest endpoint, or keep it to mint keys for further orgs later; it also gates the `/admin/*` endpoints further down this page.

### Reverse proxy and the dashboard's API origin

The two APIs stay on loopback in production; reaching them from anywhere but the host is the job of a reverse proxy you run in front of them. `infra/nginx.conf` is a starting point — it proxies `/v1/ingest` → `ingest:8001` and `/api/` → `api:8002` and serves the dashboard from `/`. Adjust it before use:

- Give the `/api/` location a trailing slash on `proxy_pass` (`proxy_pass http://api/;`) so the `/api` prefix is stripped before the request reaches the customer API, which serves `/v1/...` at its root.
- Proxy the rest of ingest's surface alongside `/v1/ingest`: the SDK also calls `/v1/policies` and `/v1/deploy` on the same base URL, OTel exporters call `/v1/otlp/traces`, and `/health` (liveness) and `/ready` (readiness) are useful for probes — but do **not** proxy `/metrics`, which is unauthenticated and for the internal scraper only.
- TLS is on you. Every key travels as a bearer token.

Serve the dashboard and the customer API from the **same origin**, path-based as that config does. In prod mode the customer API only allows CORS from `https://app.dunetrace.io`, so a dashboard on one hostname calling an API on another fails in the browser; a same-origin path needs no CORS at all.

The dashboard is static, so it is pointed at the proxied API by editing the two constants at the top of the `<script>` in `dashboard/mission-control.html`:

```js
const API = 'http://localhost:8002';   // → the API origin your proxy exposes, e.g. 'https://dunetrace.example.com/api'
const KEY = 'dt_dev_test';             // → a real key; dt_dev_* is only honoured in dev mode
```

The dashboard's config pages (keys, policies, integrations, packs, org settings) write through admin-gated endpoints, so `KEY` is normally the admin key. Agents get an ingest key and point their SDK at the proxied ingest base (`endpoint="https://dunetrace.example.com"` — the SDK appends `/v1/ingest` itself).

### Dashboard security headers

`dashboard/nginx.conf` sends four headers on every response (`always`, so error pages too): `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: same-origin` and a `Content-Security-Policy`. The CSP's `script-src` is **only the SHA-256 hash of the page's single inline `<script>`** — no `'self'`, no `'unsafe-inline'` — so an injected script tag or inline handler has no matching hash and the browser refuses to run it. `connect-src`, `style-src`, `font-src` and the rest are listed in the file with a comment per directive. Two things need a hand when you change the deployment:

- **The API origin.** `connect-src` must list the origin the page's `const API` points at. The quickstart's `http://localhost:8002` is in the file. The recommended production shape above — API proxied under the dashboard's own origin (`https://dunetrace.example.com/api`) — is covered by `'self'` and needs nothing. Any other origin has to be added to `connect-src` in `dashboard/nginx.conf`, or every API call fails in the browser with a CSP violation in the console and the dashboard shows its error banner.
- **The script hash.** Any edit to the inline script — including changing `API`/`KEY` — changes the hash, and a stale hash means the browser runs nothing (blank page, one `Content Security Policy` error in the console). Regenerate it with

  ```bash
  python scripts/check_dashboard_safety.py --fix
  docker compose restart dashboard
  ```

  CI runs the same script without `--fix` and fails a PR whose hash is stale (`.github/workflows/ci.yml`, lint job). The script also fails on any inline event-handler attribute, `javascript:` URL, `eval(`/`new Function`, or a `data-action` that names no key of the page's `ACTIONS` registry — see its docstring.

`style-src` keeps `'unsafe-inline'` on purpose: the page has one `<style>` block and several hundred inline `style=""` attributes. That does not weaken the script policy — markup injection can move things around, but cannot run code. Verify the live headers with `curl -sI http://127.0.0.1:3000/`.

`infra/nginx.conf`'s `/` location already ships all four `add_header` lines, so serving the dashboard through that reverse proxy instead of the compose `dashboard` container needs no change. Keep the two copies in sync when the CSP changes — in particular the `script-src` hash, which must be byte-identical in `infra/nginx.conf` and `dashboard/nginx.conf`. If you serve the dashboard from some *other* proxy, copy those four lines into its server block: they are part of the nginx config, not something the compose file applies.

### The startup guard

Every service — the two HTTP services and all the workers — calls `dunetrace_schemas.deploy_guard.assert_safe_deployment` at import of its `config.py`, before a pool is opened or a port is bound. It refuses exactly one pair: `ENV` in `{prod, production}` together with `AUTH_MODE` in `{dev, local, test}` (case- and whitespace-insensitive). Every other combination passes, including an unset `ENV` — the guard refuses one known-bad pair and does not try to validate the vocabulary. A refused container exits with status 1 and this on stderr:

```
dunetrace/api: refusing to start.

    ENV=prod  (a production deployment)
    AUTH_MODE=dev  (authentication disabled)

AUTH_MODE=dev skips authentication on every endpoint, so a production
deployment with it set is open to anyone who can reach the port. Either:

  * set AUTH_MODE=prod — the production override
    (docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d)
    sets it for every service, or
  * set ENV=dev if this really is a local, loopback-only instance.

See docs/operations.md, section "Deploying".
```

The first line names the service (`dunetrace/ingest`, `dunetrace/detector`, …) and a value that was not set at all prints as `<unset>`. The workers authenticate nothing, but they run the same check so a mis-set environment shows on every container in `docker compose ps`, not just on two of them.

### Schema migrations

Every table more than one service touches is declared in `packages/schemas-py/dunetrace_schemas/migrations.py`, an ordered, versioned runner with a `schema_version` table (`CURRENT_SCHEMA_VERSION` is the last migration's number). Each service's schema-init — ingest's `ensure_schema`, the detector's `ensure_detector_schema`, the API's `init_pool`, the alerts worker's `ensure_*` entry points, the semantic worker's `ensure_semantic_schema`, the integrations worker's `ensure_integrations_schema` / `ensure_elevenlabs_schema` — does the same two things in the same order before any DDL of its own: `apply_migrations` (a Postgres advisory lock serialises replicas, so N containers booting together apply each migration exactly once), then `require_schema_version(conn, CURRENT_SCHEMA_VERSION, "<service>")`.

**What `require_schema_version` refuses.** It reads `MAX(version)` from `schema_version` and raises `RuntimeError` — the container exits — when that is below the version the running code was built against:

```
detector needs schema version >= 12 but the database is at 5. Run migrations
first (any service's startup applies them); this service will not start
against an older schema.
```

Apply-then-require means the refusal is reached only when apply returned without bringing the database current: a replica that lost the advisory-lock race to a peer whose apply then failed, or a service pointed at a read-only standby or a role without DDL rights. Before the gate that case was silent — a query referencing a column another service had not yet created logged one line and returned nothing (the `policies.signature` outage). The fix is to let a service with a writable connection start; `SELECT * FROM schema_version ORDER BY version` shows how far the database got. Boot order does not matter — every service can start first on an empty database. Ingest's own DDL deliberately runs *before* the migrations (its `events` table must exist for migration 2 to index it), with the gate sitting between the migrations and its legacy org_id backfill, which reads columns migrations 5 and 7 add.

**Verifying boot order against a real Postgres.** `scripts/verify_schema_boot_order.py` boots the six schema-inits in every first-service order on fresh databases, runs six concurrent appliers, re-runs every init twice and diffs the catalog (columns, indexes, constraints) for idempotence, and upgrades a database built by the committed (`HEAD`) DDL with rows in every table the newer migrations alter. It needs asyncpg and the services' dependencies on `PATH`, and it creates and drops databases, so it refuses non-loopback hosts unless told otherwise:

```bash
# throwaway postgres:16-alpine on 127.0.0.1:55432, removed afterwards
python scripts/verify_schema_boot_order.py --start-container

# an existing local server
DATABASE_URL=postgresql://u:p@127.0.0.1:55432/postgres python scripts/verify_schema_boot_order.py
```

Run it after adding a migration or moving a table between a service and `migrations.py`. A non-zero exit names the scenario and what went wrong: the missing table, the statement a second run added, the row an upgrade lost.

---

## Observability: probes and metrics

Every service answers three unauthenticated HTTP probes, served from one shared implementation, `packages/schemas-py/dunetrace_schemas/metrics.py`. They are for the internal network only — compose healthchecks and an in-network Prometheus — and none of them exposes tenant data; do not proxy `/metrics` to the outside.

| Probe | Question it answers | 200 when | Otherwise |
|---|---|---|---|
| `GET /health` | Is the process up and serving HTTP? | Always, while the process answers. **No database round-trip** — a probe that waits on a pool connection reports a saturated-but-working process as dead, and an orchestrator acting on that restarts the one thing still draining the backlog | never fails |
| `GET /ready` | Can it do its job right now? | The database answers `SELECT 1` **and** the applied `schema_version` is at least this build's `CURRENT_SCHEMA_VERSION` (a read, never DDL) **and**, for the workers, the poll loop is fresh (below) | 503 with the same JSON body: `status`, `db` (`ok`, `no_pool`, or the exception class name), `schema_version`, `required`, `pool` sizes, and the workers' `poll` freshness fields — so a failing probe can be read from the healthcheck log alone |
| `GET /metrics` | Prometheus exposition | Always | with `prometheus_client` missing from the image the body is a one-line comment saying so, and every metric is a no-op |

Every compose file's healthchecks target `/ready`. Two things make that right: `/ready` depends on `apply_migrations()` having run — a replica that boots against a database still on an older schema is *not ready* rather than "healthy but crashing on the first query" — and, for the workers, on the loop actually turning.

### Where each service serves them

The two FastAPI services serve the probes on their own port. The five workers (detector, alerts, semantic, integrations, elevenlabs) have no HTTP server of their own, so `start_metrics_server()` runs a stdlib one on a daemon thread; its port is a setting, `0` disables it, and the opt-in workers only start it when their `*_WORKER_ENABLED` flag is on (a disabled worker exits 0 and has nothing to scrape). None of the worker ports is published by compose.

| Service | Probes at | Port setting | Default |
|---|---|---|---|
| Ingest API | `http://ingest:8001` | (service port) | 8001 |
| Customer API | `http://api:8002` | (service port) | 8002 |
| Detector worker | `http://detector:9101` | `METRICS_PORT` | 9101 |
| Alerts worker | `http://alerts:9102` | `METRICS_PORT` | 9102 |
| Semantic worker | `http://semantic:9103` | `METRICS_PORT` | 9103 |
| Integrations worker | `http://integrations:9104` | `INTEGRATIONS_METRICS_PORT` | 9104 |
| ElevenLabs worker | `http://elevenlabs:9105` | `ELEVENLABS_METRICS_PORT` | 9105 |

The last two are distinct names on purpose: both containers run from one image and read one `Settings` class, and a shared `METRICS_PORT` would make a single-container deployment of both collide on the bind. The compose files pin every port explicitly in each service's `environment` so the healthcheck URL and the server cannot drift apart.

**Poll freshness** is what turns a wedged worker — an `await` that never returns, a CPU-bound detector, a DB that vanished — into a 503 instead of a green container. The readiness check itself is scheduled onto the worker's event loop with a 5-second timeout, so a loop that cannot run it answers 503 by timing out:

| Worker | Fresh while | Before the first cycle |
|---|---|---|
| Detector | a poll **succeeded** within 3 × `POLL_INTERVAL` | ready (the window starts when the server starts) |
| Alerts | a poll **finished** (success or not) within 3 × `POLL_INTERVAL`; a poll still in flight is allowed `CLAIM_TIMEOUT_SECS`, the bound the claim design already puts on one batch's worst-case delivery | **not** ready until the first poll completes |
| Semantic, Integrations, ElevenLabs | a cycle **succeeded** within 3 × its interval (`POLL_INTERVAL` / `WAKE_INTERVAL`) | ready (seeded once the schema gate has passed) |

### Scraping

Point Prometheus at every service. The worker ports are not published, so the scraper has to be on the compose network — add it as a compose service, or run it with `--network dunetrace_default`:

```yaml
scrape_configs:
  - job_name: dunetrace
    static_configs:
      - targets:
          - ingest:8001
          - api:8002
          - detector:9101
          - alerts:9102
          - semantic:9103       # only answers while SEMANTIC_WORKER_ENABLED=true
          - integrations:9104   # only answers while INTEGRATIONS_WORKER_ENABLED=true
          - elevenlabs:9105     # only answers while ELEVENLABS_WORKER_ENABLED=true
```

Set `APP_VERSION` (or `GIT_COMMIT`) in deploy so `dunetrace_build_info` identifies the build; unset, it reports the package version.

### Metric catalogue

Names are validated to start with `dunetrace_`, counters end in `_total`, latencies are histograms in seconds ending in `_seconds`. Every label set is closed — a `path` is the route template, never the concrete URL, and anything unmatched folds into `other` — so a scanner cannot mint series. `packages/schemas-py/tests/test_metrics.py::TestMetricCatalogueDocumented` fails the build if a metric registered in any service is missing from this table.

**Every service**

| Metric | Labels | Meaning |
|---|---|---|
| `dunetrace_build_info` | `service`, `version` | Always 1; the labels identify the running build |
| `dunetrace_schema_version` | `service` | The migration version the service found at startup. `min by (service)` lagging `max` across the fleet is a replica that missed a deploy |

**Ingest API**

| Metric | Labels | Meaning |
|---|---|---|
| `dunetrace_ingest_requests_total` | `path`, `method`, `status` | Every request, including the 429/413 short-circuits that never reach a route |
| `dunetrace_ingest_rate_limited_total` | | 429s from the per-key / per-agent limiter (the middleware's; the OTLP route counts its own in `requests_total`) |
| `dunetrace_ingest_body_too_large_total` | | 413s for a body over `INGEST_MAX_BODY_BYTES` |
| `dunetrace_ingest_persist_failures_total` | `reason` = `exception` \| `shortfall` | Batches answered 503 because they were not durably written — the SDK re-sends them |
| `dunetrace_ingest_persist_seconds` | | Wall-clock of the `_persist()` call per batch, success or failure |
| `dunetrace_ingest_events_accepted_total` | | Events acknowledged with 202 (committed before the response) |

**Customer API**

| Metric | Labels | Meaning |
|---|---|---|
| `dunetrace_api_requests_total` | `path`, `method`, `status` | Every request by templated route |
| `dunetrace_api_request_seconds` | `path`, `method` | Time to serve one request |
| `dunetrace_api_llm_calls_total` | `provider`, `status` = `ok` \| `error` | Dunetrace's own LLM completions (native explain, fix diffs, custom-detector translation, issue summaries) through `llm_provider.complete` |
| `dunetrace_api_llm_call_seconds` | `provider` | Latency of one such completion |

**Detector worker**

| Metric | Labels | Meaning |
|---|---|---|
| `dunetrace_detector_backlog_runs` | | Runs (completed + stalled) the last poll found — what it *saw*, capped at `BATCH_SIZE`, not a table count |
| `dunetrace_detector_poll_saturated` | | 1 when the last poll hit `BATCH_SIZE` on either query (more work behind it, watermark held), else 0 |
| `dunetrace_detector_poll_seconds` | | One poll cycle, fetch plus every run processed |
| `dunetrace_detector_runs_processed_total` | `result` = `signals` \| `clean` \| `failed` \| `retry` | Runs handed to `process_run`: at least one signal written; none; retry budget spent and recorded with `processing_error`; left unmarked for the next poll |
| `dunetrace_detector_exceptions_total` | `where` = `poll` \| `process_run` \| `custom_detector` | Exceptions caught, by site |
| `dunetrace_detector_signals_total` | `failure_type`, `shadow` | `failure_signals` rows written (built-in, JSON-config custom and plugin) |

**Alerts worker**

| Metric | Labels | Meaning |
|---|---|---|
| `dunetrace_alerts_backlog_signals` | | Unalerted live signals claimed by the most recent poll |
| `dunetrace_alerts_processed_total` | `result` = `delivered` \| `suppressed` \| `skipped` \| `failed` | Claimed signals by outcome: at least one destination accepted; policy-pending/silenced/deduped; a lower-confidence duplicate folded into its group or nothing to send to; reconstruction, explanation or every destination failed (claim released, retried next poll). The four sum to the backlog |
| `dunetrace_alerts_poll_seconds` | | One `poll_once()` cycle, delivery included |
| `dunetrace_alerts_exceptions_total` | `where` = `poll` \| `approvals` \| `digest` \| `reconstruct` \| `explain` \| `deliver` \| `metrics_server` | Exceptions caught, by stage |
| `dunetrace_alerts_approvals_delivered_total` | | Pending approvals notified over Slack and/or the webhook |
| `dunetrace_alerts_digests_sent_total` | | Weekly digests sent (one per org) |
| `dunetrace_alerts_delivery_total` | `destination` = `slack` \| `webhook` \| `linear` \| `other`, `status` = `2xx`…`5xx` \| `error` | One count per outbound HTTP call — a retried delivery counts once per attempt, so this shows what the destination actually answered |
| `dunetrace_alerts_delivery_seconds` | `destination` | Latency of one outbound delivery call |

**Semantic worker**

| Metric | Labels | Meaning |
|---|---|---|
| `dunetrace_semantic_backlog` | | Unevaluated runs found by the most recent poll (capped at `BATCH_SIZE`) |
| `dunetrace_semantic_processed_total` | `result` = `sampled` \| `skipped` | Runs a sampling decision was recorded for |
| `dunetrace_semantic_failures_total` | `where` = `poll` \| `evaluator` \| `second_opinion` \| `conversation_evaluator` \| `conversation` | Errors the worker contained, by where |
| `dunetrace_semantic_external_calls_total` | `provider`, `status` = `ok` \| `error` | Evaluator invocations (one LLM-backed DeepEval call each), by the provider that evaluator was built with — a second opinion carries its own, usually different, provider |
| `dunetrace_semantic_external_call_seconds` | `provider` | Latency of one evaluator invocation |
| `dunetrace_semantic_poll_seconds` | | One poll cycle |

**Integrations and ElevenLabs workers** (same shape, `integrations_` / `elevenlabs_` prefix)

| Metric | Labels | Meaning |
|---|---|---|
| `dunetrace_integrations_backlog`, `dunetrace_elevenlabs_backlog` | | Integrations (accounts) due for a poll at the most recent wake |
| `dunetrace_integrations_processed_total`, `dunetrace_elevenlabs_processed_total` | `result` = `ok` \| `error` | Integrations polled, by outcome |
| `dunetrace_integrations_failures_total`, `dunetrace_elevenlabs_failures_total` | `where` = `integration` \| `cycle` (+ `correlation` for ElevenLabs) | Errors the worker contained, by where |
| `dunetrace_integrations_external_calls_total`, `dunetrace_elevenlabs_external_calls_total` | `provider`, `status` = `ok` \| `error` | Outbound provider fetches (all pages of one poll count once) |
| `dunetrace_integrations_external_call_seconds`, `dunetrace_elevenlabs_external_call_seconds` | `provider` | Latency of one such fetch |
| `dunetrace_integrations_poll_seconds`, `dunetrace_elevenlabs_poll_seconds` | | One wake cycle |

### Alerting on them

A starting set of rules — the thresholds are suggestions:

```yaml
groups:
  - name: dunetrace
    rules:
      - alert: DunetraceTargetDown
        expr: up{job="dunetrace"} == 0
        for: 2m
      - alert: DunetraceIngestNotPersisting
        expr: rate(dunetrace_ingest_persist_failures_total[5m]) > 0
        for: 5m
      - alert: DunetraceDetectorBacklogGrowing
        # a poll that keeps hitting BATCH_SIZE cannot drain the queue
        expr: dunetrace_detector_poll_saturated == 1
        for: 10m
      - alert: DunetraceDetectorRunsFailing
        expr: rate(dunetrace_detector_runs_processed_total{result="failed"}[15m]) > 0
        for: 15m
      - alert: DunetraceAlertDeliveryFailing
        expr: rate(dunetrace_alerts_delivery_total{status!~"2xx"}[15m]) > 0
        for: 15m
      - alert: DunetraceSchemaVersionSkew
        # a replica running an older build than the rest of the fleet
        expr: min(dunetrace_schema_version) < max(dunetrace_schema_version)
        for: 10m
```

`/ready` is the per-container signal; these are the fleet-level ones. Neither replaces the retention and instrumentation-health log lines in the sections below, which report conditions no counter captures.

---

## Event retention

`ingest_svc` runs a background retention pass that deletes events older than the
configured window. There are two paths, chosen automatically by whether the
`events` table is partitioned:

- **Partitioned table** (the intended form — monthly `events_YYYYMM` partitions):
  the pass drops whole partitions once all their rows are older than the window —
  an instant DDL `DROP TABLE`, no vacuum, no lock contention with ingest.
- **Non-partitioned table** (a deployment whose `events` table predates
  partitioning, i.e. it was created before the partitioning DDL and
  `CREATE TABLE IF NOT EXISTS` therefore never converted it): the pass falls back
  to a **batched `DELETE`** of old rows (10k per iteration). Retention still runs,
  but `DELETE` is heavier than a partition drop and leaves dead tuples for
  autovacuum to reclaim. On every pass the log carries a WARNING that the table
  isn't partitioned.

**Check which path you're on:**

```sql
SELECT relkind FROM pg_class WHERE relname = 'events';
-- 'p' = partitioned (partition-drop path) | 'r' = plain table (DELETE fallback)
```

**To move an existing plain table onto the efficient partition-drop path**, run
the opt-in offline migration (BACKUP + stop writers first; dry-run by default):

```bash
DATABASE_URL=postgres://... python scripts/migrate_events_to_partitioned.py        # dry run
DATABASE_URL=postgres://... python scripts/migrate_events_to_partitioned.py --yes  # apply
```

> Historical note: earlier versions silently no-opped retention on a
> non-partitioned table (it appeared configured but nothing was ever deleted).
> The DELETE fallback above ensures retention now runs everywhere.

### Configuration

```bash
# .env
EVENT_RETENTION_DAYS=90   # default; set to 0 to effectively disable pruning (nothing is ever old enough)
```

Restart `ingest` to pick up a change:

```bash
docker compose up -d --force-recreate ingest
```

### Schedule

The retention pass runs once immediately at startup (covers a service that was down long enough for pruning to matter right away), then every 24 hours, as a background task inside `ingest_svc`'s process lifetime — not a separate cron job or service. It shares the connection pool with the rest of `ingest_svc` and is deliberately co-located with partition creation (`_ensure_event_partitions()`, same file) rather than split into a separate maintenance service — both operate on the same table, the same pool, the same schema-ownership boundary.

A failed pass (e.g. a transient DB error) is logged and retried on the next scheduled tick — a single bad tick never kills the loop or the service.

### Monitoring

Every pass that actually drops something logs at `INFO`:

```
Pruned event partition events_202501 (data before 2025-02-01, ~184023 rows)
Retention pass complete: 1 partition(s) dropped, ~184023 rows freed
Retention pass took 0.14s, dropped 1 partition(s)
```

Row counts are Postgres's own planner estimate (`pg_class.reltuples`, refreshed by autovacuum/`ANALYZE`) rather than an exact `COUNT(*)` — exact counts would mean a full scan of each partition just for a log line.

**Startup staleness check**: on every boot, `ingest_svc` checks whether any partition already exceeds `EVENT_RETENTION_DAYS` *before* the scheduled pass runs. If so, it logs a `WARNING`:

```
Retention check: a partition already exceeds EVENT_RETENTION_DAYS=90 at startup — either
this is the first startup after enabling retention on older data (harmless, the prune loop
below will catch up momentarily), or the retention loop has been silently failing across
restarts. Watch for 'Retention pass' log lines after startup to confirm it catches up.
```

There's no persisted "last successful prune" timestamp anywhere — in-memory state wouldn't survive a restart, which is exactly the failure mode this exists to catch. This check is a DB-state proxy instead: if a partition this old still exists, pruning hasn't kept up, whether that's because it's never run, is broken, or this is a legitimate one-time catch-up. Both cases look identical from here; if the "Retention pass" log line doesn't show up shortly after this warning, something is actually wrong.

### Manual invocation

`POST /admin/prune-events` runs an out-of-band retention pass immediately, without waiting for the next scheduled tick — useful right after the startup staleness warning, or to reclaim space on demand. Admin-only, same pattern as `POST /v1/keys`:

```bash
curl -s -X POST "http://localhost:8001/admin/prune-events" \
  -H "Content-Type: application/json" \
  -d '{"admin_key": "<ADMIN_API_KEY>"}'

# Response:
{"partitions_dropped": 1, "signals_scrubbed": 412, "retention_days": 90}
```

Pass `retention_days` explicitly to override the configured default for this one invocation:

```bash
curl -s -X POST "http://localhost:8001/admin/prune-events" \
  -H "Content-Type: application/json" \
  -d '{"admin_key": "<ADMIN_API_KEY>", "retention_days": 30}'
```

`ADMIN_API_KEY` must be set in the environment — an unset or empty value rejects every request (closed by default), same as the key-creation endpoint.

Unlike the daily loop, a scrub failure here is **not** swallowed — the call returns a 500. The loop can afford to log and retry on the next tick; a manual invocation has no next tick, and reporting `signals_scrubbed: 0` for a pass that actually errored is indistinguishable from "there was nothing to scrub".

---

## Instrumentation health

`INSTRUMENTATION_DEGRADED` (see [detectors.md](detectors.md)) answers *"was this
run measurable?"*. This query answers *"is this agent's telemetry broken?"*,
which is only visible in aggregate: one blank LLM call is unremarkable, the same
call on 100% of an agent's traffic is a broken pipeline.

The fingerprint is a call that measurably took time and measurably produced
nothing:

```
output_length = 0 AND finish_reason = 'stop' AND completion_tokens = 0 AND latency_ms > 0
```

`latency_ms > 0` is what separates a real round-trip from a call that never
happened. A genuinely empty model response has this shape too — that is the
point. One such call is a finding; **above ~30% of an agent's calls it is not a
model answering nothing 30% of the time, it is an extractor reading the wrong
object.**

The canonical SQL lives in `services/api/api_svc/instrumentation_health.py` as a
single template rendered for both Postgres and SQLite, so the query documented
here and the one the test exercises cannot drift. Render it with
`blank_response_rate_sql("postgres")`.

`provider` comes from `llm.called`, not `llm.responded`, so the query joins the
two on `(org_id, run_id, step_index)` — `llm.responded` is emitted with
`advance=False` and therefore shares its `llm.called`'s `step_index`, which makes
that key work even for events predating `call_id`.

Both `finish_reason = 'stop'` and `finish_reason IS NULL` count. A current SDK
omits `finish_reason` when it could not read one, but events already stored — and
every agent still running a pre-provenance SDK — carry the fabricated `'stop'`.

**Worked example.** The incident this comes from: `langchain_openai` calls
`client.with_raw_response.create()`, so `Completions.create` returned a
`LegacyAPIResponse` rather than a `ChatCompletion`. The extractors hit their
fallback branches, substituted `("", "stop")`, and produced this fingerprint on
100% of calls — firing `EMPTY_LLM_RESPONSE` on every run including the control.
This query would have shown `blank_fraction = 1.0` for that agent on day one.

---

## What the SDK sends: redaction and content caps

Two controls decide what leaves the customer's process at all. They run in the
SDK, in-path, before anything is buffered — which matters, because **the ingest
service never redacts**: whatever an SDK sends is what lands in Postgres, is
rendered in the dashboard, is included in Slack alert payloads, and is sent to
Anthropic, OpenAI or Mistral when someone clicks Explain. Server-side content
caps exist only on the OpenTelemetry path (`OTLP_MAX_ATTR_CHARS`), not on the
JSON path the SDKs use.

Both SDKs implement the same rules and the same wire format, so a run reads
identically whichever one produced it.

### Redaction

Structured tool arguments are walked and any value under a secret-looking key is
replaced with `[REDACTED]` before serialisation. Keys are matched on **words**:
the name is lower-cased and `-`, `.`, whitespace and camelCase boundaries all
become `_`, then it matches when the result equals a denylist entry or ends with
`_<entry>`.

That word normalisation is load-bearing. Matching on case and `-` alone caught
`access_token` but missed `accessToken`, `clientSecret`, `refreshToken`,
`authToken` and `apiSecret` — and camelCase is the dominant convention in the
JSON tool arguments this actually sees.

| Spelling | Redacted |
|---|---|
| `Authorization`, `authorization` | yes |
| `X-Api-Key` → `x_api_key`, `APIKey` → `api_key` | yes |
| `access_token`, `db_password`, `client_secret` | yes |
| `accessToken`, `clientSecret`, `refreshToken` | yes |
| `Proxy-Authorization`, `Set-Cookie` | yes |

Built-in entries: `authorization`, `api_key`, `apikey`, `token`, `secret`,
`password`, `cookie`, `set-cookie` — the suffix rule is what lets eight entries
cover every `*_token` and `*_secret` spelling without enumerating them.

**What redaction does not cover.** It is key-based, so a secret inside a value
(a JSON blob passed as a string, a URL with a token in its query) is not
detected. Plain-text fields — LLM output, retrieval content, memory values — are
**capped, not redacted**, because there is no key to judge them by. And the walk
stops at 32 levels of nesting.

### Content caps

Every free-text field is cut at `max_field_chars` / `maxFieldChars` (default
8192, the same limit the OTLP path enforces): tool arguments and output, LLM
output, retrieval query and content, memory values, `input_text` and
`system_prompt`. A field that was cut carries two sibling keys,
`<field>_truncated: true` and `<field>_original_length: <n>`; an uncapped field
carries neither, so a small payload is byte-identical to what it would have been.

### Configuring it

```python
dt = Dunetrace(
    max_field_chars=8192,                 # 0 disables the cap entirely
    redact_keys=["x_session_id"],         # extends the built-in denylist
    redact=lambda args: {...},            # runs BEFORE the denylist
)
```

```typescript
const dt = new Dunetrace({
  maxFieldChars: 8192,
  redactKeys: ["x_session_id"],
  redact: (args) => ({ ... }),
});
```

`redact_keys` / `redactKeys` go through the same word normalisation, so
`"X-Session-Id"`, `"xSessionId"` and `"x_session_id"` are the same entry. The
`redact` hook runs first and receives the structured arguments, so it can strip
domain-specific fields the built-in list cannot know about; it runs on every
tool call, so keep it cheap.

Neither control can raise: both are wrapped so a failure ships the original
value rather than blocking the agent. That means a redaction bug fails **open**,
which is the right trade for instrumentation but is worth knowing.

---

## Signal evidence scrub

`failure_signals` has no retention policy and deliberately keeps its rows: the dashboard compares the last 30 days of signals against a days-30-to-90 baseline to decide whether a fix worked (`verified` / `likely_fixed` / `still_occurring`), so deleting aged signals would permanently truncate the baseline arm of that comparison. The table is also not partitioned, so expiry would mean a batched `DELETE` with vacuum pressure rather than an instant partition drop, and five tables carry a bare `signal_id` with no foreign key (`fixes`, `signal_feedback`, `signal_group_members`, `linear_issue_signals`) — deleting would silently orphan them.

But a signal's `evidence` dict embeds excerpts of the same raw agent content the event retention pass exists to expire — in two cases untruncated:

| Evidence key | Detector | Content |
|---|---|---|
| `args` | `TOOL_LOOP` | Full raw arguments of **every** call in the loop |
| `taint_source` | `UNGROUNDED_DESTINATION` | Objects carrying untruncated `input_text` / tool output / retrieval content / memory values |
| `destination`, `destination_host` | `UNGROUNDED_DESTINATION` | An email address or URL |
| `tool_error` | `PREMATURE_TERMINATION`, `UNREAD_TOOL_ERROR` | Raw tool error text |
| `output_snippet` | `PREMATURE_TERMINATION` | LLM output excerpt |
| `args_snippet` | `TOOL_ARGUMENT_FABRICATION`, `UNGROUNDED_DESTINATION` | Tool argument excerpt |
| `content_snippet` | `RETRIEVED_CONTENT_INJECTION` | Retrieved text excerpt |
| `value_snippet` | `MEMORY_POISONING` | Memory value excerpt |
| `fabricated_entity` | `TOOL_ARGUMENT_FABRICATION` | The fabricated value itself |
| `missing_entities` | `HANDOFF_CONTEXT_LOSS` | Entities lifted from the parent's context |
| `memory_key` | `MEMORY_POISONING`, `UNGROUNDED_DESTINATION` | Caller-chosen memory key |

So the rows stay and those keys are stripped, on the same `EVENT_RETENTION_DAYS` window as the event prune — **one content horizon**, so raw content leaves `events` and `failure_signals.evidence` at the same moment rather than on two schedules that can drift apart.

This costs no analytics. Every SQL consumer of `evidence` reads metadata (`tool`, `count`, `consecutive_fails`, `args_identical`, `growth_factor`, `conversation_id`); none reads a key in the table above. Detector-owned labels are deliberately kept — `matched_marker` and `matched_patterns` name which of Dunetrace's own constants matched, `grounded_surfaces` names surfaces (`"input_text"`), `failure_source` is a `"declared"`/`"output_text"` literal, and every `*_length` field is an integer. The explainer's display templates already null-guard the content keys, so a scrubbed signal renders without them rather than erroring.

The scrub runs as a second pass in the same daily loop (`main.py::_run_scrub_once`), kept separate from the prune so a partition-drop error can't skip it. It is idempotent: a `WHERE evidence ?| ARRAY[...]` guard means an already-scrubbed row is never rewritten, so repeat passes cost a scan and no writes. Work is batched by `ctid` (10,000 rows) so a large backlog doesn't hold one long transaction.

```
Evidence scrub complete: 412 signal(s) stripped of content keys (detected before 2026-05-21)
Evidence scrub took 0.31s, scrubbed 412 signal(s)
```

**Adding a detector**: if it puts a content excerpt in `evidence`, add the key to `CONTENT_EVIDENCE_KEYS` in `services/ingest/ingest_svc/db/postgres.py`. `TestContentEvidenceKeyCoverage` in `services/ingest/tests/test_ingest.py` walks the detector source and fails the build if a content-derived key isn't listed — it is what catches the "new detector quietly stores content that outlives the horizon forever" case, which a count assertion cannot.

---

## processed_runs retention

`processed_runs` is the detector's idempotency ledger — one row per run, recording
that the run has been analysed. It's also the anti-join target in run discovery, so
letting it grow forever slows the detector's hottest query.

The detector worker prunes it daily (shard 0 only — the table isn't
shard-partitioned, so extra replicas would only contend on the same rows). A pass
keeps deleting while batches come back full, so a long-neglected table catches up in
one pass rather than one batch per day.

**The ordering constraint runs opposite to intuition.** A `processed_runs` row may
only be deleted *after* its run's events are gone. Delete it while the events remain
and the run reads as unprocessed: the detector re-analyses it and writes a second,
duplicate set of signals — which then alert. The delete therefore carries a
`NOT EXISTS` against `events` rather than trusting a retention constant, so the
invariant holds no matter what `EVENT_RETENTION_DAYS` is set to, including values
the detector never sees.

| Env var | Default | Description |
|---|---|---|
| `PRUNE_BATCH_SIZE` | `10000` | Rows per delete batch |

**There is no age knob to tune here**, and that is the fix rather than an omission.
A `PROCESSED_RUNS_RETENTION_DAYS` bound over `processed_at` used to exist.
`processed_at` records when a run was last *analysed*, not when it happened, and it
is refreshed every time late events trigger a re-detection — so the bound never
expired rows for runs that were re-processed recently but whose events had aged out
long ago, which left permanently unprunable rows behind. It was removed rather than
left as a no-op knob. Selecting candidates by the absence-of-events test directly
also guarantees every pass makes progress; an age-bounded pass could return a full
batch of rows that all still have events and delete nothing.

So what governs the window is ingest's `EVENT_RETENTION_DAYS` (90 by default), one
service away: a `processed_runs` row becomes eligible the moment the partition
holding its events is dropped. Raise `EVENT_RETENTION_DAYS` and this table grows in
step; there is nothing to change on the detector side.

There's no manual trigger endpoint; the loop runs on worker startup and every 24h.
To verify it's working, watch for `Pruned N processed_runs row(s)` in the detector
logs, or compare `SELECT count(*) FROM processed_runs` against the retained run
count over time.

---

## Rate limiting

`ingest_svc` enforces a per-API-key sliding-window rate limit (60s window, `rate_limit_rpm` from the `api_keys` table) on `POST /v1/ingest`, `POST /v1/deploy`, and `POST /v1/otlp/traces`. Keys are org-scoped, not agent-scoped — one key can carry traffic for many agents. Without any further limiting, one runaway agent under a shared key can consume the entire key's budget and starve its siblings.

### Per-agent sub-limits

Within a key's overall budget, each distinct `agent_id` gets its own sliding-window sub-limit — by default, 20% of the key's effective rpm. An agent hitting its own sub-limit gets a 429; other agents under the same key are unaffected. The key-level limit is still checked first and always applies regardless of agent — many agents each within their own sub-limit can still collectively exhaust the key.

Agent identity for sub-limiting comes from the request body's `agent_id` field for `/v1/ingest` and `/v1/deploy` (already parsed for auth), and from the `X-Dunetrace-Agent-Id` header for `/v1/otlp/traces` (cheap to read; the `service.name` resource-attribute fallback would require decoding the OTLP body, which the rate-limit middleware deliberately avoids — an OTLP trace relying on `service.name` alone only gets key-level limiting, not a per-agent sub-limit).

**Important if a key genuinely has only one real agent**: the 20% default still applies unless overridden — it does not detect "this key only has one agent" and skip sub-limiting. A single-agent-per-key deployment relying on the full key rpm for that one agent should set an explicit override (see below) closer to 1.0, or the effective throughput for that agent will be capped at 20% of what the key alone would otherwise allow.

### Adjusting a per-agent quota

```bash
# View current quota (default or override) for an agent under a key
curl -s "http://localhost:8001/admin/keys/{key_id}/agents/{agent_id}/quota?admin_key=<ADMIN_API_KEY>"

# Response:
{"key_id": 42, "agent_id": "worker-1", "quota_pct": 0.20, "is_override": false}

# Set an override — e.g. give this agent 50% of the key's rpm
curl -s -X PUT "http://localhost:8001/admin/keys/{key_id}/agents/{agent_id}/quota" \
  -H "Content-Type: application/json" \
  -d '{"admin_key": "<ADMIN_API_KEY>", "quota_pct": 0.5}'
```

`key_id` is the numeric id from the key-creation response or `GET /v1/keys` (api_svc) — never the raw secret key string, which shouldn't appear in a URL path. Quota changes take effect within a few minutes (cached the same way `rate_limit_rpm` is, not read fresh on every request) — not instantly.

### 429 response headers

```
HTTP/1.1 429 Too Many Requests
Retry-After: 12
X-RateLimit-Key-Remaining: 0
X-RateLimit-Agent-Remaining: 0
```

`X-RateLimit-Agent-Remaining` is only present when an `agent_id` was resolvable for the request (see above) — its absence means only the key-level limit was checked, not that the agent has unlimited quota.

### Request body size limit

Rate limiting bounds *how many* requests a key or IP can make; `INGEST_MAX_BODY_BYTES` bounds how big each one can be. `POST /v1/ingest` and `POST /v1/deploy` reject any body larger than it with `413 {"detail": "Request body exceeds N bytes."}` — default 10 MiB (`10485760`). `MAX_BATCH_SIZE` still caps the event *count* per batch (default 500); this caps the bytes, so a batch of 500 events cannot smuggle in a multi-gigabyte payload one event wide.

The check runs in the outermost middleware, before anything reads the body: `Content-Length` is compared to the cap first and an oversized declaration is refused without reading a byte, then the body is read with a streaming cap, so a request with no `Content-Length` (chunked transfer encoding) or a lying one is cut off the moment it passes the limit. Nothing downstream — auth, rate-limit bucketing, JSON parsing, the route, the store — sees a rejected request. Applies on the trusted-gateway path too; the process that buffers the body is the one whose memory is at stake.

Rejections are logged at WARNING with the declared and received sizes and the client address, never the body:

```
WARNING dunetrace.ingest — Request body too large; rejected with 413. path=/v1/ingest declared=52428800 received=0 limit=10485760 client=203.0.113.7
```

`received=0` means `Content-Length` alone was enough to reject; a non-zero value is how far a streamed body got before the cap cut it off. A 413 is not counted against the rate limit — the api_key inside the body is never read, so there is no key bucket to charge.

The SDK treats 413 as permanent (no retry, batch dropped with a logged error), so one oversized batch never wedges the durable queue behind it. `POST /v1/otlp/traces` is capped separately by `OTLP_MAX_BODY_BYTES` (see `docs/integrations/otel-ingestion.md`), using the same bounded reader (`ingest_svc/body_limits.py`).
