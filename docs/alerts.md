# Alerts

Dunetrace supports two alert destinations: Slack and a generic webhook. Both can be active simultaneously.

---

## Slack

Add to your `.env`:

```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx/yyy/zzz
SLACK_CHANNEL=#agent-alerts
# Options: LOW | MEDIUM | HIGH | CRITICAL  (default: LOW — all severities alerted)
SLACK_MIN_SEVERITY=LOW
```

Get a webhook URL from [api.slack.com/messaging/webhooks](https://api.slack.com/messaging/webhooks). Restart the alerts worker to pick up changes:

```bash
docker compose up -d --force-recreate alerts
```

Each Slack alert includes: failure type, severity, confidence, what happened, why it matters, a concrete code fix targeted at the specific failure pattern detected, a one-line rate context summary showing how common this pattern is for the agent, and a **View Run** button that deep-links directly to that run's detail panel in the dashboard.

Three action buttons let you respond directly from Slack:

- **Mark resolved** — sets `resolved_at` on the signal.
- **Not a problem** — records a false positive (`agent_detector_overrides`); after 3 false positives for the same `(agent_id, failure_type)`, that detector is silenced for that agent until manually reset.
- **Snooze 24h** — mutes this `(agent_id, failure_type)` for 24 hours without affecting the false-positive count. Unlike "Not a problem," snoozing is a deliberate temporary decision ("I know about this, stop paging me today"), not accumulated feedback about detector accuracy — the two mechanisms are independent and don't interact.

When token data is available for the run, a `:moneybag:` line is included showing total tokens consumed and estimated cost (e.g. `:moneybag: *Tokens:* 581 (wasted)  *Cost:* ~$0.00`). This uses actual `prompt_tokens + completion_tokens` from SDK-recorded `llm.responded` events.

> **Note:** `docker compose restart alerts` does not re-read `.env`. Use `docker compose up -d alerts` to recreate the container and pick up env var changes.

### Rate context in Slack alerts

The rate context line appears above the "What happened" block and describes how often this failure type has affected recent runs for the same agent:

| Condition | Message |
|---|---|
| First time this failure type has appeared (last 7 days) | `:information_source: First occurrence of this pattern in the last 7 days` |
| Recurring but not yet systemic | `:bar_chart: 5/20 runs affected (25%) in the last 7 days` |
| Systemic (≥10% of runs in last 7 days affected) | `:warning: *Systemic pattern* — 8/12 runs affected (67% of runs in the last 7 days)` |

Rate context is computed per `(agent_id, failure_type)` pair at alert time from the `failure_signals` table. If the lookup fails (e.g. DB contention), the signal is still delivered without a rate context line.

---

## Generic webhook

Works with PagerDuty, Linear, or any custom endpoint.

```bash
WEBHOOK_URL=https://your-endpoint.example.com/alerts
WEBHOOK_SECRET=your-hmac-secret   # optional: enables HMAC-SHA256 signature header
```

When `WEBHOOK_SECRET` is set, each request includes an `X-Dunetrace-Signature` header containing `HMAC-SHA256(body, secret)`. Use `(run_id, failure_type, detected_at)` as the idempotency key, the alerts worker delivers at-least-once, so duplicates are possible if it crashes between sending and marking.

The webhook payload:

```json
{
  "schema_version": "1.0",
  "event": "signal.detected",
  "run_id": "...",
  "agent_id": "...",
  "failure_type": "TOOL_LOOP",
  "severity": "HIGH",
  "confidence": 0.95,
  "evidence": { "tool": "web_search", "count": 6, ... },
  "explanation": {
    "title": "...",
    "what": "...",
    "why_it_matters": "...",
    "suggested_fixes": [{ "description": "...", "language": "python", "code": "..." }]
  }
}
```

---

## Delivery guarantees

The alerts worker polls every 10 seconds for unalerted signals (`shadow=FALSE AND alerted=FALSE`). It calls the explainer, formats the payload, and POSTs with exponential backoff (up to 3 attempts). A signal is marked `alerted=TRUE` only after at least one destination succeeds.

If the worker crashes between sending and marking, the signal will be re-sent on restart. Design receivers to be idempotent.

### Running more than one alerts worker

Safe, but only because signals are **claimed** before delivery. `alerted` is set
after a successful send, so it can't serve as the claim on its own — two workers
scanning the same window would both see `alerted = FALSE`, both deliver, and you'd
get duplicate Slack messages. The `ALERT_DEDUP_WINDOW` check doesn't prevent it
either, since both read the window before either writes it.

`claim_unalerted_signals` stamps `failure_signals.alert_claimed_at` in the same
statement that selects the rows (`FOR UPDATE SKIP LOCKED`), so a row can only be
picked up by one worker. To actually scale throughput, also shard: set
`SHARD_COUNT=N` and give each replica a distinct `SHARD_INDEX`, exactly as with the
detector. Sharding on `agent_id` matters beyond throughput — the worker sends one
alert per `(org_id, agent_id, failure_type)` group, so a group must stay whole on
one worker.

Claims expire after `CLAIM_TIMEOUT_SECS` (default 300) so a worker that dies
mid-delivery doesn't strand its rows. Set it comfortably above your worst-case
delivery time for a full batch: if it expires while a worker is still alive and
working, another replica takes the row and the alert goes out twice.

See [architecture.md](architecture.md#alerts-worker-sharding-plus-claiming).

---

## Weekly digest

A weekly summary sent to Slack every Monday at 9am UTC (configurable). Covers the past 7 days:

- Top 5 failure types by affected run count (with %)
- Top 5 agents by signal volume (with dominant failure type)
- Systemic patterns — failure types affecting ≥10% of runs per agent
- Issues opened and resolved this week
- Dashboard button

Enable by adding to your `.env`:

```bash
DIGEST_ENABLED=true
DASHBOARD_URL=https://your-dashboard-url
# Optional overrides (defaults shown)
DIGEST_DAY=0          # 0=Monday … 6=Sunday
DIGEST_HOUR=9         # UTC hour
```

The digest also requires `SLACK_WEBHOOK_URL` to be set — it uses the same Slack destination as alert delivery.

Delivery is deduplicated via a `digest_log` table. If a digest was sent within the last 6 days, it will not send again even if the worker restarts. If there were no runs in the last 7 days, the digest is skipped but the sent timestamp is still logged (so it won't retry until the following week).

---

## Shadow mode

Signals in shadow mode are stored and visible in the dashboard but never delivered to Slack or webhooks. See [detectors.md](detectors.md#shadow-mode) for how to promote a detector to live.

---

## Per-detector destination routing

By default every alertable signal goes to every globally-enabled destination (Slack and/or the generic webhook). To route specific failure types to specific destinations, add a `destinations` list to that detector's block in `detectors.yml`:

```yaml
default:
  tool_loop:
    threshold: 3
    destinations: [slack]   # only Slack, even if WEBHOOK_URL is also configured
```

Valid values: `slack`, `webhook`, `linear` (Linear delivery itself is not yet implemented — a detector routed only to `linear` today delivers nowhere until that lands). Restart the alerts worker (`docker compose restart alerts`) to apply changes.
