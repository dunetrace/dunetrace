# Alerts

Dunetrace supports two alert destinations: Slack and a generic webhook. Both can be active simultaneously.

---

## Slack

Add to your `.env`:

```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx/yyy/zzz
SLACK_CHANNEL=#agent-alerts
# Options: LOW | MEDIUM | HIGH | CRITICAL  (default: HIGH)
SLACK_MIN_SEVERITY=HIGH
```

Get a webhook URL from [api.slack.com/messaging/webhooks](https://api.slack.com/messaging/webhooks). Restart the alerts worker to pick up changes:

```bash
docker compose up -d --force-recreate alerts
```

Each Slack alert includes: failure type, severity, what happened, why it matters, and a concrete code fix targeted at the specific failure pattern detected.

---

## Generic webhook

Works with PagerDuty, Linear, or any custom endpoint.

```bash
WEBHOOK_URL=https://your-endpoint.example.com/alerts
WEBHOOK_SECRET=your-hmac-secret   # optional — enables HMAC-SHA256 signature header
```

When `WEBHOOK_SECRET` is set, each request includes an `X-Dunetrace-Signature` header containing `HMAC-SHA256(body, secret)`. Use `(run_id, failure_type, detected_at)` as the idempotency key — the alerts worker delivers at-least-once, so duplicates are possible if it crashes between sending and marking.

The webhook payload:

```json
{
  "schema_version": "1",
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

---

## Shadow mode

Signals in shadow mode are stored and visible in the dashboard but never delivered to Slack or webhooks. See [detectors.md](detectors.md#shadow-mode) for how to promote a detector to live.
