# Alerts

The alerts worker watches for new failure signals and delivers them to Slack or any webhook endpoint within 10–15 seconds of detection.

---

## How It Works

The alerts pipeline runs in a 10-second polling loop:

```
1. Fetch unalerted live signals (shadow=FALSE, alerted=FALSE)
2. For each signal:
   a. explain(signal)  →  Explanation
   b. format for Slack (Block Kit) or webhook (signed JSON)
   c. POST to destination with retry
3. Mark successfully delivered signals as alerted=TRUE
```

**At-least-once delivery:** A signal is marked `alerted=TRUE` only after at least one destination succeeds. If the worker crashes between sending and marking, the signal re-delivers on restart. Design your receivers to be idempotent using `(run_id, failure_type, detected_at)` as the deduplication key.

---

## Slack Alerts

### Setup

1. Go to [api.slack.com/apps](https://api.slack.com/apps) → Create New App → From scratch
2. Add **Incoming Webhooks** → Activate → Add New Webhook to Workspace → pick a channel
3. Copy the webhook URL

```bash
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T.../B.../...
export SLACK_CHANNEL=#agent-alerts       # display only, not functional
export SLACK_MIN_SEVERITY=HIGH           # only alert on HIGH and CRITICAL
```

### What the Message Looks Like

Each Slack alert is a formatted Block Kit message containing:

- **Header** — severity emoji + failure type title
- **Context line** — agent ID, version, run ID, step index, confidence %
- **What happened** — plain-English description
- **Why it matters** — business impact
- **Evidence** — formatted code block with raw evidence values
- **Top fix** — code snippet (if < 800 characters)
- **View Run button** — deep link to `https://app.dunetrace.io/runs/{run_id}`

### Severity Filter

Only signals at or above `SLACK_MIN_SEVERITY` are sent to Slack. Severity order: `CRITICAL > HIGH > MEDIUM > LOW`.

```bash
SLACK_MIN_SEVERITY=HIGH    # sends CRITICAL + HIGH
SLACK_MIN_SEVERITY=MEDIUM  # sends CRITICAL + HIGH + MEDIUM
SLACK_MIN_SEVERITY=LOW     # sends everything
```

All live (`shadow=FALSE`) signals are sent to webhooks regardless of this filter.

---

## Webhook Alerts

### Setup

```bash
export WEBHOOK_URL=https://your-endpoint.com/dunetrace/alerts
export WEBHOOK_SECRET=your-signing-secret   # optional but recommended
```

### Payload Schema

```json
{
  "schema_version": "1.0",
  "event":          "failure_signal",
  "sent_at":        1771632965.239,
  "failure_type":   "TOOL_LOOP",
  "severity":       "HIGH",
  "run_id":         "run-f4a9b2c1",
  "agent_id":       "research-agent",
  "agent_version":  "a7f3d9b2",
  "step_index":     11,
  "confidence":     0.95,
  "evidence": {
    "tool":   "web_search",
    "count":  5,
    "window": 5
  },
  "explanation": {
    "title":            "Tool loop: `web_search` called 5× in 5 steps",
    "what":             "The agent called `web_search` 5 consecutive times...",
    "why_it_matters":   "Looping agents burn tokens and cost money...",
    "evidence_summary": "Tool `web_search` called 5× in steps 7–11. Confidence: 95%.",
    "suggested_fixes": [
      {
        "description": "Add a per-tool call limit",
        "language":    "python",
        "code":        "MAX_CALLS_PER_TOOL = 3\nif tool_call_counts[tool] > MAX_CALLS_PER_TOOL:..."
      }
    ]
  }
}
```

### Request Headers

```
Content-Type:          application/json
X-Dunetrace-Event:     failure_signal
X-Dunetrace-Version:   1.0
X-Dunetrace-Signature: sha256=<hmac>   (only if WEBHOOK_SECRET is set)
```

### Verifying Signatures

If `WEBHOOK_SECRET` is configured, every request includes an HMAC-SHA256 signature over the raw request body.

```python
import hmac, hashlib

def verify_dunetrace_signature(body: bytes, signature_header: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(
        secret.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header)
```

### Integrating with PagerDuty

Route webhook alerts to PagerDuty by setting `WEBHOOK_URL` to your PagerDuty Events API v2 endpoint and mapping the payload fields in a small bridge function. Alternatively, use Zapier or Make to route webhook events to any destination without code.

---

## Retry Behavior

HTTP delivery uses exponential backoff:

| Attempt | Delay before retry |
|---|---|
| 1 | immediate |
| 2 | 2 seconds |
| 3 | 4 seconds |
| 4 (final) | 8 seconds |

After 3 retries without success, the signal is left as `alerted=FALSE` and will be retried on the next poll cycle (10 seconds later).

---

## Configuration Reference

All configuration is via environment variables or a `.env` file in the alerts service directory.

```bash
# Slack
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...   # required to enable Slack
SLACK_CHANNEL=#agent-alerts                               # display only
SLACK_MIN_SEVERITY=HIGH                                   # LOW | MEDIUM | HIGH | CRITICAL

# Webhook
WEBHOOK_URL=https://your-endpoint.com/alerts             # required to enable webhook
WEBHOOK_SECRET=your-secret-key                           # enables HMAC signing

# Worker behavior
DATABASE_URL=postgresql://dunetrace:dunetrace@localhost:5432/dunetrace
POLL_INTERVAL=10        # seconds between cycles
BATCH_SIZE=50           # signals per cycle
MAX_RETRIES=3           # HTTP retry attempts
RETRY_BACKOFF=2.0       # seconds, doubles each retry
```

---

## Running the Alerts Worker

```bash
cd services/alerts
PYTHONPATH=.:../../packages/sdk-py \
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/... \
python -m alerts_svc.worker
```

On startup, the worker logs which destinations are enabled:

```
INFO dunetrace.alerts — Alert destinations: Slack (#agent-alerts, min=HIGH)
INFO dunetrace.alerts — Alert worker started. poll_interval=10.0s
```
