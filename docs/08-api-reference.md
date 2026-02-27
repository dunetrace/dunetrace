# REST API Reference

The Dunetrace customer API (port 8002) provides read-only access to your agent runs, events, and failure signals. All endpoints include signal explanations inline.

---

## Authentication

All endpoints require a bearer token:

```
Authorization: Bearer dt_live_your_key_here
```

In local development (`AUTH_MODE=dev`), any non-empty token is accepted:

```bash
curl -H "Authorization: Bearer dt_dev_test" http://localhost:8002/v1/agents
```

---

## Base URL

```
http://localhost:8002   # local development
https://api.dunetrace.io   # production (coming soon)
```

Interactive docs (Swagger UI) are available at `/docs` when the service is running.

---

## Endpoints

### `GET /health`

Health check. No authentication required.

**Response:**
```json
{
  "status":  "ok",
  "version": "0.1.0",
  "db":      "ok"
}
```

---

### `GET /v1/agents`

List all agents with signal counts and last-seen timestamps.

**Query Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `offset` | int | 0 | Pagination offset |
| `limit` | int | 20 | Results per page (max 100) |

**Response:**
```json
{
  "agents": [
    {
      "agent_id":       "research-agent-v2",
      "last_seen":      1771632965.0,
      "run_count":      847,
      "signal_count":   23,
      "critical_count": 2,
      "high_count":     8
    }
  ],
  "page": {
    "total":    4,
    "offset":   0,
    "limit":    20,
    "has_more": false
  }
}
```

---

### `GET /v1/agents/{agent_id}/runs`

List runs for a specific agent, newest first.

**Query Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `offset` | int | 0 | Pagination offset |
| `limit` | int | 20 | Results per page (max 100) |
| `has_signals` | bool | — | Filter to runs with (`true`) or without (`false`) signals |

**Response:**
```json
{
  "runs": [
    {
      "run_id":        "run-f4a9b2c1",
      "agent_id":      "research-agent-v2",
      "agent_version": "a7f3d9b2",
      "started_at":    1771632783.0,
      "completed_at":  1771632911.0,
      "exit_reason":   "completed",
      "step_count":    12,
      "signal_count":  2,
      "has_signals":   true
    }
  ],
  "page": { "total": 5, "offset": 0, "limit": 20, "has_more": false }
}
```

**`exit_reason` values:** `completed`, `error`, `stalled`, `max_iterations`

---

### `GET /v1/runs/{run_id}`

Full run detail including all events and signals with explanations.

**Response:**
```json
{
  "run_id":        "run-f4a9b2c1",
  "agent_id":      "research-agent-v2",
  "agent_version": "a7f3d9b2",
  "started_at":    1771632783.0,
  "completed_at":  1771632911.0,
  "exit_reason":   "completed",
  "step_count":    12,
  "events": [
    {
      "event_type":    "run.started",
      "step_index":    0,
      "timestamp":     1771632783.0,
      "payload":       { "input_hash": "e3b0c44298fc", "tools": ["web_search"] },
      "parent_run_id": null
    },
    {
      "event_type":    "tool.called",
      "step_index":    3,
      "timestamp":     1771632799.0,
      "payload":       { "tool_name": "web_search", "args_hash": "a1b2c3d4" },
      "parent_run_id": null
    }
  ],
  "signals": [
    {
      "id":               1,
      "failure_type":     "TOOL_LOOP",
      "severity":         "HIGH",
      "step_index":       11,
      "confidence":       0.95,
      "detected_at":      1771632911.0,
      "evidence":         { "tool": "web_search", "count": 5, "window": 5 },
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
  ]
}
```

---

### `GET /v1/agents/{agent_id}/signals`

List live (non-shadow) failure signals for an agent, newest first.

**Query Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `offset` | int | 0 | Pagination offset |
| `limit` | int | 20 | Results per page (max 100) |
| `severity` | string | — | Filter: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL` |
| `failure_type` | string | — | Filter: `TOOL_LOOP`, `TOOL_THRASHING`, etc. |

**Response:**
```json
{
  "signals": [
    {
      "id":               1,
      "failure_type":     "TOOL_LOOP",
      "severity":         "HIGH",
      "run_id":           "run-f4a9b2c1",
      "agent_id":         "research-agent-v2",
      "agent_version":    "a7f3d9b2",
      "step_index":       11,
      "confidence":       0.95,
      "detected_at":      1771632911.0,
      "evidence":         { "tool": "web_search", "count": 5, "window": 5 },
      "alerted":          true,
      "title":            "Tool loop: `web_search` called 5× in 5 steps",
      "what":             "...",
      "why_it_matters":   "...",
      "evidence_summary": "...",
      "suggested_fixes":  [...]
    }
  ],
  "page": { "total": 23, "offset": 0, "limit": 20, "has_more": true }
}
```

---

## Pagination

All list endpoints use offset-based pagination:

```bash
# Page 1
GET /v1/agents/my-agent/signals?offset=0&limit=20

# Page 2
GET /v1/agents/my-agent/signals?offset=20&limit=20
```

Check `page.has_more` to determine if there are more results.

---

## Timestamps

All timestamps are **Unix epoch seconds** (float). To convert:

```python
from datetime import datetime, timezone
dt = datetime.fromtimestamp(1771632911.0, tz=timezone.utc)
# → 2026-02-21 01:15:11 UTC
```

---

## Error Responses

```json
// 401 — missing or invalid API key
{ "detail": "Missing Authorization: Bearer <api_key>" }

// 404 — run not found
{ "detail": "Run 'run-xyz' not found" }

// 422 — invalid query parameter
{ "detail": [{ "loc": ["query", "limit"], "msg": "ensure this value is greater than 0" }] }
```

---

## Code Examples

### Python

```python
import requests

BASE = "http://localhost:8002"
HEADERS = {"Authorization": "Bearer dt_dev_test"}

# List agents
agents = requests.get(f"{BASE}/v1/agents", headers=HEADERS).json()

# Get runs with signals
runs = requests.get(
    f"{BASE}/v1/agents/research-agent/runs",
    params={"has_signals": True},
    headers=HEADERS,
).json()

# Full run detail
run = requests.get(f"{BASE}/v1/runs/run-f4a9b2c1", headers=HEADERS).json()
for signal in run["signals"]:
    print(f"{signal['severity']}: {signal['title']}")

# Filter signals by severity
signals = requests.get(
    f"{BASE}/v1/agents/research-agent/signals",
    params={"severity": "CRITICAL"},
    headers=HEADERS,
).json()
```

### curl

```bash
# List all agents
curl -s -H "Authorization: Bearer dt_dev_test" \
  http://localhost:8002/v1/agents | jq .

# Get recent CRITICAL signals
curl -s -H "Authorization: Bearer dt_dev_test" \
  "http://localhost:8002/v1/agents/research-agent/signals?severity=CRITICAL" | jq '.signals[].title'

# Get full run with all events
curl -s -H "Authorization: Bearer dt_dev_test" \
  http://localhost:8002/v1/runs/run-f4a9b2c1 | jq '{
    exit: .exit_reason,
    steps: .step_count,
    signals: [.signals[].title]
  }'
```

---

## Running the API

```bash
cd services/api
PYTHONPATH=.:../../packages/sdk-py \
uvicorn api_svc.main:app --reload --port 8002
```

Swagger UI: [http://localhost:8002/docs](http://localhost:8002/docs)
