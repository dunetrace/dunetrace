# Getting Started

Get the full Dunetrace pipeline running locally in about 10 minutes.

---

## Prerequisites

- Python 3.11+
- PostgreSQL (or Docker)
- Node.js 18+ (for the dashboard)

---

## 1. Start Postgres

**With Docker (recommended):**
```bash
docker run -d \
  --name dunetrace-pg \
  -e POSTGRES_USER=dunetrace \
  -e POSTGRES_PASSWORD=dunetrace \
  -e POSTGRES_DB=dunetrace \
  -p 5432:5432 \
  postgres:16-alpine
```

**Without Docker — create the user and database manually:**
```bash
psql -U $(whoami) postgres -c "CREATE USER dunetrace WITH PASSWORD 'dunetrace';"
psql -U $(whoami) postgres -c "CREATE DATABASE dunetrace OWNER dunetrace;"
```

---

## 2. Create a Virtual Environment

```bash
cd dunetrace
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install uvicorn fastapi asyncpg pydantic pydantic-settings python-dotenv \
            psycopg2-binary
```

---

## 4. Start the Services

Open four terminals. Run each from its service directory:

**Terminal 1 — Ingest API**
```bash
cd services/ingest
PYTHONPATH=. uvicorn app.main:app --reload --port 8001
```
Wait for: `Application startup complete.`

**Terminal 2 — Detector Worker**
```bash
cd services/detector
PYTHONPATH=.:../../packages/sdk-py SHADOW_MODE=false python -m app.worker
```
Wait for: `Detector worker started.`

**Terminal 3 — Alerts Worker**
```bash
cd services/alerts
PYTHONPATH=.:../../packages/sdk-py \
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T.../B.../... \
python -m alerts_svc.worker
```
Wait for: `Alert worker started.`

**Terminal 4 — Customer API**
```bash
cd services/api
PYTHONPATH=.:../../packages/sdk-py \
uvicorn api_svc.main:app --reload --port 8002
```
Wait for: `Application startup complete.`

---

## 5. Send a Test Run

```bash
curl -s -X POST localhost:8001/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "dt_dev_test",
    "agent_id": "test-agent",
    "events": [
      {"event_type":"run.started",   "run_id":"run-test-1","agent_id":"test-agent","agent_version":"abc1","step_index":0,"payload":{"input_hash":"aa","tools":["web_search"]}},
      {"event_type":"tool.called",   "run_id":"run-test-1","agent_id":"test-agent","agent_version":"abc1","step_index":1,"payload":{"tool_name":"web_search","args_hash":"bb"}},
      {"event_type":"tool.called",   "run_id":"run-test-1","agent_id":"test-agent","agent_version":"abc1","step_index":2,"payload":{"tool_name":"web_search","args_hash":"bb"}},
      {"event_type":"tool.called",   "run_id":"run-test-1","agent_id":"test-agent","agent_version":"abc1","step_index":3,"payload":{"tool_name":"web_search","args_hash":"bb"}},
      {"event_type":"tool.called",   "run_id":"run-test-1","agent_id":"test-agent","agent_version":"abc1","step_index":4,"payload":{"tool_name":"web_search","args_hash":"bb"}},
      {"event_type":"tool.called",   "run_id":"run-test-1","agent_id":"test-agent","agent_version":"abc1","step_index":5,"payload":{"tool_name":"web_search","args_hash":"bb"}},
      {"event_type":"run.completed", "run_id":"run-test-1","agent_id":"test-agent","agent_version":"abc1","step_index":6,"payload":{"exit_reason":"final_answer"}}
    ]
  }'
```

Expected response: `{"accepted":7,"batch_id":"...","queued_at":...}`

---

## 6. Verify the Pipeline

After ~15 seconds:

```bash
# Check a signal was detected
curl -s -H "Authorization: Bearer dt_dev_test" \
  http://localhost:8002/v1/agents/test-agent/signals | jq '.signals[].title'
# → "Tool loop: `web_search` called 5× in 5 steps"

# Check the full run
curl -s -H "Authorization: Bearer dt_dev_test" \
  http://localhost:8002/v1/runs/run-test-1 | jq '{
    steps: .step_count,
    exit: .exit_reason,
    signals: [.signals[].title]
  }'
```

You should also see a Slack message in your `#agent-alerts` channel.

---

## 6. Instrument a Real Agent

```python
# Install
pip install langchain langchain-openai langchain-community duckduckgo-search

# Run the example agent
cd packages/sdk-py
OPENAI_API_KEY=sk-... python examples/langchain_agent.py

# Try different failure scenarios
OPENAI_API_KEY=sk-... SCENARIO=tool_loop python examples/langchain_agent.py
OPENAI_API_KEY=sk-... SCENARIO=prompt_injection python examples/langchain_agent.py
```

---

## Environment Variables Reference

Create `.env` files in each service directory instead of exporting variables each time.

**`services/ingest/.env`**
```bash
DATABASE_URL=postgresql://dunetrace:dunetrace@localhost:5432/dunetrace
AUTH_MODE=dev
PORT=8001
```

**`services/detector/.env`**
```bash
DATABASE_URL=postgresql://dunetrace:dunetrace@localhost:5432/dunetrace
SHADOW_MODE=false
STALL_TIMEOUT=60
POLL_INTERVAL=5
```

**`services/alerts/.env`**
```bash
DATABASE_URL=postgresql://dunetrace:dunetrace@localhost:5432/dunetrace
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
SLACK_MIN_SEVERITY=HIGH
POLL_INTERVAL=10
```

**`services/api/.env`**
```bash
DATABASE_URL=postgresql://dunetrace:dunetrace@localhost:5432/dunetrace
AUTH_MODE=dev
PORT=8002
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'detector_svc'`**
You're not in the service directory, or `PYTHONPATH` isn't set.
```bash
cd services/detector
PYTHONPATH=.:../../packages/sdk-py python -m app.worker
```

**`role "dunetrace" does not exist`**
Postgres user hasn't been created. Run the `CREATE USER` command in step 1.

**`column "shadow" does not exist`**
The detector worker adds this column on first start. Start the ingest API first to create the base schema, then start the detector.
```bash
# Or add it manually:
psql -U dunetrace -d dunetrace -c \
  "ALTER TABLE failure_signals ADD COLUMN IF NOT EXISTS shadow BOOLEAN NOT NULL DEFAULT TRUE;"
```

**Events accepted but not appearing in the DB**
The ingest background task is failing. Check the ingest terminal for `ERROR` lines after the `202` log. Restart ingest with `PYTHONPATH=.:../../packages/sdk-py` to ensure the SDK is findable.

**Detector not picking up runs**
Check that `SHADOW_MODE=false` is set. Use the precision report to confirm signals are being written:
```bash
python scripts/precision_report.py
```
