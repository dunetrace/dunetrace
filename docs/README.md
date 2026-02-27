# Dunetrace Documentation

Production observability for AI agents. Detect failures, understand what went wrong, fix it fast.

---

## Contents

### Getting Started
- [What is Dunetrace?](./01-introduction.md) — overview, what problem it solves, key design decisions
- [Getting Started](./10-getting-started.md) — local setup, first test run, troubleshooting

### Concepts
- [Core Concepts](./02-core-concepts.md) — events, runs, signals, explanations, shadow mode
- [Architecture](./03-architecture.md) — service breakdown, data flow, DB schema, performance
- [How Tracing Works](./06-tracing.md) — hot path, drain thread, hashing, step indices, overhead

### Reference
- [SDK Reference](./04-sdk-reference.md) — Python SDK, context manager, LangChain adapter, privacy model
- [Detectors](./05-detectors.md) — all 6 Tier 1 detectors, thresholds, evidence, graduation guide
- [Alerts](./07-alerts.md) — Slack setup, webhook payload, signing, retry behavior, config
- [REST API](./08-api-reference.md) — all endpoints, request/response shapes, code examples
- [Dashboard](./09-dashboard.md) — agents view, runs view, signals view, timeline view

### Help
- [FAQ](./11-faq.md) — common questions on privacy, performance, detectors, and roadmap

---

## Quick Reference

**Start the pipeline locally:**
```bash
# Postgres
docker run -d --name dunetrace-pg \
  -e POSTGRES_USER=dunetrace -e POSTGRES_PASSWORD=dunetrace -e POSTGRES_DB=dunetrace \
  -p 5432:5432 postgres:16-alpine

# Services (4 terminals)
cd services/ingest  && PYTHONPATH=. uvicorn app.main:app --reload --port 8001
cd services/detector && PYTHONPATH=.:../../packages/sdk-py SHADOW_MODE=false python -m app.worker
cd services/alerts  && PYTHONPATH=.:../../packages/sdk-py SLACK_WEBHOOK_URL=... python -m alerts_svc.worker
cd services/api     && PYTHONPATH=.:../../packages/sdk-py uvicorn api_svc.main:app --reload --port 8002
```

**Instrument an agent:**
```python
from dunetrace import Dunetrace
dt = Dunetrace(api_key="dt_dev_test", agent_id="my-agent",
               ingest_url="http://localhost:8001/v1/ingest")

with dt.run(user_input, model="gpt-4o", tools=["web_search"]) as run:
    run.tool_called("web_search", {"query": "..."})
    result = search(query)
    run.tool_responded("web_search", success=True, output_length=len(result))
```

**Query signals:**
```bash
curl -s -H "Authorization: Bearer dt_dev_test" \
  "http://localhost:8002/v1/agents/my-agent/signals?severity=HIGH" | jq '.signals[].title'
```

**Check detector precision before graduating:**
```bash
python scripts/precision_report.py
python scripts/precision_report.py --inspect TOOL_LOOP
```
