# FAQ

---

## General

**What makes Dunetrace different from LangSmith or Langfuse?**

LangSmith and Langfuse are tracing tools — they record what your agent did and help you replay and debug individual runs. Dunetrace is a production monitor — it runs continuously, detects failures in real-time, and alerts you when something breaks. You would use both: LangSmith for development and debugging, Dunetrace for production alerting.

**What makes Dunetrace different from AgentWatch?**

AgentWatch is a local developer tool — you run it during development to visualize a single agent's call graph. It has no backend, no alerting, and no multi-tenant support. Dunetrace watches agents running for real users in production and pages your team when they fail.

**Does Dunetrace support frameworks other than LangChain?**

Yes. The `dunetrace.run()` context manager works with any framework — you just call the emit methods manually. The LangChain callback is a convenience wrapper. An AutoGen adapter is planned.

**Do I need an OpenAI key to use Dunetrace?**

No. The SDK, detectors, explain layer, and all infrastructure are LLM-free. You only need an OpenAI (or other) key if you're building an agent that uses one. Dunetrace itself makes no LLM calls.

---

## Privacy and Security

**Does Dunetrace see my prompts or user data?**

No. All content fields — user inputs, prompts, tool arguments, tool outputs, LLM responses — are SHA-256 hashed before leaving your process. Dunetrace sees the hashes, not the content. The only plain-text metadata sent is: event types, tool names, model names, step indices, timestamps, and token counts.

**Can I run Dunetrace completely on-premises?**

Yes. All five services (ingest, detector, alerts, customer API, dashboard) are open source and designed to run anywhere that runs Python and Postgres. Nothing phones home.

**How do I verify the webhook signature?**

```python
import hmac, hashlib

def verify(body: bytes, header: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, header)
```

---

## Detectors

**Why are all detectors in shadow mode by default?**

False positives destroy trust. If your first alert is a false alarm, the on-call engineer loses confidence in the system and may disable alerting entirely. Shadow mode lets you validate a detector on real traffic before committing to waking anyone up. See the [Detectors](/docs/detectors#shadow-mode-and-graduation) doc for the graduation process.

**How do I know when a detector is ready to graduate?**

Run `python scripts/precision_report.py --inspect TOOL_LOOP` and manually review 20 signals. If ≥ 80% are real failures (true positives), graduate the detector. TOOL_LOOP and PROMPT_INJECTION_SIGNAL typically graduate quickly; TOOL_AVOIDANCE needs more scrutiny.

**Can I write my own detector?**

Yes. Subclass `BaseDetector`, implement `detect(state: RunState) -> Optional[FailureSignal]`, and add it to the detector worker's list:

```python
from dunetrace.models import FailureSignal, FailureType, Severity
from dunetrace.detectors import BaseDetector

class MyCustomDetector(BaseDetector):
    name = "MY_CUSTOM_FAILURE"

    def detect(self, state):
        if some_condition(state):
            return FailureSignal(
                failure_type=FailureType("MY_CUSTOM_FAILURE"),
                severity=Severity.HIGH,
                confidence=0.90,
                evidence={"detail": "..."},
                ...
            )
        return None
```

**Why doesn't Dunetrace use LLMs for detection?**

Three reasons: latency (templates are <1ms vs. 500ms+ for an LLM call), cost (zero per-signal cost), and consistency (same signal always produces the same output, making testing and debugging predictable). Tier 2 semantic detectors — which require reading the actual conversation content — will use LLMs, but they're opt-in and rate-limited.

**What's the difference between Tier 1 and Tier 2 detectors?**

Tier 1 detectors are fully structural — they operate on event metadata (tool names, counts, step indices) and never touch content. They're deterministic, instant, and free. Tier 2 detectors require reading the actual conversation content and making LLM judgments (e.g., "was the user dissatisfied?"). They're slower, have per-call cost, and are not yet built.

---

## Performance

**What overhead does the SDK add to my agent?**

Less than 500μs per run in the worst case. The hot path is a single `deque.append()` call (<1μs). All I/O happens on a background drain thread. The agent thread never waits for Dunetrace.

**What happens if the ingest API is down?**

The SDK buffers up to 10,000 events in memory and retries delivery every 200ms. At normal emission rates, this covers roughly 30 minutes of outage. If the buffer fills, oldest events are dropped — your agent continues running normally.

**How does Dunetrace scale?**

The current architecture handles ~100 runs/second comfortably on a single Postgres instance. For higher throughput, the `events` table migrates to ClickHouse, and the detector worker scales horizontally with a distributed lock (Redis or Postgres advisory locks) to prevent double-processing.

---

## Operations

**How do I run all the tests?**

```bash
cd services/detector && python -m unittest tests.test_worker -v
cd services/explainer && python -m unittest tests.test_explainer -v
cd services/alerts    && python -m unittest tests.test_alerts -v
cd services/api       && python -m unittest tests.test_api -v
cd packages/sdk-py    && python -m unittest tests.test_detectors -v
```

Total: 214 tests.

**How do I add a new Slack channel or webhook destination?**

Set the environment variables and restart the alerts worker. Multiple webhooks are not yet supported — a single `WEBHOOK_URL` per worker instance. To fan out to multiple destinations, point `WEBHOOK_URL` at a router service (e.g., a small AWS Lambda or Zapier webhook).

**How do I deploy to production?**

The recommended path is a single VM (DigitalOcean, Render, Railway, or EC2) with Docker Compose orchestrating all five services. A `docker-compose.yml` is the next infrastructure milestone. See the deployment note in your account memory — you've chosen to keep things local until the pipeline is solid.

**Can I use this with a cloud-hosted Postgres (RDS, Supabase, Neon)?**

Yes. Set `DATABASE_URL` to your cloud Postgres connection string in each service. All services use `asyncpg`, which is compatible with any standard Postgres 14+ instance.

---

## Roadmap

**What's coming next?**

In priority order:

1. **Docker Compose** — one command to start the full stack
2. **ngrok / deployment guide** — running agents from anywhere, not just localhost
3. **Tier 2 semantic detectors** — USER_DISSATISFACTION, INTENT_MISALIGNMENT
4. **Patch suggester** — GPT-4o-mini generates custom fixes for your specific codebase
5. **ClickHouse migration** — events table for high-volume production use
6. **Policy engine** — block runs that match certain signal patterns in real-time
7. **AutoGen adapter** — drop-in support for Microsoft AutoGen agents
