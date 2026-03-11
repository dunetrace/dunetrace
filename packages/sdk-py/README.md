# Dunetrace SDK

Behavioral observability for AI agents at runtime. Zero-dependency Python SDK that detects failure patterns in your LLM agents in real-time.

## Install

```bash
pip install dunetrace
```

With integrations:
```bash
pip install dunetrace[langchain]   # LangChain
```

## Quickstart

```python
from dunetrace import Dunetrace

dt = Dunetrace(endpoint="http://localhost:8001")

with dt.run("my-agent", user_input="What is the capital of France?") as run:
    # your agent logic here
    run.emit("tool.called", {"tool_name": "web_search"})
```

## What it detects

- **Tool loops** — same tool called repeatedly without progress
- **Context bloat** — prompt tokens growing 3× across steps
- **Tool avoidance** — agent has tools but never uses them
- **Step count inflation** — far more steps than needed
- **First step failure** — agent fails immediately
- **RAG empty retrieval** — retrieval returns no results
- And more (tool thrashing, retry storms, cascading failures)

## Self-hosted backend

Dunetrace requires the backend to collect and analyze events:

```bash
git clone https://github.com/dunetrace/dunetrace
cd dunetrace
cp .env.example .env
docker compose up
```

## Links

- [GitHub](https://github.com/dunetrace/dunetrace)
- [Issues](https://github.com/dunetrace/dunetrace/issues)
