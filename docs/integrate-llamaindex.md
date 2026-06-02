# Integrating a LlamaIndex RAG Agent with Dunetrace

This guide covers adding Dunetrace monitoring to a LlamaIndex query engine or RAG agent. The integration wraps your agent entry point with `@dt.trace` and emits retrieval events around `query_engine.query(...)` so detectors such as `RAG_EMPTY_RETRIEVAL` have the retrieval metadata they need.

---

## How It Works

LlamaIndex query engines return a response object that includes `source_nodes`. Those source nodes are `NodeWithScore` values, so Dunetrace can record both the number of retrieved nodes and the highest retrieval score without sending raw document text.

| LlamaIndex step | Dunetrace event |
|---|---|
| Agent function starts | `RUN_STARTED` |
| Query engine is about to retrieve context | `RETRIEVAL_CALLED` |
| Query engine returns `response.source_nodes` | `RETRIEVAL_RESPONDED` with result count, top score, and latency |
| Agent returns an answer | `RUN_COMPLETED` |
| Any unhandled exception | `RUN_ERRORED` |

Dunetrace only receives structural metadata such as model name, latency, retrieval count, and top score. Hash the query before passing it as `query_hash`; do not send raw prompts, retrieved documents, or generated answers.

---

## Prerequisites

- Dunetrace backend running (`docker compose up -d`)
- Python 3.11+
- A LlamaIndex app with a query engine or retriever

> **Local dev - no API key needed.** The backend accepts requests without any API key when running locally. API keys are only required for production deployments.

---

## Step 1: Install Dependencies

```bash
pip install dunetrace llama-index
```

Install any provider packages your LlamaIndex app already uses, for example:

```bash
pip install llama-index-llms-openai llama-index-embeddings-openai
```

---

## Step 2: Create the Dunetrace Client

Create one client at process startup and flush it on exit.

```python
import atexit
import os

from dunetrace import Dunetrace

dt = Dunetrace(
    endpoint=os.environ.get("DUNETRACE_ENDPOINT", "http://localhost:8001"),
    api_key=os.environ.get("DUNETRACE_API_KEY", ""),
)
atexit.register(dt.shutdown)
```

---

## Step 3: Wrap the Query Engine Call

Use `@dt.trace` on the agent entry point, then emit `retrieval_called()` before the query and `retrieval_responded()` after the response returns.

```python
import hashlib
import time

from dunetrace import get_current_run


def query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def source_nodes(response) -> list:
    return list(getattr(response, "source_nodes", None) or [])


def top_score(nodes: list) -> float | None:
    scores = [node.score for node in nodes if getattr(node, "score", None) is not None]
    return max(scores) if scores else None


@dt.trace("llamaindex-rag-agent", model="gpt-4o-mini", tools=["llamaindex-query-engine"])
def answer_question(question: str) -> str:
    run = get_current_run()
    index_name = "llamaindex-vector-store"

    if run:
        run.retrieval_called(index_name=index_name, query_hash=query_hash(question))

    started = time.perf_counter()
    response = query_engine.query(question)
    latency_ms = int((time.perf_counter() - started) * 1000)

    nodes = source_nodes(response)
    if run:
        run.retrieval_responded(
            index_name=index_name,
            result_count=len(nodes),
            top_score=top_score(nodes),
            latency_ms=latency_ms,
        )

    return str(response)
```

`get_current_run()` returns the active Dunetrace run inside a traced function. The guard keeps the helper safe if you reuse it outside a `@dt.trace` context.

---

## Complete Example

```python
import atexit
import hashlib
import os
import time

from dunetrace import Dunetrace, get_current_run
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex


def query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def source_nodes(response) -> list:
    return list(getattr(response, "source_nodes", None) or [])


def top_score(nodes: list) -> float | None:
    scores = [node.score for node in nodes if getattr(node, "score", None) is not None]
    return max(scores) if scores else None


dt = Dunetrace(
    endpoint=os.environ.get("DUNETRACE_ENDPOINT", "http://localhost:8001"),
    api_key=os.environ.get("DUNETRACE_API_KEY", ""),
)
atexit.register(dt.shutdown)

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)


@dt.trace("llamaindex-rag-agent", model="gpt-4o-mini", tools=["llamaindex-query-engine"])
def answer_question(question: str) -> str:
    run = get_current_run()
    index_name = "llamaindex-vector-store"

    if run:
        run.retrieval_called(index_name=index_name, query_hash=query_hash(question))

    started = time.perf_counter()
    response = query_engine.query(question)
    latency_ms = int((time.perf_counter() - started) * 1000)

    nodes = source_nodes(response)
    if run:
        run.retrieval_responded(
            index_name=index_name,
            result_count=len(nodes),
            top_score=top_score(nodes),
            latency_ms=latency_ms,
        )

    return str(response)


print(answer_question("What does the product documentation say about setup?"))
```

---

## Async Query Engines

Use the same pattern with `aquery()` for async agents.

```python
@dt.trace("llamaindex-async-rag-agent", model="gpt-4o-mini", tools=["llamaindex-query-engine"])
async def answer_question_async(question: str) -> str:
    run = get_current_run()
    index_name = "llamaindex-vector-store"

    if run:
        run.retrieval_called(index_name=index_name, query_hash=query_hash(question))

    started = time.perf_counter()
    response = await query_engine.aquery(question)
    latency_ms = int((time.perf_counter() - started) * 1000)

    nodes = source_nodes(response)
    if run:
        run.retrieval_responded(
            index_name=index_name,
            result_count=len(nodes),
            top_score=top_score(nodes),
            latency_ms=latency_ms,
        )

    return str(response)
```

---

## Triggering `RAG_EMPTY_RETRIEVAL`

`RAG_EMPTY_RETRIEVAL` can fire when the agent performs a retrieval step, receives no results, and still produces an answer. For LlamaIndex, make sure `result_count` is derived from `len(response.source_nodes)`:

```python
nodes = source_nodes(response)
run.retrieval_responded(
    index_name="llamaindex-vector-store",
    result_count=len(nodes),
    top_score=top_score(nodes),
    latency_ms=latency_ms,
)
```

If the query engine returns no nodes, pass `result_count=0`. If your retriever does not populate scores, pass `top_score=None`; detectors still receive the empty retrieval signal.

---

## What Is and Isn't Captured

**Captured:**
- Run boundaries and total latency
- Retrieval result count
- Top retrieval score when LlamaIndex provides one
- Retrieval latency
- Model and tool names declared in `@dt.trace`

**Not captured:**
- Raw user queries
- Retrieved document text
- LlamaIndex response text
- Node metadata or embeddings

---

## Verify the Integration

Run your agent once, then check:

1. **Dashboard** (`http://your-dashboard:3000`) - the run should appear within 15 seconds
2. **Runs API** - `GET http://your-ingest:8002/v1/runs?agent_id=llamaindex-rag-agent`

To verify `RAG_EMPTY_RETRIEVAL`, ask a question that should not match any indexed content and confirm the query engine returns `response.source_nodes == []`.

---

## Troubleshooting

**No runs appear in the dashboard**
- Ensure the function with `query_engine.query(...)` is wrapped with `@dt.trace`
- Call `dt.shutdown()` before process exit to flush buffered events
- Confirm `DUNETRACE_ENDPOINT` points at the ingest service, usually `http://localhost:8001` in local dev

**`result_count` is always zero**
- Confirm you are reading `response.source_nodes`, not the generated answer string
- Ensure your query engine is configured to return source nodes
- Test the LlamaIndex query directly with `print(len(response.source_nodes))`

**`top_score` is always `None`**
- Some retrievers or rerankers do not populate `NodeWithScore.score`
- Pass `top_score=None`; Dunetrace can still detect empty retrievals from `result_count`

**The detector does not fire**
- Confirm your code emits both `retrieval_called()` and `retrieval_responded()`
- Confirm `result_count=0` for the empty retrieval case
- Make sure the agent still returns an answer after the empty retrieval
