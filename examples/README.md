# Examples

Dunetrace keeps runnable examples in two places:

- `examples/`: product and runtime feature demos, run from the repository root.
- `packages/sdk-py/examples/`: SDK and framework integration demos, run from
  `packages/sdk-py/` unless a row says otherwise.

The table entries below are based on each example's top-level docstring. If a
docstring does not call an example fully offline, the table does not infer it.

## Repository Examples

Run these commands from the repository root.

| Example | What it shows | How to run | Stack or offline |
| --- | --- | --- | --- |
| [`approval_agent.py`](approval_agent.py) | Human-in-the-loop `require_approval` policy gating the `wire_money` tool, including fail-closed timeout behavior. | `docker compose up -d`<br>`python examples/approval_agent.py` | Needs a local stack. Uses the Customer API and degrades gracefully if the backend is unreachable. |
| [`voice_agent.py`](voice_agent.py) | Voice pack activation, the four voice event types, and a slow-LLM voice policy that injects a recovery prompt. | `docker compose up -d`<br>`python examples/voice_agent.py` | Needs a local stack for pack activation and event ingest. |
| [`memory_agent.py`](memory_agent.py) | `memory_written()` / `memory_read()` instrumentation with clean and poisoned memory runs that demonstrate `MEMORY_POISONING`. | `docker compose up -d`<br>`python examples/memory_agent.py` | Needs a local stack. The signal appears on the dashboard. |
| [`delegation_loop_agent.py`](delegation_loop_agent.py) | Nested multi-agent runs that build a delegation graph, including a buggy loop that demonstrates `DELEGATION_LOOP`. | `docker compose up -d`<br>`python examples/delegation_loop_agent.py` | Needs a local stack. The signal appears on the dashboard. |

## SDK Python Examples

Run these commands from `packages/sdk-py/`; the commands mirror the docstrings
in that directory. If a row says to start the backend with `docker compose up`,
run that backend command from the repository root first.

| Example | What it shows | How to run | Stack or offline |
| --- | --- | --- | --- |
| [`basic_agent.py`](../packages/sdk-py/examples/basic_agent.py) | Manual instrumentation with normal, tool-loop, prompt-injection, and empty-RAG runs. | From the repository root: `docker compose up -d`<br>From `packages/sdk-py/`: `pip install dunetrace`<br>`python examples/basic_agent.py` | Needs the backend first. |
| [`decorator_agent.py`](../packages/sdk-py/examples/decorator_agent.py) | `@dt.agent()` decorator plus auto-instrumentation, with happy-path runs and failure scenarios. | From the repository root: `docker compose up -d`<br>From `packages/sdk-py/`: `pip install dunetrace`<br>`python examples/decorator_agent.py`<br>`SCENARIO=failures python examples/decorator_agent.py` | Needs the backend first. |
| [`langchain_agent.py`](../packages/sdk-py/examples/langchain_agent.py) | LangChain / LangGraph callback instrumentation, tool-loop scenario, and deploy markers. | `pip install 'dunetrace[langchain]' langchain-openai langgraph`<br>`OPENAI_API_KEY=sk-... python examples/langchain_agent.py`<br>`OPENAI_API_KEY=sk-... SCENARIO=tool_loop python examples/langchain_agent.py` | Needs an OpenAI API key for the documented runs. The docstring does not call it fully offline. |
| [`crewai_agent.py`](../packages/sdk-py/examples/crewai_agent.py) | CrewAI 1.x global LLM/tool hooks for a research-and-writing crew, including a tool-loop scenario. | `pip install 'dunetrace' crewai python-dotenv`<br>`OPENAI_API_KEY=sk-... python examples/crewai_agent.py`<br>`SCENARIO=tool_loop OPENAI_API_KEY=sk-... python examples/crewai_agent.py` | Needs an OpenAI API key for the documented runs. The docstring does not call it fully offline. |
| [`autogen_agent.py`](../packages/sdk-py/examples/autogen_agent.py) | AutoGen AssistantAgent / RoundRobinGroupChat monitoring through a model-client wrapper. | `pip install 'dunetrace' autogen-agentchat autogen-ext python-dotenv`<br>`OPENAI_API_KEY=sk-... python examples/autogen_agent.py`<br>`SCENARIO=tool_loop OPENAI_API_KEY=sk-... python examples/autogen_agent.py` | Needs an OpenAI API key for the documented runs. The docstring does not call it fully offline. |
| [`haystack_agent.py`](../packages/sdk-py/examples/haystack_agent.py) | Haystack 2.x tracer registration for a RAG pipeline, plus a tool-loop scenario and deploy markers. | `pip install 'dunetrace[haystack]' haystack-ai openai python-dotenv`<br>`python examples/haystack_agent.py`<br>`SCENARIO=tool_loop python examples/haystack_agent.py` | The docstring does not call it fully offline. |
| [`hermes_agent.py`](../packages/sdk-py/examples/hermes_agent.py) | Hermes plugin scenarios for happy path, tool loop, retry storm, goal abandonment, and real-agent runs. | `PYTHONPATH=packages/sdk-py python examples/hermes_agent.py`<br>`SCENARIO=real OPENAI_API_KEY=sk-... python examples/hermes_agent.py` | Synthetic scenarios need no LLM API key. Real scenarios need `hermes-agent` and an OpenAI API key. |
| [`openai_agents_agent.py`](../packages/sdk-py/examples/openai_agents_agent.py) | OpenAI Agents SDK trace processor instrumentation with normal and tool-loop scenarios. | `pip install 'dunetrace[openai-agents]'`<br>`pip install python-dotenv`<br>`OPENAI_API_KEY=sk-... python examples/openai_agents_agent.py`<br>`OPENAI_API_KEY=sk-... SCENARIO=tool_loop python examples/openai_agents_agent.py` | Needs an OpenAI API key for the documented runs. The docstring does not call it fully offline. |
