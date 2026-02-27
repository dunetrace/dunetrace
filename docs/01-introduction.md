# What is Dunetrace?

Dunetrace is a **production observability platform for AI agents**. It detects when your agents are misbehaving, explains what went wrong in plain English, and alerts your team before your users notice.

---

## The Problem

AI agents fail in ways that traditional monitoring can't catch.

A web service either returns a 200 or it doesn't. An agent can return a 200, produce a confident-sounding answer, and still have:

- Called the same search tool 8 times in a loop
- Answered from stale training data instead of retrieving live information
- Been manipulated by a prompt injection in user input
- Burned 15,000 tokens going in circles between two conflicting tools
- Given up halfway through a complex task and quietly returned nothing

None of these show up in your error logs. Your users just get a bad answer, and you find out from a support ticket days later.

---

## What Dunetrace Does

```
Your Agent  →  Dunetrace SDK  →  Ingest API  →  Detector Worker
                                                       ↓
                                               Explain Layer
                                                       ↓
                                          Alerts (Slack / webhook)
                                                       ↓
                                            Customer REST API
                                                       ↓
                                               Dashboard UI
```

**Instrument once.** Add a single callback or context manager to your agent. Zero change to your agent's logic.

**Detect automatically.** Six Tier 1 failure detectors run against every completed run within seconds: tool loops, thrashing, avoidance, goal abandonment, prompt injection, and RAG empty retrieval.

**Explain clearly.** Every failure signal comes with a human-readable explanation: what happened, why it matters, and a concrete code fix.

**Alert fast.** Failures above your severity threshold land in Slack (or any webhook) within 15 seconds of detection.

**Query everything.** A REST API exposes all runs, events, and signals so you can build your own dashboards, integrate with PagerDuty, or power a customer-facing status page.

---

## Who It's For

**Teams shipping agents to production** who need confidence that their agents are behaving correctly across all user inputs — not just the happy path they tested.

**Platform teams** running multiple agents across multiple customers who need a single view of what's working and what's broken.

---

## What Dunetrace Is Not

- Not a general-purpose LLM observability tool (use LangSmith or Langfuse for that)
- Not a tracing tool for debugging individual runs during development (use AgentWatch for that)
- Not an evals framework (use Braintrust or PromptFoo for that)

Dunetrace's job is **production alerting**: watching agents running for real users and telling you when something is broken.

---

## Key Design Decisions

**No raw content ever leaves your system.** All user inputs, prompts, and outputs are SHA-256 hashed before being sent to Dunetrace. You get full observability without exposing private data.

**Zero overhead on the agent thread.** The SDK appends events to an in-memory ring buffer (O(1), no locks needed in CPython). A background drain thread ships batches every 200ms. Your agent never waits for Dunetrace.

**Shadow mode by default.** New detectors run in shadow mode — signals are stored but never alerted. You validate precision on real traffic before committing to waking someone up.

**Deterministic explanations.** The explain layer uses templated text, not LLM calls. Every signal of the same type produces the same shape of explanation. This means sub-millisecond explain latency and zero API cost.
