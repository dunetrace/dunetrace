export type EventType =
  | "run.started"
  | "run.completed"
  | "run.errored"
  | "llm.called"
  | "llm.responded"
  | "tool.called"
  | "tool.responded"
  | "retrieval.called"
  | "retrieval.responded"
  | "external.signal"
  // Agent memory channel — see run.memoryWritten/memoryRead/memoryCleared.
  | "memory.written"
  | "memory.read"
  | "memory.cleared";

/** Provenance of a value written to agent memory. Feeds the server-side
 *  MEMORY_POISONING detector's risk weighting — content from an
 *  attacker-controllable channel (retrieval/tool_output/external) persisted to
 *  memory is higher risk than the agent's own reasoning. */
export type MemorySource =
  | "user_input"
  | "retrieval"
  | "tool_output"
  | "llm_output"
  | "agent_reasoning"
  | "external";

export const MEMORY_SOURCES: readonly MemorySource[] = [
  "user_input",
  "retrieval",
  "tool_output",
  "llm_output",
  "agent_reasoning",
  "external",
];

export interface AgentEvent {
  event_type:     EventType;
  run_id:         string;
  agent_id:       string;
  agent_version:  string;
  step_index:     number;
  timestamp:      number;
  payload:        Record<string, unknown>;
  parent_run_id?: string | null;
  trace_id?:      string | null;
  conversation_id?: string | null;
  // audit Finding 14: stable per-event id for ingest-side dedup of retries.
  event_id?:      string;
}

export interface RunOptions {
  systemPrompt?: string;
  model?:        string;
  tools?:        string[];
  userInput?:    string;
  parentRunId?:  string;
  /** Pre-set the run UUID. Use this to correlate a Dunetrace run with an external
   *  tracing system that was given the same ID. */
  runId?:        string;
  /** Correlation key for external evaluation integrations (Langfuse/LangSmith/
   *  Braintrust) — pass the same trace id your own instrumentation of that
   *  provider's SDK uses. Unlike runId, this doesn't change Dunetrace's own
   *  run_id — it's an independent pointer, for when you don't want the two
   *  identifiers conflated. Not folded into agentVersion's hash. */
  traceId?:      string;
  /** Groups this run with others from the same end-user interaction —
   *  pass the same id across every run() call in a multi-turn conversation.
   *  Same non-identity rationale as traceId; not folded into agentVersion's
   *  hash. */
  conversationId?: string;
}

export interface LlmRespondedOptions {
  completionTokens?: number;
  latencyMs?:        number;
  finishReason?:     string;
  outputLength?:     number;
  outputText?:       string;
  /** Pass when prompt token count is only known after the call returns
   *  (e.g. taken from the API response). Overrides the estimate given to llmCalled(). */
  promptTokens?:     number;
}

export interface ClientOptions {
  endpoint?:        string;
  apiKey?:          string;
  flushIntervalMs?: number;
  emitAsJson?:      boolean;
  bufferSize?:      number;
  timeoutMs?:       number;
  /** Custom batch-shipping strategy — see emitters.ts. Defaults to
   *  HttpBatchEmitter (same POST-to-ingest-API behavior as before this was
   *  made pluggable). Pass a DurableRetryEmitter wrapping HttpBatchEmitter to
   *  survive a backend outage across process restarts. */
  emitter?: import("./emitters.js").BatchEmitter;
}
