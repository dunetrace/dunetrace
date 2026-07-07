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
  | "external.signal";

export interface AgentEvent {
  event_type:     EventType;
  run_id:         string;
  agent_id:       string;
  agent_version:  string;
  step_index:     number;
  timestamp:      number;
  payload:        Record<string, unknown>;
  parent_run_id?: string | null;
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
