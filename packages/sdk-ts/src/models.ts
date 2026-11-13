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
  // Voice-agent events (detector pack "voice"). Emitted only by the voice
  // helpers on DunetraceRun; no built-in detector reads them until the voice
  // pack is active. Kept value-for-value in sync with the Python SDK.
  | "transcription.received"
  | "tts.generated"
  | "voice_activity.detected"
  | "turn_taking.changed"
  | "recording.available"
  // Agent memory channel — see run.memoryWritten/memoryRead/memoryCleared.
  | "memory.written"
  | "memory.read"
  | "memory.cleared";

/** VAD transition kinds accepted by run.voiceActivityDetected(). */
export type VadType = "speech_start" | "speech_end" | "silence" | "barge_in";
export const VAD_TYPES: readonly VadType[] = [
  "speech_start",
  "speech_end",
  "silence",
  "barge_in",
];

/** Conversational-floor transitions accepted by run.turnTaking(). */
export type TurnTakingAction = "agent_speaking" | "user_speaking" | "both_speaking" | "neither";
export const TURN_TAKING_ACTIONS: readonly TurnTakingAction[] = [
  "agent_speaking",
  "user_speaking",
  "both_speaking",
  "neither",
];

/** Optional metadata for run.transcriptionReceived(). */
export interface TranscriptionOptions {
  confidence?:   number;
  latencyMs?:    number;
  /** Length of the transcribed audio. Pass it for per-minute STT cost
   *  attribution (most STT providers bill per minute of audio). */
  audioSeconds?: number;
}

/** Optional metadata for run.ttsGenerated(). */
export interface TtsOptions {
  latencyMs?:  number;
  truncated?:  boolean;
  audioSeconds?: number;
  /** Provider-side correlation metadata (Phase 4.1). Pass them when your TTS
   *  runs on a provider Dunetrace can pull generation history from (ElevenLabs)
   *  so a stored generation can be matched back to this exact event. */
  voiceId?:    string;
  model?:      string;
  provider?:   string;
  providerGenerationId?: string;
}

/** Optional metadata for run.recordingMetadata(). */
export interface RecordingOptions {
  durationSeconds?:    number;
  format?:             string;
  storageProvider?:    string;
  /** When the recording began relative to the start of the call/run, used to
   *  deep-link a signal's moment into the audio. */
  startOffsetSeconds?: number;
}

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
  /** Reasoning/thinking tokens billed on top of visible completion tokens
   *  (o-series, extended-thinking models). Kept separate from completionTokens so
   *  cost accounting and the OTel exporter can attribute them distinctly. */
  reasoningTokens?:  number;
  /** Which llmCalled() this response belongs to — the number llmCalled()
   *  returned. Defaults to the most recent call, which is correct whenever
   *  called/responded are adjacent (every manual caller, every non-streaming
   *  patcher). A STREAMED call breaks that adjacency: the response is not known
   *  until the caller drains, which may be after other calls have started, so
   *  the streaming instrumentation captures the id at call time and passes it
   *  here. Without it two overlapping streams emit called(A), called(B),
   *  responded(A), responded(B) and positional pairing swaps their responses. */
  callId?:           number;
}

/** A synchronous per-event sink. Each AgentEvent is handed to it as it is
 *  emitted, in addition to the normal ship-to-ingest path. DunetraceOtelExporter
 *  implements this; the interface is duck-typed so wiring an OTel exporter never
 *  makes @opentelemetry/api a hard dependency of the core client. */
export interface EventSink {
  handle(event: AgentEvent): void;
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
  /** Optional OTel span exporter (DunetraceOtelExporter) or any EventSink. Every
   *  emitted event is also handed to it, so agent runs show up in an OTel backend
   *  alongside Dunetrace's own ingest. Additive and opt-in; a failure in the sink
   *  never touches ingest. */
  exporter?: EventSink;
  /** Per-field character cap on every free-text field the SDK ships (tool args
   *  and output, LLM output, retrieval query/content, memory values, input_text,
   *  system_prompt). Defaults to 8192 — the same cap the OTLP ingest path
   *  enforces — so a run reads the same whichever transport it arrived on. A
   *  capped field carries `<field>_truncated` / `<field>_original_length`
   *  siblings. 0 disables the cap. See redaction.ts. */
  maxFieldChars?: number;
  /** Hook applied to structured tool args before the built-in denylist, for
   *  domain-specific fields the denylist cannot know about. Gets a shallow copy
   *  and must return an object; if it throws or returns anything else its
   *  output is discarded and the built-in denylist still applies. Runs on every
   *  tool call, so keep it cheap. */
  redact?: (args: Record<string, unknown>) => Record<string, unknown>;
  /** Extra key names for the built-in denylist. Matching is on normalised
   *  words, so `"X-Session-Id"` and `"xSessionId"` are the same entry, and an
   *  entry matches a key that equals it or ends with `_<entry>`. */
  redactKeys?: string[];
  /** Flush the buffer when the process exits naturally, so a script that never
   *  calls `dt.shutdown()` still ships its events (the periodic drain timer is
   *  unref'd and a short-lived process exits before it ever fires). Defaults to
   *  true; set false to opt out. Uses Node's `beforeExit`, which never delays an
   *  exit that was going to happen anyway — see the "Exit flush" note in
   *  client.ts. `process.exit()` and fatal signals still bypass it. */
  flushOnExit?: boolean;
}
