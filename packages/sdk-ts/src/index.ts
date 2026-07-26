export { Dunetrace } from "./client.js";
export { getCurrentRun } from "./context.js";
export {
  autoInstrument,
  instrumentHttp,
  registerOwnEndpoint,
  wrapOpenAIClient,
  wrapAnthropicClient,
  type AutoInstrumentOptions,
} from "./auto.js";
export { DunetraceRun } from "./run.js";
export { agentVersion } from "./hash.js";
export {
  HttpBatchEmitter,
  NoopBatchEmitter,
  DurableRetryEmitter,
  DEFAULT_QUEUE_PATH,
  type BatchEmitter,
} from "./emitters.js";
export {
  instrumentGenerateTextOptions,
  instrumentStreamTextOptions,
  wrapGenerateText,
  wrapStreamText,
  traceGenerateText,
  traceStreamText,
  modelId,
  toolNames,
  type GenerateTextOptions,
  type StreamTextOptions,
} from "./integrations/vercel-ai.js";
export type {
  AgentEvent,
  EventType,
  RunOptions,
  LlmRespondedOptions,
  ClientOptions,
  EventSink,
  MemorySource,
  VadType,
  TurnTakingAction,
  TranscriptionOptions,
  TtsOptions,
  RecordingOptions,
} from "./models.js";
export { MEMORY_SOURCES, VAD_TYPES, TURN_TAKING_ACTIONS } from "./models.js";
