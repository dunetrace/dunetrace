import { randomUUID } from "node:crypto";
import type {
  AgentEvent,
  EventType,
  LlmRespondedOptions,
  MemorySource,
  RecordingOptions,
  TranscriptionOptions,
  TtsOptions,
  TurnTakingAction,
  VadType,
} from "./models.js";
import { MEMORY_SOURCES, TURN_TAKING_ACTIONS, VAD_TYPES } from "./models.js";
import {
  putCapped,
  readRedactionSettings,
  serializeArgs,
  type RedactionSettings,
} from "./redaction.js";

export interface EventEmitter {
  _emit(event: AgentEvent): void;
  /** Content caps + secret redaction for everything this run ships. Optional:
   *  a duck-typed emitter that does not implement it gets the documented
   *  defaults (see redaction.ts). */
  _redactionSettings?(): RedactionSettings;
}

/** True when DUNETRACE_OMIT_LLM_OUTPUT_TEXT opts out of transmitting raw LLM
 *  output text (bandwidth-sensitive deployments). Read per-call so it can be
 *  toggled at runtime / in tests. `output_length` is always sent, so size-based
 *  detectors work either way. Mirrors the Python SDK. */
function omitLlmOutputText(): boolean {
  const v = (process.env.DUNETRACE_OMIT_LLM_OUTPUT_TEXT ?? "").toLowerCase();
  return v === "1" || v === "true" || v === "yes";
}

export class DunetraceRun {
  readonly runId: string;
  /** True for a run the client opened on its own because a patched LLM call
   *  had no run to attach to (see client.ts). Its boundary is a guess, and
   *  the detector holds its signals in shadow. */
  implicit = false;
  /** Set once the terminal event has been emitted. A closed run still sitting
   *  in the async context reads as "no run" to resolveRun(). */
  closed = false;
  /** The call that opened the run when the SDK, not the caller, opened it. */
  openedBy: string | null = null;

  private _agentId:    string;
  private _version:    string;
  private _client:     EventEmitter;
  private _step        = 0;
  private _exitReason: string | null = null;
  private _events:     AgentEvent[]  = [];
  /** Next llm.called correlation id — the call's index within this run. */
  private _callSeq     = 0;
  private _redaction:  RedactionSettings;

  constructor(agentId: string, version: string, client: EventEmitter, runId?: string) {
    this.runId    = runId ?? randomUUID();
    this._agentId = agentId;
    this._version = version;
    this._client  = client;
    // Resolved once per run (with type guards, so a duck-typed emitter reads as
    // unconfigured) and read by every emit hook below.
    this._redaction = readRedactionSettings(client);
  }

  // ── LLM hooks ──────────────────────────────────────────────────────────────

  /**
   * Record the start of an LLM call. Returns the call's `call_id` — pass it to
   * `llmResponded({ callId })` when the response cannot be emitted adjacently
   * (a stream the caller drains later). See LlmRespondedOptions.callId.
   */
  llmCalled(model: string, promptTokens = 0): number {
    // Index of this call within the run, which is exactly the correlation id
    // the response needs. Wire-identical to the Python SDK's call_id, so the
    // shared server-side run builder pairs both SDKs the same way.
    const callId = this._callSeq++;
    this._emit("llm.called", { model, prompt_tokens: promptTokens, call_id: callId });
    return callId;
  }

  llmResponded(opts: LlmRespondedOptions = {}): void {
    const payload: Record<string, unknown> = {
      completion_tokens: opts.completionTokens ?? 0,
      latency_ms:        opts.latencyMs        ?? 0,
      finish_reason:     opts.finishReason      ?? "stop",
      output_length:     opts.outputLength      ?? (opts.outputText?.length ?? 0),
    };
    // Transmit the output text by default; omit it (bandwidth) when
    // DUNETRACE_OMIT_LLM_OUTPUT_TEXT is set. output_length above is the REAL
    // length; only the text itself is capped.
    if (!omitLlmOutputText()) {
      putCapped(payload, "output", opts.outputText ?? "", this._redaction.maxFieldChars);
    }
    // Echoed so the server-side builders can pair this response with its call by
    // identity rather than arrival order. Omitted (not null) when there is no
    // call to name, keeping the wire format unchanged for that case.
    const callId = this._resolveCallId(opts.callId);
    if (callId !== undefined) payload["call_id"] = callId;
    if (opts.promptTokens)    payload["prompt_tokens"]    = opts.promptTokens;
    if (opts.reasoningTokens) payload["reasoning_tokens"] = opts.reasoningTokens;
    this._emit("llm.responded", payload, false);
  }

  // ── Tool hooks ─────────────────────────────────────────────────────────────

  toolCalled(toolName: string, args: Record<string, unknown> = {}): void {
    // What leaves the process is the redacted, capped serialisation, never the
    // raw args — and serialising is non-throwing, so a circular object or a
    // BigInt in a tool argument can never fail the customer's tool call.
    const { text, truncated, originalLength } = serializeArgs(this._redaction, args);
    const payload: Record<string, unknown> = { tool_name: toolName, args: text };
    if (truncated) {
      payload["args_truncated"]       = true;
      payload["args_original_length"] = originalLength;
      // Same number under the key the shared run_builder already reads for the
      // OTLP path (ToolCall.args_length), so server-side OVERSIZED_TOOL_ARGUMENTS
      // keeps working on a payload the cap has shortened below its threshold.
      payload["args_length"]          = originalLength;
    }
    this._emit("tool.called", payload);
  }

  toolResponded(
    toolName:    string,
    success:     boolean,
    outputLength = 0,
    latencyMs    = 0,
    error?:      string,
    output       = "",
  ): void {
    // output_length is the REAL length; only the text is capped.
    const payload: Record<string, unknown> = {
      tool_name:     toolName,
      success,
      output_length: outputLength,
      latency_ms:    latencyMs,
    };
    putCapped(payload, "output", output, this._redaction.maxFieldChars);
    if (error) payload["error"] = error;
    this._emit("tool.responded", payload, false);
  }

  // ── Retrieval hooks ────────────────────────────────────────────────────────

  retrievalCalled(indexName: string, query = ""): void {
    const payload: Record<string, unknown> = { index_name: indexName };
    putCapped(payload, "query", query, this._redaction.maxFieldChars);
    this._emit("retrieval.called", payload);
  }

  retrievalResponded(
    indexName:   string,
    resultCount: number,
    topScore?:   number,
    latencyMs    = 0,
    content      = "",
  ): void {
    const payload: Record<string, unknown> = {
      index_name:   indexName,
      result_count: resultCount,
      top_score:    topScore ?? null,
      latency_ms:   latencyMs,
    };
    putCapped(payload, "content", content, this._redaction.maxFieldChars);
    this._emit("retrieval.responded", payload, false);
  }

  // ── Voice hooks (detector pack "voice") ────────────────────────────────────
  //
  // advance semantics mirror the non-voice hooks: a turn-initiating event
  // advances the step counter (like llmCalled / toolCalled), while completion
  // and high-frequency annotation events do not (like llmResponded). This keeps
  // a normal voice turn at ~one step, so the always-on built-in step detectors
  // don't false-fire on a real voice call.

  /**
   * Speech-to-text produced a transcript for one user turn. Turn-initiating —
   * advances the step counter.
   */
  transcriptionReceived(text: string, opts: TranscriptionOptions = {}): void {
    const payload: Record<string, unknown> = {
      text,
      confidence: opts.confidence ?? 1.0,
      latency_ms: opts.latencyMs ?? 0,
    };
    if (opts.audioSeconds) payload["audio_seconds"] = opts.audioSeconds;
    this._emit("transcription.received", payload);
  }

  /**
   * Text-to-speech rendered an agent response. Completes the turn — does not
   * advance the step counter (like llmResponded).
   */
  ttsGenerated(text: string, opts: TtsOptions = {}): void {
    const payload: Record<string, unknown> = {
      text,
      latency_ms: opts.latencyMs ?? 0,
      truncated:  opts.truncated ?? false,
    };
    if (opts.audioSeconds)          payload["audio_seconds"]          = opts.audioSeconds;
    if (opts.voiceId)               payload["voice_id"]               = opts.voiceId;
    if (opts.model)                 payload["model"]                  = opts.model;
    if (opts.provider)              payload["provider"]               = opts.provider;
    if (opts.providerGenerationId)  payload["provider_generation_id"] = opts.providerGenerationId;
    this._emit("tts.generated", payload, false);
  }

  /**
   * A VAD transition (speech_start / speech_end / silence / barge_in).
   * High-frequency annotation — does not advance the step counter.
   */
  voiceActivityDetected(type: VadType, durationMs = 0): void {
    if (!VAD_TYPES.includes(type)) {
      throw new Error(
        `voiceActivityDetected: type must be one of ${VAD_TYPES.join(", ")}, got ${String(type)}`,
      );
    }
    this._emit("voice_activity.detected", { type, duration_ms: durationMs }, false);
  }

  /**
   * A conversational-floor transition (agent_speaking / user_speaking /
   * both_speaking / neither). State annotation — does not advance the step counter.
   */
  turnTaking(action: TurnTakingAction, fromAgent = false, toUser = false): void {
    if (!TURN_TAKING_ACTIONS.includes(action)) {
      throw new Error(
        `turnTaking: action must be one of ${TURN_TAKING_ACTIONS.join(", ")}, got ${String(action)}`,
      );
    }
    this._emit("turn_taking.changed", { action, from_agent: fromAgent, to_user: toUser }, false);
  }

  /**
   * Record where the call audio lives. Storage-agnostic: Dunetrace stores and
   * links the URL but never fetches the audio, so a presigned or private URL is
   * fine (mind its expiry). Annotation event — does not advance the step counter.
   */
  recordingMetadata(url: string, opts: RecordingOptions = {}): void {
    const payload: Record<string, unknown> = { url };
    if (opts.durationSeconds)    payload["duration_seconds"]     = opts.durationSeconds;
    if (opts.format)             payload["format"]               = opts.format;
    if (opts.storageProvider)    payload["storage_provider"]     = opts.storageProvider;
    if (opts.startOffsetSeconds) payload["start_offset_seconds"] = opts.startOffsetSeconds;
    this._emit("recording.available", payload, false);
  }

  // ── Infrastructure signals ─────────────────────────────────────────────────

  /**
   * Emit an infrastructure context event without advancing the step counter.
   * Use for rate limits, cache misses, upstream errors, etc.
   */
  externalSignal(
    signalName: string,
    source      = "",
    meta: Record<string, unknown> = {},
  ): void {
    const payload: Record<string, unknown> = { signal_name: signalName };
    if (source) payload["source"] = source;
    Object.assign(payload, meta);
    // Bypass _emit so step counter does not advance
    const event: AgentEvent = {
      event_type:    "external.signal",
      run_id:        this.runId,
      agent_id:      this._agentId,
      agent_version: this._version,
      step_index:    this._step,
      timestamp:     Date.now() / 1000,
      payload,
    };
    this._events.push(event);
    this._client._emit(event);
  }

  /**
   * Wrap a single LLM call promise and auto-emit llm.called / llm.responded.
   * Supports OpenAI chat completions and Anthropic messages response shapes.
   *
   * @example
   * const response = await run.llm("gpt-4o",
   *   openai.chat.completions.create({ model: "gpt-4o", messages })
   * );
   */
  async llm<T extends Record<string, unknown>>(model: string, call: Promise<T>): Promise<T> {
    const t0       = Date.now();
    const response = await call;
    const latencyMs = Date.now() - t0;
    const r = response as Record<string, unknown>;

    let promptTokens     = 0;
    let completionTokens: number | undefined;
    let finishReason     = "stop";
    let outputText       = "";

    if (Array.isArray(r["choices"])) {
      // OpenAI chat completions format
      const usage = r["usage"] as Record<string, number> | undefined;
      const choice = (r["choices"] as Record<string, unknown>[])[0] ?? {};
      promptTokens     = usage?.["prompt_tokens"]     ?? 0;
      completionTokens = usage?.["completion_tokens"];
      finishReason     = (choice["finish_reason"] as string | undefined) ?? "stop";
      outputText       = ((choice["message"] as Record<string, unknown> | undefined)?.["content"] as string | undefined) ?? "";
    } else if (Array.isArray(r["content"])) {
      // Anthropic messages format
      const usage = r["usage"] as Record<string, number> | undefined;
      promptTokens     = usage?.["input_tokens"]  ?? 0;
      completionTokens = usage?.["output_tokens"];
      finishReason     = (r["stop_reason"] as string | undefined) ?? "stop";
      outputText       = ((r["content"] as Record<string, unknown>[])[0]?.["text"] as string | undefined) ?? "";
    }

    const callId = this.llmCalled(model, promptTokens);
    this.llmResponded({ completionTokens, latencyMs, finishReason, outputText, callId });
    return response;
  }

  // ── Agent memory channel ───────────────────────────────────────────────────
  //
  // Instrumentation for what an agent persists to and reads from its own memory
  // (conversation buffers, scratchpads, long-term stores). None of these advance
  // the step counter — they annotate the run, like externalSignal. The
  // server-side MEMORY_POISONING detector reads these events.

  /**
   * Record that `value` was written to agent memory under `key`.
   *
   * `source` is optional but strongly encouraged — it names where the content
   * originated so downstream detection can weigh injection risk: one of
   * "user_input", "retrieval", "tool_output", "llm_output", "agent_reasoning",
   * "external". Does not advance the step counter.
   */
  memoryWritten(key: string, value: string, source?: MemorySource): void {
    if (source !== undefined && !MEMORY_SOURCES.includes(source)) {
      throw new Error(
        `memoryWritten: source must be one of ${MEMORY_SOURCES.join(", ")} or undefined, got ${String(source)}`,
      );
    }
    const payload: Record<string, unknown> = { key };
    putCapped(payload, "value", value, this._redaction.maxFieldChars);
    if (source !== undefined) payload["source"] = source;
    this._emit("memory.written", payload, false);
  }

  /** Record that agent memory was read at `key`. Does not advance the step counter. */
  memoryRead(key: string): void {
    this._emit("memory.read", { key }, false);
  }

  /**
   * Record that agent memory was cleared — a specific `key`, or all memory when
   * `key` is omitted. Does not advance the step counter.
   */
  memoryCleared(key?: string): void {
    this._emit("memory.cleared", { key: key ?? null }, false);
  }

  finalAnswer(): void {
    this._exitReason = "final_answer";
  }

  /** Set the exit reason the client reports if the agent set none. Used for
   *  the SDK-decided endings: `idle`, `process_exit`, `explicit_run_opened`.
   *  @internal */
  _setExitReasonIfUnset(reason: string): void {
    if (this._exitReason === null) this._exitReason = reason;
  }

  // ── Accessors (internal / testing) ────────────────────────────────────────

  currentStep(): number      { return this._step; }
  exitReason():  string|null { return this._exitReason; }
  getEvents():   AgentEvent[] { return this._events; }

  // ── Private ────────────────────────────────────────────────────────────────

  /**
   * Which llm.called this response belongs to.
   *
   * An explicit id is honoured when it names a call this run actually made; an
   * out-of-range one means llm.called never landed (its emit was swallowed), so
   * the key is omitted rather than pointing at a different call. With no
   * explicit id the most recent call is correct — that is the adjacent
   * called/responded pair every manual caller emits. Matches
   * RunContext.llm_responded's call_index handling in the Python SDK.
   */
  private _resolveCallId(explicit?: number): number | undefined {
    if (explicit !== undefined) {
      return Number.isInteger(explicit) && explicit >= 0 && explicit < this._callSeq
        ? explicit
        : undefined;
    }
    return this._callSeq > 0 ? this._callSeq - 1 : undefined;
  }

  private _emit(type: EventType, payload: Record<string, unknown>, advance = true): void {
    if (advance) this._step++;
    const event: AgentEvent = {
      event_type:    type,
      run_id:        this.runId,
      agent_id:      this._agentId,
      agent_version: this._version,
      step_index:    this._step,
      timestamp:     Date.now() / 1000,
      payload,
    };
    this._events.push(event);
    this._client._emit(event);
  }
}
