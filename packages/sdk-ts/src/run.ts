import { randomUUID } from "node:crypto";
import type { AgentEvent, EventType, LlmRespondedOptions, MemorySource } from "./models.js";
import { MEMORY_SOURCES } from "./models.js";

export interface EventEmitter {
  _emit(event: AgentEvent): void;
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

  private _agentId:    string;
  private _version:    string;
  private _client:     EventEmitter;
  private _step        = 0;
  private _exitReason: string | null = null;
  private _events:     AgentEvent[]  = [];

  constructor(agentId: string, version: string, client: EventEmitter, runId?: string) {
    this.runId    = runId ?? randomUUID();
    this._agentId = agentId;
    this._version = version;
    this._client  = client;
  }

  // ── LLM hooks ──────────────────────────────────────────────────────────────

  llmCalled(model: string, promptTokens = 0): void {
    this._emit("llm.called", { model, prompt_tokens: promptTokens });
  }

  llmResponded(opts: LlmRespondedOptions = {}): void {
    const payload: Record<string, unknown> = {
      completion_tokens: opts.completionTokens ?? 0,
      latency_ms:        opts.latencyMs        ?? 0,
      finish_reason:     opts.finishReason      ?? "stop",
      output_length:     opts.outputLength      ?? (opts.outputText?.length ?? 0),
    };
    // Transmit the output text by default; omit it (bandwidth) when
    // DUNETRACE_OMIT_LLM_OUTPUT_TEXT is set. output_length is always sent.
    if (!omitLlmOutputText()) {
      payload["output"] = opts.outputText ?? "";
    }
    if (opts.promptTokens) payload["prompt_tokens"] = opts.promptTokens;
    this._emit("llm.responded", payload, false);
  }

  // ── Tool hooks ─────────────────────────────────────────────────────────────

  toolCalled(toolName: string, args: Record<string, unknown> = {}): void {
    this._emit("tool.called", {
      tool_name: toolName,
      args: JSON.stringify(args),
    });
  }

  toolResponded(
    toolName:    string,
    success:     boolean,
    outputLength = 0,
    latencyMs    = 0,
    error?:      string,
    output       = "",
  ): void {
    const payload: Record<string, unknown> = {
      tool_name:     toolName,
      success,
      output_length: outputLength,
      latency_ms:    latencyMs,
      output,
    };
    if (error) payload["error"] = error;
    this._emit("tool.responded", payload, false);
  }

  // ── Retrieval hooks ────────────────────────────────────────────────────────

  retrievalCalled(indexName: string, query = ""): void {
    this._emit("retrieval.called", {
      index_name: indexName,
      query,
    });
  }

  retrievalResponded(
    indexName:   string,
    resultCount: number,
    topScore?:   number,
    latencyMs    = 0,
    content      = "",
  ): void {
    this._emit("retrieval.responded", {
      index_name:   indexName,
      result_count: resultCount,
      top_score:    topScore ?? null,
      latency_ms:   latencyMs,
      content,
    }, false);
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

    this.llmCalled(model, promptTokens);
    this.llmResponded({ completionTokens, latencyMs, finishReason, outputText });
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
    const payload: Record<string, unknown> = { key, value };
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

  // ── Accessors (internal / testing) ────────────────────────────────────────

  currentStep(): number      { return this._step; }
  exitReason():  string|null { return this._exitReason; }
  getEvents():   AgentEvent[] { return this._events; }

  // ── Private ────────────────────────────────────────────────────────────────

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
