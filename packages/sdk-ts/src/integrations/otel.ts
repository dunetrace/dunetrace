/**
 * OTel span exporter for Dunetrace. Translates Dunetrace AgentEvents into
 * OpenTelemetry spans so agent runs show up in whatever OTel backend a customer
 * already runs (Datadog, Grafana Tempo, Honeycomb, Signoz, Jaeger).
 *
 * This runs alongside Dunetrace's own ingest, not instead of it: the same events
 * still ship to Dunetrace. OTel export is additive and opt-in.
 *
 * Pass a built exporter to the client and every event fans out to it:
 *
 *     import { trace } from "@opentelemetry/api";
 *     import { Dunetrace } from "dunetrace";
 *     import { DunetraceOtelExporter } from "dunetrace/integrations/otel";
 *
 *     const dt = new Dunetrace({
 *       exporter: new DunetraceOtelExporter({ tracer: trace.getTracer("dunetrace") }),
 *     });
 *
 * Span model (run + LLM + tool + retrieval + voice):
 *
 *     Trace (trace_id derived from run_id, so a Dunetrace run is one trace)
 *     └── span "dunetrace.run"              [dunetrace.run.id, dunetrace.run.agent_id, ...]
 *         ├── span "chat {model}"           [gen_ai.provider.name, gen_ai.request.model, ...]
 *         ├── span "dunetrace.tool.{name}"  [dunetrace.tool.*, url.full/server.address if HTTP]
 *         ├── span "dunetrace.retrieval"    [dunetrace.retrieval.vector_store, .document_count]
 *         ├── span "dunetrace.voice.transcription" / "dunetrace.voice.tts"
 *         └── event "dunetrace.voice.{vad}" [silence/speech_start/...] on the run span
 *
 * Semantic conventions match the Python SDK's exporter value-for-value:
 *   - LLM spans follow the OpenTelemetry GenAI conventions (gen_ai.*). Provider is
 *     emitted as gen_ai.provider.name, not the deprecated gen_ai.system.
 *   - HTTP-shaped tool calls use the current stable HTTP conventions
 *     (url.full, server.address, http.request.method, http.response.status_code),
 *     not the deprecated http.url / http.method / http.status_code.
 *   - Agent runs, tools, retrievals, and voice events have no standard OTel
 *     operation type, so those fields are namespaced under dunetrace.*.
 *
 * PII: content-bearing attributes (tool args, request URL, retrieval query) are
 * emitted only when captureContent is true (the default). Voice spans never carry
 * raw transcript or TTS text, only character counts and metadata.
 *
 * Correlation: traceId is uuid(run_id) as 128 bits and the root span id is its
 * lower 64 bits, both deterministic. traceIdHex()/rootSpanIdHex() expose them so
 * a caller can stamp them elsewhere and a backend can deep-link to the run. A
 * child run (parentRunId set) gets its own trace and a span Link back to the
 * parent run's root span, the OTel-correct way to express causally-related spans
 * that live in different traces.
 *
 * @opentelemetry/api is an optional peer dependency. Import this module only when
 * it is installed.
 */

import {
  trace,
  context,
  SpanKind,
  SpanStatusCode,
  TraceFlags,
  type Attributes,
  type Context,
  type HrTime,
  type Link,
  type Span,
  type SpanContext,
  type Tracer,
  type TracerProvider,
} from "@opentelemetry/api";
import type { AgentEvent } from "../models.js";

// ── ID derivation ────────────────────────────────────────────────────────────

/** Float Unix seconds to an OTel HrTime ([seconds, nanoseconds]). Unambiguous,
 *  unlike a bare epoch-millis number which the API interprets by magnitude. */
function hrTime(tsSeconds: number): HrTime {
  const seconds = Math.trunc(tsSeconds);
  const nanos = Math.round((tsSeconds - seconds) * 1e9);
  return [seconds, nanos];
}

/** True when s is a usable 32-hex-char trace id (not the all-zero invalid one). */
function isTraceIdHex(s: string): boolean {
  return /^[0-9a-f]{32}$/.test(s) && s !== "00000000000000000000000000000000";
}

/** 32-hex-char trace ID for run_id (W3C traceparent format). A Dunetrace run UUID
 *  maps to a stable trace both a backend and the dashboard can address. Returns
 *  "" when run_id is not UUID-shaped. */
export function traceIdHex(runId: string): string {
  const hex = runId.replace(/-/g, "").toLowerCase();
  return isTraceIdHex(hex) ? hex : "";
}

/** 16-hex-char root span ID for run_id (its lower 64 bits). "" when not UUID-shaped. */
export function rootSpanIdHex(runId: string): string {
  const hex = traceIdHex(runId);
  return hex ? hex.slice(16) : "";
}

/** Deterministic SpanContext for a run's root span. Seeds a run's own trace and
 *  links a child run to its parent. Returns null when run_id is not UUID-shaped,
 *  in which case the caller lets OTel assign a random trace instead of dropping
 *  the run. */
function spanContextFor(runId: string): SpanContext | null {
  const traceId = traceIdHex(runId);
  if (!traceId) return null;
  return {
    traceId,
    spanId: traceId.slice(16),
    isRemote: true,
    traceFlags: TraceFlags.SAMPLED,
  };
}

/** Context whose active span is the run's (deterministic) root span, as a
 *  non-recording remote span. Parents the real root span onto it so the root
 *  inherits the deterministic trace id. Falls back to the active context when
 *  run_id is not UUID-shaped (OTel assigns a random trace). */
function parentCtxForRun(runId: string): Context {
  const sc = spanContextFor(runId);
  return sc ? trace.setSpanContext(context.active(), sc) : context.active();
}

// ── Provider inference (gen_ai.provider.name) ────────────────────────────────
//
// The GenAI conventions want a low-cardinality provider name. We infer it from
// the model string. Prefix match; an unknown model gets no provider attribute
// rather than a wrong guess.

const PROVIDER_PREFIXES: ReadonlyArray<readonly [string, string]> = [
  ["claude", "anthropic"],
  ["gpt", "openai"],
  ["o1", "openai"],
  ["o3", "openai"],
  ["o4", "openai"],
  ["chatgpt", "openai"],
  ["gemini", "gcp.gemini"],
  ["mistral", "mistral_ai"],
  ["mixtral", "mistral_ai"],
  ["command", "cohere"],
  ["deepseek", "deepseek"],
  ["grok", "x_ai"],
];

function providerFromModel(model: string): string {
  const m = (model || "").toLowerCase();
  for (const [prefix, provider] of PROVIDER_PREFIXES) {
    if (m.startsWith(prefix)) return provider;
  }
  return "";
}

// ── Token pricing (USD per token) ────────────────────────────────────────────
// Mirrors the Python SDK's dunetrace.policies table so a span's cost_usd matches
// Dunetrace's own cost accounting. Matched by prefix; falls back to DEFAULT_PRICE.

const MODEL_PRICES: ReadonlyArray<readonly [string, { input: number; output: number }]> = [
  ["claude-opus-4", { input: 15.0e-6, output: 75.0e-6 }],
  ["claude-sonnet-4", { input: 3.0e-6, output: 15.0e-6 }],
  ["claude-haiku-4", { input: 0.8e-6, output: 4.0e-6 }],
  ["claude-3-5-sonnet", { input: 3.0e-6, output: 15.0e-6 }],
  ["claude-3-5-haiku", { input: 0.8e-6, output: 4.0e-6 }],
  ["claude-3-opus", { input: 15.0e-6, output: 75.0e-6 }],
  ["gpt-4o-mini", { input: 0.15e-6, output: 0.6e-6 }],
  ["gpt-4o", { input: 5.0e-6, output: 15.0e-6 }],
  ["gpt-4-turbo", { input: 10.0e-6, output: 30.0e-6 }],
  ["gpt-3.5-turbo", { input: 0.5e-6, output: 1.5e-6 }],
];
const DEFAULT_PRICE = { input: 3.0e-6, output: 12.0e-6 };

function priceFor(model: string): { input: number; output: number } {
  for (const [key, price] of MODEL_PRICES) {
    if (model.startsWith(key) || model.includes(key)) return price;
  }
  return DEFAULT_PRICE;
}

function llmCostUsd(model: string, inputTokens: number, outputTokens: number): number {
  const price = priceFor(model || "");
  return inputTokens * price.input + outputTokens * price.output;
}

// ── HTTP shape detection ─────────────────────────────────────────────────────

interface HttpInfo {
  url: string;
  method?: string;
}

/** Best-effort HTTP shape from a tool call's serialized args. Returns {url,
 *  method?} when the args look like an HTTP call, else null. The TS SDK stores
 *  args as JSON (run.toolCalled does JSON.stringify), so only a genuine JSON
 *  object with a string url parses; anything else is a non-HTTP tool. */
function httpInfoFromArgs(argsRepr: string): HttpInfo | null {
  if (!argsRepr || !argsRepr.includes("url")) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(argsRepr);
  } catch {
    return null;
  }
  if (typeof parsed !== "object" || parsed === null) return null;
  const url = (parsed as Record<string, unknown>)["url"];
  if (typeof url !== "string" || !url) return null;
  const info: HttpInfo = { url };
  const method = (parsed as Record<string, unknown>)["method"];
  if (typeof method === "string" && method) info.method = method.toUpperCase();
  return info;
}

function hostname(url: string): string {
  try {
    return new URL(url).hostname || "";
  } catch {
    return "";
  }
}

// ── Per-run span state ───────────────────────────────────────────────────────

interface RunSpans {
  rootSpan: Span;
  startTs: number;
  childSpan: Span | null;
  childKind: "" | "llm" | "tool" | "retrieval";
  llmModel: string;
  llmInputTokens: number;
  toolIsHttp: boolean;
}

// ── Options ──────────────────────────────────────────────────────────────────

export interface DunetraceOtelExporterOptions {
  /** A pre-built OTel Tracer (e.g. trace.getTracer("dunetrace")). */
  tracer?: Tracer;
  /** A TracerProvider to get a tracer from, when you wire your own pipeline. */
  tracerProvider?: TracerProvider;
  /** Instrumentation scope name. Default "dunetrace". */
  tracerName?: string;
  /** When false, drop content-bearing attributes (tool args, request URL,
   *  retrieval query) from spans. Default true. */
  captureContent?: boolean;
}

// ── Exporter ─────────────────────────────────────────────────────────────────

/**
 * Translates Dunetrace AgentEvents into OpenTelemetry spans. No locking is
 * needed: Node runs the event loop on a single thread, so span open/close is
 * naturally sequential per run.
 */
export class DunetraceOtelExporter {
  private readonly _tracer: Tracer;
  private readonly _captureContent: boolean;
  private readonly _runs = new Map<string, RunSpans>();

  constructor(opts: DunetraceOtelExporterOptions = {}) {
    if (opts.tracer) {
      this._tracer = opts.tracer;
    } else {
      const tp = opts.tracerProvider ?? trace.getTracerProvider();
      this._tracer = tp.getTracer(opts.tracerName ?? "dunetrace");
    }
    this._captureContent = opts.captureContent ?? true;
  }

  /** Route one AgentEvent to its span operation. Never throws: an export failure
   *  must not touch the agent's own run. Events with no span mapping (external
   *  signals, memory, turn-taking, recording metadata) are ignored. */
  handle(event: AgentEvent): void {
    try {
      switch (event.event_type) {
        case "run.started":            this._onRunStarted(event); break;
        case "run.completed":
        case "run.errored":            this._onRunEnded(event); break;
        case "llm.called":             this._onLlmCalled(event); break;
        case "llm.responded":          this._onLlmResponded(event); break;
        case "tool.called":            this._onToolCalled(event); break;
        case "tool.responded":         this._onToolResponded(event); break;
        case "retrieval.called":       this._onRetrievalCalled(event); break;
        case "retrieval.responded":    this._onRetrievalResponded(event); break;
        case "transcription.received": this._onVoiceTranscription(event); break;
        case "tts.generated":          this._onVoiceTts(event); break;
        case "voice_activity.detected": this._onVoiceActivity(event); break;
        default: break;
      }
    } catch (exc) {
      // eslint-disable-next-line no-console
      console.warn(
        `[dunetrace.otel] failed to handle ${event.event_type} for run ${event.run_id}: ${String(exc)}`,
      );
    }
  }

  // ── Run span ───────────────────────────────────────────────────────────────

  private _onRunStarted(event: AgentEvent): void {
    const parentCtx = parentCtxForRun(event.run_id);

    const attrs: Attributes = {
      "dunetrace.run.id": event.run_id,
      "dunetrace.run.agent_id": event.agent_id,
      "dunetrace.run.agent_version": event.agent_version,
      "dunetrace.run.status": "running",
    };
    if (event.conversation_id) attrs["dunetrace.run.conversation_id"] = event.conversation_id;
    if (event.parent_run_id) attrs["dunetrace.run.parent_run_id"] = event.parent_run_id;
    const model = event.payload["model"];
    if (typeof model === "string" && model) attrs["dunetrace.run.model"] = model;
    const tools = event.payload["tools"];
    if (Array.isArray(tools) && tools.length) attrs["dunetrace.run.tools"] = tools.join(",");

    // A child run lives in its own trace; link it back to the parent run's root
    // span so a backend can follow the causal edge across traces.
    const links: Link[] = [];
    if (event.parent_run_id) {
      const parentSc = spanContextFor(event.parent_run_id);
      if (parentSc) links.push({ context: parentSc });
    }

    const root = this._tracer.startSpan(
      "dunetrace.run",
      {
        kind: SpanKind.INTERNAL,
        startTime: hrTime(event.timestamp),
        attributes: attrs,
        links: links.length ? links : undefined,
      },
      parentCtx,
    );
    this._runs.set(event.run_id, {
      rootSpan: root,
      startTs: event.timestamp,
      childSpan: null,
      childKind: "",
      llmModel: "",
      llmInputTokens: 0,
      toolIsHttp: false,
    });
  }

  private _onRunEnded(event: AgentEvent): void {
    const rs = this._runs.get(event.run_id);
    if (!rs) return;
    this._runs.delete(event.run_id);

    // Close any child span left open (e.g. an LLM call with no response event).
    if (rs.childSpan) {
      rs.childSpan.setStatus({ code: SpanStatusCode.ERROR, message: "span ended without a response event" });
      rs.childSpan.end(hrTime(event.timestamp));
    }

    const root = rs.rootSpan;
    const failed = event.event_type === "run.errored";
    root.setAttribute("dunetrace.run.status", failed ? "failed" : "completed");
    root.setAttribute("dunetrace.run.duration_ms", Math.trunc((event.timestamp - rs.startTs) * 1000));
    root.setAttribute("dunetrace.run.total_steps", numAttr(event.payload["total_steps"]));
    root.setAttribute("dunetrace.run.exit_reason", strAttr(event.payload["exit_reason"]));
    root.setAttribute("dunetrace.run.tool_call_count", numAttr(event.payload["tool_call_count"]));
    if (failed) {
      const errorType = strAttr(event.payload["error_type"]) || "run errored";
      root.setAttribute("dunetrace.run.error_type", errorType);
      root.setStatus({ code: SpanStatusCode.ERROR, message: errorType });
    }
    root.end(hrTime(event.timestamp));
  }

  // ── LLM span (GenAI conventions) ─────────────────────────────────────────────

  private _onLlmCalled(event: AgentEvent): void {
    const rs = this._runs.get(event.run_id);
    if (!rs) return;
    this._closeOrphanChild(rs, event.timestamp);

    const model = strAttr(event.payload["model"]);
    const inputTokens = numAttr(event.payload["prompt_tokens"]);
    const name = model ? `chat ${model}` : "chat";

    const attrs: Attributes = {
      "gen_ai.operation.name": "chat",
      "dunetrace.run.id": event.run_id,
      "dunetrace.step_index": event.step_index,
    };
    if (model) attrs["gen_ai.request.model"] = model;
    const provider = providerFromModel(model);
    if (provider) attrs["gen_ai.provider.name"] = provider;
    if (inputTokens) attrs["gen_ai.usage.input_tokens"] = inputTokens;

    rs.childSpan = this._tracer.startSpan(
      name,
      { kind: SpanKind.CLIENT, startTime: hrTime(event.timestamp), attributes: attrs },
      trace.setSpan(context.active(), rs.rootSpan),
    );
    rs.childKind = "llm";
    rs.llmModel = model;
    rs.llmInputTokens = inputTokens;
  }

  private _onLlmResponded(event: AgentEvent): void {
    const rs = this._runs.get(event.run_id);
    if (!rs || !rs.childSpan || rs.childKind !== "llm") return;

    const p = event.payload;
    const span = rs.childSpan;
    const finishReason = strAttr(p["finish_reason"]);
    const completion = numAttr(p["completion_tokens"]);
    const reasoning = numAttr(p["reasoning_tokens"]);
    const outputTokens = completion + reasoning;
    // prompt_tokens may be back-filled here if it was only known post-call.
    const inputTokens = numAttr(p["prompt_tokens"]) || rs.llmInputTokens;
    if (inputTokens && !rs.llmInputTokens) span.setAttribute("gen_ai.usage.input_tokens", inputTokens);

    if (finishReason) {
      span.setAttribute("gen_ai.response.finish_reasons", [finishReason]);
      span.setAttribute("dunetrace.llm.output_truncated", finishReason === "length");
    }
    if (outputTokens) span.setAttribute("gen_ai.usage.output_tokens", outputTokens);
    if (reasoning) span.setAttribute("dunetrace.llm.reasoning_tokens", reasoning);
    if (numAttr(p["latency_ms"])) span.setAttribute("dunetrace.llm.latency_ms", numAttr(p["latency_ms"]));
    if (numAttr(p["output_length"])) span.setAttribute("dunetrace.llm.output_length", numAttr(p["output_length"]));

    const cost = llmCostUsd(rs.llmModel, inputTokens, completion);
    if (cost) span.setAttribute("dunetrace.llm.cost_usd", cost);

    if (p["error"]) span.setStatus({ code: SpanStatusCode.ERROR, message: String(p["error"]) });

    span.end(hrTime(event.timestamp));
    rs.childSpan = null;
    rs.childKind = "";
  }

  // ── Tool span ────────────────────────────────────────────────────────────────

  private _onToolCalled(event: AgentEvent): void {
    const rs = this._runs.get(event.run_id);
    if (!rs) return;
    this._closeOrphanChild(rs, event.timestamp);

    const toolName = strAttr(event.payload["tool_name"]);
    const argsRepr = strAttr(event.payload["args"]);
    const http = httpInfoFromArgs(argsRepr);

    const attrs: Attributes = {
      "dunetrace.tool.name": toolName,
      "dunetrace.run.id": event.run_id,
      "dunetrace.step_index": event.step_index,
    };
    if (this._captureContent && argsRepr) attrs["dunetrace.tool.args"] = argsRepr;
    if (http) {
      const host = hostname(http.url);
      if (host) attrs["server.address"] = host;
      if (http.method) attrs["http.request.method"] = http.method;
      if (this._captureContent) attrs["url.full"] = http.url;
    }

    rs.childSpan = this._tracer.startSpan(
      toolName ? `dunetrace.tool.${toolName}` : "dunetrace.tool",
      {
        kind: http ? SpanKind.CLIENT : SpanKind.INTERNAL,
        startTime: hrTime(event.timestamp),
        attributes: attrs,
      },
      trace.setSpan(context.active(), rs.rootSpan),
    );
    rs.childKind = "tool";
    rs.toolIsHttp = Boolean(http);
  }

  private _onToolResponded(event: AgentEvent): void {
    const rs = this._runs.get(event.run_id);
    if (!rs || !rs.childSpan || rs.childKind !== "tool") return;

    const p = event.payload;
    const span = rs.childSpan;
    const success = p["success"] !== false;
    span.setAttribute("dunetrace.tool.result_status", success ? "success" : "error");
    if (numAttr(p["output_length"])) span.setAttribute("dunetrace.tool.output_length", numAttr(p["output_length"]));
    if (numAttr(p["latency_ms"])) span.setAttribute("dunetrace.tool.latency_ms", numAttr(p["latency_ms"]));

    const error = p["error"];
    if (error !== undefined && error !== null && error !== "") {
      // The HTTP auto-instrumentation puts the status code in `error` on a failed
      // request; surface it as http.response.status_code when it's a bare number.
      // Otherwise it's a free-text message (gated as content).
      if (rs.toolIsHttp && /^\d+$/.test(String(error))) {
        span.setAttribute("http.response.status_code", parseInt(String(error), 10));
      } else if (this._captureContent) {
        span.setAttribute("dunetrace.tool.error_message", String(error));
      }
    }
    if (!success) span.setStatus({ code: SpanStatusCode.ERROR, message: "tool call failed" });

    span.end(hrTime(event.timestamp));
    rs.childSpan = null;
    rs.childKind = "";
    rs.toolIsHttp = false;
  }

  // ── Retrieval span ───────────────────────────────────────────────────────────

  private _onRetrievalCalled(event: AgentEvent): void {
    const rs = this._runs.get(event.run_id);
    if (!rs) return;
    this._closeOrphanChild(rs, event.timestamp);

    const attrs: Attributes = {
      "dunetrace.run.id": event.run_id,
      "dunetrace.step_index": event.step_index,
    };
    const indexName = strAttr(event.payload["index_name"]);
    if (indexName) attrs["dunetrace.retrieval.vector_store"] = indexName;
    const query = strAttr(event.payload["query"]);
    if (query && this._captureContent) attrs["dunetrace.retrieval.query"] = query;

    rs.childSpan = this._tracer.startSpan(
      "dunetrace.retrieval",
      { kind: SpanKind.CLIENT, startTime: hrTime(event.timestamp), attributes: attrs },
      trace.setSpan(context.active(), rs.rootSpan),
    );
    rs.childKind = "retrieval";
  }

  private _onRetrievalResponded(event: AgentEvent): void {
    const rs = this._runs.get(event.run_id);
    if (!rs || !rs.childSpan || rs.childKind !== "retrieval") return;

    const p = event.payload;
    const span = rs.childSpan;
    span.setAttribute("dunetrace.retrieval.document_count", numAttr(p["result_count"]));
    if (p["top_score"] !== undefined && p["top_score"] !== null) {
      span.setAttribute("dunetrace.retrieval.top_score", numAttr(p["top_score"]));
    }
    if (numAttr(p["latency_ms"])) span.setAttribute("dunetrace.retrieval.latency_ms", numAttr(p["latency_ms"]));
    // An empty retrieval is a soft signal (RAG_EMPTY_RETRIEVAL), not a failed
    // operation, so the span stays OK.

    span.end(hrTime(event.timestamp));
    rs.childSpan = null;
    rs.childKind = "";
  }

  // ── Voice spans (only fire when the voice hooks are used) ─────────────────────

  private _onVoiceTranscription(event: AgentEvent): void {
    const rs = this._runs.get(event.run_id);
    if (!rs) return;
    const p = event.payload;
    const latencyMs = numAttr(p["latency_ms"]);
    const attrs: Attributes = {
      "dunetrace.run.id": event.run_id,
      "dunetrace.voice.confidence": numAttr(p["confidence"]),
      "dunetrace.voice.char_count": strAttr(p["text"]).length,
    };
    if (latencyMs) attrs["dunetrace.voice.latency_ms"] = latencyMs;
    if (numAttr(p["audio_seconds"])) attrs["dunetrace.voice.audio_seconds"] = numAttr(p["audio_seconds"]);
    this._emitPointSpan(rs, "dunetrace.voice.transcription", event.timestamp, latencyMs, attrs);
  }

  private _onVoiceTts(event: AgentEvent): void {
    const rs = this._runs.get(event.run_id);
    if (!rs) return;
    const p = event.payload;
    const latencyMs = numAttr(p["latency_ms"]);
    const attrs: Attributes = {
      "dunetrace.run.id": event.run_id,
      "dunetrace.voice.char_count": strAttr(p["text"]).length,
      "dunetrace.voice.truncated": Boolean(p["truncated"]),
    };
    if (strAttr(p["model"])) attrs["dunetrace.voice.model"] = strAttr(p["model"]);
    if (strAttr(p["voice_id"])) attrs["dunetrace.voice.voice_id"] = strAttr(p["voice_id"]);
    if (strAttr(p["provider"])) attrs["dunetrace.voice.provider"] = strAttr(p["provider"]);
    if (latencyMs) attrs["dunetrace.voice.latency_ms"] = latencyMs;
    if (numAttr(p["audio_seconds"])) attrs["dunetrace.voice.audio_seconds"] = numAttr(p["audio_seconds"]);
    this._emitPointSpan(rs, "dunetrace.voice.tts", event.timestamp, latencyMs, attrs);
  }

  private _onVoiceActivity(event: AgentEvent): void {
    // VAD transitions are high-frequency; a span per audio frame would blow up
    // span counts, so they're span events on the run span.
    const rs = this._runs.get(event.run_id);
    if (!rs) return;
    const vadType = strAttr(event.payload["type"]) || "vad";
    const attrs: Attributes = {};
    if (numAttr(event.payload["duration_ms"])) attrs["dunetrace.voice.duration_ms"] = numAttr(event.payload["duration_ms"]);
    rs.rootSpan.addEvent(`dunetrace.voice.${vadType}`, attrs, hrTime(event.timestamp));
  }

  // ── Helpers ──────────────────────────────────────────────────────────────────

  /** Open-and-close a standalone child span for a single-event operation
   *  (transcription, TTS). Duration reflects latency when known. Doesn't touch
   *  rs.childSpan, so it can't disturb an open called/responded pair. */
  private _emitPointSpan(rs: RunSpans, name: string, ts: number, latencyMs: number, attrs: Attributes): void {
    const start = latencyMs ? ts - latencyMs / 1000 : ts;
    const span = this._tracer.startSpan(
      name,
      { kind: SpanKind.INTERNAL, startTime: hrTime(start), attributes: attrs },
      trace.setSpan(context.active(), rs.rootSpan),
    );
    span.end(hrTime(ts));
  }

  /** Close a still-open child (mismatched called/responded pair) as ERROR so a
   *  dropped response event can't leak an unclosed span. */
  private _closeOrphanChild(rs: RunSpans, ts: number): void {
    if (rs.childSpan) {
      rs.childSpan.setStatus({ code: SpanStatusCode.ERROR, message: "span ended without a response event" });
      rs.childSpan.end(hrTime(ts));
      rs.childSpan = null;
      rs.childKind = "";
    }
  }
}

// ── Attribute coercion ───────────────────────────────────────────────────────
// Payload values are unknown; coerce them to the primitive an OTel attribute
// expects rather than passing `unknown` through.

function numAttr(v: unknown): number {
  const n = typeof v === "number" ? v : typeof v === "string" ? Number(v) : 0;
  return Number.isFinite(n) ? n : 0;
}

function strAttr(v: unknown): string {
  return typeof v === "string" ? v : v == null ? "" : String(v);
}
