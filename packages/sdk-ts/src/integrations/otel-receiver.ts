/**
 * OTel span receiver for Dunetrace.
 *
 * Translates incoming OpenTelemetry spans (gen_ai.* semantic conventions) into
 * Dunetrace AgentEvents and ships them through the normal Dunetrace client, so
 * the structural detectors run on them server-side. Span content (prompts,
 * completions, tool arguments) is carried through as-is, same as the native SDK.
 *
 * Use this when an agent is already instrumented with an OTel-based tracer
 * (e.g. OpenLLMetry/Traceloop/OpenLIT) and you want Dunetrace's detectors without
 * adding manual dt.run() instrumentation. Attach it as a second span processor
 * alongside whatever exporter you already have.
 *
 *     import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
 *     import { SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
 *     import { Dunetrace } from "dunetrace";
 *     import { DunetraceOtelReceiver } from "dunetrace/integrations/otel-receiver";
 *
 *     const dt = new Dunetrace({ apiKey: "dt_live_..." });
 *     const provider = new NodeTracerProvider({
 *       spanProcessors: [new SimpleSpanProcessor(new DunetraceOtelReceiver(dt, "my-agent"))],
 *     });
 *     provider.register();
 *
 * Attributes read (Gen AI semconv first, then the OpenLLMetry / vector-store keys
 * real emitters also use) mirror the Python receiver value-for-value:
 *
 *   LLM span:   gen_ai.request.model / gen_ai.response.model / llm.request.model
 *               gen_ai.usage.input_tokens|prompt_tokens, output_tokens|completion_tokens,
 *               reasoning_tokens; gen_ai.completion / .0.content /
 *               gen_ai.output.messages / traceloop.entity.output; finish reason.
 *   Tool span:  gen_ai.tool.name / tool.name; arguments; result.
 *   Retrieval:  retrieval.index_name / vector_db.collection_name / db.name;
 *               result_count; top_score; documents.
 *
 * Voice events have no OTel convention and are best sent via the SDK directly.
 *
 * @opentelemetry/sdk-trace-base is an optional peer dependency. Import this module
 * only when it is installed.
 */

import { SpanStatusCode, type Attributes, type AttributeValue, type HrTime } from "@opentelemetry/api";
import { ExportResultCode, type ExportResult } from "@opentelemetry/core";
import type { ReadableSpan, SpanExporter } from "@opentelemetry/sdk-trace-base";
import type { Dunetrace } from "../client.js";
import type { DunetraceRun } from "../run.js";

// Finish-reason attribute — OpenLLMetry uses different keys across versions.
const FINISH_REASON_KEYS = [
  "gen_ai.completion.0.finish_reason",
  "gen_ai.response.finish_reasons.0",
  "llm.response.finish_reason",
] as const;

// Attribute keys that mark each span kind. Gen AI semconv first, then the
// OpenLLMetry / vector-store keys real emitters (OpenLIT, Traceloop) also use.
const LLM_KEYS = [
  "gen_ai.request.model",
  "gen_ai.response.model",
  "gen_ai.system",
  "gen_ai.provider.name",
  "llm.request.model",
] as const;
const TOOL_KEYS = ["gen_ai.tool.name", "tool.name"] as const;
const RETRIEVAL_KEYS = [
  "retrieval.index_name",
  "vector_db.collection_name",
  "vector_db.vendor",
  "db.name",
] as const;

/**
 * OTel SpanExporter that translates gen_ai.* spans into Dunetrace structural
 * events and ships each completed trace as one Dunetrace run. Add it as a
 * SimpleSpanProcessor (or BatchSpanProcessor) exporter alongside your existing
 * OTel pipeline — no changes to agent code required.
 */
export class DunetraceOtelReceiver implements SpanExporter {
  private readonly _dt: Dunetrace;
  private readonly _agentId: string;
  // Accumulate spans per trace until the root span arrives.
  private readonly _pending = new Map<string, ReadableSpan[]>();

  /**
   * @param dt       Dunetrace client instance.
   * @param agentId  Label for runs. Defaults to the root span name.
   */
  constructor(dt: Dunetrace, agentId = "") {
    this._dt = dt;
    this._agentId = agentId;
  }

  // ── SpanExporter interface ─────────────────────────────────────────────────

  export(spans: ReadableSpan[], resultCallback: (result: ExportResult) => void): void {
    try {
      for (const span of spans) {
        const tid = span.spanContext().traceId;
        const bucket = this._pending.get(tid);
        if (bucket) bucket.push(span);
        else this._pending.set(tid, [span]);
      }

      // Process any trace that now has its root span.
      const completed: ReadableSpan[][] = [];
      for (const [tid, batch] of this._pending) {
        if (batch.some(isRoot)) {
          completed.push(batch);
          this._pending.delete(tid);
        }
      }

      const runs = completed.map((batch) => this._processTrace(batch));
      Promise.all(runs).then(
        () => resultCallback({ code: ExportResultCode.SUCCESS }),
        () => resultCallback({ code: ExportResultCode.SUCCESS }), // translation errors must not fail the pipeline
      );
    } catch {
      resultCallback({ code: ExportResultCode.FAILED });
    }
  }

  shutdown(): Promise<void> {
    this._pending.clear();
    return Promise.resolve();
  }

  forceFlush(): Promise<void> {
    return Promise.resolve();
  }

  // ── Internal ───────────────────────────────────────────────────────────────

  private async _processTrace(batch: ReadableSpan[]): Promise<void> {
    const spans = [...batch].sort((a, b) => hrToMs(a.startTime) - hrToMs(b.startTime));
    const root = spans.find(isRoot) ?? spans[0];

    const agentId = this._agentId || (root ? root.name : "agent");
    const model = firstAttr(spans, "gen_ai.request.model") || "unknown";

    try {
      await this._dt.run(agentId, { model }, async (run) => {
        for (const span of spans) emitSpan(run, span);
        run.finalAnswer();
      });
    } catch {
      // Detection must never break on a malformed foreign trace.
    }
  }
}

// ── Span helpers ───────────────────────────────────────────────────────────────

function isRoot(span: ReadableSpan): boolean {
  return !span.parentSpanId;
}

/** HrTime ([seconds, nanos]) to float milliseconds, for ordering and durations. */
function hrToMs(t: HrTime): number {
  return t[0] * 1000 + t[1] / 1e6;
}

function firstAttr(spans: ReadableSpan[], key: string): string {
  for (const span of spans) {
    const val = span.attributes[key];
    if (val) return String(val);
  }
  return "";
}

function hasAny(attrs: Attributes, keys: readonly string[]): boolean {
  return keys.some((k) => attrs[k] !== undefined);
}

/** First present attribute among keys, as a string. Non-string values (a list of
 *  documents, a structured argument object) are JSON-serialized. */
function attrText(attrs: Attributes, keys: readonly string[]): string {
  for (const key of keys) {
    const val = attrs[key];
    if (val !== undefined && val !== null && val !== "") {
      return typeof val === "string" ? val : jsonSafe(val);
    }
  }
  return "";
}

function jsonSafe(val: unknown): string {
  try {
    return JSON.stringify(val);
  } catch {
    return String(val);
  }
}

function intAttr(val: AttributeValue | undefined): number {
  if (typeof val === "number") return Math.trunc(val);
  if (typeof val === "string") {
    const n = parseInt(val, 10);
    return Number.isFinite(n) ? n : 0;
  }
  return 0;
}

/** First present numeric-ish attribute among keys, as an int. */
function intFrom(attrs: Attributes, keys: readonly string[]): number {
  for (const key of keys) {
    const v = attrs[key];
    if (v !== undefined && v !== null && v !== "") return intAttr(v);
  }
  return 0;
}

/** Extract text from the gen_ai.output.messages structure (current GenAI
 *  convention, emitted by Traceloop): a list of messages each with `parts`
 *  [{type, content}]. Accepts a JSON string or a parsed array. */
function messagesContent(value: AttributeValue): string {
  let data: unknown;
  try {
    data = typeof value === "string" ? JSON.parse(value) : value;
  } catch {
    return "";
  }
  if (!Array.isArray(data)) return "";
  const texts: string[] = [];
  for (const msg of data) {
    if (typeof msg !== "object" || msg === null) continue;
    const m = msg as Record<string, unknown>;
    const parts = m["parts"];
    if (Array.isArray(parts)) {
      for (const part of parts) {
        if (part && typeof part === "object" && (part as Record<string, unknown>)["content"]) {
          texts.push(String((part as Record<string, unknown>)["content"]));
        }
      }
    } else if (m["content"]) {
      texts.push(String(m["content"]));
    }
  }
  return texts.filter(Boolean).join(" ");
}

/** Assistant output text across conventions: the plain-string keys, then the
 *  structured gen_ai.output.messages form. */
function llmOutput(attrs: Attributes): string {
  const text = attrText(attrs, [
    "gen_ai.completion",
    "gen_ai.completion.0.content",
    "traceloop.entity.output",
  ]);
  if (text) return text;
  const messages = attrs["gen_ai.output.messages"];
  return messages !== undefined ? messagesContent(messages) : "";
}

function finishReason(attrs: Attributes): string {
  for (const key of FINISH_REASON_KEYS) {
    const val = attrs[key];
    if (val) return String(val);
  }
  return "stop";
}

function spanIsError(span: ReadableSpan): boolean {
  return span.status.code === SpanStatusCode.ERROR;
}

/** args string to the Record shape run.toolCalled expects. A JSON object passes
 *  through; anything else (a bare string, an array) is wrapped so no content is
 *  lost and HTTP-shape detection downstream still works when a url is present. */
function toolArgsRecord(argsStr: string): Record<string, unknown> {
  if (!argsStr) return {};
  try {
    const parsed = JSON.parse(argsStr);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    return { input: parsed };
  } catch {
    return { input: argsStr };
  }
}

// ── Translation ────────────────────────────────────────────────────────────────

/** Emit the Dunetrace call(s) for one span. LLM and tool spans feed the
 *  structural detectors most heavily; retrieval spans feed the RAG detectors.
 *  Chains, agents, and other lifecycle spans have no distinct Dunetrace event and
 *  are skipped. */
function emitSpan(run: DunetraceRun, span: ReadableSpan): void {
  const attrs = span.attributes;
  const latencyMs = Math.max(0, Math.round(hrToMs(span.duration)));
  const isError = spanIsError(span);

  if (hasAny(attrs, LLM_KEYS)) {
    emitLlm(run, attrs, latencyMs, isError);
  } else if (hasAny(attrs, TOOL_KEYS)) {
    emitTool(run, attrs, latencyMs, isError, span.name);
  } else if (hasAny(attrs, RETRIEVAL_KEYS)) {
    emitRetrieval(run, attrs, latencyMs, span.name);
  }
}

function emitLlm(run: DunetraceRun, attrs: Attributes, latencyMs: number, isError: boolean): void {
  const model =
    firstOf(attrs, ["gen_ai.request.model", "gen_ai.response.model", "llm.request.model"]) || "unknown";
  // Current Gen AI semconv is input_tokens/output_tokens; the older OpenLLMetry
  // naming (prompt/completion) is accepted as a fallback so both modern and
  // legacy emitters populate token counts.
  const promptTokens = intFrom(attrs, [
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.prompt_tokens",
    "llm.usage.prompt_tokens",
  ]);
  const completionTokens = intFrom(attrs, [
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.completion_tokens",
    "llm.usage.completion_tokens",
  ]);
  const reasoningTokens = intAttr(attrs["gen_ai.usage.reasoning_tokens"]);
  const output = llmOutput(attrs);

  run.llmCalled(model, promptTokens);
  run.llmResponded({
    completionTokens,
    reasoningTokens,
    latencyMs,
    finishReason: isError ? "error" : finishReason(attrs),
    outputText: output,
    outputLength: output.length,
  });
}

function emitTool(
  run: DunetraceRun,
  attrs: Attributes,
  latencyMs: number,
  isError: boolean,
  spanName: string,
): void {
  const toolName = firstOf(attrs, ["gen_ai.tool.name", "tool.name"]) || spanName || "tool";
  const args = attrText(attrs, ["gen_ai.tool.call.arguments", "tool.arguments", "traceloop.entity.input"]);
  const output = attrText(attrs, ["gen_ai.tool.call.result", "tool.result", "traceloop.entity.output"]);
  run.toolCalled(toolName, toolArgsRecord(args));
  run.toolResponded(toolName, !isError, output.length, latencyMs, undefined, output);
}

function emitRetrieval(run: DunetraceRun, attrs: Attributes, latencyMs: number, spanName: string): void {
  const index =
    firstOf(attrs, ["retrieval.index_name", "vector_db.collection_name", "db.name"]) ||
    spanName ||
    "retrieval";
  const resultCount = intFrom(attrs, ["retrieval.result_count", "db.result_count"]);
  const topScoreRaw = attrs["retrieval.top_score"];
  const topScore = typeof topScoreRaw === "number" ? topScoreRaw : undefined;
  const content = attrText(attrs, ["retrieval.documents", "traceloop.entity.output"]);
  run.retrievalCalled(index);
  run.retrievalResponded(index, resultCount, topScore, latencyMs, content);
}

/** First present string-valued attribute among keys. */
function firstOf(attrs: Attributes, keys: readonly string[]): string {
  for (const key of keys) {
    const val = attrs[key];
    if (val) return String(val);
  }
  return "";
}
