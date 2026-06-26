/**
 * Vercel AI SDK integration for Dunetrace.
 *
 * Wraps `generateText` / `streamText` (from the `ai` package) to emit
 * llm.called, llm.responded, tool.called, and tool.responded events
 * automatically when called inside a `dt.run()` context.
 *
 * Requires the `ai` package (peer dependency) when using this module.
 */

import type {
  generateText,
  streamText,
  StepResult,
  ToolSet,
  LanguageModel,
  LanguageModelUsage,
  TypedToolCall,
  TypedToolResult,
  TypedToolError,
} from "ai";
import { getCurrentRun, type Dunetrace } from "../client.js";
import { resultLength } from "../util.js";
import type { DunetraceRun } from "../run.js";
import type { RunOptions } from "../models.js";

/** Options accepted by `generateText` from the Vercel AI SDK. */
export type GenerateTextOptions = Parameters<typeof generateText>[0];

/** Options accepted by `streamText` from the Vercel AI SDK. */
export type StreamTextOptions = Parameters<typeof streamText>[0];

// ── Instrumentation ───────────────────────────────────────────────────────────

/** Inject Dunetrace callbacks into generateText options. No-op outside a run. */
export function instrumentGenerateTextOptions<OPTIONS extends GenerateTextOptions>(
  opts: OPTIONS,
): OPTIONS {
  const run = getCurrentRun();
  if (!run) return opts;

  const model            = modelId(opts.model);
  const userOnStepFinish = opts.onStepFinish;
  let   stepStart        = Date.now();

  return {
    ...opts,
    onStepFinish: async (step) => {
      const latencyMs = Date.now() - stepStart;
      emitStepEvents(run, step, model, latencyMs);
      stepStart = Date.now();
      if (userOnStepFinish) await userOnStepFinish(step);
    },
  };
}

/**
 * Inject Dunetrace callbacks into streamText options. No-op outside a run.
 *
 * Events are emitted from `onStepFinish`, which the AI SDK fires once per step
 * (including the final step). We deliberately do NOT also emit from `onFinish`
 * — doing so would double-count the last step. Any user-supplied `onStepFinish`
 * is chained after ours; a user-supplied `onFinish` is preserved untouched.
 */
export function instrumentStreamTextOptions<OPTIONS extends StreamTextOptions>(
  opts: OPTIONS,
): OPTIONS {
  const run = getCurrentRun();
  if (!run) return opts;

  const model            = modelId(opts.model);
  const userOnStepFinish = opts.onStepFinish;
  let   stepStart        = Date.now();

  return {
    ...opts,
    onStepFinish: async (step) => {
      const latencyMs = Date.now() - stepStart;
      emitStepEvents(run, step, model, latencyMs);
      stepStart = Date.now();
      if (userOnStepFinish) await userOnStepFinish(step);
    },
  };
}

/** Wrap a `generateText` import to auto-instrument every call. */
export function wrapGenerateText(generateTextFn: typeof generateText): typeof generateText {
  const wrapped = (async (opts: GenerateTextOptions) =>
    generateTextFn(instrumentGenerateTextOptions(opts))) as typeof generateText;
  return wrapped;
}

/**
 * Wrap a `streamText` import to auto-instrument every call.
 *
 * `streamText` is synchronous in the AI SDK (it returns a `StreamTextResult`
 * immediately, before the stream is consumed), so the wrapper is synchronous too
 * — wrapping it in an async function would change the return type to a Promise
 * and break drop-in usage like `wrapped(opts).textStream`.
 */
export function wrapStreamText(streamTextFn: typeof streamText): typeof streamText {
  return ((opts: StreamTextOptions) =>
    streamTextFn(instrumentStreamTextOptions(opts))) as typeof streamText;
}

/** Open a Dunetrace run, call generateText with instrumentation, and close the run. */
export async function traceGenerateText(
  dt: Dunetrace,
  agentId: string,
  runOpts: RunOptions,
  generateTextFn: typeof generateText,
  textOpts: GenerateTextOptions,
): Promise<Awaited<ReturnType<typeof generateText>>> {
  const userInput = runOpts.userInput ?? deriveUserInput(textOpts);
  const model     = runOpts.model ?? modelId(textOpts.model);
  const tools     = runOpts.tools ?? toolNames(textOpts.tools);

  return dt.run(agentId, { ...runOpts, userInput, model, tools }, async (run) => {
    const result = await generateTextFn(instrumentGenerateTextOptions(textOpts));
    run.finalAnswer();
    return result;
  });
}

/**
 * Open a Dunetrace run, call streamText with instrumentation, and close the run.
 *
 * Because `onStepFinish` only fires while the stream is being consumed, this
 * helper drains the stream (`consumeStream()`) before emitting `run.completed`,
 * so the per-step `llm.*` / `tool.*` events are ordered inside the run boundary.
 * As a result the returned stream is already consumed — use this when you only
 * need the final result (`result.text`, `result.usage`). For incremental
 * streaming to a client, use `wrapStreamText` / `instrumentStreamTextOptions`
 * inside an explicit `dt.run()` and call `run.finalAnswer()` yourself.
 */
export async function traceStreamText(
  dt: Dunetrace,
  agentId: string,
  runOpts: RunOptions,
  streamTextFn: typeof streamText,
  textOpts: StreamTextOptions,
): Promise<Awaited<ReturnType<typeof streamText>>> {
  const userInput = runOpts.userInput ?? deriveUserInput(textOpts);
  const model     = runOpts.model ?? modelId(textOpts.model);
  const tools     = runOpts.tools ?? toolNames(textOpts.tools);

  return dt.run(agentId, { ...runOpts, userInput, model, tools }, async (run) => {
    const result = await streamTextFn(instrumentStreamTextOptions(textOpts));
    await result.consumeStream();
    run.finalAnswer();
    return result;
  });
}

// ── Event emission ────────────────────────────────────────────────────────────

function emitStepEvents(
  run: DunetraceRun,
  step: StepResult<ToolSet>,
  model: string,
  latencyMs: number,
): void {
  const usage = normalizeUsage(step.usage);
  run.llmCalled(model, usage.promptTokens);
  run.llmResponded({
    promptTokens:     usage.promptTokens,
    completionTokens: usage.completionTokens,
    latencyMs,
    finishReason:     step.finishReason ?? "stop",
    outputText:       step.text ?? "",
  });

  for (const tc of step.toolCalls) {
    emitToolCalled(run, tc);
  }
  for (const tr of step.toolResults) {
    emitToolResult(run, tr);
  }
  // Tool executions that threw surface as `tool-error` parts in the step content.
  for (const part of step.content) {
    if (part.type === "tool-error") emitToolError(run, part);
  }
}

function emitToolCalled(run: DunetraceRun, tc: TypedToolCall<ToolSet>): void {
  const name = tc.toolName ?? "tool";
  const raw  = tc.input;
  const args = (typeof raw === "object" && raw !== null && !Array.isArray(raw))
    ? raw as Record<string, unknown>
    : { value: raw };
  run.toolCalled(name, args);
}

// AI SDK's StepResult exposes no per-tool timing, so latency is reported as 0.
function emitToolResult(run: DunetraceRun, tr: TypedToolResult<ToolSet>): void {
  run.toolResponded(
    tr.toolName ?? "tool",
    true,
    resultLength(tr.output),
    0,
  );
}

function emitToolError(run: DunetraceRun, te: TypedToolError<ToolSet>): void {
  run.toolResponded(
    te.toolName ?? "tool",
    false,
    0,
    0,
    String(te.error ?? "tool error"),
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

export function modelId(model: LanguageModel | undefined): string {
  if (model == null) return "unknown";
  if (typeof model === "string") return model;
  if (typeof model === "object") {
    if ("modelId" in model && typeof model.modelId === "string") return model.modelId;
    if ("model" in model && typeof (model as { model: unknown }).model === "string") {
      return (model as { model: string }).model;
    }
  }
  return "unknown";
}

export function toolNames(tools: ToolSet | undefined): string[] {
  if (tools == null) return [];
  return Object.keys(tools);
}

/**
 * Best-effort run input fingerprint from generateText/streamText options.
 *
 * Prefers `prompt` (string form). Falls back to the last message in a
 * `messages` array — the common chat pattern — so multi-turn calls don't
 * produce an empty input_hash. The text content of that message is used when
 * available; otherwise the message is JSON-stringified.
 */
function deriveUserInput(opts: { prompt?: unknown; messages?: unknown }): string {
  if (opts.prompt != null) return String(opts.prompt);

  const messages = opts.messages;
  if (Array.isArray(messages) && messages.length > 0) {
    const last = messages[messages.length - 1] as { content?: unknown };
    const content = last?.content;
    if (typeof content === "string") return content;
    if (content != null) {
      try { return JSON.stringify(content); } catch { /* fall through */ }
    }
    try { return JSON.stringify(last); } catch { return ""; }
  }
  return "";
}

function normalizeUsage(usage: LanguageModelUsage): { promptTokens: number; completionTokens: number } {
  return {
    promptTokens:     usage.inputTokens ?? 0,
    completionTokens: usage.outputTokens ?? 0,
  };
}

