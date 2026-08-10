/**
 * Auto-instrumentation for the OpenAI and Anthropic Node SDKs.
 *
 * `autoInstrument()` patches the shared resource prototypes so every client
 * instance emits llm.* events inside a `dt.run()` — including clients constructed
 * by code you don't control. Compare `dt.wrapOpenAI(client)`, which instruments
 * one instance you already hold.
 *
 * WHY PROTOTYPE PATCHING. The Python SDK rebinds module attributes
 * (`openai.resources.chat.Completions.create = wrapper`) and every existing
 * reference sees it. Node has no portable equivalent: under ESM an imported
 * binding is read-only, so `mod.create = wrapper` throws, and `require`-cache
 * interception only works for CommonJS consumers.
 *
 * What *is* portable is that ESM freezes the binding, not the object. Both SDKs
 * define their methods on resource-class prototypes shared by every instance, so
 * `Completions.prototype.create = wrapper` mutates a plain object and behaves
 * identically under ESM, CommonJS, and bundlers. That's the mechanism here, and
 * it's why this works without a loader hook or a require shim.
 *
 * COVERED: non-streaming and streaming calls for both SDKs, plus outbound HTTP
 * via the global `fetch` (the Node counterpart to the Python SDK's httpx and
 * requests patches).
 *
 * Streaming can't be measured at call time — usage and the finish reason only
 * exist once the stream is drained, and draining it here would consume it out
 * from under the caller. So `llm.called` is emitted immediately and the stream is
 * handed back through a pass-through proxy that observes chunks *as the caller
 * pulls them*, emitting `llm.responded` when it ends. A stream that is never
 * consumed therefore reports no `llm.responded` — there is nothing to report.
 *
 * NOT COVERED:
 *   - LangChain.js, which needs a callback-handler integration rather than a
 *     patch (see docs/integrate-typescript-agent.md).
 *   - CrewAI, which has no JavaScript port — it is Python-only.
 *   - The Vercel AI SDK, which already has its own integration
 *     (see integrations/vercel-ai.ts).
 */

import { getCurrentRun, httpInstrumentationSuppressed, httpSuppression } from "./context.js";

/** Marks a function as already instrumented, so re-patching is a no-op. */
const INSTRUMENTED = Symbol.for("dunetrace.instrumented");

/** Targets patched in this process, so repeat calls stay idempotent. */
const _patched = new Set<string>();

type AnyFn = (...args: unknown[]) => unknown;
type AsyncFn = (...args: unknown[]) => Promise<unknown>;

export interface AutoInstrumentOptions {
  /**
   * The `openai` module, the `OpenAI` class, or any client instance. Omit to
   * auto-detect via `require("openai")`, which only resolves under CommonJS.
   */
  openai?: unknown;
  /**
   * The `@anthropic-ai/sdk` module, the `Anthropic` class, or a client instance.
   * Omit to auto-detect.
   */
  anthropic?: unknown;
  /**
   * The `@mistralai/mistralai` module, the `Mistral` class, or a client
   * instance. Omit to auto-detect.
   */
  mistral?: unknown;
  /** Restrict to a subset, e.g. `["openai"]`. Defaults to every known target. */
  targets?: string[];
  /** Throw instead of warning when a requested target can't be patched. */
  strict?: boolean;
}

const KNOWN_TARGETS = ["openai", "anthropic", "mistral", "http"] as const;

function warn(message: string, err?: unknown): void {
  const suffix = err instanceof Error ? `: ${err.message}` : err ? `: ${String(err)}` : "";
  console.warn(`[dunetrace] ${message}${suffix}`);
}

/**
 * Run an emission and swallow anything it throws.
 *
 * Auto-instrumentation sits in the middle of a call the host application depends
 * on. A bug in our event emission must never turn a working LLM call into a
 * failed one, so every emit goes through here — the same guarantee the Python
 * SDK's `_safe_emit` provides.
 */
function safeEmit(emit: () => void): void {
  try {
    emit();
  } catch (err) {
    warn("auto-instrumentation failed to emit an event (call itself is unaffected)", err);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function num(value: unknown): number | undefined {
  return typeof value === "number" ? value : undefined;
}

function str(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

// ── Response readers ──────────────────────────────────────────────────────────
// Tolerant by construction: a shape change in either vendor SDK should cost a
// field in one event, never an exception inside the host's call path.

function emitOpenAIResponse(model: string, resp: unknown, startedAt: number): void {
  const run = getCurrentRun();
  if (!run) return;
  safeEmit(() => {
    const body = isRecord(resp) ? resp : {};
    const usage = isRecord(body["usage"]) ? body["usage"] : {};
    const choices = Array.isArray(body["choices"]) ? body["choices"] : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice["message"]) ? choice["message"] : {};

    run.llmCalled(model, num(usage["prompt_tokens"]) ?? 0);
    run.llmResponded({
      completionTokens: num(usage["completion_tokens"]),
      latencyMs: Date.now() - startedAt,
      finishReason: str(choice["finish_reason"]) ?? "stop",
      outputText: str(message["content"]) ?? "",
    });
  });
}

function emitAnthropicResponse(model: string, resp: unknown, startedAt: number): void {
  const run = getCurrentRun();
  if (!run) return;
  safeEmit(() => {
    const body = isRecord(resp) ? resp : {};
    const usage = isRecord(body["usage"]) ? body["usage"] : {};
    const content = Array.isArray(body["content"]) ? body["content"] : [];
    const first = isRecord(content[0]) ? content[0] : {};

    run.llmCalled(model, num(usage["input_tokens"]) ?? 0);
    run.llmResponded({
      completionTokens: num(usage["output_tokens"]),
      latencyMs: Date.now() - startedAt,
      finishReason: str(body["stop_reason"]) ?? "stop",
      outputText: str(first["text"]) ?? "",
    });
  });
}

/**
 * Mistral's JS SDK deserialises to camelCase (`promptTokens`, `finishReason`),
 * unlike the snake_case its HTTP API puts on the wire and unlike the Python
 * SDK — verified against @mistralai/mistralai. Both spellings are read anyway so
 * a raw-response shape still produces numbers rather than silently zeroing.
 */
function emitMistralResponse(model: string, resp: unknown, startedAt: number): void {
  const run = getCurrentRun();
  if (!run) return;
  safeEmit(() => {
    const body = isRecord(resp) ? resp : {};
    const usage = isRecord(body["usage"]) ? body["usage"] : {};
    const choices = Array.isArray(body["choices"]) ? body["choices"] : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice["message"]) ? choice["message"] : {};

    run.llmCalled(model, num(usage["promptTokens"]) ?? num(usage["prompt_tokens"]) ?? 0);
    run.llmResponded({
      completionTokens: num(usage["completionTokens"]) ?? num(usage["completion_tokens"]),
      latencyMs: Date.now() - startedAt,
      finishReason: str(choice["finishReason"]) ?? str(choice["finish_reason"]) ?? "stop",
      outputText: mistralContentText(message["content"]),
    });
  });
}

/** Mistral content is a string, or a list of chunks for multimodal replies. */
function mistralContentText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.map((c) => (isRecord(c) ? str(c["text"]) ?? "" : "")).join("");
}

// ── Streaming ─────────────────────────────────────────────────────────────────

/** Accumulates what a stream reveals only as it is consumed. */
interface StreamCollector {
  observe(chunk: unknown): void;
  result(): {
    promptTokens: number;
    completionTokens?: number;
    finishReason?: string;
    outputText: string;
  };
}

function openAIStreamCollector(): StreamCollector {
  let text = "";
  let promptTokens = 0;
  let completionTokens: number | undefined;
  let finishReason: string | undefined;

  return {
    observe(chunk) {
      if (!isRecord(chunk)) return;
      // Present only when the caller passed stream_options.include_usage.
      const usage = isRecord(chunk["usage"]) ? chunk["usage"] : undefined;
      if (usage) {
        promptTokens = num(usage["prompt_tokens"]) ?? promptTokens;
        completionTokens = num(usage["completion_tokens"]) ?? completionTokens;
      }
      const choices = Array.isArray(chunk["choices"]) ? chunk["choices"] : [];
      const choice = isRecord(choices[0]) ? choices[0] : undefined;
      if (!choice) return;
      const delta = isRecord(choice["delta"]) ? choice["delta"] : undefined;
      text += str(delta?.["content"]) ?? "";
      finishReason = str(choice["finish_reason"]) ?? finishReason;
    },
    result: () => ({ promptTokens, completionTokens, finishReason, outputText: text }),
  };
}

/**
 * Mistral wraps each streamed chunk in a CompletionEvent, so the payload sits
 * at `event.data`. Usage arrives on the final chunk by default — no opt-in flag
 * the way OpenAI needs `stream_options.include_usage`.
 */
function mistralStreamCollector(): StreamCollector {
  let text = "";
  let promptTokens = 0;
  let completionTokens: number | undefined;
  let finishReason: string | undefined;

  return {
    observe(event) {
      if (!isRecord(event)) return;
      const chunk = isRecord(event["data"]) ? event["data"] : event;
      const usage = isRecord(chunk["usage"]) ? chunk["usage"] : undefined;
      if (usage) {
        promptTokens = num(usage["promptTokens"]) ?? num(usage["prompt_tokens"]) ?? promptTokens;
        completionTokens =
          num(usage["completionTokens"]) ?? num(usage["completion_tokens"]) ?? completionTokens;
      }
      const choices = Array.isArray(chunk["choices"]) ? chunk["choices"] : [];
      const choice = isRecord(choices[0]) ? choices[0] : undefined;
      if (!choice) return;
      const delta = isRecord(choice["delta"]) ? choice["delta"] : undefined;
      if (delta) text += mistralContentText(delta["content"]);
      finishReason = str(choice["finishReason"]) ?? str(choice["finish_reason"]) ?? finishReason;
    },
    result: () => ({ promptTokens, completionTokens, finishReason, outputText: text }),
  };
}

function anthropicStreamCollector(): StreamCollector {
  let text = "";
  let promptTokens = 0;
  let completionTokens: number | undefined;
  let finishReason: string | undefined;

  return {
    observe(chunk) {
      if (!isRecord(chunk)) return;
      switch (str(chunk["type"])) {
        case "message_start": {
          const message = isRecord(chunk["message"]) ? chunk["message"] : {};
          const usage = isRecord(message["usage"]) ? message["usage"] : {};
          promptTokens = num(usage["input_tokens"]) ?? promptTokens;
          break;
        }
        case "content_block_delta": {
          const delta = isRecord(chunk["delta"]) ? chunk["delta"] : {};
          text += str(delta["text"]) ?? "";
          break;
        }
        case "message_delta": {
          const delta = isRecord(chunk["delta"]) ? chunk["delta"] : {};
          const usage = isRecord(chunk["usage"]) ? chunk["usage"] : {};
          finishReason = str(delta["stop_reason"]) ?? finishReason;
          completionTokens = num(usage["output_tokens"]) ?? completionTokens;
          break;
        }
      }
    },
    result: () => ({ promptTokens, completionTokens, finishReason, outputText: text }),
  };
}

/**
 * Return a stand-in for `stream` that reports what passes through it.
 *
 * The caller owns the stream: reading it here to collect usage would consume the
 * iterator out from under them and they'd receive nothing. So this observes
 * chunks *as the caller pulls them* and never pulls on its own.
 *
 * A Proxy rather than an async generator, because both SDKs return a rich object
 * — OpenAI's `Stream` has `.tee()`, `.controller`, `.toReadableStream()` — and
 * handing back a bare generator would silently drop all of it. Only
 * `Symbol.asyncIterator` is intercepted; everything else passes through.
 *
 * `onDone` fires exactly once: on normal end, on error, or on early `break`
 * (which reaches us as `return()`), so a partially-read stream still reports the
 * text it produced.
 */
function observeStream(stream: unknown, collector: StreamCollector, onDone: (r: ReturnType<StreamCollector["result"]>) => void): unknown {
  if (!isRecord(stream) || typeof (stream as Record<symbol, unknown>)[Symbol.asyncIterator] !== "function") {
    // Not an async iterable — nothing to observe; hand it back untouched.
    return stream;
  }

  let settled = false;
  const finish = (): void => {
    if (settled) return;
    settled = true;
    onDone(collector.result());
  };

  return new Proxy(stream as object, {
    get(target, prop): unknown {
      if (prop === Symbol.asyncIterator) {
        return function (): AsyncIterator<unknown> {
          const inner = (
            (target as Record<symbol, () => AsyncIterator<unknown>>)[Symbol.asyncIterator]
          ).call(target);
          return {
            async next(...args: [] | [unknown]): Promise<IteratorResult<unknown>> {
              try {
                const step = await inner.next(...(args as []));
                if (step.done) finish();
                else collector.observe(step.value);
                return step;
              } catch (err) {
                finish();
                throw err;
              }
            },
            async return(value?: unknown): Promise<IteratorResult<unknown>> {
              finish();
              return inner.return
                ? inner.return(value)
                : { done: true, value: value as never };
            },
            async throw(err?: unknown): Promise<IteratorResult<unknown>> {
              finish();
              if (inner.throw) return inner.throw(err);
              throw err;
            },
            [Symbol.asyncIterator]() {
              return this;
            },
          } as AsyncIterator<unknown>;
        };
      }
      // Read off the target, not the proxy: routing through `receiver` breaks
      // class private fields (`#x`), which both SDKs use internally.
      const value = Reflect.get(target, prop) as unknown;
      return typeof value === "function" ? (value as AnyFn).bind(target) : value;
    },
  });
}

// ── Method wrappers ───────────────────────────────────────────────────────────

type Emitter = (model: string, resp: unknown, startedAt: number) => void;
type CollectorFactory = () => StreamCollector;

/**
 * Wrap a `create`-shaped method so successful non-streaming calls emit events.
 *
 * Errors propagate untouched and emit nothing: a call that threw produced no
 * usage, no finish reason, and no output, so there is no llm.responded to
 * describe. The failure is already visible to the caller and to whatever
 * error handling the host has.
 */
function instrumentCreate(
  orig: AsyncFn,
  emit: Emitter,
  collectorFor: CollectorFactory,
  alwaysStream = false,
): AsyncFn {
  if ((orig as unknown as Record<symbol, unknown>)[INSTRUMENTED]) return orig;

  const wrapped = async function (this: unknown, ...args: unknown[]): Promise<unknown> {
    const opts = isRecord(args[0]) ? args[0] : undefined;
    const model = str(opts?.["model"]) ?? "unknown";
    const startedAt = Date.now();

    // The SDK issues its request through fetch; suppress HTTP instrumentation for
    // the duration so one LLM call doesn't also register as a tool call.
    const resp = await httpSuppression.run(true, () => orig.apply(this, args));

    // openai/anthropic express streaming as an option on one method; Mistral
    // has a separate `stream` method that always streams.
    if (!alwaysStream && !opts?.["stream"]) {
      emit(model, resp, startedAt);
      return resp;
    }

    // Streaming: usage and finish reason only exist once the caller drains the
    // stream, so emit llm.called now and llm.responded when it finishes.
    const run = getCurrentRun();
    if (!run) return resp;
    safeEmit(() => { run.llmCalled(model, 0); });

    return observeStream(resp, collectorFor(), (result) => {
      safeEmit(() => {
        run.llmResponded({
          completionTokens: result.completionTokens,
          latencyMs: Date.now() - startedAt,
          finishReason: result.finishReason ?? "stop",
          outputText: result.outputText,
        });
      });
    });
  };

  Object.defineProperty(wrapped, INSTRUMENTED, { value: true, enumerable: false });
  // Keep the original name/arity so anything reflecting over the SDK still works.
  Object.defineProperty(wrapped, "name", { value: (orig as AnyFn).name, configurable: true });
  return wrapped as AsyncFn;
}

// ── Instance wrapping ─────────────────────────────────────────────────────────

/**
 * Instrument the `create` method on one resource object, in place.
 *
 * Deliberately does not `.bind()` the original: a bound function is a *new*
 * function object and doesn't carry the INSTRUMENTED marker, so binding first
 * would defeat the idempotency check and double-count every call for a client
 * that `autoInstrument()` had already covered via its prototype. Binding is also
 * unnecessary — the wrapper forwards `this`, which is the resource object when
 * called as `client.chat.completions.create(...)`.
 */
function instrumentResource(
  resource: Record<string, unknown>,
  emit: Emitter,
  collectorFor: CollectorFactory,
): void {
  const existing = resource["create"] as AsyncFn;
  if ((existing as unknown as Record<symbol, unknown>)[INSTRUMENTED]) return;
  resource["create"] = instrumentCreate(existing, emit, collectorFor);
}

/** Instrument one OpenAI client instance in place. Returns the same object. */
export function wrapOpenAIClient<T>(client: T): T {
  const target = isRecord(client) ? client : undefined;
  const chat = isRecord(target?.["chat"]) ? target["chat"] : undefined;
  const completions = isRecord(chat?.["completions"]) ? chat["completions"] : undefined;
  if (!completions || typeof completions["create"] !== "function") {
    warn("wrapOpenAI: client has no chat.completions.create — leaving it untouched");
    return client;
  }
  instrumentResource(completions, emitOpenAIResponse, openAIStreamCollector);
  return client;
}

/** Instrument one Anthropic client instance in place. Returns the same object. */
export function wrapAnthropicClient<T>(client: T): T {
  const target = isRecord(client) ? client : undefined;
  const messages = isRecord(target?.["messages"]) ? target["messages"] : undefined;
  if (!messages || typeof messages["create"] !== "function") {
    warn("wrapAnthropic: client has no messages.create — leaving it untouched");
    return client;
  }
  instrumentResource(messages, emitAnthropicResponse, anthropicStreamCollector);
  return client;
}

// ── Prototype resolution ──────────────────────────────────────────────────────

/**
 * Find the prototype carrying `create` for a vendor SDK, from whatever the caller
 * handed us: the module namespace, the client class, or a live client instance.
 *
 * Patching the *prototype* rather than an instance is what makes this "auto" —
 * every client sharing it is covered, including ones built inside libraries you
 * don't control, and ones constructed after this call.
 */
function resolvePrototype(
  candidate: unknown,
  spec: TargetSpec,
): Record<string, unknown> | null {
  const probe = spec.methods[0].name;
  const carries = (obj: unknown): boolean =>
    isRecord(obj) && typeof (obj as Record<string, unknown>)[probe] === "function";

  // Unwrap a module namespace to its default/named export.
  let cls = candidate;
  if (isRecord(cls) && typeof cls !== "function") {
    cls =
      (cls as Record<string, unknown>)[spec.exportName] ??
      (cls as Record<string, unknown>)["default"] ??
      cls;
  }

  // Path 1 — a class exposing its resource classes as statics.
  if (spec.staticPath.length > 0) {
    let node: unknown = cls;
    for (const key of spec.staticPath) {
      node = isRecord(node) || typeof node === "function"
        ? (node as Record<string, unknown>)[key]
        : undefined;
      if (node === undefined) break;
    }
    if (typeof node === "function" && isRecord((node as AnyFn).prototype)) {
      return (node as unknown as { prototype: Record<string, unknown> }).prototype;
    }
  }

  // Path 2 — a live instance: walk to the resource object and take its
  // prototype, which is the same object every other instance shares.
  const fromInstance = (root: unknown): Record<string, unknown> | null => {
    let inst: unknown = root;
    for (const key of spec.instancePath) {
      inst = isRecord(inst) ? inst[key] : undefined;
      if (inst === undefined) break;
    }
    if (!isRecord(inst)) return null;
    const proto = Object.getPrototypeOf(inst) as Record<string, unknown> | null;
    if (carries(proto)) return proto;
    // Instance owns the method directly rather than inheriting it.
    if (carries(inst)) return inst;
    return null;
  };

  const direct = fromInstance(candidate);
  if (direct) return direct;

  // Path 3 — construct one. Needed when the SDK never exports the resource
  // class and only exposes it as a getter on a client instance, which is how
  // @mistralai/mistralai is laid out. The throwaway client does no I/O; its
  // resource object's prototype is shared with every real client.
  if (spec.instantiate && typeof cls === "function") {
    try {
      const probeClient = new (cls as new (opts: unknown) => unknown)({
        apiKey: "dunetrace-probe",
      });
      return fromInstance(probeClient);
    } catch {
      return null;
    }
  }

  return null;
}

function tryRequire(moduleName: string): unknown {
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    return (eval("require") as (id: string) => unknown)(moduleName);
  } catch {
    return null;
  }
}

interface TargetMethod {
  /** Method on the resolved prototype. */
  name: string;
  /** true when the method always returns a stream, rather than taking a flag. */
  alwaysStream?: boolean;
}

interface TargetSpec {
  name: string;
  moduleName: string;
  /** Named export to unwrap when handed a module namespace. */
  exportName: string;
  staticPath: readonly string[];
  instancePath: readonly string[];
  /**
   * Construct the client to reach its resource object when the SDK does not
   * export the resource class. Mistral needs this: `chat` is a lazily-cached
   * getter on Mistral.prototype and the Chat class is not exported, so there is
   * no static path to walk. Constructing does no I/O.
   */
  instantiate?: boolean;
  methods: readonly TargetMethod[];
  emit: Emitter;
  collector: CollectorFactory;
  /** Named in the "could not patch" message. */
  fallbackHint: string;
}

const SPECS: Record<string, TargetSpec> = {
  openai: {
    name: "openai",
    moduleName: "openai",
    exportName: "OpenAI",
    staticPath: ["Chat", "Completions"],
    instancePath: ["chat", "completions"],
    methods: [{ name: "create" }],
    emit: emitOpenAIResponse,
    collector: openAIStreamCollector,
    fallbackHint: "dt.wrapOpenAI(client)",
  },
  anthropic: {
    name: "anthropic",
    moduleName: "@anthropic-ai/sdk",
    exportName: "Anthropic",
    staticPath: ["Messages"],
    instancePath: ["messages"],
    methods: [{ name: "create" }],
    emit: emitAnthropicResponse,
    collector: anthropicStreamCollector,
    fallbackHint: "dt.wrapAnthropic(client)",
  },
  mistral: {
    name: "mistral",
    moduleName: "@mistralai/mistralai",
    exportName: "Mistral",
    // The Chat class is not exported, so there is no static path — resolution
    // goes through a throwaway instance. Verified against @mistralai/mistralai.
    staticPath: [],
    instancePath: ["chat"],
    instantiate: true,
    // Two methods rather than one flagged method: `parse`/`parseStream` are
    // deliberately left alone because they call these internally and would
    // double-count.
    methods: [{ name: "complete" }, { name: "stream", alwaysStream: true }],
    emit: emitMistralResponse,
    collector: mistralStreamCollector,
    fallbackHint: "manual run.llmCalled()/run.llmResponded() calls",
  },
};

/**
 * Patch supported LLM SDKs so their calls emit events inside a `dt.run()`.
 *
 * Pass the imported module or class — that works everywhere, and is required
 * under ESM and bundlers where the auto-detect `require` can't resolve:
 *
 *     import OpenAI from "openai";
 *     autoInstrument({ openai: OpenAI });
 *
 * With no arguments it tries to `require` each known SDK and patches whatever it
 * finds, skipping the rest silently — the closest equivalent to the Python SDK's
 * zero-argument `auto_instrument()`, and subject to the CommonJS caveat above.
 *
 * Idempotent: patching an already-patched target is a no-op, so calling this
 * twice (or alongside `dt.wrapOpenAI`) won't double-count events.
 *
 * @returns the target names actually patched.
 */
export function autoInstrument(options: AutoInstrumentOptions = {}): string[] {
  const requested = options.targets ?? [...KNOWN_TARGETS];
  const patched: string[] = [];

  for (const name of requested) {
    if (name === "http") {
      if (instrumentHttp()) patched.push("http");
      continue;
    }
    const spec = SPECS[name];
    if (!spec) {
      warn(`autoInstrument: unknown target "${name}" — known targets: ${KNOWN_TARGETS.join(", ")}`);
      continue;
    }
    if (_patched.has(name)) {
      patched.push(name);
      continue;
    }

    const supplied = (options as Record<string, unknown>)[name];
    const candidate = supplied ?? tryRequire(spec.moduleName);
    if (!candidate) {
      if (options.strict) {
        throw new Error(
          `autoInstrument: could not load "${spec.moduleName}". Pass it explicitly, ` +
            `e.g. autoInstrument({ ${name}: ${spec.exportName} }).`,
        );
      }
      continue;
    }

    const proto = resolvePrototype(candidate, spec);
    const missing = spec.methods.filter((m) => !proto || typeof proto[m.name] !== "function");
    if (!proto || missing.length === spec.methods.length) {
      const message =
        `autoInstrument: found "${spec.moduleName}" but could not locate its ` +
        `${[...spec.instancePath, spec.methods[0].name].join(".")} prototype — ` +
        `the SDK layout may have changed. Fall back to ${spec.fallbackHint}.`;
      if (options.strict) throw new Error(message);
      warn(message);
      continue;
    }

    for (const method of spec.methods) {
      const orig = proto[method.name];
      if (typeof orig !== "function") continue;
      if ((orig as unknown as Record<symbol, unknown>)[INSTRUMENTED]) continue;
      // No .bind() here: the method must keep receiving the calling instance as
      // `this`, since one prototype serves every client.
      proto[method.name] = instrumentCreate(
        orig as AsyncFn,
        spec.emit,
        spec.collector,
        method.alwaysStream ?? false,
      );
    }
    _patched.add(name);
    patched.push(name);
  }

  return patched;
}

// ── HTTP (global fetch) ───────────────────────────────────────────────────────

/**
 * Base URLs belonging to Dunetrace itself.
 *
 * The client ships events over `fetch`. Instrumenting our own ingest POST would
 * emit a tool.called describing it, which buffers another event, which ships…
 * Registering the endpoint keeps the wrapper off its own tail.
 */
const _ownEndpoints = new Set<string>();

/** Record an endpoint as Dunetrace's own, so HTTP instrumentation skips it. */
export function registerOwnEndpoint(url: string | null | undefined): void {
  if (!url) return;
  try {
    _ownEndpoints.add(new URL(url).origin);
  } catch {
    /* not a parseable URL — nothing to exclude */
  }
}

function requestUrl(input: unknown): string | null {
  if (typeof input === "string") return input;
  if (input instanceof URL) return input.toString();
  if (isRecord(input) && typeof input["url"] === "string") return input["url"];
  return null;
}

/** Hostname as the tool name — stable, and keeps full URLs out of the event. */
function httpToolName(url: string): string {
  try {
    return new URL(url).hostname || "http";
  } catch {
    return "http";
  }
}

function instrumentFetch(originalFetch: typeof fetch): typeof fetch {
  const wrapped = async function (
    this: unknown,
    input: Parameters<typeof fetch>[0],
    init?: Parameters<typeof fetch>[1],
  ): Promise<Response> {
    const run = getCurrentRun();
    const url = requestUrl(input);

    // Skip when: outside a run, inside an LLM call already reporting itself, the
    // URL is unreadable, or the request is Dunetrace's own telemetry.
    const skip =
      !run ||
      httpInstrumentationSuppressed() ||
      !url ||
      isOwnEndpoint(url);
    if (skip) return originalFetch.call(globalThis, input, init);

    const tool = httpToolName(url);
    const startedAt = Date.now();
    // Not inside safeEmit's swallow: a policy on this tool call must still be
    // able to stop the request before it goes out.
    safeEmit(() => { run.toolCalled(tool, { url }); });

    try {
      const resp = await originalFetch.call(globalThis, input, init);
      safeEmit(() => {
        const status = resp.status;
        const ok = status >= 200 && status < 400;
        const length = Number(resp.headers.get("content-length") ?? 0) || 0;
        run.toolResponded(
          tool,
          ok,
          length,
          Date.now() - startedAt,
          ok ? undefined : String(status),
        );
      });
      return resp;
    } catch (err) {
      safeEmit(() => {
        run.toolResponded(
          tool,
          false,
          0,
          Date.now() - startedAt,
          err instanceof Error ? err.message : String(err),
        );
      });
      throw err;
    }
  } as unknown as typeof fetch;

  Object.defineProperty(wrapped, INSTRUMENTED, { value: true, enumerable: false });
  return wrapped;
}

function isOwnEndpoint(url: string): boolean {
  try {
    return _ownEndpoints.has(new URL(url).origin);
  } catch {
    return false;
  }
}

/**
 * Instrument the global `fetch`, emitting tool.called / tool.responded per request.
 *
 * The Node counterpart to the Python SDK's httpx and requests patches: `fetch` is
 * what the vast majority of Node HTTP goes through, including both vendor LLM
 * SDKs. Requests made by those SDKs are deliberately excluded — see
 * `httpSuppression` — so one LLM call doesn't also count as a tool call and trip
 * tool-loop detection.
 *
 * Only requests made inside a `dt.run()` are recorded; everything else passes
 * through with the original `fetch` semantics.
 */
export function instrumentHttp(): boolean {
  if (_patched.has("http")) return true;
  const current = globalThis.fetch;
  if (typeof current !== "function") {
    warn("instrumentHttp: no global fetch on this runtime — skipping");
    return false;
  }
  if ((current as unknown as Record<symbol, unknown>)[INSTRUMENTED]) {
    _patched.add("http");
    return true;
  }
  globalThis.fetch = instrumentFetch(current);
  _patched.add("http");
  return true;
}

/** Test seam: forget what has been patched. Does not un-patch anything. */
export function _resetAutoInstrumentState(): void {
  _patched.clear();
  _ownEndpoints.clear();
}
