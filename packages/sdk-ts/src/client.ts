import { randomUUID } from "node:crypto";
import { agentVersion } from "./hash.js";
import { DunetraceRun } from "./run.js";
import { EventBuffer } from "./buffer.js";
import {
  resultLength as _resultLength,
  resultText as _resultText,
  safeEmit as _safeEmit,
} from "./util.js";
import {
  capText,
  compileDenylist,
  resolveMaxFieldChars,
  type RedactionSettings,
} from "./redaction.js";
import { HttpBatchEmitter, type BatchEmitter } from "./emitters.js";
import { runStorage as _runStorage, getCurrentRun } from "./context.js";
import { registerOwnEndpoint, wrapAnthropicClient, wrapOpenAIClient } from "./auto.js";
import type { AgentEvent, ClientOptions, EventSink, RunOptions } from "./models.js";

export { getCurrentRun };

// ── Exit flush ────────────────────────────────────────────────────────────────
//
// A TypeScript agent that exits without calling `dt.shutdown()` used to ship
// NOTHING. The only drain path is the 200ms `setInterval` below, deliberately
// `unref()`'d so instrumentation never holds a process open — and a script that
// awaits one `dt.run()` and returns finishes well inside one tick, so that timer
// never fires and the whole run dies in the buffer. That is the normal shape of
// a CLI agent, a cron job and a Lambda handler. The Python client has never had
// this problem: it registers a weakref'd `atexit` flush (sdk-py client.py), so
// no explicit call is needed there either.
//
// `beforeExit` is Node's equivalent and the only hook that can do this: it fires
// when the loop has drained and the process is about to exit *naturally*, and
// async work started from it keeps the loop alive until that work settles — then
// fires again. So the flush completes, and the process still exits exactly when
// it would have. `exit` cannot be used (synchronous only, so the HTTP POST could
// never complete) and signal handlers are deliberately NOT installed: adding one
// suppresses Node's default SIGINT/SIGTERM termination, which would keep the
// process alive longer than it otherwise would run. `process.exit()` and a fatal
// signal therefore still lose the buffer, exactly as `atexit` does in Python.
//
// ONE listener for the whole module, iterating a registry of live clients, not
// one listener per client: a test file (or any app) that constructs a few dozen
// clients would otherwise trip Node's MaxListenersExceededWarning. Clients are
// held by WeakRef and a FinalizationRegistry drops the dead entries, so the
// registry never keeps a client alive the way a captured closure would — the
// same reason the Python hook holds a weakref.

const _liveClients = new Set<WeakRef<Dunetrace>>();

const _clientFinalizer =
  typeof FinalizationRegistry === "function"
    ? new FinalizationRegistry<WeakRef<Dunetrace>>((ref) => { _liveClients.delete(ref); })
    : null;

let _exitHookInstalled = false;

function _installExitFlushHook(): void {
  if (_exitHookInstalled) return;
  if (typeof process === "undefined" || typeof process.on !== "function") return;
  _exitHookInstalled = true;
  process.on("beforeExit", () => {
    for (const ref of _liveClients) {
      ref.deref()?._flushBeforeExit();
    }
  });
}

export class Dunetrace {
  private _ingestUrl:  string | null;
  private _apiKey:     string;
  /** Run-aware outbound queue — sheds the oldest whole run under overload and
   *  never sheds a run terminal. See buffer.ts. */
  private _buffer:     EventBuffer;
  private _timeoutMs:  number;
  private _redaction:  RedactionSettings;
  private _drainTimer: ReturnType<typeof setInterval> | null = null;
  private _emitJson:   boolean;
  private _emitter:    BatchEmitter;
  private _exporter:   EventSink | null;
  /** This client's entry in the module-level exit-flush registry — see the
   *  "Exit flush" note above. null when the hook is off (`flushOnExit: false`)
   *  or already withdrawn by shutdown(). */
  private _exitRef:    WeakRef<Dunetrace> | null = null;
  /** Guards the exit flush against re-entry: `beforeExit` fires again once the
   *  flush's own async work settles, and without this it would start a second
   *  flush on top of the first. */
  private _exitFlushing = false;

  constructor(opts: ClientOptions = {}) {
    const base      = (opts.endpoint ?? "http://localhost:8001").replace(/\/$/, "");
    this._ingestUrl = base + "/v1/ingest";
    // Tell HTTP instrumentation to ignore our own traffic. Without this, shipping
    // a batch would emit a tool.called describing the ship, which buffers another
    // event, which ships — a feedback loop against our own ingest.
    registerOwnEndpoint(base);
    this._apiKey    = opts.apiKey ?? "";
    this._emitJson  = opts.emitAsJson ?? false;
    this._buffer     = new EventBuffer(opts.bufferSize ?? 10_000);
    this._timeoutMs  = opts.timeoutMs  ?? 5_000;
    // Content caps and redaction (redaction.ts), read by every run's emit hooks.
    if (opts.redact !== undefined && typeof opts.redact !== "function") {
      throw new TypeError("redact must be a function (args => args)");
    }
    this._redaction  = {
      maxFieldChars: resolveMaxFieldChars(opts.maxFieldChars),
      redact:        opts.redact ?? null,
      denylist:      compileDenylist(opts.redactKeys),
    };
    this._emitter    = opts.emitter ?? new HttpBatchEmitter(base, this._apiKey, this._timeoutMs);
    this._exporter   = opts.exporter ?? null;

    const interval = opts.flushIntervalMs ?? 200;
    this._drainTimer = setInterval(() => { this._drain(); }, interval);
    // Don't prevent the process from exiting naturally
    if (typeof this._drainTimer.unref === "function") {
      this._drainTimer.unref();
    }

    // Flush whatever is still buffered when the process exits on its own. The
    // drain timer above is unref'd, so without this a process that finishes
    // before the first tick ships nothing at all. See the "Exit flush" note
    // above for why this is `beforeExit` and nothing else.
    if (opts.flushOnExit !== false) {
      this._exitRef = new WeakRef(this);
      _liveClients.add(this._exitRef);
      _clientFinalizer?.register(this, this._exitRef, this._exitRef);
      _installExitFlushHook();
    }
  }

  // ── Run context ────────────────────────────────────────────────────────────

  async run<T>(
    agentId: string,
    opts:    RunOptions,
    fn:      (run: DunetraceRun) => Promise<T>,
  ): Promise<T> {
    const model   = opts.model        ?? "unknown";
    const tools   = opts.tools        ?? [];
    const version = agentVersion(opts.systemPrompt ?? "", model, tools);
    const run     = new DunetraceRun(agentId, version, this, opts.runId);

    // Auto-thread parent_run_id: if this run opens while another run is already
    // active (this call is nested inside an enclosing run's fn, so
    // _runStorage.getStore() returns that parent) and the caller didn't pass
    // parentRunId, inherit the active run's id. This links nested multi-agent
    // runs into a parent/child graph with no manual id threading — the substrate
    // the server-side DELEGATION_LOOP and HANDOFF_CONTEXT_LOSS detectors consume.
    // An explicit parentRunId always wins. Propagation follows AsyncLocalStorage,
    // so it survives awaits within the same async context.
    const parentRunId = opts.parentRunId ?? _runStorage.getStore()?.runId ?? null;

    // Content caps, applied AFTER the version hash above so grouping stays
    // stable however long the prompt is. Markers only when a cut happened, so
    // an ordinary run.started is byte-identical to before.
    const maxChars = this._redaction.maxFieldChars;
    const input    = capText(opts.userInput ?? "", maxChars);
    const prompt   = capText(opts.systemPrompt ?? "", maxChars);
    const startedPayload: Record<string, unknown> = {
      input_text:    input.text,
      system_prompt: prompt.text,
      model,
      tools,
    };
    if (input.truncated) {
      startedPayload["input_text_truncated"]       = true;
      startedPayload["input_text_original_length"] = input.originalLength;
    }
    if (prompt.truncated) {
      startedPayload["system_prompt_truncated"]       = true;
      startedPayload["system_prompt_original_length"] = prompt.originalLength;
    }

    _safeEmit(() => { this._emit({
      event_type:    "run.started",
      run_id:        run.runId,
      agent_id:      agentId,
      agent_version: version,
      step_index:    0,
      timestamp:     Date.now() / 1000,
      payload: startedPayload,
      parent_run_id: parentRunId,
      trace_id:      opts.traceId ?? null,
      conversation_id: opts.conversationId ?? null,
    }); }, "dt.run");

    let result: T;
    try {
      result = await _runStorage.run(run, () => fn(run));
    } catch (err) {
      _safeEmit(() => { this._emit({
        event_type:    "run.errored",
        run_id:        run.runId,
        agent_id:      agentId,
        agent_version: version,
        step_index:    run.currentStep(),
        timestamp:     Date.now() / 1000,
        payload: this._terminalPayload(run.runId, {
          error_type: (err instanceof Error) ? err.name : "Error",
          error:      String(err),
          step_index: run.currentStep(),
        }),
      }); }, "dt.run");
      throw err;
    }

    _safeEmit(() => { this._emit({
      event_type:    "run.completed",
      run_id:        run.runId,
      agent_id:      agentId,
      agent_version: version,
      step_index:    run.currentStep(),
      timestamp:     Date.now() / 1000,
      payload: this._terminalPayload(run.runId, {
        total_steps:     run.currentStep(),
        exit_reason:     run.exitReason() ?? "completed",
        tool_call_count: run.getEvents().filter(e => e.event_type === "tool.called").length,
      }),
    }); }, "dt.run");

    return result;
  }

  // ── Tool wrapper ───────────────────────────────────────────────────────────

  /**
   * Wrap a function to auto-emit tool.called / tool.responded around each call.
   * No-op when called outside a dt.run() context — the function still runs.
   *
   * @example
   * const search = dt.tool(webSearch);
   * const search = dt.tool(webSearch, "web_search");
   * const search = dt.tool(async (q: string) => fetchResults(q), "search");
   */
  tool<T extends (...args: unknown[]) => unknown>(fn: T, name?: string): T {
    const toolName = name ?? fn.name ?? "tool";
    const isAsync  = fn.constructor.name === "AsyncFunction";

    // Every emission goes through _safeEmit. Instrumentation wraps a function the
    // host application depends on: a tool argument JSON cannot serialise (an ORM
    // entity, an Express req/res, a DB client, a BigInt), or any other defect in
    // our own event path, must never turn a working call into a failed one. The
    // tool.called emit in particular sat OUTSIDE the try below, so anything it
    // threw propagated to the caller before the tool body had even run.
    if (isAsync) {
      const wrapper = async (...args: Parameters<T>): Promise<unknown> => {
        const run = _runStorage.getStore();
        if (run) _safeEmit(() => { run.toolCalled(toolName, _argsRecord(fn, args)); }, "dt.tool");
        const t0 = Date.now();
        try {
          const result = await (fn as (...a: unknown[]) => Promise<unknown>)(...args);
          if (run) _safeEmit(() => { run.toolResponded(toolName, true, _resultLength(result), Date.now() - t0, undefined, _resultText(result)); }, "dt.tool");
          return result;
        } catch (err) {
          if (run) _safeEmit(() => { run.toolResponded(toolName, false, 0, Date.now() - t0, String(err)); }, "dt.tool");
          throw err;
        }
      };
      return wrapper as unknown as T;
    } else {
      const wrapper = (...args: Parameters<T>): unknown => {
        const run = _runStorage.getStore();
        if (run) _safeEmit(() => { run.toolCalled(toolName, _argsRecord(fn, args)); }, "dt.tool");
        const t0 = Date.now();
        try {
          const result = (fn as (...a: unknown[]) => unknown)(...args);
          if (run) _safeEmit(() => { run.toolResponded(toolName, true, _resultLength(result), Date.now() - t0, undefined, _resultText(result)); }, "dt.tool");
          return result;
        } catch (err) {
          if (run) _safeEmit(() => { run.toolResponded(toolName, false, 0, Date.now() - t0, String(err)); }, "dt.tool");
          throw err;
        }
      };
      return wrapper as unknown as T;
    }
  }

  // ── Agent wrapper ──────────────────────────────────────────────────────────

  /**
   * Wrap an async function to automatically open and close a run context.
   * The first parameter is used as userInput.
   *
   * @example
   * const agent = dt.trace(myAgent, "my-agent", { model: "gpt-4o" });
   * const agent = dt.trace(myAgent); // agentId defaults to function name
   */
  trace<T extends (...args: unknown[]) => Promise<unknown>>(
    fn:       T,
    agentId?: string,
    opts:     Omit<RunOptions, "userInput"> = {},
  ): T {
    const _agentId = agentId ?? fn.name ?? "agent";
    const wrapper  = async (...args: Parameters<T>): Promise<unknown> => {
      const userInput = args[0] != null ? String(args[0]) : "";
      return this.run(_agentId, { ...opts, userInput }, async (run) => {
        const result = await fn(...args);
        run.finalAnswer();
        return result;
      });
    };
    return wrapper as unknown as T;
  }

  // ── Auto-instrumentation ───────────────────────────────────────────────────

  /**
   * Patch an OpenAI client so every chat.completions.create() call inside a
   * dt.run() context is tracked automatically — no manual llmCalled /
   * llmResponded calls needed. Mutates and returns the same client instance.
   * Streaming calls (stream: true) are not patched and must be tracked manually.
   *
   * @example
   * const openai = dt.wrapOpenAI(new OpenAI());
   */
  wrapOpenAI<T extends { chat: { completions: { create: (...args: unknown[]) => Promise<unknown> } } }>(client: T): T {
    return wrapOpenAIClient(client);
  }

  /**
   * Patch an Anthropic client so every messages.create() call inside a
   * dt.run() context is tracked automatically. Streaming calls are skipped.
   *
   * @example
   * const anthropic = dt.wrapAnthropic(new Anthropic());
   */
  wrapAnthropic<T extends { messages: { create: (...args: unknown[]) => Promise<unknown> } }>(client: T): T {
    return wrapAnthropicClient(client);
  }

  // ── Deploy markers ─────────────────────────────────────────────────────────

  /** Fire-and-forget deploy marker. Call from CI/CD or app startup. */
  markDeploy(agentId: string, version: string, meta: Record<string, unknown> = {}): void {
    if (!this._ingestUrl) return;
    const base = this._ingestUrl.replace("/v1/ingest", "");
    const body = JSON.stringify({ api_key: this._apiKey, agent_id: agentId, version, meta });
    fetch(`${base}/v1/deploy`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body,
    }).catch(err => {
      process.stderr.write(`[dunetrace] markDeploy failed: ${err}\n`);
    });
  }

  // ── Flush / shutdown ───────────────────────────────────────────────────────

  async flush(): Promise<void> {
    const batch = this._buffer.drainAll();
    if (batch.length > 0) await this._emitter.ship(batch);
  }

  async shutdown(timeoutMs = 5000): Promise<void> {
    if (this._drainTimer !== null) {
      clearInterval(this._drainTimer);
      this._drainTimer = null;
    }
    // Withdraw from the exit-flush registry: an explicit shutdown is the clean
    // path and must not be followed by a second flush at `beforeExit`.
    if (this._exitRef !== null) {
      _liveClients.delete(this._exitRef);
      _clientFinalizer?.unregister(this._exitRef);
      this._exitRef = null;
    }
    const deadline = Date.now() + timeoutMs;
    while (this._buffer.length > 0 && Date.now() < deadline) {
      await this.flush();
    }
  }

  // ── Internal ───────────────────────────────────────────────────────────────

  _emit(event: AgentEvent): void {
    // Non-throwing end to end, like the Python client's _emit. Everything below
    // runs inside the customer's agent process; a defect here costs an event,
    // never the call that produced it.
    try {
      // audit Finding 14: stamp a stable id once, at buffer entry, so a retry of
      // the same buffered event ships the same id and the ingest side dedups it.
      if (!event.event_id) event.event_id = randomUUID();
      if (this._emitJson) this._writeJsonLine(event);
      // Fan out to the optional OTel exporter. It never throws (see handle()), but
      // guard anyway so a sink defect can't break the agent's own event path.
      if (this._exporter) {
        try {
          this._exporter.handle(event);
        } catch (err) {
          process.stderr.write(`[dunetrace] exporter failed: ${err}\n`);
        }
      }
      // Run-aware shedding: a full buffer drops the OLDEST WHOLE RUN, never the
      // event that happened to arrive next — which used to include run terminals.
      this._buffer.push(event);
    } catch (err) {
      process.stderr.write(`[dunetrace] failed to emit ${event?.event_type}: ${err}\n`);
    }
  }

  /** Cap + redaction settings, read once per run by DunetraceRun. */
  _redactionSettings(): RedactionSettings {
    return this._redaction;
  }

  /**
   * Stamp `dropped_events` onto a run.completed / run.errored payload.
   *
   * Reserving the terminal's slot *first* means any shedding needed to fit it is
   * already reflected in the count. The key is omitted when nothing was dropped,
   * so a healthy run's wire format is unchanged. The detector reads it into
   * RunState.dropped_events and holds the run's signals in shadow. Never throws
   * — a failure here costs the marker, not the event.
   */
  private _terminalPayload(runId: string, payload: Record<string, unknown>): Record<string, unknown> {
    let dropped = 0;
    try {
      dropped = this._buffer.reserveTerminal(runId);
    } catch {
      dropped = 0;
    }
    if (dropped > 0) payload["dropped_events"] = dropped;
    return payload;
  }

  /**
   * Ship whatever is buffered, from the module's `beforeExit` listener.
   *
   * Fire-and-forget on purpose: `beforeExit` cannot be awaited, but the promise
   * started here is itself pending work, so Node keeps the loop alive until it
   * settles and then fires `beforeExit` again. The second pass finds an empty
   * buffer, schedules nothing, and the process exits — no busy loop, and no
   * delay beyond the one in-flight request. A failed ship is swallowed: the
   * events are already out of the buffer, and there is no one left to tell.
   *
   * @internal — public only because the module-level listener calls it.
   */
  _flushBeforeExit(): void {
    if (this._exitFlushing || this._buffer.length === 0) return;
    this._exitFlushing = true;
    void this.flush()
      .catch(() => {})
      .finally(() => { this._exitFlushing = false; });
  }

  private _drain(): void {
    if (!this._ingestUrl) return;
    const batch = this._buffer.drain(100);
    if (batch.length === 0) return;
    this._emitter.ship(batch).catch(() => {});
  }

  private _writeJsonLine(event: AgentEvent): void {
    const ts   = new Date(event.timestamp * 1000).toISOString();
    const line = JSON.stringify({
      ts,
      level:         "info",
      logger:        "dunetrace",
      event_type:    event.event_type,
      agent_id:      event.agent_id,
      run_id:        event.run_id,
      agent_version: event.agent_version,
      step_index:    event.step_index,
      payload:       event.payload,
    });
    process.stdout.write(line + "\n");
  }
}

function _argsRecord(fn: (...args: unknown[]) => unknown, args: unknown[]): Record<string, unknown> {
  // Best-effort: use parameter names from function source if available
  try {
    const match = fn.toString().match(/^[^(]*\(([^)]*)\)/);
    if (match) {
      const names = match[1].split(",").map(s => s.trim().replace(/=.*$/, "").replace(/^\.\.\./, ""));
      const result: Record<string, unknown> = {};
      names.forEach((name, i) => { if (name) result[name] = args[i]; });
      return result;
    }
  } catch {
    // ignore
  }
  return Object.fromEntries(args.map((v, i) => [`arg${i}`, v]));
}
