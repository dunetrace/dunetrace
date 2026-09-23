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
import { runStorage as _runStorage, getCurrentRun, setRunOpener, type EntryRunHandle } from "./context.js";
import { parseDsn, dsnHost } from "./dsn.js";
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

// ── Crash hook ───────────────────────────────────────────────────────────────
//
// `uncaughtExceptionMonitor` observes without changing what Node does next:
// the process still crashes exactly as before. Installed once per process for
// the clients that open runs on their own; each turns the crash into
// run.errored on its open implicit runs, or on a one-event run when none is
// open, so an unhandled exception is never silent. Exceptions `dt.run()`
// already recorded are skipped.
const _crashClients = new Set<WeakRef<Dunetrace>>();
const _recordedErrors = new WeakSet<object>();
let _crashHookInstalled = false;

function _markRecorded(err: unknown): void {
  if (typeof err === "object" && err !== null) _recordedErrors.add(err);
}

function _installCrashHook(): void {
  if (_crashHookInstalled) return;
  if (typeof process === "undefined" || typeof process.on !== "function") return;
  _crashHookInstalled = true;
  process.on("uncaughtExceptionMonitor", (err: unknown) => {
    for (const ref of _crashClients) {
      try { ref.deref()?._recordCrash(err); } catch { /* never throw from a crash hook */ }
    }
  });
}

function _envBool(name: string, fallback: boolean): boolean {
  const raw = process.env[name];
  if (raw === undefined) return fallback;
  return ["1", "true", "yes", "on"].includes(raw.trim().toLowerCase());
}

function _scriptName(): string {
  try {
    const argv1 = process.argv[1] ?? "";
    const base = argv1.split(/[\\/]/).pop() ?? "";
    return base || "implicit";
  } catch {
    return "implicit";
  }
}

interface ImplicitHandle {
  run: DunetraceRun;
  finish: (err?: unknown) => void;
  timer: ReturnType<typeof setTimeout> | null;
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
  /** Runs this client opened on its own, by run id, with their idle timers.
   *  See _openImplicitRun for how they end. */
  private _implicitOpen = new Map<string, ImplicitHandle>();
  private _implicitRuns: boolean;
  private _implicitIdleMs: number;
  private _defaultAgentId: string;
  private _implicitAnnounced = false;
  private _crashRef: WeakRef<Dunetrace> | null = null;

  constructor(opts: ClientOptions = {}) {
    // A DSN carries endpoint and key in one value. Explicit options and the
    // dedicated env vars win over it, so adding one never changes an
    // existing configuration.
    const dsn = opts.dsn ?? process.env.DUNETRACE_DSN ?? "";
    let dsnEndpoint = "";
    let dsnKey = "";
    if (dsn) {
      try {
        ({ endpoint: dsnEndpoint, apiKey: dsnKey } = parseDsn(dsn));
      } catch (err) {
        process.stderr.write(`[dunetrace] ignoring DUNETRACE_DSN for host ${dsnHost(dsn)}: ${err instanceof Error ? err.message : String(err)}\n`);
      }
    }
    const base      = (opts.endpoint ?? (process.env.DUNETRACE_ENDPOINT || dsnEndpoint || "http://localhost:8001")).replace(/\/$/, "");
    this._ingestUrl = base + "/v1/ingest";
    // Tell HTTP instrumentation to ignore our own traffic. Without this, shipping
    // a batch would emit a tool.called describing the ship, which buffers another
    // event, which ships — a feedback loop against our own ingest.
    registerOwnEndpoint(base);
    this._apiKey    = opts.apiKey ?? (process.env.DUNETRACE_API_KEY || dsnKey || "");
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

    // Runs that open themselves. See _openImplicitRun for the rules on when
    // they end. The most recent client is the one instrumentation asks.
    this._implicitRuns   = opts.implicitRuns ?? _envBool("DUNETRACE_IMPLICIT_RUNS", true);
    const idleS          = opts.implicitRunIdleS ?? Number(process.env.DUNETRACE_IMPLICIT_RUN_IDLE_S);
    this._implicitIdleMs = Math.max(0.01, Number.isFinite(idleS) && idleS > 0 ? idleS : 30) * 1000;
    this._defaultAgentId = opts.defaultAgentId ?? process.env.DUNETRACE_AGENT_ID ?? "";
    if (this._implicitRuns) {
      setRunOpener({
        implicit: (openedBy) => this._openImplicitRun(openedBy),
        entry:    (openedBy, agentId) => this._openEntryRun(openedBy, agentId),
      });
      this._crashRef = new WeakRef(this);
      _crashClients.add(this._crashRef);
      _installCrashHook();
    }

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
    const { run, finish } = this._startRun(agentId, opts);
    let result: T;
    try {
      result = await _runStorage.run(run, () => fn(run));
    } catch (err) {
      finish(err);
      _markRecorded(err);
      throw err;
    }
    finish();
    return result;
  }

  /**
   * Open a run: build the DunetraceRun, emit run.started, and hand back a
   * `finish` that emits the terminal event once. `run()` wraps this in the
   * async context; the implicit and entry-point paths below use it directly
   * because their end is decided elsewhere (an idle timer, a framework
   * callback, process exit).
   */
  private _startRun(
    agentId: string,
    opts:    RunOptions,
    extra:   { openedBy?: string; implicit?: boolean } = {},
  ): { run: DunetraceRun; finish: (err?: unknown) => void } {
    const model   = opts.model        ?? "unknown";
    const tools   = opts.tools        ?? [];
    const version = agentVersion(opts.systemPrompt ?? "", model, tools);
    const run     = new DunetraceRun(agentId, version, this, opts.runId);
    run.openedBy  = extra.openedBy ?? null;
    run.implicit  = extra.implicit === true;

    // A declared run never nests under a guessed one: close the implicit run
    // active in this context first, so the events that follow belong to the
    // run the agent declared.
    const active = _runStorage.getStore();
    if (!run.implicit && active && active.implicit && !active.closed) {
      this._closeImplicit(active.runId, "explicit_run_opened");
    }

    // Auto-thread parent_run_id: if this run opens while another run is already
    // active (this call is nested inside an enclosing run's fn, so
    // _runStorage.getStore() returns that parent) and the caller didn't pass
    // parentRunId, inherit the active run's id. This links nested multi-agent
    // runs into a parent/child graph with no manual id threading — the substrate
    // the server-side DELEGATION_LOOP and HANDOFF_CONTEXT_LOSS detectors consume.
    // An explicit parentRunId always wins. Propagation follows AsyncLocalStorage,
    // so it survives awaits within the same async context. A closed or implicit
    // run is never a parent.
    const parent = active && !active.closed && !active.implicit ? active : null;
    const parentRunId = opts.parentRunId ?? parent?.runId ?? null;

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
    // A run the SDK opened says so, and names the call that opened it. The
    // detector reads `implicit` and holds the run's signals in shadow.
    if (run.openedBy) startedPayload["opened_by"] = run.openedBy;
    if (run.implicit) startedPayload["implicit"]  = true;

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

    const finish = (err?: unknown): void => {
      if (run.closed) return;
      run.closed = true;
      if (err !== undefined) {
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
        return;
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
    };

    return { run, finish };
  }

  // ── Runs that open themselves ──────────────────────────────────────────────
  //
  // Every event belongs to a run, and a run needs a start and an end. When
  // instrumentation finds no run, the client decides both. How the run ENDS,
  // in the order tried:
  //
  //   1. A framework says so. `wrapGenerateText` / `wrapStreamText` open an
  //      exact run around the call (_openEntryRun) and close it when the call
  //      settles. Ordinary runs; nothing is guessed.
  //   2. The caller's own `dt.run()` closes an implicit run active in the same
  //      context (`exit_reason: explicit_run_opened`), so a real run never
  //      nests under a guessed one.
  //   3. Idle. A bare LLM call with no boundary in sight opens an *implicit*
  //      run and enters it into this async context; the calls that follow
  //      attach to it. It closes after implicitRunIdleS seconds with no
  //      events (`exit_reason: idle`). This is the rule for scripts. It is a
  //      guess, and the run says so on the wire (`implicit: true`,
  //      `opened_by`); the detector holds its signals in shadow.
  //   4. Process exit. shutdown() and the exit flush close whatever implicit
  //      runs are open (`exit_reason: process_exit`).
  //   5. A crash. An uncaught exception errors the open implicit runs, or a
  //      one-event run when none is open.
  //
  // The known weak spot is a long-lived server with no request boundary:
  // calls from many requests in one async context share an implicit run
  // until it goes idle. That is why implicit runs are shadowed, and why the
  // entry-point rules run first.

  _openImplicitRun(openedBy: string): DunetraceRun | null {
    if (!this._implicitRuns) return null;
    const agentId = this._defaultAgentId || _scriptName();
    const { run, finish } = this._startRun(agentId, {}, { openedBy, implicit: true });
    // enterWith: the rest of this synchronous execution and every async
    // continuation from it sees this run, which is how the calls that follow
    // attach without any code at the call site.
    _runStorage.enterWith(run);
    const handle: ImplicitHandle = { run, finish, timer: null };
    this._implicitOpen.set(run.runId, handle);
    this._armIdle(handle);
    if (!this._implicitAnnounced) {
      this._implicitAnnounced = true;
      process.stderr.write(
        `[dunetrace] no run was open when ${openedBy} was called, so one was opened for agent ` +
        `"${agentId}". It closes after ${Math.round(this._implicitIdleMs / 1000)}s without events or at ` +
        `process exit, and its signals are held in shadow. Wrap the agent in dt.run() or use ` +
        `wrapGenerateText() for exact run boundaries.\n`,
      );
    }
    return run;
  }

  _openEntryRun(openedBy: string, agentId?: string): EntryRunHandle | null {
    if (!this._implicitRuns) return null;
    const id = agentId || this._defaultAgentId || openedBy.split(".")[0] || "agent";
    const { run, finish } = this._startRun(id, {}, { openedBy });
    return { run, finish };
  }

  private _armIdle(handle: ImplicitHandle): void {
    if (handle.timer !== null) clearTimeout(handle.timer);
    handle.timer = setTimeout(() => { this._closeImplicit(handle.run.runId, "idle"); }, this._implicitIdleMs);
    // Never keep the process alive just to close a guessed run; exit closes it.
    if (typeof handle.timer.unref === "function") handle.timer.unref();
  }

  private _closeImplicit(runId: string, exitReason: string, err?: unknown): boolean {
    const handle = this._implicitOpen.get(runId);
    if (!handle) return false;
    this._implicitOpen.delete(runId);
    if (handle.timer !== null) clearTimeout(handle.timer);
    if (err === undefined) handle.run._setExitReasonIfUnset(exitReason);
    handle.finish(err);
    return true;
  }

  /** Close every implicit run this client has open. @internal */
  _closeImplicitRuns(exitReason: string, err?: unknown): number {
    let n = 0;
    for (const runId of [...this._implicitOpen.keys()]) {
      if (this._closeImplicit(runId, exitReason, err)) n += 1;
    }
    return n;
  }

  /** How many implicit runs are open. @internal */
  _implicitOpenCount(): number {
    return this._implicitOpen.size;
  }

  /** Turn an uncaught exception into run.errored. @internal */
  _recordCrash(err: unknown): number {
    if (typeof err === "object" && err !== null && _recordedErrors.has(err)) return 0;
    let n = this._closeImplicitRuns("crash", err);
    if (n === 0 && this._implicitRuns) {
      const { finish } = this._startRun(this._defaultAgentId || _scriptName(), {}, {
        openedBy: "uncaughtException",
        implicit: true,
      });
      finish(err);
      n = 1;
    }
    if (n > 0) _markRecorded(err);
    return n;
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
    if (this._crashRef !== null) {
      _crashClients.delete(this._crashRef);
      this._crashRef = null;
    }
    // Terminal events for runs the SDK opened on its own, before the buffer
    // drains, so they leave in the same flush.
    this._closeImplicitRuns("process_exit");
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
      // Activity on an implicit run pushes its idle deadline out.
      const implicit = this._implicitOpen.get(event.run_id);
      if (implicit && event.event_type !== "run.completed" && event.event_type !== "run.errored") {
        this._armIdle(implicit);
      }
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
    if (this._exitFlushing) return;
    this._closeImplicitRuns("process_exit");
    if (this._buffer.length === 0) return;
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
