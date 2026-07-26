import { randomUUID } from "node:crypto";
import { agentVersion } from "./hash.js";
import { DunetraceRun } from "./run.js";
import { resultLength as _resultLength, resultText as _resultText } from "./util.js";
import { HttpBatchEmitter, type BatchEmitter } from "./emitters.js";
import { runStorage as _runStorage, getCurrentRun } from "./context.js";
import { registerOwnEndpoint, wrapAnthropicClient, wrapOpenAIClient } from "./auto.js";
import type { AgentEvent, ClientOptions, EventSink, RunOptions } from "./models.js";

export { getCurrentRun };

export class Dunetrace {
  private _ingestUrl:  string | null;
  private _apiKey:     string;
  private _buffer:     AgentEvent[]       = [];
  private _bufferSize: number;
  private _timeoutMs:  number;
  private _drainTimer: ReturnType<typeof setInterval> | null = null;
  private _emitJson:   boolean;
  private _emitter:    BatchEmitter;
  private _exporter:   EventSink | null;

  constructor(opts: ClientOptions = {}) {
    const base      = (opts.endpoint ?? "http://localhost:8001").replace(/\/$/, "");
    this._ingestUrl = base + "/v1/ingest";
    // Tell HTTP instrumentation to ignore our own traffic. Without this, shipping
    // a batch would emit a tool.called describing the ship, which buffers another
    // event, which ships — a feedback loop against our own ingest.
    registerOwnEndpoint(base);
    this._apiKey    = opts.apiKey ?? "";
    this._emitJson  = opts.emitAsJson ?? false;
    this._bufferSize = opts.bufferSize ?? 10_000;
    this._timeoutMs  = opts.timeoutMs  ?? 5_000;
    this._emitter    = opts.emitter ?? new HttpBatchEmitter(base, this._apiKey, this._timeoutMs);
    this._exporter   = opts.exporter ?? null;

    const interval = opts.flushIntervalMs ?? 200;
    this._drainTimer = setInterval(() => { this._drain(); }, interval);
    // Don't prevent the process from exiting naturally
    if (typeof this._drainTimer.unref === "function") {
      this._drainTimer.unref();
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

    this._emit({
      event_type:    "run.started",
      run_id:        run.runId,
      agent_id:      agentId,
      agent_version: version,
      step_index:    0,
      timestamp:     Date.now() / 1000,
      payload: {
        input_text:    opts.userInput ?? "",
        system_prompt: opts.systemPrompt ?? "",
        model,
        tools,
      },
      parent_run_id: parentRunId,
      trace_id:      opts.traceId ?? null,
      conversation_id: opts.conversationId ?? null,
    });

    let result: T;
    try {
      result = await _runStorage.run(run, () => fn(run));
    } catch (err) {
      this._emit({
        event_type:    "run.errored",
        run_id:        run.runId,
        agent_id:      agentId,
        agent_version: version,
        step_index:    run.currentStep(),
        timestamp:     Date.now() / 1000,
        payload: {
          error_type: (err instanceof Error) ? err.name : "Error",
          error:      String(err),
          step_index: run.currentStep(),
        },
      });
      throw err;
    }

    this._emit({
      event_type:    "run.completed",
      run_id:        run.runId,
      agent_id:      agentId,
      agent_version: version,
      step_index:    run.currentStep(),
      timestamp:     Date.now() / 1000,
      payload: {
        total_steps:     run.currentStep(),
        exit_reason:     run.exitReason() ?? "completed",
        tool_call_count: run.getEvents().filter(e => e.event_type === "tool.called").length,
      },
    });

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

    if (isAsync) {
      const wrapper = async (...args: Parameters<T>): Promise<unknown> => {
        const run = _runStorage.getStore();
        if (run) run.toolCalled(toolName, _argsRecord(fn, args));
        const t0 = Date.now();
        try {
          const result = await (fn as (...a: unknown[]) => Promise<unknown>)(...args);
          if (run) run.toolResponded(toolName, true, _resultLength(result), Date.now() - t0, undefined, _resultText(result));
          return result;
        } catch (err) {
          if (run) run.toolResponded(toolName, false, 0, Date.now() - t0, String(err));
          throw err;
        }
      };
      return wrapper as unknown as T;
    } else {
      const wrapper = (...args: Parameters<T>): unknown => {
        const run = _runStorage.getStore();
        if (run) run.toolCalled(toolName, _argsRecord(fn, args));
        const t0 = Date.now();
        try {
          const result = (fn as (...a: unknown[]) => unknown)(...args);
          if (run) run.toolResponded(toolName, true, _resultLength(result), Date.now() - t0, undefined, _resultText(result));
          return result;
        } catch (err) {
          if (run) run.toolResponded(toolName, false, 0, Date.now() - t0, String(err));
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
    const batch = this._buffer.splice(0);
    if (batch.length > 0) await this._emitter.ship(batch);
  }

  async shutdown(timeoutMs = 5000): Promise<void> {
    if (this._drainTimer !== null) {
      clearInterval(this._drainTimer);
      this._drainTimer = null;
    }
    const deadline = Date.now() + timeoutMs;
    while (this._buffer.length > 0 && Date.now() < deadline) {
      await this.flush();
    }
  }

  // ── Internal ───────────────────────────────────────────────────────────────

  _emit(event: AgentEvent): void {
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
    if (this._buffer.length >= this._bufferSize) return;
    this._buffer.push(event);
  }

  private _drain(): void {
    const batch = this._buffer.splice(0, 100);
    if (batch.length === 0 || !this._ingestUrl) return;
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
