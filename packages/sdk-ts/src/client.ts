import { AsyncLocalStorage } from "node:async_hooks";
import { hashContent, agentVersion } from "./hash.js";
import { DunetraceRun } from "./run.js";
import type { AgentEvent, ClientOptions, RunOptions } from "./models.js";

const _runStorage = new AsyncLocalStorage<DunetraceRun>();

function _resultLength(v: unknown): number {
  if (v == null) return 0;
  if (typeof v === "string") return v.length;
  try { return JSON.stringify(v).length; } catch { return 0; }
}

export class Dunetrace {
  private _ingestUrl:  string | null;
  private _apiKey:     string;
  private _buffer:     AgentEvent[]       = [];
  private _drainTimer: ReturnType<typeof setInterval> | null = null;
  private _emitJson:   boolean;

  constructor(opts: ClientOptions = {}) {
    const base      = (opts.endpoint ?? "http://localhost:8001").replace(/\/$/, "");
    this._ingestUrl = base + "/v1/ingest";
    this._apiKey    = opts.apiKey ?? "";
    this._emitJson  = opts.emitAsJson ?? false;

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

    this._emit({
      event_type:    "run.started",
      run_id:        run.runId,
      agent_id:      agentId,
      agent_version: version,
      step_index:    0,
      timestamp:     Date.now() / 1000,
      payload: {
        input_hash:   opts.userInput ? hashContent(opts.userInput) : "",
        model,
        tools,
      },
      parent_run_id: opts.parentRunId ?? null,
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
          error_hash: hashContent(String(err)),
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
          if (run) run.toolResponded(toolName, true, _resultLength(result), Date.now() - t0);
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
          if (run) run.toolResponded(toolName, true, _resultLength(result), Date.now() - t0);
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
    if (batch.length > 0) await this._ship(batch);
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
    if (this._emitJson) this._writeJsonLine(event);
    this._buffer.push(event);
  }

  private _drain(): void {
    const batch = this._buffer.splice(0, 100);
    if (batch.length === 0 || !this._ingestUrl) return;
    this._ship(batch).catch(() => {});
  }

  private async _ship(batch: AgentEvent[]): Promise<void> {
    if (!this._ingestUrl) return;
    try {
      await fetch(this._ingestUrl, {
        method:  "POST",
        headers: {
          "Content-Type":      "application/json",
          "X-Dunetrace-Agent": batch[0]?.agent_id ?? "",
        },
        body: JSON.stringify({
          api_key:  this._apiKey,
          agent_id: batch[0]?.agent_id ?? "",
          events:   batch,
        }),
      });
    } catch (err) {
      process.stderr.write(`[dunetrace] Failed to ship ${batch.length} events: ${err}\n`);
    }
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

/** Return the current DunetraceRun from async context, or null. */
export function getCurrentRun(): DunetraceRun | null {
  return _runStorage.getStore() ?? null;
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
