/**
 * Content caps and secret redaction for everything the SDK ships off-process.
 *
 * The TypeScript port of `packages/sdk-py/dunetrace/redaction.py`. Two
 * independent controls, both applied in the emit hooks (`DunetraceRun`) and on
 * the `run.started` payload (`Dunetrace.run`):
 *
 * * **Caps** — every free-text field (tool args, tool output, LLM output,
 *   retrieval query/content, memory values, `input_text`, `system_prompt`) is
 *   cut at `maxFieldChars` (default {@link DEFAULT_MAX_FIELD_CHARS}, the same
 *   8192 the OTLP ingest path already enforces). A capped field carries two
 *   sibling wire keys, `<field>_truncated: true` and
 *   `<field>_original_length: N`; an uncapped field carries neither, so small
 *   payloads are byte-identical to before.
 *
 * * **Redaction** — structured tool args are walked and any value under a
 *   secret-looking key is replaced with `"[REDACTED]"` before serialisation.
 *   A key matches when its normalised form *equals* a denylist entry **or ends
 *   with** `_<entry>`, so the built-in list catches `Authorization`,
 *   `X-Api-Key`, `access_token`, `client_secret`, `db_password` and the like
 *   without enumerating them. A customer hook (`new Dunetrace({ redact })`)
 *   runs *before* the denylist, so it can strip domain-specific fields the
 *   built-in list cannot know about.
 *
 * NORMALISATION SPLITS camelCase, which the Python implementation does not.
 * Python normalises case and `-` → `_` only, so it matches `access_token` but
 * misses `accessToken`, `clientSecret`, `refreshToken` — and camelCase is the
 * dominant convention in JS/TS tool arguments, i.e. exactly the keys this SDK
 * sees. Matching is on *words*: `accessToken` normalises to `access_token`
 * before the comparison, so both spellings hit the same denylist entry.
 *
 * Everything here runs inside the customer's agent process, on the
 * instrumentation hot path, and must never throw.
 */

// Mirrors OTLP_MAX_ATTR_CHARS in services/ingest/ingest_svc/config.py so a run
// reads the same whichever transport it arrived on.
export const DEFAULT_MAX_FIELD_CHARS = 8192;

export const REDACTED = "[REDACTED]";

/** Matched against normalised keys (lower-cased, word-separated by `_`). */
export const DEFAULT_DENYLIST: readonly string[] = [
  "authorization",
  "api_key",
  "apikey",
  "token",
  "secret",
  "password",
  "cookie",
  "set_cookie",
];

/**
 * How deep {@link redactValue} descends. Deep enough for any real tool payload;
 * shallow enough that a pathological self-similar structure costs bounded work.
 * Nodes below the limit are passed through untouched (not dropped) — the cap
 * must never hide that data was sent, it just stops inspecting it.
 */
export const MAX_REDACT_DEPTH = 32;

export interface CompiledDenylist {
  exact: Set<string>;
  suffixes: string[];
}

/** `(text, truncated, originalLength)` — the result of a capped serialisation. */
export interface CappedText {
  text: string;
  truncated: boolean;
  originalLength: number;
}

/** Per-run content-cap + redaction configuration, resolved off the client. */
export interface RedactionSettings {
  maxFieldChars: number;
  redact: ((args: Record<string, unknown>) => Record<string, unknown>) | null;
  denylist: CompiledDenylist;
}

/**
 * Normalise a key to its word form: lower case, words separated by `_`.
 *
 * `accessToken` → `access_token`, `X-Api-Key` → `x_api_key`, `APIKey` →
 * `api_key`. The camelCase split is what the Python matcher is missing; see the
 * module docstring.
 */
export function normaliseKey(key: string): string {
  return key
    // lower/digit → upper boundary: "accessToken" → "access_Token"
    .replace(/([a-z0-9])([A-Z])/g, "$1_$2")
    // acronym → word boundary: "APIKey" → "API_Key"
    .replace(/([A-Z]+)([A-Z][a-z])/g, "$1_$2")
    // header/path separators are word separators too
    .replace(/[-.\s]+/g, "_")
    .toLowerCase();
}

/**
 * Build the (exact-match set, suffix list) pair the matcher uses.
 *
 * `extraKeys` extends {@link DEFAULT_DENYLIST}; both sides are normalised the
 * same way keys are at match time, so `redactKeys: ["X-Session-Id"]` and
 * `["xSessionId"]` are the same denylist. The suffix list is `"_" + entry` for
 * every entry, which is what makes `access_token` match `token` and
 * `x_api_key` match `api_key`.
 */
export function compileDenylist(extraKeys?: Iterable<string> | null): CompiledDenylist {
  const entries = new Set<string>();
  for (const k of DEFAULT_DENYLIST) entries.add(normaliseKey(k));
  if (extraKeys) {
    for (const k of extraKeys) {
      if (typeof k === "string" && k.trim()) entries.add(normaliseKey(k.trim()));
    }
  }
  const suffixes: string[] = [];
  for (const e of entries) suffixes.push("_" + e);
  suffixes.sort();
  return { exact: entries, suffixes };
}

const DEFAULT_COMPILED: CompiledDenylist = compileDenylist();

/** The settings a run uses when its client configured none. */
export const DEFAULT_REDACTION_SETTINGS: RedactionSettings = {
  maxFieldChars: DEFAULT_MAX_FIELD_CHARS,
  redact: null,
  denylist: DEFAULT_COMPILED,
};

/** True when `key` names a value that must not leave the process. */
export function keyIsSensitive(key: unknown, compiled?: CompiledDenylist | null): boolean {
  if (typeof key !== "string") return false;
  const { exact, suffixes } = compiled ?? DEFAULT_COMPILED;
  const nk = normaliseKey(key);
  if (exact.has(nk)) return true;
  for (const suffix of suffixes) {
    if (nk.endsWith(suffix)) return true;
  }
  return false;
}

/**
 * Objects {@link redactValue} descends into.
 *
 * Plain objects, arrays and class instances (an ORM entity carrying an
 * `apiKey` field is exactly the case this exists for) are walked, because
 * `JSON.stringify` will serialise their own enumerable properties. Types
 * `JSON.stringify` turns into something other than an object literal — Date,
 * RegExp, Error, Map/Set, typed arrays, anything with its own `toJSON` — are
 * passed through untouched, since walking them would change what ships.
 */
function isWalkable(value: unknown): boolean {
  if (typeof value !== "object" || value === null) return false;
  if (Array.isArray(value)) return true;
  if (value instanceof Date || value instanceof RegExp || value instanceof Error) return false;
  if (value instanceof Map || value instanceof Set) return false;
  if (value instanceof ArrayBuffer || ArrayBuffer.isView(value)) return false;
  if (typeof (value as { toJSON?: unknown }).toJSON === "function") return false;
  return true;
}

/**
 * Return a copy of `value` with every value under a sensitive key replaced by
 * {@link REDACTED}.
 *
 * Never mutates its input, never descends past `maxDepth`, and never throws: on
 * any internal failure the *original* value is returned, so the hook still
 * ships something rather than blocking the agent.
 */
export function redactValue(
  value: unknown,
  denylist?: CompiledDenylist | null,
  maxDepth: number = MAX_REDACT_DEPTH,
): unknown {
  try {
    return redactNode(value, denylist ?? DEFAULT_COMPILED, maxDepth);
  } catch {
    return value;
  }
}

function redactNode(node: unknown, compiled: CompiledDenylist, depth: number): unknown {
  if (depth <= 0 || !isWalkable(node)) return node;
  if (Array.isArray(node)) {
    return node.map((v) => redactNode(v, compiled, depth - 1));
  }
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(node as Record<string, unknown>)) {
    out[k] = keyIsSensitive(k, compiled) ? REDACTED : redactNode(v, compiled, depth - 1);
  }
  return out;
}

/**
 * `JSON.stringify` that cannot throw into the customer's call path.
 *
 * `JSON.stringify` throws on a circular reference and on a BigInt — both
 * routine in real tool arguments (an ORM entity, an Express `req`, a DB client,
 * a BigInt id). The replacer converts what JSON has no representation for and
 * substitutes `"[Circular]"` for a genuine cycle, detected against the
 * *ancestor chain* rather than a seen-set, so a value referenced twice in a DAG
 * still serialises normally. If serialisation fails anyway (a throwing getter,
 * a throwing `toJSON`) it falls back to `String(value)`, and to a type-name
 * placeholder if that throws too.
 */
export function safeJsonStringify(value: unknown): string {
  const ancestors: unknown[] = [];
  try {
    const text = JSON.stringify(value, function (this: unknown, _key: string, val: unknown) {
      if (typeof val === "bigint") return val.toString();
      if (typeof val === "function") return `<function ${(val as { name?: string }).name || "anonymous"}>`;
      if (typeof val === "symbol") return String(val);
      if (typeof val !== "object" || val === null) return val;
      // `this` is the holder of `val`; unwind to it, then the remaining stack
      // is exactly val's ancestor chain.
      while (ancestors.length > 0 && ancestors[ancestors.length - 1] !== this) ancestors.pop();
      if (ancestors.indexOf(val) !== -1) return "[Circular]";
      ancestors.push(val);
      return val;
    });
    return text === undefined ? safeString(value) : text;
  } catch {
    return safeString(value);
  }
}

export function safeString(value: unknown): string {
  try {
    return String(value);
  } catch {
    try {
      return Object.prototype.toString.call(value);
    } catch {
      return "<unserializable>";
    }
  }
}

/**
 * `(text, truncated, originalLength)` for a plain string. A `maxChars` of 0 or
 * less disables the cap. A non-string input is coerced first.
 */
export function capText(text: unknown, maxChars: number): CappedText {
  const s = typeof text === "string" ? text : safeString(text);
  const n = s.length;
  if (maxChars > 0 && n > maxChars) {
    return { text: s.slice(0, maxChars), truncated: true, originalLength: n };
  }
  return { text: s, truncated: false, originalLength: n };
}

/** `(text, truncated, originalLength)` for a structured value. */
export function serializeCapped(value: unknown, maxChars: number): CappedText {
  return capText(safeJsonStringify(value), maxChars);
}

/**
 * Store `value` under `payload[field]`, capped when it is a string, setting the
 * `<field>_truncated` / `<field>_original_length` markers only when a cut
 * actually happened. Returns what was stored.
 *
 * Non-string values are stored untouched — the hooks type these fields as
 * `string`, but callers pass what they have, and changing the type of an
 * already-odd value is not this layer's job.
 */
export function putCapped(
  payload: Record<string, unknown>,
  field: string,
  value: unknown,
  maxChars: number,
): unknown {
  if (typeof value !== "string") {
    payload[field] = value;
    return value;
  }
  const { text, truncated, originalLength } = capText(value, maxChars);
  payload[field] = text;
  if (truncated) {
    payload[field + "_truncated"] = true;
    payload[field + "_original_length"] = originalLength;
  }
  return text;
}

// A customer redact hook that throws is warned about exactly once per process,
// then stays silent — the hook fires on every tool call, and one bad hook must
// not turn an agent's logs into a wall of stack traces.
let redactHookWarned = false;

/** Reset the once-per-process hook warning. Testing seam. */
export function _resetRedactHookWarning(): void {
  redactHookWarned = false;
}

/**
 * Customer hook first, then the built-in denylist. Never throws.
 *
 * The hook gets a shallow copy and is expected to return a new object; if it
 * throws, or returns something that is not an object, its output is discarded
 * and the *original* args continue into the denylist step — the built-in
 * protection still applies, and the agent is never blocked by its own redaction
 * code. The hook is retried on every call rather than disabled after one
 * failure: a hook that chokes on one odd payload still protects every other.
 */
export function applyRedaction(
  settings: RedactionSettings,
  args: Record<string, unknown>,
): unknown {
  let value: unknown = args;
  if (settings.redact) {
    try {
      const out = settings.redact({ ...args });
      if (out !== null && typeof out === "object") {
        value = out;
      } else {
        throw new TypeError(`redact hook returned ${typeof out}, expected an object`);
      }
    } catch (err) {
      if (!redactHookWarned) {
        redactHookWarned = true;
        console.warn(
          "[dunetrace] redact hook failed " +
            `(${err instanceof Error ? err.message : String(err)}); shipping args with the ` +
            "built-in denylist only. Logged once per process.",
        );
      }
    }
  }
  return redactValue(value, settings.denylist);
}

/**
 * Redact + serialise + cap a `tool.called` `args` value.
 *
 * An object goes through the hook and the denylist and is then JSON-serialised;
 * an array skips the (object-typed) hook but is still denylist-walked; anything
 * else is stringified and capped as-is. `null`/`undefined` is the empty-args
 * `"{}"`, matching the Python SDK's wire format.
 */
export function serializeArgs(settings: RedactionSettings, args: unknown): CappedText {
  const max = settings.maxFieldChars;
  if (args === null || args === undefined) {
    return { text: "{}", truncated: false, originalLength: 2 };
  }
  if (typeof args === "string") return capText(args, max);
  if (Array.isArray(args)) return serializeCapped(redactValue(args, settings.denylist), max);
  if (typeof args === "object") {
    return serializeCapped(applyRedaction(settings, args as Record<string, unknown>), max);
  }
  return capText(safeString(args), max);
}

/** Coerce a caller-supplied `maxFieldChars` into a usable cap. */
export function resolveMaxFieldChars(value: unknown): number {
  if (value === undefined || value === null) return DEFAULT_MAX_FIELD_CHARS;
  if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
    console.warn(
      `[dunetrace] bad maxFieldChars=${safeString(value)}, using default ` +
        `${DEFAULT_MAX_FIELD_CHARS} (0 disables the cap)`,
    );
    return DEFAULT_MAX_FIELD_CHARS;
  }
  return value;
}

/**
 * Read the cap + redaction settings off a client.
 *
 * `Dunetrace` implements `_redactionSettings()`, but a `DunetraceRun` must also
 * work on any duck-typed emitter — tests and framework adapters pass a bare
 * `{ _emit }`. Every field is type-checked, and anything that is not the real
 * thing reads as "unconfigured": the documented default cap, no hook, the
 * built-in denylist. Read once per run, never on the hot path.
 */
export function readRedactionSettings(client: unknown): RedactionSettings {
  const accessor = (client as { _redactionSettings?: unknown } | null)?._redactionSettings;
  if (typeof accessor !== "function") return DEFAULT_REDACTION_SETTINGS;
  let raw: unknown;
  try {
    raw = (accessor as () => unknown).call(client);
  } catch {
    return DEFAULT_REDACTION_SETTINGS;
  }
  if (raw === null || typeof raw !== "object") return DEFAULT_REDACTION_SETTINGS;
  const s = raw as Partial<RedactionSettings>;
  const denylist =
    s.denylist && s.denylist.exact instanceof Set && Array.isArray(s.denylist.suffixes)
      ? s.denylist
      : DEFAULT_COMPILED;
  return {
    maxFieldChars:
      typeof s.maxFieldChars === "number" && Number.isInteger(s.maxFieldChars) && s.maxFieldChars >= 0
        ? s.maxFieldChars
        : DEFAULT_MAX_FIELD_CHARS,
    redact: typeof s.redact === "function" ? s.redact : null,
    denylist,
  };
}
