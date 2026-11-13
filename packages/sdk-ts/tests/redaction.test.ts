/**
 * Secret redaction and content caps.
 *
 * The SDK is the only line of defence here: the server never redacts, and caps
 * only the OTLP path. An Authorization header left in tool args reaches
 * Postgres verbatim, renders in the dashboard, ships in Slack payloads and is
 * sent to an LLM provider on Explain.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  DEFAULT_MAX_FIELD_CHARS,
  REDACTED,
  capText,
  compileDenylist,
  keyIsSensitive,
  normaliseKey,
  putCapped,
  redactValue,
  safeJsonStringify,
  serializeArgs,
  _resetRedactHookWarning,
  type RedactionSettings,
} from "../src/redaction.js";

const settings = (over: Partial<RedactionSettings> = {}): RedactionSettings => ({
  maxFieldChars: DEFAULT_MAX_FIELD_CHARS,
  redact: null,
  denylist: compileDenylist(),
  ...over,
});

beforeEach(() => {
  _resetRedactHookWarning();
});

describe("key normalisation — words, not just case", () => {
  it("splits camelCase, which is the dominant convention in JS tool args", () => {
    expect(normaliseKey("accessToken")).toBe("access_token");
    expect(normaliseKey("clientSecret")).toBe("client_secret");
    expect(normaliseKey("refreshToken")).toBe("refresh_token");
    expect(normaliseKey("authToken")).toBe("auth_token");
    expect(normaliseKey("apiSecret")).toBe("api_secret");
  });

  it("splits acronym boundaries and header separators", () => {
    expect(normaliseKey("APIKey")).toBe("api_key");
    expect(normaliseKey("X-Api-Key")).toBe("x_api_key");
    expect(normaliseKey("Proxy-Authorization")).toBe("proxy_authorization");
    expect(normaliseKey("Set-Cookie")).toBe("set_cookie");
  });
});

describe("keyIsSensitive", () => {
  it("matches the camelCase spellings the Python matcher misses", () => {
    // Python normalises only case and '-' → '_', so these five slipped past it.
    for (const key of ["accessToken", "clientSecret", "refreshToken", "authToken", "apiSecret"]) {
      expect(keyIsSensitive(key), key).toBe(true);
    }
  });

  it("matches the snake_case and header spellings too", () => {
    for (const key of [
      "Authorization",
      "authorization",
      "X-Api-Key",
      "api_key",
      "apiKey",
      "access_token",
      "client_secret",
      "db_password",
      "Proxy-Authorization",
      "Cookie",
      "set-cookie",
      "token",
      "secret",
    ]) {
      expect(keyIsSensitive(key), key).toBe(true);
    }
  });

  it("leaves innocuous keys alone", () => {
    for (const key of [
      "userId",
      "maxTokens",
      "tokenCount",
      "secretary",
      "passwordless",
      "query",
      "toolName",
      "promptTokens",
    ]) {
      expect(keyIsSensitive(key), key).toBe(false);
    }
  });

  it("honours extra denylist keys in either spelling", () => {
    const compiled = compileDenylist(["X-Session-Id"]);
    expect(keyIsSensitive("xSessionId", compiled)).toBe(true);
    expect(keyIsSensitive("x_session_id", compiled)).toBe(true);
    expect(keyIsSensitive("xSessionId")).toBe(false); // not in the default list
  });
});

describe("redactValue", () => {
  it("replaces the value, not the key", () => {
    const out = redactValue({ url: "/v1/x", accessToken: "sk-live-1234" }) as Record<string, unknown>;
    expect(out["accessToken"]).toBe(REDACTED);
    expect(out["url"]).toBe("/v1/x");
  });

  it("walks nested objects and arrays of header dicts", () => {
    const out = redactValue({
      request: { headers: [{ Authorization: "Bearer abc" }, { "X-Api-Key": "k" }] },
    }) as Record<string, Record<string, Record<string, string>[]>>;
    expect(out["request"]["headers"][0]["Authorization"]).toBe(REDACTED);
    expect(out["request"]["headers"][1]["X-Api-Key"]).toBe(REDACTED);
  });

  it("never mutates its input", () => {
    const input = { apiKey: "sk-live" };
    redactValue(input);
    expect(input.apiKey).toBe("sk-live");
  });

  it("redacts a class instance's fields (an ORM entity carrying a token)", () => {
    class User {
      constructor(public id = 7, public refreshToken = "rt-secret") {}
    }
    const out = redactValue(new User()) as Record<string, unknown>;
    expect(out["refreshToken"]).toBe(REDACTED);
    expect(out["id"]).toBe(7);
  });

  it("passes Dates through untouched rather than flattening them to {}", () => {
    const when = new Date("2026-01-01T00:00:00Z");
    const out = redactValue({ when }) as Record<string, unknown>;
    expect(out["when"]).toBe(when);
  });

  it("stops at the depth limit instead of recursing forever", () => {
    const deep: Record<string, unknown> = {};
    let node = deep;
    for (let i = 0; i < 200; i++) {
      const next: Record<string, unknown> = {};
      node["child"] = next;
      node = next;
    }
    node["token"] = "leaked-too-deep";
    expect(() => redactValue(deep)).not.toThrow();
  });
});

describe("safeJsonStringify — never throws into customer code", () => {
  it("serialises a circular object instead of throwing", () => {
    const a: Record<string, unknown> = { name: "a" };
    a["self"] = a;
    const text = safeJsonStringify(a);
    expect(text).toContain("[Circular]");
    expect(() => JSON.parse(text)).not.toThrow();
  });

  it("serialises a BigInt instead of throwing", () => {
    expect(safeJsonStringify({ id: 10n })).toBe('{"id":"10"}');
  });

  it("does not mistake a shared reference in a DAG for a cycle", () => {
    const shared = { k: 1 };
    expect(safeJsonStringify({ a: shared, b: shared })).toBe('{"a":{"k":1},"b":{"k":1}}');
  });

  it("falls back to a string when even the replacer cannot save it", () => {
    const hostile = {
      get boom() {
        throw new Error("getter exploded");
      },
    };
    expect(() => safeJsonStringify(hostile)).not.toThrow();
    expect(typeof safeJsonStringify(hostile)).toBe("string");
  });
});

describe("capText / putCapped — the wire markers", () => {
  it("leaves a short field byte-identical, with no markers", () => {
    const payload: Record<string, unknown> = {};
    putCapped(payload, "output", "hello", 100);
    expect(payload).toEqual({ output: "hello" });
  });

  it("cuts a long field and records the original length", () => {
    const payload: Record<string, unknown> = {};
    putCapped(payload, "output", "x".repeat(50), 10);
    expect(payload["output"]).toBe("x".repeat(10));
    expect(payload["output_truncated"]).toBe(true);
    expect(payload["output_original_length"]).toBe(50);
  });

  it("treats a cap of 0 as no cap", () => {
    expect(capText("x".repeat(99), 0).truncated).toBe(false);
  });

  it("defaults to the 8192 the OTLP ingest path enforces", () => {
    expect(DEFAULT_MAX_FIELD_CHARS).toBe(8192);
  });
});

describe("serializeArgs", () => {
  it("redacts before serialising, so the secret is never in the string", () => {
    const { text } = serializeArgs(settings(), {
      url: "https://api.example.com",
      headers: { Authorization: "Bearer sk-live-supersecret" },
    });
    expect(text).not.toContain("sk-live-supersecret");
    expect(text).toContain(REDACTED);
  });

  it("caps the serialised form and reports the pre-cap length", () => {
    const { text, truncated, originalLength } = serializeArgs(settings({ maxFieldChars: 32 }), {
      blob: "y".repeat(500),
    });
    expect(text).toHaveLength(32);
    expect(truncated).toBe(true);
    expect(originalLength).toBeGreaterThan(500);
  });

  it("empty args are the Python SDK's '{}'", () => {
    expect(serializeArgs(settings(), null).text).toBe("{}");
    expect(serializeArgs(settings(), undefined).text).toBe("{}");
  });

  it("runs the customer hook before the built-in denylist", () => {
    const s = settings({
      redact: (args) => ({ ...args, ssn: "[dropped by hook]" }),
    });
    const { text } = serializeArgs(s, { ssn: "123-45-6789", apiKey: "sk-live" });
    expect(text).toContain("[dropped by hook]");
    expect(text).not.toContain("123-45-6789");
    expect(text).toContain(REDACTED); // denylist still ran on apiKey
  });

  it("a throwing hook does not block the emit, and the denylist still applies", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const s = settings({
      redact: () => {
        throw new Error("hook is broken");
      },
    });
    const { text } = serializeArgs(s, { apiKey: "sk-live-secret" });
    expect(text).toContain(REDACTED);
    expect(text).not.toContain("sk-live-secret");
    expect(warn).toHaveBeenCalledOnce();

    // Warned once per process, not once per tool call.
    serializeArgs(s, { apiKey: "another" });
    expect(warn).toHaveBeenCalledOnce();
    warn.mockRestore();
  });

  it("a hook returning a non-object is discarded, not shipped", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const s = settings({ redact: (() => "nope") as unknown as RedactionSettings["redact"] });
    const { text } = serializeArgs(s!, { apiKey: "sk-live" });
    expect(text).toContain(REDACTED);
    expect(JSON.parse(text)).toEqual({ apiKey: REDACTED });
    warn.mockRestore();
  });
});
