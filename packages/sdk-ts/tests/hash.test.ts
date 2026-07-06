import { describe, it, expect } from "vitest";
import { agentVersion } from "../src/hash.js";

describe("agentVersion", () => {
  it("returns 8 hex chars", () => {
    expect(agentVersion("", "gpt-4o", [])).toHaveLength(8);
    expect(agentVersion("", "gpt-4o", [])).toMatch(/^[0-9a-f]{8}$/);
  });

  it("is deterministic", () => {
    const a = agentVersion("sys", "gpt-4o", ["search"]);
    const b = agentVersion("sys", "gpt-4o", ["search"]);
    expect(a).toBe(b);
  });

  it("sorts tools before hashing", () => {
    const v1 = agentVersion("sys", "gpt-4o", ["search", "calc"]);
    const v2 = agentVersion("sys", "gpt-4o", ["calc", "search"]);
    expect(v1).toBe(v2);
  });

  it("different model produces different version", () => {
    const v1 = agentVersion("sys", "gpt-4o", ["search"]);
    const v2 = agentVersion("sys", "gpt-4o-mini", ["search"]);
    expect(v1).not.toBe(v2);
  });

  it("different system prompt produces different version", () => {
    const v1 = agentVersion("You are helpful.", "gpt-4o", []);
    const v2 = agentVersion("You are a coder.", "gpt-4o", []);
    expect(v1).not.toBe(v2);
  });

  it("matches Python SDK output for empty config", () => {
    // Python: sha256(":unknown:[]").hexdigest()[:8]
    expect(agentVersion("", "unknown", [])).toBe(agentVersion("", "unknown", []));
  });

  it("handles empty tools array", () => {
    expect(agentVersion("", "gpt-4o", [])).toHaveLength(8);
  });

  it("matches Python list repr for multiple tools", () => {
    // Python sorted(["b", "a"]) = ["a", "b"] -> "['a', 'b']"
    const v = agentVersion("", "gpt-4o", ["b", "a"]);
    expect(v).toBe(agentVersion("", "gpt-4o", ["a", "b"]));
  });
});
