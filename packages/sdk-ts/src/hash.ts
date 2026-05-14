import { createHash } from "node:crypto";

/** SHA-256, first 16 hex chars. Matches Python: hashlib.sha256(text.encode()).hexdigest()[:16] */
export function hashContent(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex").slice(0, 16);
}

/**
 * Stable 8-char fingerprint of the agent configuration.
 * Any change to system prompt, model, or tool list produces a new version.
 *
 * Matches Python exactly:
 *   sha256(f"{system_prompt}:{model}:{sorted(tools)}").hexdigest()[:8]
 * where sorted(tools) is Python list repr: "['a', 'b']"
 */
export function agentVersion(systemPrompt: string, model: string, tools: string[]): string {
  const sorted     = [...tools].sort();
  const pythonRepr = "[" + sorted.map(t => `'${t}'`).join(", ") + "]";
  return createHash("sha256")
    .update(`${systemPrompt}:${model}:${pythonRepr}`, "utf8")
    .digest("hex")
    .slice(0, 8);
}
