/** Length of a value's string form: string length, else JSON length, else 0. */
export function resultLength(v: unknown): number {
  if (v == null) return 0;
  if (typeof v === "string") return v.length;
  try { return JSON.stringify(v).length; } catch { return 0; }
}
