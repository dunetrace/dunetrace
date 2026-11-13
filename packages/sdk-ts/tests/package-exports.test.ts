/**
 * Real Node module-resolution tests for the published `exports` map.
 *
 * The map declared only `"require"` and `"types"`. Node resolves a bare
 * specifier `import` with the conditions `["node", "import"]`; neither matched
 * and there was no `"default"`, so `import { Dunetrace } from "dunetrace"`
 * threw ERR_PACKAGE_PATH_NOT_EXPORTED for every ESM consumer while `require`
 * worked fine. Reading the map cannot catch that — only Node's own resolver
 * can, so these tests build a throwaway `node_modules/dunetrace` from the REAL
 * package.json with stub `dist` files and let a child Node process resolve it.
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { execFileSync } from "node:child_process";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const PKG_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const pkg = JSON.parse(readFileSync(join(PKG_ROOT, "package.json"), "utf8")) as {
  name: string;
  exports: Record<string, Record<string, string>>;
};

/** Subpath -> the marker the stub module exports, so we can prove *what* resolved. */
const SUBPATHS: Record<string, string> = {
  ".": "root",
  "./integrations/otel": "otel",
  "./integrations/otel-receiver": "otel-receiver",
};

let sandbox: string;

beforeAll(() => {
  sandbox = mkdtempSync(join(tmpdir(), "dunetrace-exports-"));
  const installed = join(sandbox, "node_modules", pkg.name);
  mkdirSync(installed, { recursive: true });
  // The real package.json — this is the artefact under test.
  writeFileSync(join(installed, "package.json"), readFileSync(join(PKG_ROOT, "package.json")));

  // Stub every file the exports map points at, so resolution has something to
  // load without a build. Each stub reports which subpath it came from.
  for (const [subpath, marker] of Object.entries(SUBPATHS)) {
    const entry = pkg.exports[subpath];
    expect(entry, `exports map is missing "${subpath}"`).toBeTruthy();
    for (const target of new Set(Object.values(entry))) {
      if (target.endsWith(".d.ts")) continue; // types are not a runtime target
      const file = join(installed, target);
      mkdirSync(dirname(file), { recursive: true });
      writeFileSync(file, `module.exports = { marker: ${JSON.stringify(marker)} };\n`);
    }
  }
});

afterAll(() => {
  rmSync(sandbox, { recursive: true, force: true });
});

function runProbe(filename: string, source: string): string {
  const file = join(sandbox, filename);
  writeFileSync(file, source);
  return execFileSync(process.execPath, [file], { cwd: sandbox, encoding: "utf8" }).trim();
}

describe("published exports map — real Node resolution", () => {
  it("every subpath resolves under an ESM import (the ERR_PACKAGE_PATH_NOT_EXPORTED bug)", () => {
    const imports = Object.entries(SUBPATHS)
      .map(([subpath, marker], i) => {
        const specifier = subpath === "." ? pkg.name : `${pkg.name}/${subpath.slice(2)}`;
        return `import m${i} from ${JSON.stringify(specifier)};\nmarkers.push([m${i}.marker, ${JSON.stringify(marker)}]);`;
      })
      .join("\n");
    const out = runProbe(
      "probe.mjs",
      `const markers = [];\n${imports}\nconsole.log(JSON.stringify(markers));\n`,
    );
    const markers = JSON.parse(out) as [string, string][];
    expect(markers).toHaveLength(Object.keys(SUBPATHS).length);
    for (const [actual, expected] of markers) expect(actual).toBe(expected);
  });

  it("every subpath still resolves under require()", () => {
    const requires = Object.entries(SUBPATHS)
      .map(([subpath, marker]) => {
        const specifier = subpath === "." ? pkg.name : `${pkg.name}/${subpath.slice(2)}`;
        return `markers.push([require(${JSON.stringify(specifier)}).marker, ${JSON.stringify(marker)}]);`;
      })
      .join("\n");
    const out = runProbe(
      "probe.cjs",
      `const markers = [];\n${requires}\nconsole.log(JSON.stringify(markers));\n`,
    );
    const markers = JSON.parse(out) as [string, string][];
    expect(markers).toHaveLength(Object.keys(SUBPATHS).length);
    for (const [actual, expected] of markers) expect(actual).toBe(expected);
  });

  it("named ESM imports work through Node's CommonJS interop", () => {
    // What a consumer actually writes: `import { Dunetrace } from "dunetrace"`.
    // Node's cjs-module-lexer has to find the named export on the CJS module.
    const installed = join(sandbox, "node_modules", pkg.name);
    writeFileSync(
      join(installed, pkg.exports["."]["default"]),
      "exports.Dunetrace = class Dunetrace {};\n",
    );
    const out = runProbe(
      "named.mjs",
      `import { Dunetrace } from ${JSON.stringify(pkg.name)};\n` +
        "console.log(typeof Dunetrace);\n",
    );
    expect(out).toBe("function");
  });

  it('lists "types" first in every entry, as resolvers require', () => {
    for (const [subpath, entry] of Object.entries(pkg.exports)) {
      const keys = Object.keys(entry);
      expect(keys[0], `"${subpath}" must list "types" first`).toBe("types");
      // "default" matches every condition, so anything after it is dead.
      expect(keys[keys.length - 1], `"${subpath}" must list "default" last`).toBe("default");
    }
  });
});
