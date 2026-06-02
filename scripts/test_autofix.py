#!/usr/bin/env python3
"""
End-to-end test for the autofix feature (explain + copy + apply-fix).

Uses signal 348 (TOOL_LOOP, langfuse-example-agent) which is already in
the running DB. Exercises each new endpoint in sequence:

  1. POST /v1/signals/{id}/explain  — root_cause + fix_content + prompt detection
  2. POST /v1/signals/{id}/record-copy — clipboard-path tracking
  3. GET  /v1/signals/{id}/fix-status  — recurrence verdict
  4. (apply-fix skipped unless --apply flag passed, as it writes to Langfuse)

Usage:
    python scripts/test_autofix.py
    python scripts/test_autofix.py --apply   # also tests apply-fix (writes to Langfuse)
"""

from __future__ import annotations

import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

# ── Load .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

API = "http://localhost:8002"
AUTH_HEADER = {"Authorization": "Bearer dt_dev_test", "Content-Type": "application/json"}
SIGNAL_ID = 348  # TOOL_LOOP on langfuse-example-agent, steps 2-7
APPLY = "--apply" in sys.argv

_pass = 0
_fail = 0


def ok(label):
    global _pass
    _pass += 1
    print(f"  PASS  {label}")


def fail(label, detail=""):
    global _fail
    _fail += 1
    print(f"  FAIL  {label}" + (f": {detail}" if detail else ""))


def get(path):
    req = urllib.request.Request(f"{API}{path}", headers=AUTH_HEADER)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read()), r.status


def post(path, body=None):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        f"{API}{path}",
        data=data,
        headers=AUTH_HEADER,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        body_bytes = e.read()
        try:
            detail = json.loads(body_bytes).get("detail", body_bytes.decode())
        except Exception:
            detail = body_bytes.decode()
        return {"detail": detail, "_status": e.code}, e.code


# ── Step 0: Verify signal exists ───────────────────────────────────────────────

print(f"\n{'=' * 60}")
print(f"Autofix end-to-end test  (signal_id={SIGNAL_ID})")
print(f"{'=' * 60}\n")

print("[0] Verify signal exists")
data, status = get(f"/v1/agents/langfuse-example-agent/signals?limit=500")
sig = next((s for s in data.get("signals", []) if s["id"] == SIGNAL_ID), None)
if sig:
    ok(f"Signal {SIGNAL_ID} found: {sig['failure_type']} ({sig['severity']})")
    print(f"      run_id = {sig['run_id']}")
    print(f"      evidence_summary = {sig['evidence_summary']}")
else:
    fail(f"Signal {SIGNAL_ID} not found — run the langfuse_agent.py example first")
    sys.exit(1)


# ── Step 1: Explain ────────────────────────────────────────────────────────────

print("\n[1] POST /v1/signals/{id}/explain")
explain_data, explain_status = post(f"/v1/signals/{SIGNAL_ID}/explain")

if explain_status == 200:
    ok(f"explain returned 200")

    if "root_cause" in explain_data and explain_data["root_cause"]:
        ok(f"root_cause present ({len(explain_data['root_cause'])} chars)")
        print(f"\n      ROOT CAUSE:\n      {explain_data['root_cause'][:300]}\n")
    else:
        fail("root_cause missing or empty", str(explain_data))

    if "fix_content" in explain_data:
        fc = explain_data["fix_content"]
        if fc:
            ok(f"fix_content present: {fc!r}")
        else:
            fail("fix_content is empty — LLM did not return structured JSON")
    else:
        fail("fix_content key missing from response")

    pname = explain_data.get("langfuse_prompt_name")
    pver = explain_data.get("langfuse_prompt_version")
    if pname:
        ok(f"langfuse_prompt_name = {pname!r}  version = {pver}")
    else:
        print(f"  NOTE  langfuse_prompt_name = null — trace didn't use a managed prompt")
        print(f"        'Apply via Langfuse' button will NOT be shown (correct behaviour)")

    fix_content = explain_data.get("fix_content", "")
    prompt_name = explain_data.get("langfuse_prompt_name")

elif explain_status == 503:
    detail = explain_data.get("detail", "")
    if "Langfuse" in detail:
        print(f"  SKIP  Langfuse not configured ({detail})")
    elif "LLM" in detail or "API key" in detail:
        print(f"  SKIP  LLM not configured ({detail})")
    else:
        fail(f"explain 503", detail)
    fix_content = sig["suggested_fixes"][0]["code"] if sig.get("suggested_fixes") else ""
    prompt_name = None
else:
    fail(f"explain returned {explain_status}", explain_data.get("detail", ""))
    fix_content = ""
    prompt_name = None


# ── Step 2: record-copy ────────────────────────────────────────────────────────

print("\n[2] POST /v1/signals/{id}/record-copy")
if not fix_content:
    fix_content = (
        sig["suggested_fixes"][0]["code"] if sig.get("suggested_fixes") else "Add deduplication"
    )

copy_data, copy_status = post(
    f"/v1/signals/{SIGNAL_ID}/record-copy",
    {
        "fix_content": fix_content,
        "langfuse_prompt_name": prompt_name or "",
        "applied_via": "clipboard",
    },
)
if copy_status == 200:
    ok(f"record-copy returned 200  (fix_id={copy_data.get('fix_id')})")
else:
    fail(f"record-copy returned {copy_status}", copy_data.get("detail", ""))


# ── Step 3: fix-status ─────────────────────────────────────────────────────────

print("\n[3] GET /v1/signals/{id}/fix-status")
status_data, status_code = get(f"/v1/signals/{SIGNAL_ID}/fix-status")
if status_code == 200:
    ok(f"fix-status returned 200")
    fix_applied = status_data.get("fix_applied")
    if fix_applied:
        ok(f"fix_applied = True  →  verdict = {status_data.get('verdict')!r}")
        print(f"      runs_after_fix        = {status_data.get('runs_after_fix')}")
        print(f"      recurrences_after_fix = {status_data.get('recurrences_after_fix')}")
    else:
        fail("fix_applied = False — record-copy did not persist")
else:
    fail(f"fix-status returned {status_code}", status_data.get("detail", ""))


# ── Step 4: apply-fix (optional) ──────────────────────────────────────────────

if APPLY:
    print("\n[4] POST /v1/signals/{id}/apply-fix  (--apply flag set)")
    if not prompt_name:
        print("  SKIP  No langfuse_prompt_name — cannot apply without a managed prompt")
    elif not fix_content:
        print("  SKIP  No fix_content from explain step")
    else:
        apply_data, apply_status = post(
            f"/v1/signals/{SIGNAL_ID}/apply-fix",
            {
                "fix_content": fix_content,
                "langfuse_prompt_name": prompt_name,
                "applied_via": "langfuse",
            },
        )
        if apply_status == 200:
            ok(f"apply-fix returned 200")
            ok(
                f"new_version = {apply_data.get('new_version')}  url = {apply_data.get('prompt_url')}"
            )
        else:
            fail(f"apply-fix returned {apply_status}", apply_data.get("detail", ""))
else:
    print("\n[4] apply-fix  (skipped — pass --apply to write to Langfuse)")


# ── Summary ────────────────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print(f"Results:  {_pass} passed  {_fail} failed")
if _fail:
    print(f"{'=' * 60}\n")
    sys.exit(1)
print(f"{'=' * 60}\n")
