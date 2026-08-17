#!/usr/bin/env python3
"""
End-to-end test for the autofix feature (explain + copy + fix-status + open-pr).

Root-cause analysis is fully native — no external tracing system involved.
Exercises each endpoint in sequence against a real signal from the running DB:

  1. POST /v1/signals/{id}/explain     — root_cause + fix_content/fix_patch
  2. POST /v1/signals/{id}/record-copy — clipboard-path tracking
  3. GET  /v1/signals/{id}/fix-status  — recurrence verdict
  4. (open-pr skipped unless --open-pr flag passed, as it writes to GitHub)

Usage:
    python scripts/test_autofix.py --agent-id my-agent --signal-id 42
    python scripts/test_autofix.py --agent-id my-agent --signal-id 42 --open-pr
"""

from __future__ import annotations

import argparse
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
AUTH_HEADER = {
    "Authorization": "Bearer dt_dev_test",
    "Content-Type": "application/json",
}

parser = argparse.ArgumentParser()
parser.add_argument("--agent-id", default="", help="Agent ID to pick a recent signal from")
parser.add_argument("--signal-id", type=int, default=0, help="Exact signal ID to test")
parser.add_argument("--open-pr", action="store_true", help="Also test the open-pr endpoint")
args = parser.parse_args()

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


# ── Step 0: Find a signal to test ──────────────────────────────────────────────

print(f"\n{'=' * 60}")
print("Autofix end-to-end test")
print(f"{'=' * 60}\n")

print("[0] Find a signal to test")
if args.signal_id and args.agent_id:
    data, status = get(f"/v1/agents/{args.agent_id}/signals?limit=500")
    sig = next((s for s in data.get("signals", []) if s["id"] == args.signal_id), None)
    if not sig:
        fail(f"Signal {args.signal_id} not found for agent {args.agent_id!r}")
        sys.exit(1)
elif args.agent_id:
    data, status = get(f"/v1/agents/{args.agent_id}/signals?limit=1")
    sigs = data.get("signals", [])
    if not sigs:
        fail(f"No signals found for agent {args.agent_id!r} — run any instrumented agent first")
        sys.exit(1)
    sig = sigs[0]
else:
    fail("Pass --agent-id (and optionally --signal-id) — see --help")
    sys.exit(1)

SIGNAL_ID = sig["id"]
ok(f"Signal {SIGNAL_ID} found: {sig['failure_type']} ({sig['severity']})")
print(f"      run_id = {sig['run_id']}")
print(f"      evidence_summary = {sig['evidence_summary']}")


# ── Step 1: Explain ────────────────────────────────────────────────────────────

print("\n[1] POST /v1/signals/{id}/explain")
explain_data, explain_status = post(f"/v1/signals/{SIGNAL_ID}/explain")

fix_content = ""

if explain_status == 200:
    ok("explain returned 200")

    if "root_cause" in explain_data and explain_data["root_cause"]:
        ok(f"root_cause present ({len(explain_data['root_cause'])} chars)")
        print(f"\n      ROOT CAUSE:\n      {explain_data['root_cause'][:300]}\n")
    else:
        fail("root_cause missing or empty", str(explain_data))

    if explain_data.get("fix_category") == "dunetrace_native":
        ok(
            f"fix_category = dunetrace_native — suggested_policy: {explain_data.get('suggested_policy')}"
        )
        fix_content = explain_data.get("root_cause", "")
    else:
        fc = explain_data.get("fix_content")
        if fc:
            ok(f"fix_content present: {fc!r}")
            fix_content = fc
        else:
            fail("fix_content is empty — LLM did not return structured JSON")
        print(
            f"      fix_type = {explain_data.get('fix_type')}  apply_blocked = {explain_data.get('apply_blocked')}"
        )

elif explain_status == 503:
    detail = explain_data.get("detail", "")
    if "LLM" in detail or "API key" in detail:
        print(f"  SKIP  LLM not configured ({detail})")
    else:
        fail("explain 503", detail)
    fix_content = sig["suggested_fixes"][0]["code"] if sig.get("suggested_fixes") else ""
else:
    fail(f"explain returned {explain_status}", explain_data.get("detail", ""))


# ── Step 2: record-copy ────────────────────────────────────────────────────────

print("\n[2] POST /v1/signals/{id}/record-copy")
if not fix_content:
    fix_content = (
        sig["suggested_fixes"][0]["code"] if sig.get("suggested_fixes") else "Add deduplication"
    )

copy_data, copy_status = post(
    f"/v1/signals/{SIGNAL_ID}/record-copy",
    {"fix_content": fix_content},
)
if copy_status == 200:
    ok(f"record-copy returned 200  (fix_id={copy_data.get('fix_id')})")
else:
    fail(f"record-copy returned {copy_status}", copy_data.get("detail", ""))


# ── Step 3: fix-status ─────────────────────────────────────────────────────────

print("\n[3] GET /v1/signals/{id}/fix-status")
status_data, status_code = get(f"/v1/signals/{SIGNAL_ID}/fix-status")
if status_code == 200:
    ok("fix-status returned 200")
    fix_applied = status_data.get("fix_applied")
    if fix_applied:
        ok(f"fix_applied = True  →  verdict = {status_data.get('verdict')!r}")
        print(f"      runs_after_fix        = {status_data.get('runs_after_fix')}")
        print(f"      recurrences_after_fix = {status_data.get('recurrences_after_fix')}")
    else:
        fail("fix_applied = False — record-copy did not persist")
else:
    fail(f"fix-status returned {status_code}", status_data.get("detail", ""))


# ── Step 4: open-pr (optional) ─────────────────────────────────────────────────

if args.open_pr:
    print("\n[4] POST /v1/signals/{id}/open-pr  (--open-pr flag set)")
    pr_data, pr_status = post(
        f"/v1/signals/{SIGNAL_ID}/open-pr",
        {
            "root_cause": explain_data.get("root_cause", ""),
            "fix_content": fix_content,
            "fix_patch": explain_data.get("fix_patch", ""),
        },
    )
    if pr_status == 200:
        ok(f"open-pr returned 200 — {pr_data.get('pr_url')}")
    elif pr_status == 503:
        print(f"  SKIP  GitHub not configured ({pr_data.get('detail', '')})")
    else:
        fail(f"open-pr returned {pr_status}", pr_data.get("detail", ""))
else:
    print("\n[4] open-pr  (skipped — pass --open-pr to write to GitHub)")


# ── Summary ────────────────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print(f"Results:  {_pass} passed  {_fail} failed")
if _fail:
    print(f"{'=' * 60}\n")
    sys.exit(1)
print(f"{'=' * 60}\n")
