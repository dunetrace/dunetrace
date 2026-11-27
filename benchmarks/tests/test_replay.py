"""Replay through the real ingest persist path and detector worker.

Needs a Postgres. Skipped unless BENCH_TEST_DATABASE_URL names an admin DSN on
a local host; each test creates and drops its own databases under it:

    python -m bench_tau2 db up
    BENCH_TEST_DATABASE_URL=postgresql://postgres:bench@127.0.0.1:55433/postgres \\
        make test-bench

Every replay runs the CLI in a subprocess, so each one imports the services
fresh with its own pinned environment, exactly as a real invocation does.
"""

import asyncio
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

import fixtures

ADMIN_DSN = os.environ.get("BENCH_TEST_DATABASE_URL")
BENCH_DIR = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not ADMIN_DSN, reason="set BENCH_TEST_DATABASE_URL to run replay tests"
)

OUTPUTS = ("runs.jsonl", "signals.jsonl", "signals_evidence.jsonl")


def _drop(names):
    import asyncpg

    async def go():
        conn = await asyncpg.connect(ADMIN_DSN)
        try:
            for name in names:
                await conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        finally:
            await conn.close()

    asyncio.run(go())


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("bench")
    airline = [fixtures.two_turn_airline(f"air-{n}") for n in range(3)]
    # 30 runs, interleaved in time, mostly 1 lookup with every fifth one long.
    lookups = [
        fixtures.lookup_conversation(f"look-{n:02d}", 6 if n % 5 == 4 else 1, start=n * 7.0)
        for n in range(30)
    ]
    manifest = fixtures.write_corpus(
        root,
        [
            ("lookups", "airline", fixtures.results("airline", lookups)),
            ("synthetic", "airline", fixtures.results("airline", airline)),
            ("synthetic", "retail", fixtures.results("retail", [fixtures.retry_storm_retail()])),
            ("synthetic", "telecom", fixtures.results("telecom", [fixtures.old_format_telecom()])),
        ],
    )
    created = []

    def replay(out_name, db_name=None, *extra, expect_ok=True):
        db_name = db_name or f"bench_test_{uuid.uuid4().hex[:10]}"
        if db_name not in created:
            created.append(db_name)
        out = root / out_name
        cmd = [
            sys.executable, "-m", "bench_tau2",
            "--manifest", str(manifest), "--cache-dir", str(root / "data"),
            "replay", "--database-url", ADMIN_DSN, "--db-name", db_name, "--out", str(out), *extra,
        ]  # fmt: skip
        proc = subprocess.run(cmd, cwd=BENCH_DIR, capture_output=True, text=True, timeout=600)
        if expect_ok:
            assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
        summary = None
        for line in proc.stdout.splitlines():
            if line.startswith("{"):
                summary = json.loads(line)
        return proc, summary, out, db_name

    replay.root = root
    baseline = replay("baseline", None, "--fresh")
    replay.baseline = baseline
    yield replay
    _drop(created)


def _read(out, name):
    return (out / name).read_text()


def _jsonl(out, name):
    return [json.loads(line) for line in _read(out, name).splitlines()]


def test_every_run_is_processed_and_signals_map_to_messages(world):
    _, summary, out, _ = world.baseline
    runs = _jsonl(out, "runs.jsonl")
    assert summary["runs"] == len(runs) == 30 + 3 * 2 + 1 + 1
    assert summary["processed"] == len(runs)
    assert summary["processing_errors"] == 0
    assert all(r["processed"] and r["processing_error"] is None for r in runs)

    signals = _jsonl(out, "signals.jsonl")
    storm = [s for s in signals if s["failure_type"] == "RETRY_STORM"]
    assert len(storm) == 1, signals
    s = storm[0]
    assert (s["agent_id"], s["sim_id"], s["label"]) == ("tau2.retail.synthetic", "sim-retry", "HF")
    # RETRY_STORM reports the run's last step; that step is the closing reply.
    assert s["step_index"] == 7
    assert s["message_indices"] == [8]
    assert s["shadow"] is False  # RETRY_STORM is live
    evidence = [
        e for e in _jsonl(out, "signals_evidence.jsonl") if e["failure_type"] == "RETRY_STORM"
    ]
    assert evidence[0]["evidence"]["first_fail_step"] == 2
    assert "evidence" not in s  # content stays out of the committable file


def test_baselines_form_and_baseline_detectors_fire(world):
    """Without this, determinism and resume would only be shown for detectors
    that ignore history. STEP_COUNT_INFLATION needs 20 earlier runs."""
    _, _, out, _ = world.baseline
    inflation = [
        s for s in _jsonl(out, "signals.jsonl") if s["failure_type"] == "STEP_COUNT_INFLATION"
    ]
    assert inflation, "no baseline-dependent signal fired"
    assert {s["agent_id"] for s in inflation} == {"tau2.airline.lookups"}
    fired = sorted(int(s["sim_id"].split("-")[1]) for s in inflation)
    assert all(n % 5 == 4 and n >= 20 for n in fired), (
        fired
    )  # only long runs, only once history exists


def test_replay_is_deterministic(world):
    _, _, first, _ = world.baseline
    _, _, second, _ = world("again", None, "--fresh")
    for name in OUTPUTS:
        assert _read(first, name) == _read(second, name), name


def test_resume_after_interruption_equals_an_uninterrupted_replay(world):
    # Stop inside the lookups agent, after its baseline has started forming.
    _, partial, out, db = world("resumed", None, "--fresh", "--limit", "22")
    assert partial["processed"] == 22
    runs = _jsonl(out, "runs.jsonl")
    assert sum(r["processed"] for r in runs) == 22
    _, resumed, out, _ = world("resumed", db)
    assert resumed["processed_this_invocation"] == resumed["runs"] - 22
    for name in OUTPUTS:
        assert _read(out, name) == _read(world.baseline[2], name), name


def test_finished_replay_is_idempotent(world):
    _, _, out, db = world.baseline
    before = {name: _read(out, name) for name in OUTPUTS}
    _, summary, out, _ = world("baseline", db)
    assert summary["processed_this_invocation"] == 0
    assert {name: _read(out, name) for name in OUTPUTS} == before


def test_refuses_to_resume_with_a_different_selection(world):
    _, _, _, db = world.baseline
    proc, _, _, _ = world("baseline", db, "--agent", "tau2.retail.synthetic", expect_ok=False)
    assert proc.returncode != 0
    assert "refusing to resume" in proc.stdout + proc.stderr
