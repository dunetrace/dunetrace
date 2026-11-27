"""Command line: fetch, check, db, replay.

cd benchmarks
python -m bench_tau2 fetch                 # download + verify the 27 files
python -m bench_tau2 check                 # convert everything, lossless check, counts
python -m bench_tau2 db up                 # local Postgres for replay (docker)
python -m bench_tau2 replay --fresh        # full replay into a new database
python -m bench_tau2 replay                # resume where it stopped
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from bench_tau2 import corpus

CONTAINER = "dunetrace-bench-pg"
VOLUME = "dunetrace-bench-pg"
IMAGE = "postgres:16-alpine"
PORT = 55433


def cmd_fetch(args) -> int:
    manifest = corpus.load_manifest(args.manifest)
    corpus.fetch(manifest, args.cache_dir)
    problems = corpus.verify(manifest, args.cache_dir)
    for p in problems:
        print(p)
    return 1 if problems else 0


def cmd_check(args) -> int:
    """Convert the whole corpus without a database and report what it contains.

    Fails when the trajectory counts differ from the paper's Table 1 or any run
    is not lossless.
    """
    from bench_tau2.convert import Converter, file_context
    from bench_tau2.verify import verify_run

    manifest = corpus.load_manifest(args.manifest)
    counts: Counter = Counter()
    label_counts: Counter = Counter()
    stats: Counter = Counter()
    problems: Counter = Counter()
    examples = {}
    started = time.time()
    with Converter() as converter:
        for cf in corpus.iter_files(manifest, args.cache_dir):
            ctx = file_context(cf)
            sims = {s["id"]: s for s in cf.simulations}
            runs = converter.convert_file(cf)
            counts[(cf.entry.domain, "excluded")] += cf.excluded_count
            for sim in cf.simulations:
                reward = sim["reward_info"]["reward"]
                counts[(cf.entry.domain, "total")] += 1
                counts[(cf.entry.domain, "success" if reward == 1 else "failure")] += 1
            for run in runs:
                stats.update(run.stats)
                stats["runs"] += 1
                if run.turn_index == 0:
                    label_counts[run.label] += 1
                for p in verify_run(run, ctx, sims[run.sim_id]):
                    key = p.split(":", 1)[0]
                    problems[key] += 1
                    examples.setdefault(
                        key, f"{run.agent_id} {run.sim_id} turn {run.turn_index}: {p}"
                    )
            conversations_without_runs = len(sims) - len({r.sim_id for r in runs})
            stats["conversations_without_runs"] += conversations_without_runs
            print(f"{cf.entry.agent_id}: {len(cf.simulations)} trajectories, {len(runs)} runs")

    failed = False
    print("\ncorpus vs arXiv 2606.09863 Table 1")
    for domain, want in manifest.expected.items():
        got = {k: counts[(domain, k)] for k in ("total", "success", "failure")}
        ok = got == want
        failed |= not ok
        print(f"  {domain:8s} {got}  {'ok' if ok else 'MISMATCH, expected ' + json.dumps(want)}")
    print(f"  excluded: {sum(v for (d, k), v in counts.items() if k == 'excluded')}")
    print(f"\nlabels (one per trajectory): {dict(sorted(label_counts.items()))}")
    print(f"conversion: {dict(sorted(stats.items()))}")
    if problems:
        failed = True
        print(f"\nNOT LOSSLESS: {dict(problems)}")
        for key, example in examples.items():
            print(f"  {example}")
    else:
        print("\nlossless: every run rebuilds to its trajectory")
    print(f"({time.time() - started:.0f}s)")
    return 1 if failed else 0


def cmd_db(args) -> int:
    if args.action == "up":
        running = subprocess.run(
            ["docker", "ps", "-q", "-f", f"name=^{CONTAINER}$"], capture_output=True, text=True
        ).stdout.strip()
        if running:
            print(f"{CONTAINER} already running on 127.0.0.1:{PORT}")
            return 0
        subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
        subprocess.run(
            [
                "docker", "run", "-d", "--name", CONTAINER,
                "-e", "POSTGRES_PASSWORD=bench",
                "-p", f"127.0.0.1:{PORT}:5432",
                "-v", f"{VOLUME}:/var/lib/postgresql/data",
                IMAGE,
            ],
            check=True,
        )  # fmt: skip
        for _ in range(60):
            ready = subprocess.run(
                ["docker", "exec", CONTAINER, "pg_isready", "-U", "postgres"], capture_output=True
            )
            if ready.returncode == 0:
                print(f"{CONTAINER} ready on 127.0.0.1:{PORT}")
                return 0
            time.sleep(1)
        print("postgres did not become ready", file=sys.stderr)
        return 1
    if args.action == "down":
        subprocess.run(["docker", "rm", "-f", CONTAINER], check=False)
        return 0
    if args.action == "destroy":
        subprocess.run(["docker", "rm", "-f", CONTAINER], check=False)
        subprocess.run(["docker", "volume", "rm", VOLUME], check=False)
        return 0
    return 2


def cmd_replay(args) -> int:
    from bench_tau2.replay import ReplayConfig, main_replay

    config = ReplayConfig(
        admin_dsn=args.database_url,
        db_name=args.db_name,
        out_dir=Path(args.out),
        cache_dir=Path(args.cache_dir),
        manifest_path=Path(args.manifest),
        fresh=args.fresh,
        limit=args.limit,
        agents=set(args.agent) if args.agent else None,
        check_hash=not args.no_hash_check,
        allow_remote=args.allow_remote,
    )
    summary = main_replay(config)
    return 1 if summary["processing_errors"] else 0


def build_parser() -> argparse.ArgumentParser:
    from bench_tau2.replay import DEFAULT_ADMIN_DSN, DEFAULT_DB_NAME, REPO_ROOT

    parser = argparse.ArgumentParser(prog="bench_tau2", description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", default=str(corpus.MANIFEST_PATH))
    parser.add_argument("--cache-dir", default=str(corpus.DEFAULT_CACHE_DIR))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch", help="download and verify the corpus").set_defaults(func=cmd_fetch)
    sub.add_parser("check", help="convert everything and check it").set_defaults(func=cmd_check)

    db = sub.add_parser("db", help="local Postgres container for replay")
    db.add_argument("action", choices=["up", "down", "destroy"])
    db.set_defaults(func=cmd_db)

    rp = sub.add_parser("replay", help="replay through ingest and the detector worker")
    rp.add_argument(
        "--database-url",
        default=DEFAULT_ADMIN_DSN,
        help="admin DSN; a database is created under it",
    )
    rp.add_argument("--db-name", default=DEFAULT_DB_NAME)
    rp.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / ".cache" / "replay"))
    rp.add_argument("--fresh", action="store_true", help="drop the database and start over")
    rp.add_argument("--limit", type=int, help="process at most N more runs, then export")
    rp.add_argument("--agent", action="append", help="replay only this agent_id (repeatable)")
    rp.add_argument("--no-hash-check", action="store_true", help="skip per-file SHA-256 checks")
    rp.add_argument("--allow-remote", action="store_true")
    rp.set_defaults(func=cmd_replay)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
