"""Replay converted runs through the production ingest and detector code.

For each run, in a fixed order:

1. The events are validated by ingest's own request model (`IngestRequest`,
   which is `dunetrace_schemas.AgentEventSchema` per event) and written by
   `ingest_svc.routers.ingest._persist`, the function `POST /v1/ingest` calls
   once auth and rate limiting have passed.
2. The detector worker's `process_run` runs on it, unchanged: baselines,
   cross-run checks, LIVE_DETECTORS shadow gating, the risk engine,
   co-occurrence boosts, issue tracking and baseline writes all happen as in
   production.

Runs are processed one at a time. The worker normally processes a poll's runs
concurrently, which is fine in production but would make baselines depend on
scheduling. Within an agent, runs are ordered by completion time, which is
the order a live worker would have seen them. Each manifest file is its own
agent, so order across files changes nothing.

Resumable: a run already in `processed_runs` is skipped, and replay refuses to
resume with different code, config or corpus (see replay_meta.json). Because
runs are processed strictly in order, the processed set is always a prefix of
the order, and resuming is equivalent to never having stopped. That is
checked, not assumed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

from bench_tau2 import corpus, detector_config, labels
from bench_tau2._paths import REPO_ROOT, install_repo_paths
from bench_tau2.convert import CONVERTER_VERSION, ConvertedRun, Converter
from bench_tau2.tools import TOOLS_PATH

ORG_ID = "bench-tau2"
DEFAULT_DB_NAME = "dunetrace_bench"
DEFAULT_ADMIN_DSN = "postgresql://postgres:bench@127.0.0.1:55433/postgres"
TRIGGER = "completed"
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}

# Source trees whose behaviour decides what fires. A change to any of them
# invalidates a partial replay.
CODE_TREES = [
    "packages/sdk-py/dunetrace",
    "packages/schemas-py/dunetrace_schemas",
    "services/ingest/ingest_svc",
    "services/detector/detector_svc",
    "benchmarks/bench_tau2",
]


@dataclass
class ReplayConfig:
    admin_dsn: str = DEFAULT_ADMIN_DSN
    db_name: str = DEFAULT_DB_NAME
    out_dir: Path = REPO_ROOT / "benchmarks" / ".cache" / "replay"
    cache_dir: Path = corpus.DEFAULT_CACHE_DIR
    manifest_path: Path = corpus.MANIFEST_PATH
    fresh: bool = False
    limit: Optional[int] = None
    agents: Optional[set] = None
    check_hash: bool = True
    allow_remote: bool = False


# ── Fingerprint ──────────────────────────────────────────────────────────────


def code_sha256() -> str:
    digest = hashlib.sha256()
    for tree in CODE_TREES:
        for path in sorted((REPO_ROOT / tree).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            digest.update(str(path.relative_to(REPO_ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def fingerprint(manifest: corpus.Manifest, config_sha: str, agents: List[str]) -> dict:
    """Everything a replay's results are a function of. Resuming requires equality."""
    return {
        "converter_version": CONVERTER_VERSION,
        "label_rule_version": labels.LABEL_RULE_VERSION,
        "manifest_sha256": manifest.sha256,
        "tau2_tools_sha256": hashlib.sha256(TOOLS_PATH.read_bytes()).hexdigest(),
        "detector_config_sha256": config_sha,
        "code_sha256": code_sha256(),
        "agents": sorted(agents),
        "org_id": ORG_ID,
    }


# ── Database ─────────────────────────────────────────────────────────────────


def _dsn_for(admin_dsn: str, db_name: str) -> str:
    return urlunsplit(urlsplit(admin_dsn)._replace(path=f"/{db_name}"))


async def _prepare_database(config: ReplayConfig) -> str:
    import asyncpg

    host = urlsplit(config.admin_dsn).hostname
    if host not in _LOCAL_HOSTS and not config.allow_remote:
        raise SystemExit(f"refusing to create or drop databases on non-local host {host!r}")
    conn = await asyncpg.connect(config.admin_dsn)
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", config.db_name)
        if exists and config.fresh:
            await conn.execute(f'DROP DATABASE "{config.db_name}" WITH (FORCE)')
            exists = False
        if not exists:
            await conn.execute(f'CREATE DATABASE "{config.db_name}"')
    finally:
        await conn.close()
    return _dsn_for(config.admin_dsn, config.db_name)


def pin_environment(dsn: str, detector_config_path: Path) -> None:
    """Set before any service module is imported. Their config modules read the
    repo .env with setdefault, and that file may name a real database."""
    os.environ["DATABASE_URL"] = dsn
    os.environ["DETECTOR_CONFIG"] = str(detector_config_path)
    os.environ["METRICS_PORT"] = "0"
    os.environ.pop("ENV", None)  # never trip the deployment guard


class Services:
    """The production modules replay drives, imported after the environment is pinned."""

    def __init__(self, dsn: str) -> None:
        install_repo_paths()
        import detector_svc.db as detector_db
        import ingest_svc.db.postgres as ingest_db
        from detector_svc.detectors import get_detectors
        from detector_svc.worker import process_run
        from ingest_svc.routers.ingest import _persist
        from ingest_svc.schemas import IngestRequest

        ingest_db.settings.DATABASE_URL = dsn
        detector_db.settings.DATABASE_URL = dsn
        self.ingest_db = ingest_db
        self.detector_db = detector_db
        self.persist = _persist
        self.IngestRequest = IngestRequest
        self.process_run = process_run
        self.get_detectors = get_detectors

    async def start(self) -> None:
        await self.ingest_db.init_pool()
        await self.ingest_db.ensure_schema()
        await self.detector_db.init_pool()
        await self.detector_db.ensure_detector_schema()

    async def close(self) -> None:
        await self.detector_db.close_pool()
        await self.ingest_db.close_pool()

    async def check_battery(self, agents: Dict[str, str]) -> None:
        """Every agent runs the 37 built-ins with shipped settings, except
        UNRESOLVED_AMBIGUITY's declared tools. No plugins, no packs."""
        from dunetrace.detectors import DETECTOR_KEYS

        from bench_tau2.tools import irreversible_tools

        default = await self.get_detectors("default", ORG_ID)
        names = sorted(type(d).__name__ for d in default)
        expected = sorted(c.__name__ for c in DETECTOR_KEYS.values())
        if names != expected:
            raise RuntimeError(f"default battery is not the {len(expected)} built-ins: {names}")
        baseline = {type(d).__name__: _tunables(d) for d in default}
        for agent_id, domain in agents.items():
            battery = await self.get_detectors(agent_id, ORG_ID)
            got = {type(d).__name__: _tunables(d) for d in battery}
            if sorted(got) != expected:
                raise RuntimeError(f"{agent_id}: battery differs from the built-ins")
            ua = got["UnresolvedAmbiguityDetector"].pop("IRREVERSIBLE_TOOLS", None)
            if list(ua or []) != irreversible_tools(domain):
                raise RuntimeError(f"{agent_id}: IRREVERSIBLE_TOOLS not applied ({ua!r})")
            want = {k: dict(v) for k, v in baseline.items()}
            want["UnresolvedAmbiguityDetector"].pop("IRREVERSIBLE_TOOLS", None)
            if got != want:
                raise RuntimeError(f"{agent_id}: a setting other than IRREVERSIBLE_TOOLS differs")

    async def processed(self) -> Dict[str, Optional[str]]:  # run_id -> processing_error
        async with self.detector_db._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT run_id, processing_error FROM processed_runs WHERE org_id = $1", ORG_ID
            )
        return {r["run_id"]: r["processing_error"] for r in rows}

    async def replay_run(self, run: ConvertedRun) -> Optional[str]:
        """Persist and detect one run. Returns its processing_error, None if clean."""
        body = self.IngestRequest(agent_id=run.agent_id, events=run.events)
        await self.persist(body.events, f"bench-{run.run_id}", ORG_ID)
        # The worker's own retry budget: a failed attempt leaves the run
        # unmarked, and the third records processing_error.
        for _ in range(self.detector_db.MAX_PROCESSING_ATTEMPTS):
            await self.process_run(run.run_id, run.agent_id, run.agent_version, TRIGGER, ORG_ID)
            async with self.detector_db._pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT processing_error FROM processed_runs WHERE org_id = $1 AND run_id = $2",
                    ORG_ID,
                    run.run_id,
                )
            if row is not None:
                return row["processing_error"]
        raise RuntimeError(f"run {run.run_id} was never marked processed")

    async def signals(self) -> list:
        async with self.detector_db._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT run_id, failure_type, severity, step_index, confidence,
                       evidence, shadow, co_signal_count
                FROM failure_signals WHERE org_id = $1
                """,
                ORG_ID,
            )
        out = []
        for r in rows:
            evidence = r["evidence"]
            out.append(
                {
                    **dict(r),
                    "evidence": json.loads(evidence) if isinstance(evidence, str) else evidence,
                }
            )
        return out


def _tunables(detector) -> dict:
    """A detector's UPPERCASE settings as it will run, class defaults included."""
    return {
        name: getattr(detector, name)
        for name in dir(detector)
        if name.isupper() and not name.startswith("_") and not callable(getattr(detector, name))
    }


# ── Replay ───────────────────────────────────────────────────────────────────


def _dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _run_record(run: ConvertedRun) -> dict:
    return {
        "run_id": run.run_id,
        "conversation_id": run.conversation_id,
        "agent_id": run.agent_id,
        "agent_version": run.agent_version,
        "submission": run.submission,
        "domain": run.domain,
        "sim_id": run.sim_id,
        "task_id": run.task_id,
        "trial": run.trial,
        "turn_index": run.turn_index,
        "n_turns": run.n_turns,
        "reward": run.reward,
        "label": run.label,
        "termination_reason": run.termination_reason,
        "n_events": len(run.events),
        "step_messages": {str(k): v for k, v in sorted(run.step_messages.items())},
        "stats": run.stats,
    }


async def replay(config: ReplayConfig, log: Callable[[str], None] = print) -> dict:
    manifest = corpus.load_manifest(config.manifest_path)
    entries = [e for e in manifest.files if config.agents is None or e.agent_id in config.agents]
    if not entries:
        raise SystemExit("no manifest entries selected")
    agents = {e.agent_id: e.domain for e in entries}

    config.out_dir.mkdir(parents=True, exist_ok=True)
    config_path = config.out_dir / "detectors.bench.yml"
    config_sha = detector_config.write(config_path, agents.items())
    meta_path = config.out_dir / "replay_meta.json"
    fp = fingerprint(manifest, config_sha, list(agents))
    fp["database"] = config.db_name
    if meta_path.exists() and not config.fresh:
        previous = json.loads(meta_path.read_text()).get("fingerprint")
        if previous != fp:
            changed = sorted(k for k in fp if (previous or {}).get(k) != fp[k])
            raise SystemExit(
                f"refusing to resume: {', '.join(changed)} changed since this replay started. "
                "Pass --fresh to start over."
            )

    dsn = await _prepare_database(config)
    pin_environment(dsn, config_path)
    services = Services(dsn)
    await services.start()
    try:
        await services.check_battery(agents)
        done = await services.processed()
        if done and not meta_path.exists():
            raise SystemExit(
                f"database {config.db_name} already holds {len(done)} processed runs but "
                f"{meta_path} is missing, so what produced them is unknown. Pass --fresh."
            )
        if config.fresh or not meta_path.exists():
            meta_path.write_text(json.dumps({"fingerprint": fp}, indent=2, sort_keys=True) + "\n")

        index: Dict[str, dict] = {}
        order: List[str] = []
        processed_now = 0
        started = time.time()
        with Converter() as converter:
            for corpus_file in corpus.iter_files(
                manifest, config.cache_dir, check_hash=config.check_hash, agents=set(agents)
            ):
                runs = converter.convert_file(corpus_file)
                runs.sort(key=lambda r: (r.end_ts, r.run_id))
                for run in runs:
                    index[run.run_id] = _run_record(run)
                    order.append(run.run_id)
                    if run.run_id in done:
                        continue
                    if config.limit is not None and processed_now >= config.limit:
                        continue
                    done[run.run_id] = await services.replay_run(run)
                    processed_now += 1
                    if processed_now % 1000 == 0:
                        rate = processed_now / max(time.time() - started, 1e-9)
                        log(f"  {processed_now} runs processed ({rate:.0f}/s)")
                log(f"{corpus_file.entry.agent_id}: {len(runs)} runs")

        _check_prefix(order, done)
        signals = await services.signals()
    finally:
        await services.close()

    summary = export(config.out_dir, index, done, signals)
    summary["processed_this_invocation"] = processed_now
    log(_dumps(summary))
    return summary


def _check_prefix(order: List[str], done: Dict[str, Optional[str]]) -> None:
    """The processed runs must be exactly the first len(done) runs of the order.
    Anything else means two replays interleaved, and baselines are unreliable."""
    n = sum(1 for run_id in order if run_id in done)
    if set(order[:n]) != {run_id for run_id in order if run_id in done}:
        raise RuntimeError("processed runs are not a prefix of the replay order")
    stray = set(done) - set(order)
    if stray:
        raise RuntimeError(f"{len(stray)} processed runs are not in this corpus selection")


def export(
    out_dir: Path, index: Dict[str, dict], done: Dict[str, Optional[str]], signals: list
) -> dict:
    """Canonical, sorted outputs. Byte-identical across replays of the same
    fingerprint; that is what the determinism test compares.

    signals.jsonl carries no evidence content (only its keys and a hash), so it
    can be committed without redistributing trajectory text. The evidence
    itself goes to signals_evidence.jsonl, which stays in the cache.
    """
    by_run: Dict[str, List[dict]] = {}
    for s in signals:
        by_run.setdefault(s["run_id"], []).append(s)

    run_lines, signal_lines, evidence_lines = [], [], []
    for run_id, record in index.items():
        record = dict(record)
        record["processed"] = run_id in done
        record["processing_error"] = done.get(run_id)
        record["signal_count"] = len(by_run.get(run_id, []))
        run_lines.append(record)
        for s in by_run.get(run_id, []):
            evidence = _dumps(s["evidence"])
            step = s["step_index"]
            signal_lines.append(
                {
                    "run_id": run_id,
                    "conversation_id": record["conversation_id"],
                    "agent_id": record["agent_id"],
                    "submission": record["submission"],
                    "domain": record["domain"],
                    "sim_id": record["sim_id"],
                    "turn_index": record["turn_index"],
                    "label": record["label"],
                    "failure_type": s["failure_type"],
                    "severity": s["severity"],
                    "shadow": s["shadow"],
                    "step_index": step,
                    "message_indices": record["step_messages"].get(str(step), []),
                    "confidence": round(float(s["confidence"]), 6),
                    "co_signal_count": s["co_signal_count"],
                    "evidence_keys": sorted(s["evidence"] or {}),
                    "evidence_sha256": hashlib.sha256(evidence.encode()).hexdigest(),
                }
            )
            evidence_lines.append(
                {
                    "run_id": run_id,
                    "failure_type": s["failure_type"],
                    "step_index": step,
                    "evidence": s["evidence"],
                }
            )

    run_lines.sort(key=_run_key)
    signal_lines.sort(key=_signal_key)
    evidence_lines.sort(key=_evidence_key)

    for name, lines in (
        ("runs.jsonl", run_lines),
        ("signals.jsonl", signal_lines),
        ("signals_evidence.jsonl", evidence_lines),
    ):
        with open(out_dir / name, "w") as fh:
            for line in lines:
                fh.write(_dumps(line) + "\n")

    errors = sum(1 for r in run_lines if r["processing_error"])
    return {
        "runs": len(run_lines),
        "processed": sum(1 for r in run_lines if r["processed"]),
        "processing_errors": errors,
        "signals": len(signal_lines),
        "conversations": len({r["conversation_id"] for r in run_lines}),
    }


def _run_key(r: dict) -> tuple:
    return (r["agent_id"], r["sim_id"], r["turn_index"])


def _signal_key(s: dict) -> tuple:
    return (
        s["agent_id"],
        s["sim_id"],
        s["turn_index"],
        s["failure_type"],
        s["step_index"],
        s["evidence_sha256"],
    )


def _evidence_key(e: dict) -> tuple:
    return (e["run_id"], e["failure_type"], e["step_index"], _dumps(e["evidence"]))


def environment_info() -> dict:
    """Recorded beside results for provenance. Not part of the fingerprint."""
    import asyncpg

    return {
        "git_head": _git_head(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "asyncpg": asyncpg.__version__,
    }


def main_replay(config: ReplayConfig) -> dict:
    summary = asyncio.run(replay(config))
    (config.out_dir / "environment.json").write_text(
        json.dumps(environment_info(), indent=2, sort_keys=True) + "\n"
    )
    return summary
