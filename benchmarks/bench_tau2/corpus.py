"""The corpus: the committed manifest, hash-verified download, and iteration.

The raw trajectories are not committed. Sierra publishes no data license
separate from the τ²-bench repo's MIT license, and the files total 968 MB. The
manifest pins each file by S3 key and SHA-256, so a fetch either reproduces the
exact bytes this harness was built on or fails.

One manifest entry is one (submission, domain) file, which is also one agent
for replay: every simulation in a file ran the same model, prompt and tools.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional

from bench_tau2._paths import REPO_ROOT

DATA_DIR = Path(__file__).resolve().parent / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"
DEFAULT_CACHE_DIR = REPO_ROOT / "benchmarks" / ".cache" / "tau2"


@dataclass(frozen=True)
class FileEntry:
    submission: str
    domain: str
    s3_key: str
    sha256: str
    size: int
    tau2_git_commit: str
    agent_llm: str

    @property
    def agent_id(self) -> str:
        return f"tau2.{self.domain}.{self.submission}"

    @property
    def file_name(self) -> str:
        return f"{self.submission}__{self.s3_key.rsplit('/', 1)[-1]}"


@dataclass(frozen=True)
class Manifest:
    source_bucket: str
    files: List[FileEntry]
    expected: dict
    sha256: str


def load_manifest(path: Optional[Path] = None) -> Manifest:
    path = Path(path or MANIFEST_PATH)
    raw_bytes = path.read_bytes()
    raw = json.loads(raw_bytes)
    files = [
        FileEntry(
            submission=f["submission"],
            domain=f["domain"],
            s3_key=f["s3_key"],
            sha256=f["sha256"],
            size=int(f["size"]),
            tau2_git_commit=f.get("tau2_git_commit") or "unknown",
            agent_llm=f["agent_llm"],
        )
        for f in raw["files"]
    ]
    return Manifest(
        source_bucket=raw["source_bucket"],
        files=files,
        expected=raw.get("expected", {}),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_path(entry: FileEntry, cache_dir: Path) -> Path:
    return Path(cache_dir) / entry.file_name


def fetch(
    manifest: Manifest,
    cache_dir: Path,
    log: Callable[[str], None] = print,
    retries: int = 5,
) -> None:
    """Download every manifest file not already present with the right hash.

    A file whose bytes do not match the manifest is an error, not a warning:
    the S3 objects are mutable, and a silent substitution would change every
    number downstream.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for entry in manifest.files:
        dest = local_path(entry, cache_dir)
        if dest.exists() and sha256_file(dest) == entry.sha256:
            log(f"ok       {entry.file_name}")
            continue
        url = manifest.source_bucket + entry.s3_key
        tmp = dest.with_suffix(dest.suffix + ".part")
        for attempt in range(1, retries + 1):
            try:
                with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:
                    for chunk in iter(lambda: resp.read(1 << 20), b""):
                        out.write(chunk)
                break
            except OSError as exc:
                if attempt == retries:
                    raise RuntimeError(f"download failed after {retries} attempts: {url}") from exc
                time.sleep(2 * attempt)
        got = sha256_file(tmp)
        if got != entry.sha256:
            tmp.unlink()
            raise RuntimeError(
                f"{entry.s3_key}: sha256 {got} does not match the manifest's {entry.sha256}. "
                "The published object has changed; the corpus is no longer the one measured."
            )
        tmp.replace(dest)
        log(f"fetched  {entry.file_name}")


def verify(manifest: Manifest, cache_dir: Path) -> List[str]:
    """Problems with the local copy, empty when every file matches."""
    problems = []
    for entry in manifest.files:
        dest = local_path(entry, cache_dir)
        if not dest.exists():
            problems.append(f"missing  {dest}")
        elif sha256_file(dest) != entry.sha256:
            problems.append(f"hash mismatch  {dest}")
    return problems


def is_excluded(sim: dict) -> bool:
    """The paper's filter: no reward, or no message structure to replay.

    Removes exactly the 132 simulations that separate the 10,008 published
    simulations from the paper's 9,876.
    """
    reward = (sim.get("reward_info") or {}).get("reward")
    return reward is None or not sim.get("messages")


@dataclass
class CorpusFile:
    entry: FileEntry
    info: dict
    simulations: List[dict]  # excluded ones removed
    excluded_count: int


def load_file(entry: FileEntry, cache_dir: Path, *, check_hash: bool = True) -> CorpusFile:
    path = local_path(entry, cache_dir)
    if check_hash and sha256_file(path) != entry.sha256:
        raise RuntimeError(f"{path} does not match the manifest hash; run `fetch` again")
    raw = json.loads(path.read_bytes())
    info = raw["info"]
    domain = (info.get("environment_info") or {}).get("domain_name")
    if domain != entry.domain:
        raise RuntimeError(f"{path}: domain {domain!r} but the manifest says {entry.domain!r}")
    kept = [s for s in raw["simulations"] if not is_excluded(s)]
    return CorpusFile(
        entry=entry,
        info=info,
        simulations=kept,
        excluded_count=len(raw["simulations"]) - len(kept),
    )


def iter_files(
    manifest: Manifest,
    cache_dir: Path,
    *,
    check_hash: bool = True,
    agents: Optional[set] = None,
) -> Iterator[CorpusFile]:
    """One file at a time, in manifest order. Loading all 27 at once needs
    several GB; nothing downstream needs more than one agent in memory."""
    for entry in manifest.files:
        if agents is not None and entry.agent_id not in agents:
            continue
        yield load_file(entry, cache_dir, check_hash=check_hash)
