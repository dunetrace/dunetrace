import json
import os
from pathlib import Path

import pytest

import fixtures
from bench_tau2 import corpus


def test_committed_manifest_is_the_papers_selection():
    m = corpus.load_manifest()
    assert len(m.files) == 27
    assert len({f.submission for f in m.files}) == 9
    assert {f.domain for f in m.files} == {"airline", "retail", "telecom"}
    assert len({f.agent_id for f in m.files}) == 27
    for f in m.files:
        assert len(f.sha256) == 64 and int(f.sha256, 16) >= 0
        assert f.s3_key.startswith(f"submissions/{f.submission}/trajectories/")
    total = sum(d["total"] for d in m.expected.values())
    success = sum(d["success"] for d in m.expected.values())
    assert (total, success, total - success) == (9876, 8146, 1730)


def test_gpt_5_2_appears_twice():
    subs = {f.submission for f in corpus.load_manifest().files}
    assert {"gpt-5-2_sierra_2026-02-26", "gpt-5-2-none_sierra_2026-02-26"} <= subs


@pytest.mark.parametrize(
    "sim, excluded",
    [
        ({"reward_info": {"reward": 1.0}, "messages": [{"role": "user"}]}, False),
        ({"reward_info": {"reward": 0.0}, "messages": [{"role": "user"}]}, False),
        ({"reward_info": None, "messages": [{"role": "user"}]}, True),
        ({"reward_info": {"reward": None}, "messages": [{"role": "user"}]}, True),
        ({"reward_info": {"reward": 1.0}, "messages": []}, True),
        ({"reward_info": {"reward": 1.0}, "messages": None}, True),
    ],
)
def test_exclusion_rule(sim, excluded):
    assert corpus.is_excluded(sim) is excluded


def test_load_file_drops_excluded_and_checks_hash_and_domain(tmp_path):
    body = fixtures.results(
        "airline",
        [
            fixtures.two_turn_airline("a"),
            fixtures.simulation("b", [fixtures.user("x", 1)], reward=None),
        ],
    )
    manifest = corpus.load_manifest(fixtures.write_corpus(tmp_path, [("sub", "airline", body)]))
    cf = corpus.load_file(manifest.files[0], tmp_path / "data")
    assert [s["id"] for s in cf.simulations] == ["a"]
    assert cf.excluded_count == 1

    path = corpus.local_path(manifest.files[0], tmp_path / "data")
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="manifest hash"):
        corpus.load_file(manifest.files[0], tmp_path / "data")

    wrong = json.loads(Path(tmp_path / "manifest.json").read_text())
    wrong["files"][0]["domain"] = "retail"
    (tmp_path / "manifest.json").write_text(json.dumps(wrong))
    with pytest.raises(RuntimeError, match="domain"):
        corpus.load_file(
            corpus.load_manifest(tmp_path / "manifest.json").files[0],
            tmp_path / "data",
            check_hash=False,
        )


@pytest.mark.skipif(
    not os.environ.get("BENCH_TAU2_DATA"), reason="set BENCH_TAU2_DATA to the fetched corpus"
)
def test_real_corpus_matches_table_1():
    """Opt-in: the fetched files reproduce arXiv 2606.09863 Table 1 exactly."""
    m = corpus.load_manifest()
    got = {}
    for cf in corpus.iter_files(m, Path(os.environ["BENCH_TAU2_DATA"])):
        d = got.setdefault(cf.entry.domain, {"total": 0, "success": 0, "failure": 0})
        for sim in cf.simulations:
            d["total"] += 1
            d["success" if sim["reward_info"]["reward"] == 1 else "failure"] += 1
    assert got == m.expected
