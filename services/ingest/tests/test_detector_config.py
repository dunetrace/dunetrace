"""
GET /v1/detector-config — server-authoritative thresholds for the SDK's
client-side detector pass. DB is mocked; the yaml fixture is a temp file.

Run:
    PYTHONPATH=packages/sdk-py:packages/schemas-py:services/ingest \\
        python -m pytest services/ingest/tests/test_detector_config.py -v
"""

from __future__ import annotations

import builtins
import os
import sys
import tempfile
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient, ASGITransport

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytestmark = pytest.mark.asyncio

YAML = """
default:
  tool_loop:
    threshold: 2
    window: 5
  retry_storm:
    threshold: 3
    severity: high
  empty_llm_response:
    max_cost_ns: 250000
web-research:
  tool_loop:
    threshold: 5
  reasoning_stall:
    inflation_factor: 3.0
custom_detectors:
  evaluation_budget_ms: 25
"""

BASELINES = {
    "p75_steps": 12.0,
    "p75_latency_tool_ms": 800.0,
    "p75_latency_llm_ms": 2100.0,
    "p75_token_growth": 1.4,
    "p75_llm_tool_ratio": 2.0,
    "p75_total_tokens": 9000.0,
    "p75_duration_s": 31.5,
}


@pytest.fixture
def yaml_path():
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False)
    f.write(YAML)
    f.close()
    yield f.name
    os.unlink(f.name)


@pytest.fixture
def env(monkeypatch, yaml_path):
    """Mocked DB + fixture yaml + reset caches. Yields the mocks so a test can
    assert on calls or swap side effects."""
    import ingest_svc.detector_config as dc

    monkeypatch.setattr("ingest_svc.db.postgres._pool", object())
    monkeypatch.setattr("ingest_svc.db.postgres.init_pool", AsyncMock())
    monkeypatch.setattr("ingest_svc.db.postgres.close_pool", AsyncMock())
    monkeypatch.setattr("ingest_svc.db.postgres.ensure_schema", AsyncMock())
    monkeypatch.setattr("ingest_svc.db.postgres.check_db", AsyncMock(return_value="ok"))
    monkeypatch.setattr(
        "ingest_svc.routers.ingest.verify_api_key", AsyncMock(return_value="org-test")
    )
    monkeypatch.setattr("ingest_svc.config.settings.DETECTOR_CONFIG", yaml_path)
    packs = AsyncMock(return_value=["voice"])
    latest = AsyncMock(return_value="v-latest")
    baselines = AsyncMock(return_value=dict(BASELINES))
    monkeypatch.setattr(dc, "fetch_org_enabled_packs", packs)
    monkeypatch.setattr(dc, "fetch_latest_agent_version", latest)
    monkeypatch.setattr(dc, "fetch_agent_baselines", baselines)
    dc.reset_caches()
    yield {"packs": packs, "latest": latest, "baselines": baselines, "dc": dc}
    dc.reset_caches()


@pytest.fixture
async def client(env):
    from ingest_svc.main import create_app

    async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test") as c:
        yield c


AUTH = {"Authorization": "Bearer dt_live_header"}


async def _get(client, **params):
    return await client.get("/v1/detector-config", params=params, headers=AUTH)


# ── Auth: identical to /v1/policies ───────────────────────────────────────────


class TestAuth:
    async def test_bearer_header_authenticates(self, client):
        r = await _get(client, agent_id="agent-xyz")
        assert r.status_code == 200
        assert r.json()["agent_id"] == "agent-xyz"

    async def test_x_dunetrace_api_key_header_authenticates(self, client):
        r = await client.get(
            "/v1/detector-config",
            params={"agent_id": "agent-xyz"},
            headers={"X-Dunetrace-API-Key": "dt_live_header"},
        )
        assert r.status_code == 200

    async def test_query_param_still_accepted_but_warned(self, client, caplog):
        with caplog.at_level("WARNING", logger="dunetrace.ingest"):
            r = await client.get(
                "/v1/detector-config", params={"agent_id": "agent-xyz", "api_key": "dt_live_legacy"}
            )
        assert r.status_code == 200
        assert any("Deprecated: /v1/detector-config" in rec.getMessage() for rec in caplog.records)

    async def test_header_wins_when_both_are_present(self, client, monkeypatch):
        seen = []

        async def _spy(key):
            seen.append(key)
            return "org-1"

        monkeypatch.setattr("ingest_svc.routers.ingest.verify_api_key", _spy)
        await client.get(
            "/v1/detector-config",
            params={"agent_id": "agent-xyz", "api_key": "from-query"},
            headers={"Authorization": "Bearer from-header"},
        )
        assert seen == ["from-header"]

    async def test_no_key_at_all_is_rejected(self, client):
        r = await client.get("/v1/detector-config", params={"agent_id": "agent-xyz"})
        assert r.status_code == 401

    async def test_invalid_key_is_rejected(self, client, monkeypatch):
        monkeypatch.setattr(
            "ingest_svc.routers.ingest.verify_api_key", AsyncMock(return_value=None)
        )
        r = await _get(client, agent_id="agent-xyz")
        assert r.status_code == 401

    async def test_agent_id_is_required(self, client):
        r = await client.get("/v1/detector-config", headers=AUTH)
        assert r.status_code == 422

    async def test_trusted_gateway_path_uses_org_header(self, client, monkeypatch, env):
        monkeypatch.setattr("ingest_svc.routers.ingest.is_trusted", lambda request: True)
        monkeypatch.setattr("ingest_svc.main.is_trusted", lambda request: True)
        r = await client.get(
            "/v1/detector-config",
            params={"agent_id": "agent-xyz"},
            headers={"Authorization": "Bearer anything", "x-org-id": "org-from-gateway"},
        )
        assert r.status_code == 200
        env["packs"].assert_awaited_once_with("org-from-gateway")

    async def test_org_scoped_reads(self, client, env):
        await _get(client, agent_id="agent-xyz", agent_version="v1")
        env["packs"].assert_awaited_once_with("org-test")
        env["baselines"].assert_awaited_once_with("org-test", "agent-xyz", "v1")


# ── Shape ─────────────────────────────────────────────────────────────────────


class TestShape:
    async def test_envelope(self, client):
        body = (await _get(client, agent_id="agent-xyz")).json()
        assert body["schema_version"] == 1
        assert body["agent_id"] == "agent-xyz"
        assert body["ttl_s"] == 60
        assert isinstance(body["generated_at"], float)
        assert set(body) == {
            "schema_version",
            "agent_id",
            "agent_version",
            "generated_at",
            "ttl_s",
            "detectors",
            "packs",
            "baselines",
        }

    async def test_default_section_for_unknown_category_mapped_to_uppercase_kwargs(self, client):
        body = (await _get(client, agent_id="agent-xyz")).json()
        assert body["detectors"] == {
            "tool_loop": {"THRESHOLD": 2, "WINDOW": 5},
            "retry_storm": {"THRESHOLD": 3, "SEVERITY": "HIGH"},
            "empty_llm_response": {"MAX_COST_NS": 250000},
        }

    async def test_category_section_merged_over_default(self, client):
        body = (await _get(client, agent_id="web-research")).json()
        assert body["detectors"] == {
            "tool_loop": {"THRESHOLD": 5, "WINDOW": 5},
            "retry_storm": {"THRESHOLD": 3, "SEVERITY": "HIGH"},
            "empty_llm_response": {"MAX_COST_NS": 250000},
            "reasoning_stall": {"INFLATION_FACTOR": 3.0},
        }

    async def test_merge_matches_detector_worker_kwargs(self, client):
        """What the endpoint serves for a category must be exactly the kwargs
        detector_svc's _instantiate would pass for it. Same parser, same merge
        — checked here through the detector's own adapter and merge rule."""
        from dunetrace_schemas.detector_config import effective_detector_kwargs
        from ingest_svc.detector_config import get_yaml_config

        body = (await _get(client, agent_id="web-research")).json()
        config = get_yaml_config()
        # detector_svc.detectors.get_detectors: {**default_cfg.get(key, {}), **category_cfg.get(key, {})}
        expected = {}
        for key in set(config["default"]) | set(config["web-research"]):
            kw = {**config["default"].get(key, {}), **config["web-research"].get(key, {})}
            expected[key] = {k: (v.value if hasattr(v, "value") else v) for k, v in kw.items()}
        assert body["detectors"] == expected
        assert effective_detector_kwargs(config, "web-research").keys() == expected.keys()

    async def test_packs_and_baselines(self, client):
        body = (await _get(client, agent_id="agent-xyz", agent_version="v1")).json()
        assert body["packs"] == ["voice"]
        assert body["baselines"] == BASELINES
        assert body["agent_version"] == "v1"

    async def test_omitted_agent_version_resolves_latest_recorded(self, client, env):
        body = (await _get(client, agent_id="agent-xyz")).json()
        env["latest"].assert_awaited_once_with("org-test", "agent-xyz")
        env["baselines"].assert_awaited_once_with("org-test", "agent-xyz", "v-latest")
        assert body["agent_version"] == "v-latest"

    async def test_agent_with_no_recorded_runs_has_null_baselines(self, client, env):
        env["latest"].return_value = None
        body = (await _get(client, agent_id="brand-new")).json()
        assert body["baselines"] is None
        assert body["agent_version"] is None
        env["baselines"].assert_not_awaited()

    async def test_insufficient_history_is_null(self, client, env):
        env["baselines"].return_value = None
        body = (await _get(client, agent_id="agent-xyz", agent_version="v1")).json()
        assert body["baselines"] is None
        assert body["detectors"]  # yaml overrides still served


# ── Degradation: never fail the request ───────────────────────────────────────


class TestFailOpen:
    async def test_missing_yaml_file_gives_empty_detectors(self, client, monkeypatch, env):
        monkeypatch.setattr(
            "ingest_svc.config.settings.DETECTOR_CONFIG", "/nonexistent/detectors.yml"
        )
        env["dc"].reset_caches()
        body = (await _get(client, agent_id="agent-xyz")).json()
        assert body["detectors"] == {}
        assert body["packs"] == ["voice"]

    async def test_missing_pyyaml_gives_empty_detectors_with_one_warning(
        self, client, monkeypatch, env, caplog
    ):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "yaml" or name.startswith("yaml."):
                raise ImportError("No module named 'yaml'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setitem(sys.modules, "yaml", None)
        monkeypatch.setattr(builtins, "__import__", fake_import)
        env["dc"].reset_caches()
        with caplog.at_level("WARNING", logger="dunetrace.detector_config"):
            r1 = await _get(client, agent_id="agent-xyz")
            r2 = await _get(client, agent_id="agent-xyz")
        assert r1.status_code == r2.status_code == 200
        assert r1.json()["detectors"] == {}
        warnings = [r for r in caplog.records if "PyYAML not installed" in r.getMessage()]
        assert len(warnings) == 1  # parsed once, warned once

    async def test_baseline_query_failure_is_null_with_warning(self, client, env, caplog):
        env["baselines"].side_effect = RuntimeError("relation run_baseline_metrics does not exist")
        with caplog.at_level("WARNING", logger="dunetrace.ingest.detector_config"):
            r = await _get(client, agent_id="agent-xyz", agent_version="v1")
        assert r.status_code == 200
        assert r.json()["baselines"] is None
        assert r.json()["detectors"]["tool_loop"] == {"THRESHOLD": 2, "WINDOW": 5}
        assert any("baseline read failed" in rec.getMessage() for rec in caplog.records)

    async def test_latest_version_failure_is_null_with_warning(self, client, env, caplog):
        env["latest"].side_effect = RuntimeError("boom")
        with caplog.at_level("WARNING", logger="dunetrace.ingest.detector_config"):
            r = await _get(client, agent_id="agent-xyz")
        assert r.status_code == 200
        assert r.json()["baselines"] is None
        assert any("baseline read failed" in rec.getMessage() for rec in caplog.records)

    async def test_packs_query_failure_is_empty_with_warning(self, client, env, caplog):
        env["packs"].side_effect = RuntimeError("relation org_enabled_packs does not exist")
        with caplog.at_level("WARNING", logger="dunetrace.ingest.detector_config"):
            r = await _get(client, agent_id="agent-xyz")
        assert r.status_code == 200
        assert r.json()["packs"] == []
        assert r.json()["baselines"] == BASELINES
        assert any("enabled packs" in rec.getMessage() for rec in caplog.records)


# ── Caching ───────────────────────────────────────────────────────────────────


class TestCaching:
    async def test_packs_and_baselines_cached_per_org_agent_for_ttl(self, client, env):
        for _ in range(3):
            await _get(client, agent_id="agent-xyz", agent_version="v1")
        assert env["packs"].await_count == 1
        assert env["baselines"].await_count == 1

    async def test_cache_keyed_by_agent_and_version(self, client, env):
        await _get(client, agent_id="agent-a", agent_version="v1")
        await _get(client, agent_id="agent-b", agent_version="v1")
        await _get(client, agent_id="agent-a", agent_version="v2")
        assert env["packs"].await_count == 1  # per org
        assert env["baselines"].await_count == 3  # per (org, agent, version)

    async def test_expired_entry_is_refetched(self, client, env, monkeypatch):
        import ingest_svc.detector_config as dc

        await _get(client, agent_id="agent-xyz", agent_version="v1")
        now = [0.0]
        base = __import__("time").monotonic()
        now[0] = base + dc.TTL_S + 1
        monkeypatch.setattr(dc.time, "monotonic", lambda: now[0])
        await _get(client, agent_id="agent-xyz", agent_version="v1")
        assert env["packs"].await_count == 2
        assert env["baselines"].await_count == 2

    async def test_yaml_parsed_once(self, client, env, monkeypatch):
        import ingest_svc.detector_config as dc

        spy = AsyncMock()  # never awaited; just a call counter via wraps below
        calls = []
        real = dc.load_detector_kwargs

        def counting(*a, **kw):
            calls.append(1)
            return real(*a, **kw)

        monkeypatch.setattr(dc, "load_detector_kwargs", counting)
        dc.reset_caches()
        await _get(client, agent_id="agent-xyz")
        await _get(client, agent_id="web-research")
        assert len(calls) == 1
        del spy

    async def test_not_rate_limited(self, client):
        """Like /v1/policies: an exhausted ingest quota must not block config."""
        from ingest_svc.rate_limiter import get_limiter

        # clear(), not reassignment: _windows is a defaultdict(deque) and every
        # later request test relies on that (a plain dict raises KeyError).
        get_limiter()._windows.clear()
        get_limiter()._agent_windows.clear()
        r = await _get(client, agent_id="agent-xyz")
        assert r.status_code == 200
