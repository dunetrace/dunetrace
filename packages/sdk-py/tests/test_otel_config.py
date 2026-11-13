"""
Phase 1 tests for dunetrace.otel — the OTel export config/bootstrap module.

Covers env-var parsing, TracerProvider construction, failure isolation (the
circuit breaker and the never-raise contract), and backward compat (SDK
untouched when OTel is not configured). No real OTLP endpoint is contacted: the
provider is built and inspected in-process, and the circuit breaker is tested
against a fake exporter.
"""

from __future__ import annotations

import unittest
import uuid

import pytest

try:
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from dunetrace import otel
    from dunetrace.integrations.otel import DunetraceOTelExporter
    from dunetrace.models import AgentEvent, EventType
    from dunetrace.otel import OTelConfig, _CircuitBreakerExporter

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

if not _OTEL_AVAILABLE:
    raise unittest.SkipTest("opentelemetry not installed — skipping OTel config tests")


_OTEL_ENV_VARS = (
    "DUNETRACE_OTEL_ENABLED",
    "DUNETRACE_OTEL_ENDPOINT",
    "DUNETRACE_OTEL_HEADERS",
    "DUNETRACE_OTEL_PROTOCOL",
    "DUNETRACE_OTEL_SERVICE_NAME",
    "DUNETRACE_OTEL_SAMPLING_RATIO",
    "DUNETRACE_OTEL_CAPTURE_CONTENT",
    "DUNETRACE_ORG_ID",
)


@pytest.fixture(autouse=True)
def _clean_otel_env(monkeypatch):
    """Every test starts from a clean env and a reset global tracer, and leaves
    them clean, so tests don't leak enabled state into each other."""
    for var in _OTEL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    otel._reset_for_tests()
    yield
    otel._reset_for_tests()


# ── Fake exporter for circuit-breaker tests ─────────────────────────────────────


class _FakeExporter:
    """SpanExporter stub whose export() result is scripted. Records how many
    times export() was actually invoked so we can prove the open circuit stops
    calling through."""

    def __init__(self, result=SpanExportResult.SUCCESS, raises: bool = False):
        self.result = result
        self.raises = raises
        self.calls = 0

    def export(self, spans):
        self.calls += 1
        if self.raises:
            raise RuntimeError("collector unreachable")
        return self.result

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=30_000):
        return True


class _Clock:
    """Injectable monotonic clock for deterministic cooldown tests."""

    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


# ── Config parsing ──────────────────────────────────────────────────────────────


class TestConfigParsing:
    def test_defaults_when_env_empty(self):
        cfg = OTelConfig.from_env()
        assert cfg.enabled is False
        assert cfg.endpoint == ""
        assert cfg.headers == {}
        assert cfg.protocol == "grpc"
        assert cfg.service_name == "dunetrace"
        assert cfg.sampling_ratio == 1.0
        assert cfg.org_id == ""

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_enabled_truthy_variants(self, monkeypatch, value):
        monkeypatch.setenv("DUNETRACE_OTEL_ENABLED", value)
        assert OTelConfig.from_env().enabled is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_enabled_falsy_variants(self, monkeypatch, value):
        monkeypatch.setenv("DUNETRACE_OTEL_ENABLED", value)
        assert OTelConfig.from_env().enabled is False

    def test_endpoint_read_and_stripped(self, monkeypatch):
        monkeypatch.setenv("DUNETRACE_OTEL_ENDPOINT", "  https://otlp.example.com:4317  ")
        assert OTelConfig.from_env().endpoint == "https://otlp.example.com:4317"

    def test_headers_parsed(self, monkeypatch):
        monkeypatch.setenv("DUNETRACE_OTEL_HEADERS", "DD-API-KEY=abc123, X-Extra = v ")
        assert OTelConfig.from_env().headers == {"DD-API-KEY": "abc123", "X-Extra": "v"}

    def test_headers_ignores_malformed_entries(self, monkeypatch):
        monkeypatch.setenv("DUNETRACE_OTEL_HEADERS", "good=1,,broken,=novalue,another=2")
        assert OTelConfig.from_env().headers == {"good": "1", "another": "2"}

    def test_protocol_http(self, monkeypatch):
        monkeypatch.setenv("DUNETRACE_OTEL_PROTOCOL", "http/protobuf")
        assert OTelConfig.from_env().protocol == "http/protobuf"

    def test_protocol_invalid_falls_back_to_grpc(self, monkeypatch):
        monkeypatch.setenv("DUNETRACE_OTEL_PROTOCOL", "smoke-signals")
        assert OTelConfig.from_env().protocol == "grpc"

    def test_service_name_custom(self, monkeypatch):
        monkeypatch.setenv("DUNETRACE_OTEL_SERVICE_NAME", "my-agent")
        assert OTelConfig.from_env().service_name == "my-agent"

    def test_service_name_blank_falls_back(self, monkeypatch):
        monkeypatch.setenv("DUNETRACE_OTEL_SERVICE_NAME", "   ")
        assert OTelConfig.from_env().service_name == "dunetrace"

    @pytest.mark.parametrize(
        "raw,expected",
        [("0.5", 0.5), ("0", 0.0), ("1", 1.0), ("2.0", 1.0), ("-1", 0.0), ("abc", 1.0)],
    )
    def test_sampling_ratio_parse_and_clamp(self, monkeypatch, raw, expected):
        monkeypatch.setenv("DUNETRACE_OTEL_SAMPLING_RATIO", raw)
        assert OTelConfig.from_env().sampling_ratio == expected

    def test_org_id_read(self, monkeypatch):
        monkeypatch.setenv("DUNETRACE_ORG_ID", "org_42")
        assert OTelConfig.from_env().org_id == "org_42"

    def test_capture_content_defaults_true(self):
        assert OTelConfig.from_env().capture_content is True

    def test_capture_content_disabled(self, monkeypatch):
        monkeypatch.setenv("DUNETRACE_OTEL_CAPTURE_CONTENT", "false")
        assert OTelConfig.from_env().capture_content is False

    def test_service_version_is_sdk_version(self):
        # Non-empty and stable; exact string depends on install vs source.
        assert isinstance(OTelConfig.from_env().service_version, str)
        assert OTelConfig.from_env().service_version != ""

    def test_overrides_win_over_env(self, monkeypatch):
        monkeypatch.setenv("DUNETRACE_OTEL_ENABLED", "1")
        monkeypatch.setenv("DUNETRACE_OTEL_PROTOCOL", "grpc")
        cfg = OTelConfig.from_env(enabled=False, protocol="http/protobuf")
        assert cfg.enabled is False
        assert cfg.protocol == "http/protobuf"


# ── Provider construction ───────────────────────────────────────────────────────


class TestBuildProvider:
    def test_none_when_disabled(self):
        cfg = OTelConfig(enabled=False, endpoint="http://localhost:4317")
        assert otel.build_tracer_provider(cfg) is None

    def test_none_when_no_endpoint(self):
        cfg = OTelConfig(enabled=True, endpoint="")
        assert otel.build_tracer_provider(cfg) is None

    def test_returns_provider_when_configured_grpc(self):
        cfg = OTelConfig(enabled=True, endpoint="http://localhost:4317", protocol="grpc")
        provider = otel.build_tracer_provider(cfg)
        try:
            assert provider is not None
        finally:
            if provider is not None:
                provider.shutdown()

    def test_returns_provider_when_configured_http(self):
        cfg = OTelConfig(
            enabled=True,
            endpoint="http://localhost:4318/v1/traces",
            protocol="http/protobuf",
        )
        provider = otel.build_tracer_provider(cfg)
        try:
            assert provider is not None
        finally:
            if provider is not None:
                provider.shutdown()

    def test_resource_has_service_name_and_version(self):
        cfg = OTelConfig(
            enabled=True,
            endpoint="http://localhost:4317",
            service_name="checkout-agent",
            service_version="9.9.9",
        )
        provider = otel.build_tracer_provider(cfg)
        try:
            attrs = provider.resource.attributes
            assert attrs["service.name"] == "checkout-agent"
            assert attrs["dunetrace.sdk.version"] == "9.9.9"
            assert "service.version" not in attrs
        finally:
            provider.shutdown()

    def test_resource_has_org_id_only_when_set(self):
        with_org = otel.build_tracer_provider(
            OTelConfig(enabled=True, endpoint="http://localhost:4317", org_id="org_7")
        )
        without_org = otel.build_tracer_provider(
            OTelConfig(enabled=True, endpoint="http://localhost:4317")
        )
        try:
            assert with_org.resource.attributes["dunetrace.org_id"] == "org_7"
            assert "dunetrace.org_id" not in without_org.resource.attributes
        finally:
            with_org.shutdown()
            without_org.shutdown()

    def _run_span_count(self, monkeypatch, sampling_ratio, run_id):
        # Keep the production BatchSpanProcessor off the network: without this,
        # provider.shutdown() flushes a sampled span to the real OTLP endpoint.
        monkeypatch.setattr(otel, "_build_span_exporter", lambda _config: InMemorySpanExporter())
        cfg = OTelConfig(
            enabled=True, endpoint="http://localhost:4317", sampling_ratio=sampling_ratio
        )
        provider = otel.build_tracer_provider(cfg)
        memory = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(memory))
        try:
            # Wired the way client.py wires the env-driven exporter.
            exporter = DunetraceOTelExporter(
                tracer_provider=provider, sampling_ratio=cfg.sampling_ratio
            )
            for et, payload in (
                (EventType.RUN_STARTED, {"model": "gpt-4o", "tools": []}),
                (EventType.RUN_COMPLETED, {}),
            ):
                exporter.handle(
                    AgentEvent(
                        event_type=et,
                        run_id=run_id,
                        agent_id="a",
                        agent_version="v",
                        step_index=0,
                        payload=payload,
                    )
                )
            return len(memory.get_finished_spans())
        finally:
            provider.shutdown()

    def test_sampling_ratio_zero_drops_run_spans(self, monkeypatch):
        # DunetraceOTelExporter starts every run under a sampled remote parent,
        # so a provider-level ParentBased sampler alone kept every run whatever
        # the ratio. The exporter now decides per run.
        assert self._run_span_count(monkeypatch, 0.0, str(uuid.uuid4())) == 0

    def test_sampling_ratio_one_keeps_run_spans(self, monkeypatch):
        assert self._run_span_count(monkeypatch, 1.0, str(uuid.uuid4())) == 1

    def test_partial_ratio_follows_run_sampled(self, monkeypatch):
        for _ in range(20):
            run_id = str(uuid.uuid4())
            expected = 1 if otel.run_sampled(run_id, 0.5) else 0
            assert self._run_span_count(monkeypatch, 0.5, run_id) == expected

    def test_provider_sampler_still_respects_parents(self):
        cfg = OTelConfig(enabled=True, endpoint="http://localhost:4317", sampling_ratio=0.25)
        provider = otel.build_tracer_provider(cfg)
        try:
            # Customer spans opened on get_tracer() under their own trace keep
            # standard OTel semantics: children follow the parent's decision.
            assert provider.sampler.__class__.__name__ == "ParentBased"
        finally:
            provider.shutdown()

    def test_build_swallows_exporter_construction_error(self, monkeypatch):
        def boom(_config):
            raise RuntimeError("no exporter for you")

        monkeypatch.setattr(otel, "_build_span_exporter", boom)
        cfg = OTelConfig(enabled=True, endpoint="http://localhost:4317")
        # Must degrade to None, not raise.
        assert otel.build_tracer_provider(cfg) is None


# ── Circuit breaker / failure isolation ─────────────────────────────────────────


class TestCircuitBreaker:
    def test_passes_through_success(self):
        fake = _FakeExporter(result=SpanExportResult.SUCCESS)
        breaker = _CircuitBreakerExporter(fake)
        assert breaker.export(["span"]) == SpanExportResult.SUCCESS
        assert fake.calls == 1

    def test_opens_after_threshold_failures_and_stops_calling_through(self):
        fake = _FakeExporter(result=SpanExportResult.FAILURE)
        breaker = _CircuitBreakerExporter(fake)

        for _ in range(_CircuitBreakerExporter.FAILURE_THRESHOLD):
            assert breaker.export(["span"]) == SpanExportResult.FAILURE
        calls_at_trip = fake.calls
        assert calls_at_trip == _CircuitBreakerExporter.FAILURE_THRESHOLD

        # Circuit is now open: further exports are dropped without touching the
        # wrapped exporter.
        assert breaker.export(["span"]) == SpanExportResult.FAILURE
        assert fake.calls == calls_at_trip  # no new call

    def test_raising_exporter_is_swallowed(self):
        fake = _FakeExporter(raises=True)
        breaker = _CircuitBreakerExporter(fake)
        # Must not propagate the RuntimeError.
        assert breaker.export(["span"]) == SpanExportResult.FAILURE

    def test_recovers_after_cooldown(self):
        clock = _Clock(0.0)
        fake = _FakeExporter(result=SpanExportResult.FAILURE)
        breaker = _CircuitBreakerExporter(fake, now=clock)

        for _ in range(_CircuitBreakerExporter.FAILURE_THRESHOLD):
            breaker.export(["span"])
        calls_at_trip = fake.calls

        # Still within cooldown: dropped, no new call.
        clock.t = _CircuitBreakerExporter.COOLDOWN - 1
        breaker.export(["span"])
        assert fake.calls == calls_at_trip

        # Past cooldown: it tries the wrapped exporter again.
        clock.t = _CircuitBreakerExporter.COOLDOWN + 1
        fake.result = SpanExportResult.SUCCESS
        assert breaker.export(["span"]) == SpanExportResult.SUCCESS
        assert fake.calls == calls_at_trip + 1

    def test_success_resets_failure_count(self):
        clock = _Clock(0.0)
        fake = _FakeExporter(result=SpanExportResult.FAILURE)
        breaker = _CircuitBreakerExporter(fake, now=clock)

        # Four failures (one short of the threshold), then a success clears them.
        for _ in range(_CircuitBreakerExporter.FAILURE_THRESHOLD - 1):
            breaker.export(["span"])
        fake.result = SpanExportResult.SUCCESS
        breaker.export(["span"])

        # Four more failures should not trip (count was reset by the success).
        fake.result = SpanExportResult.FAILURE
        for _ in range(_CircuitBreakerExporter.FAILURE_THRESHOLD - 1):
            assert breaker.export(["span"]) == SpanExportResult.FAILURE
        calls = fake.calls
        # Circuit still closed: the next export calls through rather than dropping.
        breaker.export(["span"])
        assert fake.calls == calls + 1


# ── Global init / tracer accessor ────────────────────────────────────────────────


class TestInit:
    def test_disabled_returns_false_and_no_tracer(self):
        assert otel.init() is False
        assert otel.get_tracer() is None
        assert otel.is_enabled() is False

    def test_enabled_sets_tracer(self, monkeypatch):
        monkeypatch.setenv("DUNETRACE_OTEL_ENABLED", "1")
        monkeypatch.setenv("DUNETRACE_OTEL_ENDPOINT", "http://localhost:4317")
        try:
            assert otel.init() is True
            assert otel.get_tracer() is not None
            assert otel.is_enabled() is True
        finally:
            otel.shutdown()

    def test_init_idempotent(self, monkeypatch):
        monkeypatch.setenv("DUNETRACE_OTEL_ENABLED", "1")
        monkeypatch.setenv("DUNETRACE_OTEL_ENDPOINT", "http://localhost:4317")
        try:
            assert otel.init() is True
            provider_first = otel.get_tracer_provider()
            assert otel.init() is True
            assert otel.get_tracer_provider() is provider_first  # first provider wins
        finally:
            otel.shutdown()

    def test_shutdown_clears_state(self, monkeypatch):
        monkeypatch.setenv("DUNETRACE_OTEL_ENABLED", "1")
        monkeypatch.setenv("DUNETRACE_OTEL_ENDPOINT", "http://localhost:4317")
        otel.init()
        otel.shutdown()
        assert otel.get_tracer() is None
        assert otel.is_enabled() is False


# ── Backward compatibility ───────────────────────────────────────────────────────


class TestBackwardCompat:
    def test_no_env_means_disabled(self):
        # With a clean env, nothing about OTel activates.
        assert otel.init() is False
        assert otel.is_enabled() is False

    def test_client_construction_unaffected_when_otel_unset(self):
        from dunetrace import Dunetrace
        from dunetrace.emitters import NoopBatchingEmitter

        dt = Dunetrace(emitter=NoopBatchingEmitter())
        try:
            with dt.run("compat-agent", user_input="hi") as run:
                run.llm_called("gpt-4o", prompt_tokens=5)
                run.llm_responded(completion_tokens=3, finish_reason="stop")
                run.final_answer()
        finally:
            dt.shutdown()
        # OTel stayed disabled throughout.
        assert otel.is_enabled() is False


class TestHttpTracesUrl:
    def test_bare_base_url_gets_the_traces_path(self):
        assert (
            otel._http_traces_url("https://ingest.eu.signoz.cloud:443")
            == "https://ingest.eu.signoz.cloud:443/v1/traces"
        )
        assert otel._http_traces_url("http://collector:4318/") == "http://collector:4318/v1/traces"

    def test_explicit_path_is_kept(self):
        assert otel._http_traces_url("http://c:4318/v1/traces") == "http://c:4318/v1/traces"
        assert otel._http_traces_url("https://x/otlp/v1/traces") == "https://x/otlp/v1/traces"

    def test_non_url_is_left_for_the_exporter(self):
        assert otel._http_traces_url("collector:4318") == "collector:4318"
# ── run_sampled: the per-run decision both SDKs and the detector share ───────


class TestRunSampled:
    def test_ratio_is_a_real_probability_over_uuid_run_ids(self):
        """The bug this replaced: TraceIdRatioBased read the UUID's low 64 bits,
        whose top two are always the variant bits 10, so 0.1/0.25/0.5 exported
        0% and 0.9 exported 100%."""
        from dunetrace.otel import run_sampled

        ids = [str(uuid.uuid4()) for _ in range(4000)]
        for ratio in (0.1, 0.25, 0.5, 0.75, 0.9):
            kept = sum(run_sampled(r, ratio) for r in ids) / len(ids)
            assert abs(kept - ratio) < 0.04, (ratio, kept)

    def test_uuid1_run_ids_are_not_clustered(self):
        from dunetrace.otel import run_sampled

        ids = [str(uuid.uuid1()) for _ in range(2000)]
        kept = sum(run_sampled(r, 0.3) for r in ids) / len(ids)
        assert abs(kept - 0.3) < 0.05

    def test_deterministic_and_monotonic(self):
        from dunetrace.otel import run_sampled

        rid = str(uuid.uuid4())
        assert run_sampled(rid, 0.4) == run_sampled(rid, 0.4)
        # A run kept at a lower ratio is kept at every higher one.
        kept_at = [r for r in (0.1, 0.3, 0.5, 0.7, 0.9) if run_sampled(rid, r)]
        if kept_at:
            assert all(run_sampled(rid, r) for r in (0.1, 0.3, 0.5, 0.7, 0.9) if r >= kept_at[0])

    def test_edges(self):
        from dunetrace.otel import run_sampled

        assert run_sampled("anything", 1.0) is True
        assert run_sampled("anything", 0.0) is False
        assert run_sampled("not-a-uuid", 0.999999) in (True, False)

    def test_known_vector(self):
        """Pins the algorithm so the TS SDK can assert the same values."""
        from dunetrace.otel import run_sampled

        # sha256 first 8 bytes: 0xdb8055e0e0307d5a and 0x11e594f481958c10.
        assert run_sampled("00000000-0000-4000-8000-000000000000", 0.5) is False
        assert run_sampled("00000000-0000-4000-8000-000000000001", 0.5) is True


class TestParseRatioNonFinite:
    def test_nan_and_inf_fall_back_to_the_default(self):
        from dunetrace.otel import _parse_ratio

        assert _parse_ratio("nan") == 1.0
        assert _parse_ratio("inf") == 1.0
        assert _parse_ratio("-inf") == 1.0
        assert _parse_ratio("0.3") == 0.3
