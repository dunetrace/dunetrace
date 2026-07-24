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

import pytest

try:
    from opentelemetry.sdk.trace.export import SpanExportResult

    from dunetrace import otel
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
            assert attrs["service.version"] == "9.9.9"
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

    def test_sampler_is_parent_based_ratio(self):
        cfg = OTelConfig(enabled=True, endpoint="http://localhost:4317", sampling_ratio=0.25)
        provider = otel.build_tracer_provider(cfg)
        try:
            # ParentBased wraps a TraceIdRatioBased root; assert we didn't get
            # the always-on default sampler.
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
