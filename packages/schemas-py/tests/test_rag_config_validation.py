import pytest

from dunetrace_schemas.detector_config import load_detector_kwargs


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("min_retrievals", "true"),
        ("min_retrievals", "0"),
        ("min_retrievals", "1.5"),
        ("min_retrievals", "'3'"),
        ("drop_threshold", "false"),
        ("drop_threshold", "-.1"),
        ("drop_threshold", "1.1"),
        ("drop_threshold", ".nan"),
        ("drop_threshold", ".inf"),
        ("drop_threshold", "'0.3'"),
    ],
)
def test_invalid_rag_override_is_ignored(tmp_path, caplog, parameter, value):
    config = tmp_path / "detectors.yml"
    config.write_text(
        f"default:\n  rag_quality_degradation:\n    {parameter}: {value}\n",
        encoding="utf-8",
    )
    assert load_detector_kwargs(str(config)) == {"default": {}}
    assert "Ignoring override" in caplog.text


@pytest.mark.parametrize("threshold", [0, 0.3, 1])
def test_valid_rag_overrides_retained(tmp_path, threshold):
    config = tmp_path / "detectors.yml"
    config.write_text(
        "default:\n  rag_quality_degradation:\n"
        f"    min_retrievals: 4\n    drop_threshold: {threshold}\n",
        encoding="utf-8",
    )
    assert load_detector_kwargs(str(config))["default"]["rag_quality_degradation"] == {
        "MIN_RETRIEVALS": 4,
        "DROP_THRESHOLD": threshold,
    }
