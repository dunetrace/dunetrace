import fixtures
from bench_tau2 import detector_config, tools
from bench_tau2._paths import REPO_ROOT, install_repo_paths

install_repo_paths()

from dunetrace_schemas.detector_config import load_detector_kwargs  # noqa: E402


def test_irreversible_tools_are_taus_write_annotations():
    assert tools.irreversible_tools("airline") == [
        "book_reservation",
        "cancel_reservation",
        "send_certificate",
        "update_reservation_baggages",
        "update_reservation_flights",
        "update_reservation_passengers",
    ]
    assert "get_order_details" not in tools.irreversible_tools("retail")
    assert "cancel_pending_order" in tools.irreversible_tools("retail")
    assert "refuel_data" in tools.irreversible_tools("telecom")
    assert "transfer_to_human_agents" not in tools.irreversible_tools("telecom")


def test_available_tools_adds_tools_that_answered_successfully_only():
    sim = fixtures.simulation(
        "s",
        [
            fixtures.greeting(0),
            fixtures.user("hi", 1),
            fixtures.assistant(
                "",
                2,
                [
                    fixtures.call("a", "get_item_details", {"item_id": "1"}),
                    fixtures.call("b", "$INVENTED_TOOL", {}),
                ],
            ),
            fixtures.tool("a", '{"item_id": "1"}', 3),
            fixtures.tool("b", "Error: Tool '$INVENTED_TOOL' not found.", 3, error=True),
            fixtures.assistant("done", 4),
        ],
    )
    available = tools.available_tools("retail", [sim])
    assert "get_item_details" in available
    assert "$INVENTED_TOOL" not in available
    assert set(tools.source_agent_tools("retail")) <= set(available)


def test_system_prompt_is_taus_template():
    prompt = tools.system_prompt("THE POLICY")
    assert prompt.startswith("<instructions>\nYou are a customer service agent")
    assert prompt.endswith("<policy>\nTHE POLICY\n</policy>")


def test_bench_config_is_shipped_file_plus_one_value_per_agent(tmp_path):
    agents = [("tau2.airline.x", "airline"), ("tau2.telecom.y", "telecom")]
    path = tmp_path / "detectors.bench.yml"
    detector_config.write(path, agents)
    shipped_text = (REPO_ROOT / "detectors.yml").read_text()
    assert path.read_text().startswith(shipped_text)

    shipped = load_detector_kwargs(str(REPO_ROOT / "detectors.yml"))
    bench = load_detector_kwargs(str(path))
    for category, section in shipped.items():
        assert bench[category] == section, f"shipped section {category} changed"
    assert bench["tau2.airline.x"] == {
        "unresolved_ambiguity": {"IRREVERSIBLE_TOOLS": tools.irreversible_tools("airline")}
    }
    assert bench["tau2.telecom.y"] == {
        "unresolved_ambiguity": {"IRREVERSIBLE_TOOLS": tools.irreversible_tools("telecom")}
    }
