"""Converter: turn boundaries, event shapes, step indices, timestamps, and the
lossless round trip through the production build_run_state."""

import copy
import json
import os

import pytest

import fixtures
from bench_tau2 import corpus, labels
from bench_tau2.convert import Converter, file_context
from bench_tau2.verify import verify_run


def convert(domain, sims, *, submission="test-sub", llm="openai/test-model", policy="POLICY"):
    body = fixtures.results(domain, sims, llm=llm, policy=policy)
    entry = corpus.FileEntry(submission, domain, "k", "0" * 64, 0, "test", llm)
    cf = corpus.CorpusFile(
        entry=entry,
        info=body["info"],
        simulations=[s for s in sims if not corpus.is_excluded(s)],
        excluded_count=0,
    )
    with Converter() as converter:
        runs = converter.convert_file(cf)
    return cf, runs


def shape(run):
    return [(e["step_index"], e["event_type"]) for e in run.events]


def by_type(run, event_type):
    return [e for e in run.events if e["event_type"] == event_type]


# ── Turn boundaries ─────────────────────────────────────────────────────────


def test_one_run_per_answered_user_turn():
    cf, runs = convert("airline", [fixtures.two_turn_airline()])
    assert [r.turn_index for r in runs] == [0, 1]
    assert all(r.n_turns == 2 for r in runs)
    # The greeting is not an LLM call and the STOP turn gets no run.
    assert runs[0].stats["greeting_messages_dropped"] == 1
    assert runs[0].stats["unanswered_user_blocks_dropped"] == 1
    assert by_type(runs[0], "run.started")[0]["payload"]["input_text"].startswith(
        "I want to cancel"
    )
    assert by_type(runs[1], "run.started")[0]["payload"]["input_text"] == "Yes please."
    # One conversation, distinct runs, one agent.
    assert runs[0].conversation_id == runs[1].conversation_id
    assert runs[0].run_id != runs[1].run_id
    assert {e["conversation_id"] for r in runs for e in r.events} == {runs[0].conversation_id}
    assert {r.agent_id for r in runs} == {"tau2.airline.test-sub"}


def test_sdk_event_shape_and_step_indices():
    _, runs = convert("airline", [fixtures.two_turn_airline()])
    assert shape(runs[0]) == [
        (0, "run.started"),
        (1, "llm.called"),
        (1, "llm.responded"),
        (2, "tool.called"),
        (2, "tool.responded"),
        (3, "tool.called"),
        (3, "tool.responded"),
        (4, "llm.called"),
        (4, "llm.responded"),
        (4, "run.completed"),
    ]
    # Each step points back at the trajectory messages that produced it.
    assert runs[0].step_messages == {0: [1], 1: [2], 2: [2, 3], 3: [2, 4], 4: [5]}
    assert runs[1].step_messages == {0: [6], 1: [7], 2: [7, 8], 3: [9]}
    called = by_type(runs[0], "llm.called")
    responded = by_type(runs[0], "llm.responded")
    assert [c["payload"]["call_id"] for c in called] == [0, 1]
    assert [r["payload"]["call_id"] for r in responded] == [0, 1]
    completed = by_type(runs[0], "run.completed")[0]["payload"]
    assert completed["exit_reason"] == "final_answer"
    assert completed["tool_call_count"] == 2
    assert "dropped_events" not in completed


def test_tool_calls_pair_with_their_own_results():
    _, runs = convert("airline", [fixtures.two_turn_airline()])
    tools_called = [
        (e["payload"]["tool_name"], json.loads(e["payload"]["args"]))
        for e in by_type(runs[0], "tool.called")
    ]
    outputs = [e["payload"]["output"] for e in by_type(runs[0], "tool.responded")]
    assert tools_called == [
        ("get_user_details", {"user_id": "mia_li_3668"}),
        ("get_reservation_details", {"reservation_id": "ABC123"}),
    ]
    assert '"mia_li_3668"' in outputs[0] and '"ABC123"' in outputs[1]


def test_run_started_carries_tau_prompt_tools_and_model():
    _, runs = convert(
        "airline", [fixtures.two_turn_airline()], llm="anthropic/claude-x", policy="FLY SAFE"
    )
    payload = by_type(runs[0], "run.started")[0]["payload"]
    assert payload["model"] == "claude-x"  # litellm routing prefix removed
    assert payload["system_prompt"].endswith("<policy>\nFLY SAFE\n</policy>")
    assert "cancel_reservation" in payload["tools"]
    assert (
        payload["source_file"] == "benchmarks/bench_tau2/convert.py"
    )  # repo-relative, not a home path
    assert all(c["payload"]["model"] == "claude-x" for c in by_type(runs[0], "llm.called"))


# ── Field mapping ───────────────────────────────────────────────────────────


def test_tokens_finish_reason_and_reasoning_tokens():
    sim = fixtures.simulation(
        "s",
        [
            fixtures.greeting(0),
            fixtures.user("hi", 1),
            fixtures.assistant(
                "hello", 2, prompt_tokens=4321, completion_tokens=12, reasoning_tokens=7
            ),
            fixtures.user("###STOP###", 3),
        ],
    )
    _, runs = convert("retail", [sim])
    called = by_type(runs[0], "llm.called")[0]["payload"]
    responded = by_type(runs[0], "llm.responded")[0]["payload"]
    assert called["prompt_tokens"] == 4321
    assert responded["prompt_tokens"] == 4321
    assert responded["completion_tokens"] == 12
    assert responded["reasoning_tokens"] == 7
    assert responded["finish_reason"] == "stop"
    assert responded["output"] == "hello"
    assert responded["latency_ms"] == 1000


def test_missing_provider_response_leaves_finish_reason_unknown():
    _, runs = convert("telecom", [fixtures.old_format_telecom()])
    for responded in by_type(runs[0], "llm.responded"):
        assert "finish_reason" not in responded["payload"]


def test_raw_provider_arguments_are_kept_verbatim():
    """A malformed argument string must reach MALFORMED_TOOL_CALL as written."""
    bad = "{'reservation_id': 'ABC123', }"
    sim = fixtures.simulation(
        "s",
        [
            fixtures.greeting(0),
            fixtures.user("cancel ABC123", 1),
            fixtures.assistant(
                "",
                2,
                [fixtures.call("c1", "cancel_reservation", {"reservation_id": "ABC123"})],
                raw_arguments={"c1": bad},
                finish_reason="tool_calls",
            ),
            fixtures.tool("c1", "ok", 3),
            fixtures.assistant("Done.", 4),
            fixtures.user("###STOP###", 5),
        ],
    )
    _, runs = convert("airline", [sim])
    assert by_type(runs[0], "tool.called")[0]["payload"]["args"] == bad
    assert runs[0].stats["tool_args_raw"] == 1


def test_failed_tool_result_is_a_failed_tool_call():
    _, runs = convert("retail", [fixtures.retry_storm_retail()])
    responded = by_type(runs[0], "tool.responded")
    assert [r["payload"]["success"] for r in responded] == [False, False, False]
    assert all(r["payload"]["error"] == "Error: order not found" for r in responded)
    assert runs[0].label == labels.HONEST_FAILURE


def test_user_device_actions_never_become_agent_events():
    _, runs = convert("telecom", [fixtures.old_format_telecom()])
    run = runs[0]
    assert [e["payload"]["tool_name"] for e in by_type(run, "tool.called")] == [
        "get_customer_by_phone"
    ]
    assert by_type(run, "run.started")[0]["payload"]["input_text"] == "My phone shows no service."
    assert run.stats["user_tool_calls_dropped"] == 1
    # Old format: no ids, no requestor. Paired by position, parsed args serialised as JSON.
    assert json.loads(by_type(run, "tool.called")[0]["payload"]["args"]) == {
        "phone_number": "555-0100"
    }
    assert by_type(run, "tool.responded")[0]["payload"]["output"] == '{"customer_id": "C1"}'
    assert run.stats["tool_args_parsed"] == 1


def test_abnormal_termination_errors_only_the_last_run():
    sim = fixtures.two_turn_airline("s")
    sim["termination_reason"] = "too_many_errors"
    sim["messages"] = sim["messages"][:-2]  # ends on the cancel tool result, no reply, no STOP
    _, runs = convert("airline", [sim])
    assert runs[0].events[-1]["event_type"] == "run.completed"
    assert runs[1].events[-1]["event_type"] == "run.errored"
    assert runs[1].events[-1]["payload"]["error_type"] == "Tau2RunTerminated"


def test_tool_call_without_result_is_recorded_unanswered():
    sim = fixtures.simulation(
        "s",
        [
            fixtures.greeting(0),
            fixtures.user("cancel it", 1),
            fixtures.assistant(
                "", 2, [fixtures.call("c1", "cancel_reservation", {"reservation_id": "X"})]
            ),
        ],
        termination="max_steps",
        reward=0.0,
    )
    _, runs = convert("airline", [sim])
    assert shape(runs[0])[-2:] == [(2, "tool.called"), (2, "run.errored")]
    assert runs[0].stats["tool_calls_without_result"] == 1


# ── Timestamps ──────────────────────────────────────────────────────────────


def test_timestamps_come_from_the_trajectory():
    _, runs = convert("airline", [fixtures.two_turn_airline()])
    stamps = {(e["step_index"], e["event_type"]): e["timestamp"] for e in runs[0].events}
    assert stamps[(0, "run.started")] == fixtures.epoch(2)
    assert stamps[(1, "llm.called")] == fixtures.epoch(2)
    assert stamps[(1, "llm.responded")] == fixtures.epoch(5)
    assert stamps[(2, "tool.called")] == fixtures.epoch(5)
    assert stamps[(2, "tool.responded")] == fixtures.epoch(5.5)
    assert stamps[(3, "tool.called")] == fixtures.epoch(5.5)
    assert stamps[(3, "tool.responded")] == fixtures.epoch(6)
    assert stamps[(4, "run.completed")] == fixtures.epoch(9)
    latencies = [e["payload"]["latency_ms"] for e in runs[0].events if "latency_ms" in e["payload"]]
    assert latencies == [3000, 500, 500, 3000]
    assert "synthesized_timestamps" not in runs[0].stats


def test_missing_or_backwards_timestamps_are_clamped_and_counted():
    sim = fixtures.two_turn_airline()
    sim["messages"][2]["timestamp"] = fixtures.ts(1)  # assistant before the user message
    del sim["messages"][3]["timestamp"]
    _, runs = convert("airline", [sim])
    stamps = [e["timestamp"] for e in runs[0].events]
    assert stamps == sorted(stamps)
    assert runs[0].stats["synthesized_timestamps"] == 2


# ── Determinism, environment, caps ──────────────────────────────────────────


def test_conversion_is_deterministic():
    sims = [fixtures.two_turn_airline(), fixtures.retry_storm_retail()]
    _, first = convert("airline", copy.deepcopy(sims))
    _, second = convert("airline", copy.deepcopy(sims))
    assert json.dumps([r.events for r in first], sort_keys=True) == json.dumps(
        [r.events for r in second], sort_keys=True
    )
    event_ids = [e["event_id"] for r in first for e in r.events]
    assert len(event_ids) == len(set(event_ids))


def test_caller_environment_cannot_change_what_is_emitted(monkeypatch):
    monkeypatch.setenv("DUNETRACE_OMIT_LLM_OUTPUT_TEXT", "1")
    monkeypatch.setenv("DUNETRACE_API_KEY", "dt_live_should_not_be_used")
    _, runs = convert("airline", [fixtures.two_turn_airline()])
    assert by_type(runs[0], "llm.responded")[1]["payload"]["output"].startswith(
        "I found reservation"
    )
    assert "DUNETRACE_API_KEY" not in os.environ


def test_sdk_field_cap_applies_and_is_counted():
    sim = fixtures.two_turn_airline()
    sim["messages"][3]["content"] = "x" * 9000
    cf, runs = convert("airline", [sim])
    responded = by_type(runs[0], "tool.responded")[0]["payload"]
    assert responded["output_length"] == 9000
    assert len(responded["output"]) == 8192
    assert runs[0].stats["capped_fields"] == 1
    assert verify_run(runs[0], file_context(cf), sim) == []


# ── Lossless round trip ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "domain, make",
    [
        ("airline", fixtures.two_turn_airline),
        ("retail", fixtures.retry_storm_retail),
        ("telecom", fixtures.old_format_telecom),
    ],
)
def test_round_trip_is_lossless(domain, make):
    sim = make()
    cf, runs = convert(domain, [sim])
    ctx = file_context(cf)
    assert runs
    for run in runs:
        assert verify_run(run, ctx, sim) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: by_type(r, "tool.responded")[0]["payload"].update(output="tampered"),
        lambda r: by_type(r, "tool.responded")[0]["payload"].update(success=False),
        lambda r: by_type(r, "llm.responded")[0]["payload"].update(output="tampered"),
        lambda r: by_type(r, "llm.responded")[0]["payload"].update(prompt_tokens=1),
        lambda r: by_type(r, "run.started")[0]["payload"].update(input_text="tampered"),
        lambda r: r.events.pop(3),
    ],
)
def test_verifier_catches_a_lossy_conversion(mutate):
    sim = fixtures.two_turn_airline()
    cf, runs = convert("airline", [sim])
    broken = copy.deepcopy(runs[0])
    mutate(broken)
    assert verify_run(broken, file_context(cf), sim) != []
