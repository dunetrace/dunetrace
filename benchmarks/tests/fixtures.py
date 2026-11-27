"""Synthetic trajectories in τ²-bench's results format.

The real corpus is not committed (no data license beyond the repo's), so the
tests build trajectories by hand with the same field names and shapes the
leaderboard files use, including the older format without tool-message ids
or requestors.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import List, Optional

BASE = datetime.datetime(2026, 1, 1, 12, 0, 0)


def ts(seconds: float) -> str:
    return (BASE + datetime.timedelta(seconds=seconds)).isoformat()


def epoch(seconds: float) -> float:
    return (
        (BASE + datetime.timedelta(seconds=seconds))
        .replace(tzinfo=datetime.timezone.utc)
        .timestamp()
    )


def greeting(at: float = 0.0) -> dict:
    return {
        "role": "assistant",
        "content": "Hi! How can I help you today?",
        "turn_idx": 0,
        "timestamp": ts(at),
    }


def user(content: str, at: float, tool_calls: Optional[list] = None) -> dict:
    return {
        "role": "user",
        "content": content,
        "tool_calls": tool_calls,
        "timestamp": ts(at),
        "usage": {"completion_tokens": 0, "prompt_tokens": 0},
    }


def call(call_id: str, name: str, arguments: dict, requestor: str = "assistant") -> dict:
    return {"id": call_id, "name": name, "arguments": arguments, "requestor": requestor}


def assistant(
    content: str,
    at: float,
    tool_calls: Optional[list] = None,
    *,
    prompt_tokens: int = 1000,
    completion_tokens: int = 50,
    finish_reason: Optional[str] = "stop",
    raw_arguments: Optional[dict] = None,
    raw: bool = True,
    reasoning_tokens: int = 0,
) -> dict:
    message = {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
        "timestamp": ts(at),
        "usage": {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens},
    }
    if raw:
        raw_calls = []
        for tc in tool_calls or []:
            args = (raw_arguments or {}).get(tc["id"], json.dumps(tc["arguments"]))
            raw_calls.append(
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": args},
                }
            )
        usage = {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens}
        if reasoning_tokens:
            usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
        message["raw_data"] = {
            "model": "test-model",
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": raw_calls or None,
                    },
                }
            ],
            "usage": usage,
        }
    return message


def tool(
    call_id: str, content: str, at: float, *, error: bool = False, requestor: str = "assistant"
) -> dict:
    return {
        "id": call_id,
        "role": "tool",
        "content": content,
        "requestor": requestor,
        "error": error,
        "timestamp": ts(at),
    }


def simulation(
    sim_id: str,
    messages: List[dict],
    *,
    reward: Optional[float] = 1.0,
    termination: str = "user_stop",
    task_id: str = "0",
    trial: int = 0,
) -> dict:
    return {
        "id": sim_id,
        "task_id": task_id,
        "trial": trial,
        "start_time": messages[0]["timestamp"] if messages else ts(0),
        "end_time": messages[-1]["timestamp"] if messages else ts(0),
        "duration": 1.0,
        "termination_reason": termination,
        "reward_info": None if reward is None else {"reward": reward},
        "messages": messages,
    }


def results(
    domain: str, simulations: List[dict], *, llm: str = "openai/test-model", policy: str = "POLICY"
) -> dict:
    return {
        "timestamp": ts(0),
        "info": {
            "git_commit": "test",
            "num_trials": 1,
            "max_steps": 200,
            "max_errors": 10,
            "user_info": {"implementation": "user_simulator", "llm": "test-user"},
            "agent_info": {"implementation": "llm_agent", "llm": llm},
            "environment_info": {"domain_name": domain, "policy": policy, "tool_defs": None},
            "seed": 1,
        },
        "tasks": [],
        "simulations": simulations,
    }


def write_corpus(root: Path, files: List[tuple]) -> Path:
    """Write [(submission, domain, results_dict)] plus a manifest. Returns the manifest path."""
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    entries = []
    for submission, domain, body in files:
        name = f"{submission}_{domain}.json"
        path = data / f"{submission}__{name}"
        raw = json.dumps(body).encode()
        path.write_bytes(raw)
        entries.append(
            {
                "submission": submission,
                "domain": domain,
                "s3_key": f"submissions/{submission}/trajectories/{name}",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
                "tau2_git_commit": "test",
                "agent_llm": body["info"]["agent_info"]["llm"],
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps({"source_bucket": "file://", "expected": {}, "files": entries}, indent=2)
    )
    return manifest


# ── Canned conversations ────────────────────────────────────────────────────


def two_turn_airline(sim_id: str = "sim-two-turn") -> dict:
    """Greeting, a lookup turn with two parallel tool calls, a cancel turn, STOP."""
    return simulation(
        sim_id,
        [
            greeting(0),
            user("I want to cancel a trip. My user id is mia_li_3668.", 2),
            assistant(
                "Let me look you up.",
                5,
                [
                    call("c1", "get_user_details", {"user_id": "mia_li_3668"}),
                    call("c2", "get_reservation_details", {"reservation_id": "ABC123"}),
                ],
                finish_reason="tool_calls",
            ),
            tool("c1", '{"user_id": "mia_li_3668", "reservations": ["ABC123"]}', 5.5),
            tool("c2", '{"reservation_id": "ABC123", "status": "active"}', 6.0),
            assistant("I found reservation ABC123. Shall I cancel it?", 9, prompt_tokens=1500),
            user("Yes please.", 12),
            assistant(
                "",
                14,
                [call("c3", "cancel_reservation", {"reservation_id": "ABC123"})],
                finish_reason="tool_calls",
                prompt_tokens=1800,
            ),
            tool("c3", '{"reservation_id": "ABC123", "status": "cancelled"}', 14.2),
            assistant("Your reservation has been cancelled.", 16, prompt_tokens=1900),
            user("Thanks! ###STOP###", 18),
        ],
    )


def retry_storm_retail(sim_id: str = "sim-retry") -> dict:
    """One turn where the same tool fails three times in a row, then a refusal."""
    messages = [greeting(0), user("Look up my order #W0001.", 1)]
    t = 2.0
    for n in range(3):
        cid = f"r{n}"
        messages.append(
            assistant(
                "",
                t,
                [call(cid, "get_order_details", {"order_id": f"#W000{n + 1}"})],
                finish_reason="tool_calls",
            )
        )
        messages.append(tool(cid, "Error: order not found", t + 0.1, error=True))
        t += 2
    messages.append(assistant("I'm sorry, I cannot find that order.", t))
    messages.append(user("###STOP###", t + 1))
    return simulation(sim_id, messages, reward=0.0)


def old_format_telecom(sim_id: str = "sim-old") -> dict:
    """τ²-bench 0.1.x shape: tool messages without ids or requestors, plus
    a user device action (dual control) that must not become an agent event."""
    messages = [
        greeting(0),
        user("", 1, tool_calls=[{"id": "", "name": "check_status_bar", "arguments": {}}]),
        {"role": "tool", "content": "No service", "timestamp": ts(1.2), "error": False},
        user("My phone shows no service.", 2),
        assistant(
            "Let me check your line.",
            4,
            [
                {
                    "id": "",
                    "name": "get_customer_by_phone",
                    "arguments": {"phone_number": "555-0100"},
                }
            ],
            finish_reason="tool_calls",
            raw=False,
        ),
        {"role": "tool", "content": '{"customer_id": "C1"}', "timestamp": ts(4.5), "error": False},
        assistant("Your line is active. Please reboot your phone.", 7, raw=False),
        user("###STOP###", 9),
    ]
    return simulation(sim_id, messages)


def lookup_conversation(sim_id: str, lookups: int, start: float) -> dict:
    """One turn with `lookups` sequential reservation lookups, then a reply.

    A corpus of these with mostly short runs and a few long ones gives the
    detector worker enough clean history (20+ runs) to form step-count
    baselines, so baseline-dependent detectors fire and resume is tested across
    that state.
    """
    t = start
    # The ids are in the request, so TOOL_ARGUMENT_FABRICATION (live) stays quiet.
    # Baselines only learn from clean runs: a run with any live signal is excluded.
    ids = " ".join(f"R{n:04d}" for n in range(lookups))
    messages = [greeting(t), user(f"Check my reservations {ids} please.", t + 1)]
    t += 2
    for n in range(lookups):
        cid = f"{sim_id}-l{n}"
        messages.append(
            assistant(
                "",
                t,
                [call(cid, "get_reservation_details", {"reservation_id": f"R{n:04d}"})],
                finish_reason="tool_calls",
            )
        )
        messages.append(
            tool(cid, json.dumps({"reservation_id": f"R{n:04d}", "status": "active"}), t + 0.2)
        )
        t += 1
    messages.append(assistant("All your reservations are active.", t + 1))
    messages.append(user("###STOP###", t + 2))
    return simulation(sim_id, messages)
