"""τ²-bench tool definitions and the agent's system prompt.

The trajectory files leave `environment_info.tool_defs` null, so the tool list
an agent was given has to come from τ²-bench source. data/tau2_tools.json holds
the tool names and their read/write annotations, and the agent prompt template,
taken from the newest public commit the corpus ran on (see its `source` block).

Three of the corpus's commits are not public and two runs did not record one.
Older τ²-bench versions had tools the public source lacks (retail's
get_item_details answers successfully 191 times). A tool that returned a
successful result in a file demonstrably existed in that environment, so
available_tools() adds those per file. A call to a tool the source does not
define that only ever errored is the agent inventing a name, and is not added.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List

TOOLS_PATH = Path(__file__).resolve().parent / "data" / "tau2_tools.json"


@lru_cache(maxsize=1)
def load_tool_metadata() -> dict:
    return json.loads(TOOLS_PATH.read_text())


def source_agent_tools(domain: str) -> Dict[str, str]:
    """{tool_name: "read" | "write" | "generic" | "think"} for the domain's agent."""
    return dict(load_tool_metadata()["agent_tools"][domain])


def irreversible_tools(domain: str) -> List[str]:
    """The agent tools τ²-bench itself annotates as ToolType.WRITE.

    Used as UNRESOLVED_AMBIGUITY's IRREVERSIBLE_TOOLS. The list comes from the
    benchmark's own annotations so the choice is not made by looking at
    results.
    """
    return sorted(name for name, kind in source_agent_tools(domain).items() if kind == "write")


def available_tools(domain: str, simulations: Iterable[dict]) -> List[str]:
    """Source tools for the domain, plus any tool with a successful call in the file."""
    names = set(source_agent_tools(domain))
    for sim in simulations:
        names |= _successful_agent_tools(sim.get("messages") or [])
    return sorted(names)


def _successful_agent_tools(messages: list) -> set:
    calls: Dict[str, str] = {}
    found = set()
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if call.get("id"):
                    calls[call["id"]] = call.get("name") or ""
        elif message.get("role") == "tool" and not message.get("error"):
            if message.get("requestor", "assistant") != "assistant":
                continue
            name = calls.get(message.get("id") or "")
            if name:
                found.add(name)
    return found


def system_prompt(policy: str) -> str:
    """The prompt τ²-bench's LLMAgent sends: its fixed instruction plus the domain policy."""
    meta = load_tool_metadata()
    return meta["system_prompt_template"].format(
        agent_instruction=meta["agent_instruction"], domain_policy=policy or ""
    )
