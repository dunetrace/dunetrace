"""τ²-bench trajectory to Dunetrace events, through the real SDK.

The converter does not write events by hand. It drives the Python SDK's own
`dt.run()` / `RunContext` API with what the trajectory recorded and captures
what the SDK emits, so payload shapes, step indices, `call_id` pairing, field
caps, `agent_version` hashing and the in-path prompt-injection scan are the
production code's, not a copy of it. Four SDK-generated values are then
overwritten so a replay is deterministic: `run_id`, `event_id`, `timestamp`
(taken from the trajectory) and `run.started.payload.source_file` (made
repo-relative).

Run boundary: one run per user turn. A deployed chat agent instrumented with
the SDK's request middleware produces exactly this: each user message is a
request, the request is a run, and the middleware calls `final_answer()` when
it returns. `conversation_id` ties a trajectory's runs together.

Mapping, per trajectory:

- Assistant messages before the first user message are τ²-bench's canned
  greeting. They carry no usage and no provider response, so they are not LLM
  calls and are dropped.
- A turn is a block of user-side messages and the agent-side messages that
  answer it. User-side means role "user", plus tool messages the user
  requested (telecom's dual-control device actions). Those device actions are
  the user's, not the agent's, and never become agent tool events. The block's
  user text becomes `input_text`.
- The final user block, which carries τ²-bench's stop token and gets no
  answer, produces no run.
- Each assistant message is one `llm_called` / `llm_responded` pair: tokens
  from `usage`, finish_reason, reasoning tokens from the provider response in
  `raw_data`, output from `content`. A missing provider response leaves
  finish_reason unknown, which the SDK omits from the wire.
- Each tool call is a `tool_called` / `tool_responded` pair emitted when its
  result arrives, the order τ²-bench executes them in. Arguments are the raw
  provider string when the provider response has the call, else τ²-bench's
  parsed dict (which the SDK serialises as JSON).
- A tool call with no result (the run hit a step limit) is a `tool_called`
  with no response.
- A trajectory that ended on anything but a user or agent stop ends its last
  run with an exception inside `dt.run()`, so the SDK records `run.errored`.

Timestamps: `run.started` and the first `llm.called` take the user message's
time, `llm.responded` the assistant message's, `tool.called` the previous
event's and `tool.responded` the tool message's. Latencies are those gaps. A
missing or backwards timestamp is replaced by the previous one and counted.
"""

from __future__ import annotations

import datetime
import os
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from bench_tau2 import labels, tools
from bench_tau2._paths import REPO_ROOT, install_repo_paths
from bench_tau2.corpus import CorpusFile

install_repo_paths()

from dunetrace.client import DunetraceClient  # noqa: E402
from dunetrace.models import CallableExporter  # noqa: E402

CONVERTER_VERSION = "1"

# Fixed namespace for every id the converter derives. Changing it changes every
# run_id, so it is part of CONVERTER_VERSION's contract.
NAMESPACE = uuid.UUID("3f6b2a52-8f0e-5c5d-9b8e-2a1d7c4e9b10")

COMPLETED_TERMINATIONS = {"user_stop", "agent_stop"}


class Tau2RunTerminated(Exception):
    """Raised inside dt.run() for a turn τ²-bench ended abnormally, so the SDK
    records run.errored the way it would for an agent framework that raised."""


@dataclass
class ConvertedRun:
    run_id: str
    conversation_id: str
    agent_id: str
    agent_version: str
    submission: str
    domain: str
    sim_id: str
    task_id: str
    trial: Optional[int]
    turn_index: int
    n_turns: int
    reward: float
    label: str
    termination_reason: str
    end_ts: float
    events: List[dict]
    # step_index -> trajectory message indices that produced that step's events
    step_messages: Dict[int, List[int]]
    user_indices: List[int]
    agent_indices: List[int]
    stats: Dict[str, int] = field(default_factory=dict)


@dataclass
class _Call:
    """One tool call as τ²-bench recorded it, with its result if any."""

    message_index: int
    owner: str  # "user" | "agent"
    call_id: str
    name: str
    arguments: Any
    raw_arguments: Optional[str]
    result_index: Optional[int] = None


@dataclass
class FileContext:
    submission: str
    domain: str
    agent_id: str
    model: str
    system_prompt: str
    tools: List[str]


def file_context(corpus_file: CorpusFile) -> FileContext:
    entry, info = corpus_file.entry, corpus_file.info
    policy = (info.get("environment_info") or {}).get("policy") or ""
    return FileContext(
        submission=entry.submission,
        domain=entry.domain,
        agent_id=entry.agent_id,
        model=configured_model(info),
        system_prompt=tools.system_prompt(policy),
        tools=tools.available_tools(entry.domain, corpus_file.simulations),
    )


def configured_model(info: dict) -> str:
    """The agent model as configured, minus litellm's routing prefix
    ("anthropic/", "vertex_ai/", "openai/"). One value per file: τ²-bench sent
    the same model on every call, and the SDK records the requested model."""
    llm = ((info.get("agent_info") or {}).get("llm") or "unknown").strip()
    return llm.rsplit("/", 1)[-1] or "unknown"


def parse_ts(value: Any) -> Optional[float]:
    if not value or not isinstance(value, str):
        return None
    try:
        ts = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=datetime.timezone.utc)
    return ts.timestamp()


def _raw_choice(message: dict) -> dict:
    raw = message.get("raw_data")
    if not isinstance(raw, dict):
        return {}
    choices = raw.get("choices") or []
    return choices[0] if choices and isinstance(choices[0], dict) else {}


def _raw_arguments(message: dict) -> Dict[str, str]:
    """{tool_call_id: provider argument string} from the provider response."""
    out = {}
    for call in (_raw_choice(message).get("message") or {}).get("tool_calls") or []:
        fn = call.get("function") or {}
        if call.get("id") and isinstance(fn.get("arguments"), str):
            out[call["id"]] = fn["arguments"]
    return out


def _reasoning_tokens(message: dict) -> int:
    raw = message.get("raw_data")
    usage = (raw or {}).get("usage") if isinstance(raw, dict) else None
    details = (usage or {}).get("completion_tokens_details") or {}
    return int(details.get("reasoning_tokens") or 0)


def pair_tool_calls(messages: list) -> Tuple[List[_Call], Dict[int, _Call]]:
    """Every tool call in the trajectory, and the call each tool message answers.

    Matched by id. Older τ²-bench files leave tool-message ids empty; those take
    the oldest unanswered call, which is the order τ²-bench executes them in.
    """
    calls: List[_Call] = []
    pending: List[_Call] = []
    by_result: Dict[int, _Call] = {}
    for index, message in enumerate(messages):
        role = message.get("role")
        if role in ("assistant", "user"):
            raw = _raw_arguments(message) if role == "assistant" else {}
            for tc in message.get("tool_calls") or []:
                call = _Call(
                    message_index=index,
                    owner="agent" if role == "assistant" else "user",
                    call_id=tc.get("id") or "",
                    name=tc.get("name") or "unknown",
                    arguments=tc.get("arguments"),
                    raw_arguments=raw.get(tc.get("id") or ""),
                )
                calls.append(call)
                pending.append(call)
        elif role == "tool":
            match = None
            if message.get("id"):
                match = next((c for c in pending if c.call_id == message["id"]), None)
            if match is None and pending:
                match = pending[0]
            if match is not None:
                pending.remove(match)
                match.result_index = index
                by_result[index] = match
    return calls, by_result


def message_sides(messages: list, by_result: Dict[int, _Call]) -> List[str]:
    """ "user" | "agent" | "other" per message."""
    sides = []
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "user":
            sides.append("user")
        elif role == "assistant":
            sides.append("agent")
        elif role == "tool":
            requestor = message.get("requestor")
            if requestor in ("user", "assistant"):
                sides.append("user" if requestor == "user" else "agent")
            elif index in by_result:
                sides.append(by_result[index].owner)
            else:
                sides.append("agent")
        else:
            sides.append("other")
    return sides


def split_turns(sides: List[str]) -> Tuple[List[Tuple[List[int], List[int]]], Counter]:
    """[(user_indices, agent_indices)] for every turn the agent answered."""
    stats: Counter = Counter()
    turns: List[Tuple[List[int], List[int]]] = []
    user: List[int] = []
    agent: List[int] = []
    seen_user = False
    for index, side in enumerate(sides):
        if side == "other":
            continue
        if side == "user":
            if agent:
                turns.append((user, agent))
                user, agent = [], []
            user.append(index)
            seen_user = True
        elif not seen_user:
            stats["greeting_messages_dropped"] += 1
        else:
            agent.append(index)
    if agent:
        turns.append((user, agent))
    elif user:
        stats["unanswered_user_blocks_dropped"] += 1
    return turns, stats


class _Capture:
    """Records each SDK event with the trajectory time and message it maps to."""

    def __init__(self) -> None:
        self.events: list = []
        self.meta: List[Tuple[float, Optional[int]]] = []

    def handle(self, event) -> None:
        self.events.append(event)

    def mark(self, ts: float, message_index: Optional[int]) -> None:
        new = len(self.events) - len(self.meta)
        if new != 1:
            raise RuntimeError(f"expected one SDK event per call, got {new}")
        self.meta.append((ts, message_index))


class _Clock:
    """Trajectory time, never moving backwards. Fills gaps with the last time."""

    def __init__(self, start: float) -> None:
        self.now = start
        self.synthesized = 0

    def at(self, message: dict) -> float:
        ts = parse_ts(message.get("timestamp"))
        if ts is None or ts < self.now:
            self.synthesized += 1
            return self.now
        self.now = ts
        return ts


def _clean_sdk_environment() -> None:
    """Run the SDK on its defaults. Any DUNETRACE_* variable in the caller's
    shell (an API key, an output-text opt-out, a field cap) would otherwise
    change what is emitted or send it somewhere."""
    for key in list(os.environ):
        if key.startswith("DUNETRACE_"):
            del os.environ[key]


class Converter:
    def __init__(self) -> None:
        _clean_sdk_environment()
        self._capture = _Capture()
        self._client = DunetraceClient(
            api_key=None,
            # Never shed: a shed run is marked incomplete and its signals are
            # held in shadow, which would be a harness artifact.
            buffer_size=10**8,
            exporters=[CallableExporter(self._capture.handle)],
            implicit_runs=False,
        )
        self._client._ship = lambda batch: None  # nothing leaves the process

    def close(self) -> None:
        self._client.shutdown(timeout=5)

    def __enter__(self) -> "Converter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def convert_file(self, corpus_file: CorpusFile) -> List[ConvertedRun]:
        ctx = file_context(corpus_file)
        ids = [s.get("id") for s in corpus_file.simulations]
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"{corpus_file.entry.file_name}: duplicate simulation ids")
        runs: List[ConvertedRun] = []
        for sim in corpus_file.simulations:
            runs.extend(self.convert_trajectory(ctx, sim))
        return runs

    def convert_trajectory(self, ctx: FileContext, sim: dict) -> List[ConvertedRun]:
        messages = sim.get("messages") or []
        calls, by_result = pair_tool_calls(messages)
        sides = message_sides(messages, by_result)
        turns, convo_stats = split_turns(sides)
        convo_stats["user_tool_calls_dropped"] = sum(1 for c in calls if c.owner == "user")
        reward = (sim.get("reward_info") or {}).get("reward")
        sim_label = labels.label(reward, messages)
        termination = sim.get("termination_reason") or "unknown"
        sim_key = f"{ctx.submission}/{ctx.domain}/{sim.get('id')}"
        conversation_id = str(uuid.uuid5(NAMESPACE, sim_key))
        clock = _Clock(parse_ts(sim.get("start_time")) or 0.0)

        runs = []
        for turn_index, (user_idx, agent_idx) in enumerate(turns):
            final = turn_index == len(turns) - 1
            errored = final and termination not in COMPLETED_TERMINATIONS
            run = self._convert_turn(
                ctx,
                sim,
                messages,
                by_result,
                calls,
                user_idx,
                agent_idx,
                clock,
                errored,
                conversation_id,
            )
            run.run_id = str(uuid.uuid5(NAMESPACE, f"{sim_key}/{turn_index}"))
            run.conversation_id = conversation_id
            run.turn_index = turn_index
            run.n_turns = len(turns)
            run.reward = float(reward)
            run.label = sim_label
            run.termination_reason = termination
            run.sim_id = str(sim.get("id"))
            run.task_id = str(sim.get("task_id"))
            run.trial = sim.get("trial")
            _finalise_ids(run)
            runs.append(run)
        if runs:
            for key, value in convo_stats.items():
                runs[0].stats[key] = runs[0].stats.get(key, 0) + value
        return runs

    def _convert_turn(
        self,
        ctx: FileContext,
        sim: dict,
        messages: list,
        by_result: Dict[int, _Call],
        calls: List[_Call],
        user_idx: List[int],
        agent_idx: List[int],
        clock: _Clock,
        errored: bool,
        conversation_id: str,
    ) -> ConvertedRun:
        cap = self._capture
        cap.events, cap.meta = [], []
        stats: Counter = Counter()
        synthesized_before = clock.synthesized

        user_text = "\n".join(
            messages[i]["content"]
            for i in user_idx
            if messages[i].get("role") == "user" and (messages[i].get("content") or "").strip()
        )
        start_ts = clock.at(messages[user_idx[-1]]) if user_idx else clock.now
        agent_set = set(agent_idx)
        turn_calls = [c for c in calls if c.owner == "agent" and c.message_index in agent_set]

        try:
            with self._client.run(
                ctx.agent_id,
                user_input=user_text,
                system_prompt=ctx.system_prompt,
                model=ctx.model,
                tools=ctx.tools,
                conversation_id=conversation_id,
            ) as run:
                cap.mark(start_ts, user_idx[-1] if user_idx else None)
                prev_ts = start_ts
                for index in agent_idx:
                    message = messages[index]
                    if message.get("role") == "assistant":
                        ts = clock.at(message)
                        usage = message.get("usage") or {}
                        prompt_tokens = int(usage.get("prompt_tokens") or 0)
                        content = message.get("content") or ""
                        run.llm_called(model=ctx.model, prompt_tokens=prompt_tokens)
                        cap.mark(prev_ts, index)
                        run.llm_responded(
                            completion_tokens=int(usage.get("completion_tokens") or 0),
                            reasoning_tokens=_reasoning_tokens(message),
                            latency_ms=max(0, round((ts - prev_ts) * 1000)),
                            finish_reason=_raw_choice(message).get("finish_reason"),
                            output=content,
                            output_length=len(content),
                            prompt_tokens=prompt_tokens,
                        )
                        cap.mark(ts, index)
                        stats["llm_calls"] += 1
                        prev_ts = ts
                    elif index in by_result and by_result[index].owner == "agent":
                        call = by_result[index]
                        ts = clock.at(message)
                        self._tool_called(run, call, stats)
                        cap.mark(prev_ts, call.message_index)
                        content = message.get("content") or ""
                        failed = bool(message.get("error"))
                        run.tool_responded(
                            call.name,
                            success=not failed,
                            error=(content or "error") if failed else None,
                            output=content,
                            latency_ms=max(0, round((ts - prev_ts) * 1000)),
                        )
                        cap.mark(ts, index)
                        prev_ts = ts
                    else:
                        stats["agent_tool_messages_unmatched"] += 1
                for call in turn_calls:
                    if call.result_index is None or call.result_index not in agent_set:
                        self._tool_called(run, call, stats)
                        cap.mark(prev_ts, call.message_index)
                        stats["tool_calls_without_result"] += 1
                if errored:
                    raise Tau2RunTerminated(
                        f"tau2 termination_reason={sim.get('termination_reason')}"
                    )
                if messages[agent_idx[-1]].get("role") == "assistant":
                    run.final_answer()
                else:
                    stats["turns_ending_without_reply"] += 1
        except Tau2RunTerminated:
            pass
        cap.mark(prev_ts, agent_idx[-1])

        events: List[dict] = []
        step_messages: Dict[int, List[int]] = defaultdict(list)
        for event, (ts, message_index) in zip(cap.events, cap.meta):
            d = event.to_dict()
            d["timestamp"] = ts
            events.append(d)
            if message_index is not None and message_index not in step_messages[d["step_index"]]:
                step_messages[d["step_index"]].append(message_index)
        stats["synthesized_timestamps"] = clock.synthesized - synthesized_before
        stats["events"] = len(events)
        stats["capped_fields"] = _count_capped(events)
        if events[-1]["payload"].get("dropped_events"):
            raise RuntimeError("the SDK buffer shed events; the run would be marked incomplete")

        return ConvertedRun(
            run_id="",
            conversation_id="",
            agent_id=ctx.agent_id,
            agent_version=events[0]["agent_version"],
            submission=ctx.submission,
            domain=ctx.domain,
            sim_id="",
            task_id="",
            trial=None,
            turn_index=0,
            n_turns=0,
            reward=0.0,
            label="",
            termination_reason="",
            end_ts=events[-1]["timestamp"],
            events=events,
            step_messages=dict(step_messages),
            user_indices=list(user_idx),
            agent_indices=list(agent_idx),
            stats={k: v for k, v in stats.items() if v},
        )

    def _tool_called(self, run, call: _Call, stats: Counter) -> None:
        if call.raw_arguments is not None:
            run.tool_called(call.name, call.raw_arguments)
            stats["tool_args_raw"] += 1
        else:
            run.tool_called(call.name, call.arguments)
            stats["tool_args_parsed"] += 1


def _finalise_ids(run: ConvertedRun) -> None:
    """Replace the SDK's random run and event ids with ids derived from the
    trajectory, and the machine-specific caller path with a repo-relative one."""
    for seq, event in enumerate(run.events):
        event["run_id"] = run.run_id
        event["event_id"] = str(uuid.uuid5(NAMESPACE, f"{run.run_id}/{seq}"))
        payload = event["payload"]
        if event["event_type"] == "run.started" and payload.get("source_file"):
            payload["source_file"] = _repo_relative(payload["source_file"])


def _repo_relative(path: str) -> str:
    try:
        return os.path.relpath(path, REPO_ROOT)
    except ValueError:
        return os.path.basename(path)


def _count_capped(events: List[dict]) -> int:
    capped = 0
    for event in events:
        payload = event["payload"]
        for marker in ("args_truncated", "input_text_truncated", "system_prompt_truncated"):
            if payload.get(marker):
                capped += 1
        if event["event_type"] in ("llm.responded", "tool.responded"):
            if (payload.get("output_length") or 0) > len(payload.get("output") or ""):
                capped += 1
    return capped
