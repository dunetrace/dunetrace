"""
Multi-agent delegation-loop example (Capability 2).

Two agents — a planner and an executor — delegate to each other. Each agent is
its own dt.run(); because a nested run auto-inherits the active run's id as its
parent_run_id (no manual threading — see docs/multi-agent.md), the recursive
delegation builds a run graph: planner -> executor -> planner -> executor -> ...

The buggy version never converges, so the chain keeps growing and the
DELEGATION_LOOP detector fires once ~2.5 round trips accumulate (5 runs). The
"fixed" version converges after one hand-off and does not.

Run it against a local stack:
    docker compose up -d
    python examples/delegation_loop_agent.py

The signal shows up on the dashboard as a live alert (DELEGATION_LOOP is in
detector_svc's LIVE_DETECTORS).
"""

from __future__ import annotations

from dunetrace import Dunetrace

dt = Dunetrace(api_key="dt_dev_test", api_url="http://localhost:8001")

MAX_DEPTH = 8  # safety stop so the demo terminates even though the "agents" don't


def planner(goal: str, converge: bool, depth: int = 0) -> str:
    with dt.run("planner", user_input=goal):
        if depth >= MAX_DEPTH:
            return "gave up"
        # The planner always hands the goal to the executor.
        return executor(goal, converge, depth)


def executor(goal: str, converge: bool, depth: int = 0) -> str:
    with dt.run("executor", user_input=goal):
        if depth >= MAX_DEPTH:
            return "gave up"
        if converge:
            # Fixed: the executor finishes the work instead of bouncing it back.
            return f"done: {goal}"
        # Buggy: can't finish, delegates straight back to the planner -> loop.
        return planner(goal, converge, depth + 1)


def main() -> None:
    # Converging system: planner -> executor -> done. One hand-off, no loop.
    print("healthy run:", planner("summarize the Q3 report", converge=True))

    # Looping system: planner <-> executor forever (until MAX_DEPTH). The nested
    # runs form a planner->executor->planner->... chain that DELEGATION_LOOP
    # flags once it has gone around enough times.
    print("looping run:", planner("summarize the Q3 report", converge=False))

    dt.shutdown(timeout=5)
    print("\nDone. Check the looping run's leaf on the dashboard for DELEGATION_LOOP.")


if __name__ == "__main__":
    main()
