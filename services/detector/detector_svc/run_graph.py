"""
Delegation run-graph construction for the DELEGATION_LOOP detector.

A multi-agent run graph is a forest: each run has at most one parent (its
`parent_run_id`, auto-threaded by the SDK — see client.py), and run ids are
unique, so the *run* graph can never contain a cycle. The cycle lives in the
*agent* dimension: agent A's run spawns a run of agent B, whose run spawns a run
of A again, and so on — A → B → A → B, an infinite mutual-delegation loop.

This module walks a run's `parent_run_id` chain up to the root (one lightweight
lineage fetch per hop) into an ordered ancestor chain. The DELEGATION_LOOP
detector derives the agent-delegation graph from that chain and runs cycle
detection on it. Keeping the traversal here — pure but for the injected async
fetch function — makes it unit-testable without a database.

The traversal is defensive: a `visited` set breaks any run-id-level cycle that
should be structurally impossible but could arise from corrupt data, and
`max_depth` caps work on a pathologically deep chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional

# Default cap on how many ancestors to walk. A genuine delegation loop reveals
# itself within a handful of hops; anything past this is either pathological or
# not a loop, and not worth unbounded sequential fetches on the worker.
DEFAULT_MAX_DEPTH = 20


@dataclass(frozen=True)
class RunNode:
    run_id: str
    agent_id: str
    agent_version: str
    parent_run_id: Optional[str]


# An async fetcher: run_id -> lineage dict (run_id/agent_id/agent_version/
# parent_run_id) or None. In the worker this is db.fetch_run_lineage; tests
# inject a dict-backed fake.
LineageFetcher = Callable[[str], Awaitable[Optional[dict]]]


def _node_from_lineage(lineage: dict) -> Optional[RunNode]:
    run_id = lineage.get("run_id")
    agent_id = lineage.get("agent_id")
    if not run_id or not agent_id:
        return None
    return RunNode(
        run_id=run_id,
        agent_id=agent_id,
        agent_version=lineage.get("agent_version") or "",
        parent_run_id=lineage.get("parent_run_id"),
    )


async def build_ancestor_chain(
    start: RunNode,
    fetch_lineage: LineageFetcher,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> List[RunNode]:
    """Walk `start`'s parent chain to the root.

    Returns nodes ordered leaf-first: ``[start, parent, grandparent, …, root]``.
    A single-run (no parent) start returns ``[start]``. Stops at the root, at
    `max_depth` hops, on a missing ancestor (parent id with no `run.started`
    on record — e.g. not yet ingested), or on a repeated run id (corrupt data).
    """
    chain: List[RunNode] = [start]
    visited = {start.run_id}
    parent_id = start.parent_run_id
    depth = 0

    while parent_id and depth < max_depth:
        if parent_id in visited:
            break  # run-id-level cycle (should be impossible) — stop defensively
        lineage = await fetch_lineage(parent_id)
        if lineage is None:
            break  # ancestor not on record (not yet ingested, or pruned)
        node = _node_from_lineage(lineage)
        if node is None:
            break
        chain.append(node)
        visited.add(node.run_id)
        parent_id = node.parent_run_id
        depth += 1

    return chain


def build_agent_delegation_edges(chain: List[RunNode]) -> dict[str, set[str]]:
    """Derive the directed agent-delegation graph from an ancestor chain.

    For each adjacent (child, parent) pair in the leaf-first chain, the parent
    run spawned the child run, so the parent's agent *delegated to* the child's
    agent — edge ``parent_agent -> child_agent``. When agents repeat along the
    chain (A ← B ← A → …) this yields a cycle in the agent graph.
    """
    edges: dict[str, set[str]] = {}
    for child, parent in zip(chain, chain[1:]):
        edges.setdefault(parent.agent_id, set()).add(child.agent_id)
    return edges


def agent_sequence(chain: List[RunNode]) -> List[str]:
    """The agent ids along the chain, root-first — the delegation order that
    actually happened (root delegated to its child, …, down to the leaf). This
    preserves repetition (unlike the deduplicated edge set), so the detector can
    measure how many times a loop went around, not just that it exists."""
    return [n.agent_id for n in reversed(chain)]


def find_cycle(edges: dict[str, set[str]]) -> Optional[List[str]]:
    """DFS for a directed cycle in the agent-delegation graph.

    Returns the cycle as an ordered agent-id list that starts and ends on the
    same agent (e.g. ``['A', 'B', 'A']`` for A → B → A), or None if the graph is
    acyclic. Standard three-colour (white/grey/black) DFS: reaching a grey
    (currently-on-the-recursion-stack) node closes a cycle.
    """
    WHITE, GREY, BLACK = 0, 1, 2
    colour: dict[str, int] = {}
    for src, succs in edges.items():
        colour.setdefault(src, WHITE)
        for s in succs:
            colour.setdefault(s, WHITE)

    path: List[str] = []

    def dfs(node: str) -> Optional[List[str]]:
        colour[node] = GREY
        path.append(node)
        for nxt in edges.get(node, ()):
            if colour.get(nxt, WHITE) == GREY:
                return path[path.index(nxt) :] + [nxt]  # close the cycle
            if colour.get(nxt, WHITE) == WHITE:
                found = dfs(nxt)
                if found:
                    return found
        path.pop()
        colour[node] = BLACK
        return None

    for start in list(colour):
        if colour[start] == WHITE:
            found = dfs(start)
            if found:
                return found
    return None
