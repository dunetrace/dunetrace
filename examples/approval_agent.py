"""
Human-in-the-loop approval example (Capability 2).

A `require_approval` policy gates the `wire_money` tool: when the agent tries to
call it, the run blocks until a human approves in Slack or the dashboard — or
the request times out, which is fail-closed (the tool is blocked, not allowed).

Run it against a local stack:
    docker compose up -d
    python examples/approval_agent.py

Then approve or deny the request on the dashboard's **Approvals** page (or in
Slack, if the org's Slack integration is configured). Approvals use the Customer
API (:8002); this script uses a dev placeholder key and degrades gracefully if
the backend isn't reachable.
"""

from __future__ import annotations

import urllib.error

from dunetrace import ApprovalDenied, Dunetrace

dt = Dunetrace(api_key="dt_dev_test", api_url="http://localhost:8002")

# Gate the wire_money tool behind human approval. A short timeout so this demo
# doesn't block forever if nobody decides (fail-closed → ApprovalDenied).
dt.add_policy(
    name="approve-wires",
    condition={"trigger": "before_tool_call", "operator": "eq", "value": "wire_money"},
    action={"type": "require_approval", "params": {"timeout_s": 60}},
)


@dt.tool("wire_money")
def wire_money(to: str, amount: int) -> str:
    # This body only runs if a human approves the call.
    print(f"  >> wiring ${amount} to {to}")
    return f"transferred ${amount} to {to}"


@dt.tool("check_balance")
def check_balance(account: str) -> int:
    # Not guarded — runs immediately, no approval needed.
    return 12000


def main() -> None:
    with dt.run("billing-agent", user_input="pay invoice 8842", model="gpt-4o") as run:
        # Unguarded tool: runs right away.
        bal = check_balance("acct_123")
        print(f"balance: ${bal}")

        # Guarded tool: blocks here until a human decides.
        print("requesting approval to wire funds — approve/deny in the dashboard...")
        try:
            result = wire_money("acct_999", 5000)
            print(f"approved: {result}")
        except ApprovalDenied as exc:
            # Denied outright, or nobody approved before the 60s timeout.
            print(f"blocked: {exc.status} (tool did not run)")
            if exc.note:
                # The human's correction, delivered into the run with
                # provenance. Feed it into the next planning step and retry
                # inside THIS run — see docs/approvals.md.
                print(f"  human note from {exc.decided_by or 'operator'}: {exc.note}")
        except (urllib.error.URLError, RuntimeError) as exc:
            print(f"(could not reach the approval API — is the stack running? {exc})")

        run.final_answer()

    dt.shutdown(timeout=5)


if __name__ == "__main__":
    main()
