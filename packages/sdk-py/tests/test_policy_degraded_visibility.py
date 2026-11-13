"""
Phase 1B: a downgraded policy must not look like a configured one.

Downgrading an unverifiable remote policy to log-only is correct and stays.
The defect was that the downgrade rewrote ``action`` in place and kept no
record, so a policy forced to log-only was byte-identical to one someone chose
to make log-only. The only evidence was a single WARNING, emitted on a daemon
thread at fetch time, on a logger most users never configure.

These tests pin the three things that make the degraded state legible:
the record keeps what was asked for, the engine can list what is degraded, and
the summary fires on change rather than on every refresh.
"""

from __future__ import annotations

import unittest

from dunetrace.policies import Policy, PolicyEngine

_SECRET = "s3cret"


def _remote(name, action_type, policy_id=1, **params):
    return {
        "id": policy_id,
        "name": name,
        "agent_id": "a",
        "condition": {"trigger": "tool_call_count", "operator": "gt", "value": 5},
        "action": {"type": action_type, **({"params": params} if params else {})},
    }


class TestDowngradeIsRecorded(unittest.TestCase):
    def test_enforcing_remote_policy_without_a_secret_is_marked(self):
        e = PolicyEngine()
        e.load([_remote("stop-runaway", "stop")], secret="", agent_id="a")
        p = e._policies[0]
        self.assertEqual(p.action["type"], "log", "the downgrade itself must still happen")
        self.assertTrue(p.is_degraded)
        self.assertEqual(p.downgraded_from, "stop")

    def test_deliberately_log_only_is_not_marked(self):
        """The distinction that did not exist before. Both end up action=log."""
        e = PolicyEngine()
        e.load([_remote("chosen-log", "log")], secret="", agent_id="a")
        p = e._policies[0]
        self.assertEqual(p.action["type"], "log")
        self.assertFalse(p.is_degraded)
        self.assertIsNone(p.downgraded_from)

    def test_every_enforcing_action_is_recorded_by_name(self):
        for action in (
            "stop",
            "require_approval",
            "escalate_to_human",
            "switch_model",
            "inject_prompt",
            "inject_recovery_prompt",
            "stop_current_tts",
        ):
            with self.subTest(action=action):
                e = PolicyEngine()
                spec = _remote(f"p-{action}", action)
                if action == "require_approval":
                    spec["condition"] = {
                        "trigger": "before_tool_call",
                        "operator": "eq",
                        "value": "wire",
                    }
                e.load([spec], secret="", agent_id="a")
                self.assertEqual(e._policies[0].downgraded_from, action)

    def test_in_process_policies_are_never_downgraded(self):
        """add_policy() is the customer's own code. Nothing to verify."""
        e = PolicyEngine()
        e.add(
            Policy(
                name="local",
                agent_id="a",
                condition={"trigger": "tool_call_count", "operator": "gt", "value": 5},
                action={"type": "stop"},
            )
        )
        p = e._policies[0]
        self.assertEqual(p.action["type"], "stop")
        self.assertFalse(p.is_degraded)

    def test_a_signed_policy_is_not_downgraded(self):
        import hashlib
        import hmac

        from dunetrace.policies import _policy_canonical, sig_version_for_condition

        spec = _remote("signed-stop", "stop")
        version = sig_version_for_condition(spec["condition"])
        canonical = _policy_canonical(
            version,
            spec["id"],
            spec["agent_id"],
            spec["name"],
            spec["condition"],
            spec["action"],
            True,
            100,
        )
        spec["signature"] = hmac.new(
            _SECRET.encode(), canonical.encode(), hashlib.sha256
        ).hexdigest()
        spec["sig_version"] = version

        e = PolicyEngine()
        e.load([spec], secret=_SECRET, agent_id="a")
        self.assertEqual(len(e._policies), 1, "signature should verify")
        self.assertEqual(e._policies[0].action["type"], "stop")
        self.assertFalse(e._policies[0].is_degraded)


class TestEngineExposesDegraded(unittest.TestCase):
    def test_degraded_lists_only_the_degraded(self):
        e = PolicyEngine()
        e.load(
            [
                _remote("stop-runaway", "stop", 1),
                _remote("chosen-log", "log", 2),
                _remote("swap-model", "switch_model", 3),
            ],
            secret="",
            agent_id="a",
        )
        self.assertEqual(sorted(p.name for p in e.degraded()), ["stop-runaway", "swap-model"])

    def test_degraded_is_empty_when_nothing_is(self):
        e = PolicyEngine()
        e.load([_remote("chosen-log", "log")], secret="", agent_id="a")
        self.assertEqual(e.degraded(), [])


class TestStartupSummary(unittest.TestCase):
    def test_summary_names_count_policies_and_both_secrets(self):
        e = PolicyEngine()
        with self.assertLogs("dunetrace.policies", level="WARNING") as cm:
            e.load([_remote("stop-runaway", "stop")], secret="", agent_id="a")
        summary = [m for m in cm.output if "DEGRADED" in m]
        self.assertEqual(len(summary), 1, "exactly one summary line")
        text = summary[0]
        self.assertIn("stop-runaway", text)
        self.assertIn("wanted stop", text)
        self.assertIn("DUNETRACE_POLICY_SECRET", text)
        self.assertIn("POLICY_SIGNING_SECRET", text)

    def test_nothing_repeats_on_an_unchanged_refresh(self):
        """load() runs every 60s. Both the per-policy line and the summary are
        keyed on the degraded set, so a steady state is completely silent.
        Warning every minute about an unchanged condition trains people to
        filter the message, which costs more than the message is worth."""
        e = PolicyEngine()
        bundle = [_remote("stop-runaway", "stop")]
        with self.assertLogs("dunetrace.policies", level="WARNING") as first:
            e.load(bundle, secret="", agent_id="a")
        self.assertEqual(len([m for m in first.output if "DEGRADED" in m]), 1)
        self.assertEqual(len([m for m in first.output if "Remote policy" in m]), 1)

        with self.assertNoLogs("dunetrace.policies", level="WARNING"):
            e.load(bundle, secret="", agent_id="a")

    def test_only_the_newly_degraded_policy_is_named_again(self):
        """A second policy degrading must not re-announce the first."""
        e = PolicyEngine()
        with self.assertLogs("dunetrace.policies", level="WARNING"):
            e.load([_remote("stop-runaway", "stop", 1)], secret="", agent_id="a")
        with self.assertLogs("dunetrace.policies", level="WARNING") as cm:
            e.load(
                [_remote("stop-runaway", "stop", 1), _remote("swap", "switch_model", 2)],
                secret="",
                agent_id="a",
            )
        per_policy = [m for m in cm.output if "Remote policy" in m]
        self.assertEqual(len(per_policy), 1, "re-announced an already-degraded policy")
        self.assertIn("swap", per_policy[0])
        # The summary still restates the full set, which is the point of a summary.
        summary = [m for m in cm.output if "DEGRADED" in m][0]
        self.assertIn("stop-runaway", summary)
        self.assertIn("swap", summary)

    def test_summary_fires_again_when_the_set_changes(self):
        e = PolicyEngine()
        with self.assertLogs("dunetrace.policies", level="WARNING"):
            e.load([_remote("stop-runaway", "stop", 1)], secret="", agent_id="a")
        with self.assertLogs("dunetrace.policies", level="WARNING") as cm:
            e.load(
                [_remote("stop-runaway", "stop", 1), _remote("swap", "switch_model", 2)],
                secret="",
                agent_id="a",
            )
        self.assertEqual(len([m for m in cm.output if "DEGRADED" in m]), 1)

    def test_recovery_is_announced(self):
        e = PolicyEngine()
        with self.assertLogs("dunetrace.policies", level="WARNING"):
            e.load([_remote("stop-runaway", "stop")], secret="", agent_id="a")
        with self.assertLogs("dunetrace.policies", level="INFO") as cm:
            e.load([_remote("chosen-log", "log")], secret="", agent_id="a")
        self.assertTrue(any("restored" in m for m in cm.output))


if __name__ == "__main__":
    unittest.main(verbosity=2)
