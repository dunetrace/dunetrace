"""Failure isolation: a Dunetrace-internal error must never break the host app.

This is the cardinal property of an in-path observability SDK. Instrumentation
is strictly additive — if something on our side breaks, the agent it is watching
must still run, still get its result, and still see its own exceptions unchanged.

Two things are deliberately NOT isolated, and are asserted here too:
  * the caller's own exceptions propagate untouched;
  * PolicyViolation / ApprovalDenied propagate, because those are the
    runtime-prevention feature working, not a failure.

No network required.
"""

import types
import unittest
from unittest.mock import patch

import dunetrace.auto as auto
from dunetrace.client import Dunetrace
from dunetrace.policies import PolicyViolation

_ANSWER = "real answer"


def _fake_openai_module():
    """Minimal stand-in for openai.resources.chat.completions.

    Counts calls so a test can prove the underlying (billable) request actually
    went out and its result survived.
    """
    calls = {"n": 0}

    class Completions:
        def create(self, *, messages=None, model="unknown", **kwargs):
            calls["n"] += 1
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=_ANSWER),
                        finish_reason="stop",
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

    return Completions, calls


class TestAutoInstrumentIsolation(unittest.TestCase):
    """auto.py wraps calls the host depends on — an SDK bug must not break them."""

    def setUp(self):
        import sys

        self._Completions, self._calls = _fake_openai_module()
        mod = types.ModuleType("openai.resources.chat.completions")
        mod.Completions = self._Completions
        self._added = {
            "openai": types.ModuleType("openai"),
            "openai.resources": types.ModuleType("openai.resources"),
            "openai.resources.chat": types.ModuleType("openai.resources.chat"),
            "openai.resources.chat.completions": mod,
        }
        self._saved = {k: sys.modules.get(k) for k in self._added}
        sys.modules.update(self._added)
        auto._PATCHED.discard("openai")
        self.dt = Dunetrace()
        auto._patch_openai(self.dt)
        self.client = self._Completions()

    def tearDown(self):
        import sys

        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        auto._PATCHED.discard("openai")

    def _call_under_fault(self, target):
        with patch(target, side_effect=RuntimeError("dunetrace-internal")):
            with self.dt.run("agent") as run:
                return self.client.create(
                    messages=[{"role": "user", "content": "hi"}], model="gpt-4o"
                )

    def test_emit_failure_before_call_does_not_prevent_the_call(self):
        resp = self._call_under_fault("dunetrace.run_context.RunContext.llm_called")
        self.assertEqual(self._calls["n"], 1)
        self.assertEqual(resp.choices[0].message.content, _ANSWER)

    def test_emit_failure_after_call_does_not_discard_the_result(self):
        # The worst case: the request was made and billed. Losing the response
        # to an instrumentation bug costs the caller real money.
        resp = self._call_under_fault("dunetrace.auto._emit_openai_response")
        self.assertEqual(self._calls["n"], 1)
        self.assertEqual(resp.choices[0].message.content, _ANSWER)

    def test_token_estimation_failure_does_not_prevent_the_call(self):
        resp = self._call_under_fault("dunetrace.auto._estimate_tokens")
        self.assertEqual(self._calls["n"], 1)
        self.assertEqual(resp.choices[0].message.content, _ANSWER)

    def test_provider_exception_is_not_replaced_by_ours(self):
        """On the error path we emit too — the provider's error must still win."""

        class ProviderError(Exception):
            pass

        def _boom(self, *, messages=None, model="unknown", **kwargs):
            raise ProviderError("rate limited")

        with patch.object(self._Completions, "create", _boom):
            auto._PATCHED.discard("openai")
            auto._patch_openai(self.dt)
            with patch(
                "dunetrace.run_context.RunContext.llm_responded",
                side_effect=RuntimeError("dunetrace-internal"),
            ):
                with self.assertRaises(ProviderError):
                    with self.dt.run("agent") as run:
                        self._Completions().create(messages=[], model="gpt-4o")


class TestPolicyControlFlowStillPropagates(unittest.TestCase):
    """Guarding emits must not accidentally swallow runtime prevention."""

    def test_stop_policy_still_raises(self):
        dt = Dunetrace()
        dt.add_policy(
            "halt",
            {"trigger": "tool_call_count", "operator": "gte", "value": 1},
            {"type": "stop"},
        )
        with self.assertRaises(PolicyViolation):
            with dt.run("agent") as run:
                run.tool_called("x", {})
                run.tool_responded("x", success=True)


class TestRunSetupIsolation(unittest.TestCase):
    """dt.run()'s own setup/teardown must not raise into the caller."""

    def setUp(self):
        self.dt = Dunetrace()

    def test_non_str_user_input_does_not_raise(self):
        # Callers pass whatever their agent has: a message dict, a list of
        # messages, an id, bytes off a socket. None of it may break the run.
        for value in ({"q": "hi"}, ["a", "b"], 42, b"hi", None, 3.5, object()):
            with self.subTest(value=type(value).__name__):
                with self.dt.run("agent", user_input=value) as run:
                    run.final_answer()

    def test_non_str_system_prompt_does_not_raise(self):
        with self.dt.run("agent", system_prompt={"role": "system"}) as run:
            run.final_answer()

    def test_unstringable_input_does_not_raise(self):
        class Hostile:
            def __str__(self):
                raise ValueError("nope")

            __repr__ = __str__

        with self.dt.run("agent", user_input=Hostile()) as run:
            run.final_answer()

    def test_injection_scan_failure_is_contained(self):
        with patch(
            "dunetrace.detectors.PromptInjectionDetector.check_input",
            side_effect=RuntimeError("dunetrace-internal"),
        ):
            with self.dt.run("agent", user_input="hello") as run:
                run.final_answer()

    def test_agent_version_failure_is_contained(self):
        with patch(
            "dunetrace.client.agent_version", side_effect=RuntimeError("dunetrace-internal")
        ):
            with self.dt.run("agent", user_input="hello") as run:
                run.final_answer()

    def test_completion_path_failure_is_contained(self):
        """A failure *after* the caller's block succeeded must not surface —
        previously this was caught by the generic handler, mis-recorded as a run
        error, and re-raised into a caller whose code had actually succeeded."""
        with patch(
            "dunetrace.run_context.RunContext._warn_unread_advisory",
            side_effect=RuntimeError("dunetrace-internal"),
        ):
            with self.dt.run("agent") as run:
                run.final_answer()

    def test_caller_exception_still_propagates(self):
        class CallerBug(Exception):
            pass

        with self.assertRaises(CallerBug):
            with self.dt.run("agent") as run:
                raise CallerBug("the agent's own bug")

    def test_caller_exception_survives_a_failing_error_emit(self):
        """Even if recording the error fails, the caller's exception wins."""

        class CallerBug(Exception):
            pass

        with patch.object(self.dt, "_emit", side_effect=RuntimeError("dunetrace-internal")):
            with self.assertRaises(CallerBug):
                with self.dt.run("agent") as run:
                    raise CallerBug("the agent's own bug")


if __name__ == "__main__":
    unittest.main()
