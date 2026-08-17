"""
Tests for the semantic worker's orchestration: wiring the sampling decision
to structural/retrieval/agent-config lookups, budget enforcement, and (Phase
1.4.1) running the configured evaluators on sampled runs. DB and DeepEval are
mocked — nothing running required.

Run:
    cd services/semantic
    pytest tests/ -v
"""

from __future__ import annotations

import contextlib
import unittest
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import semantic_svc.worker  # must be imported before patch() can resolve "semantic_svc.worker.*"
from semantic_svc.worker import (
    _build_second_opinion_evaluators,
    _mistral_second_opinion_model,
    _opposite_provider,
    _provider_has_key,
    _run_evaluators,
    _severity_for_confidence,
    poll_once,
    process_run,
    run_worker,
)


@contextlib.contextmanager
def _mocked_deps(
    structural=False,
    retrieval=False,
    agent_config=None,
    consume_budget_result=True,
    signals_written=0,
    org_quota_settings=None,
    consume_org_quota_result=True,
    conversation_signals_written=0,
):
    with contextlib.ExitStack() as stack:
        yield {
            "structural": stack.enter_context(
                patch(
                    "semantic_svc.worker.has_structural_signal",
                    AsyncMock(return_value=structural),
                )
            ),
            "retrieval": stack.enter_context(
                patch("semantic_svc.worker.has_retrieval_event", AsyncMock(return_value=retrieval))
            ),
            "config": stack.enter_context(
                patch(
                    "semantic_svc.worker.fetch_agent_semantic_config",
                    AsyncMock(return_value=agent_config),
                )
            ),
            "budget": stack.enter_context(
                patch(
                    "semantic_svc.worker.consume_budget",
                    AsyncMock(return_value=consume_budget_result),
                )
            ),
            "quota": stack.enter_context(
                patch("semantic_svc.worker.write_quota_exceeded_signal", AsyncMock())
            ),
            "mark": stack.enter_context(
                patch("semantic_svc.worker.mark_run_processed", AsyncMock())
            ),
            "run_evaluators": stack.enter_context(
                patch(
                    "semantic_svc.worker._run_evaluators",
                    AsyncMock(return_value=signals_written),
                )
            ),
            "org_quota_settings": stack.enter_context(
                patch(
                    "semantic_svc.worker.fetch_org_semantic_quota_settings",
                    AsyncMock(
                        return_value=org_quota_settings or {"quota": 1000, "allow_overage": False}
                    ),
                )
            ),
            "org_quota_consume": stack.enter_context(
                patch(
                    "semantic_svc.worker.consume_org_semantic_quota",
                    AsyncMock(return_value=consume_org_quota_result),
                )
            ),
            "conversation_eval": stack.enter_context(
                patch(
                    "semantic_svc.worker._maybe_run_conversation_evaluator",
                    AsyncMock(return_value=conversation_signals_written),
                )
            ),
        }


class TestSeverityForConfidence(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(_severity_for_confidence(0.9), "CRITICAL")
        self.assertEqual(_severity_for_confidence(0.85), "CRITICAL")
        self.assertEqual(_severity_for_confidence(0.75), "HIGH")
        self.assertEqual(_severity_for_confidence(0.70), "HIGH")
        self.assertEqual(_severity_for_confidence(0.55), "MEDIUM")
        self.assertEqual(_severity_for_confidence(0.50), "MEDIUM")
        self.assertEqual(_severity_for_confidence(0.49), "LOW")
        self.assertEqual(_severity_for_confidence(0.0), "LOW")


class TestProcessRun(unittest.IsolatedAsyncioTestCase):
    async def test_structural_signal_always_sampled(self):
        with _mocked_deps(structural=True) as mocks:
            sampled, signals = await process_run("run-1", "agent-1", "v1", "org-1")

        self.assertTrue(sampled)
        self.assertEqual(signals, 0)
        mocks["mark"].assert_called_once_with(
            "run-1", "agent-1", "v1", "org-1", True, "structural_signal"
        )
        mocks["run_evaluators"].assert_called_once_with("run-1", "agent-1", "v1", "org-1", None)

    async def test_no_signal_no_retrieval_no_override_uses_baseline(self):
        with _mocked_deps() as mocks:
            await process_run("run-baseline", "agent-1", "v1", "org-1")

        args = mocks["mark"].call_args.args
        self.assertEqual(args[:4], ("run-baseline", "agent-1", "v1", "org-1"))
        self.assertEqual(args[5], "baseline")

    async def test_evaluators_not_run_when_not_sampled(self):
        agent_config = {"sample_rate": 0.0, "budget_monthly": None, "semantic_critical": False}
        with _mocked_deps(agent_config=agent_config) as mocks:
            sampled, signals = await process_run("run-skip", "agent-1", "v1", "org-1")

        self.assertFalse(sampled)
        self.assertEqual(signals, 0)
        mocks["run_evaluators"].assert_not_called()

    async def test_signals_written_propagated_from_run_evaluators(self):
        with _mocked_deps(structural=True, signals_written=2) as mocks:
            sampled, signals = await process_run("run-1", "agent-1", "v1", "org-1")

        self.assertTrue(sampled)
        self.assertEqual(signals, 2)

    async def test_evaluator_override_list_passed_through(self):
        agent_config = {
            "sample_rate": None,
            "budget_monthly": None,
            "semantic_critical": True,
            "evaluators": ["HALLUCINATION"],
        }
        with _mocked_deps(agent_config=agent_config) as mocks:
            await process_run("run-1", "agent-1", "v1", "org-1")

        mocks["run_evaluators"].assert_called_once_with(
            "run-1", "agent-1", "v1", "org-1", ["HALLUCINATION"]
        )

    async def test_budget_exhausted_downgrades_sampling_and_emits_quota_signal(self):
        # semantic_critical=True forces sampled=True from decide_sampling, then
        # budget enforcement should downgrade it back to False.
        agent_config = {
            "sample_rate": None,
            "budget_monthly": 100,
            "evaluators": [],
            "semantic_critical": True,
        }
        with _mocked_deps(agent_config=agent_config, consume_budget_result=False) as mocks:
            sampled, signals = await process_run("run-2", "agent-2", "v1", "org-1")

        self.assertFalse(sampled)
        self.assertEqual(signals, 0)
        mocks["mark"].assert_called_once_with(
            "run-2", "agent-2", "v1", "org-1", False, "budget_exceeded"
        )
        mocks["quota"].assert_called_once_with(
            "org-1", "agent-2", "v1", "run-2", ANY, "agent_budget", 100
        )
        mocks["run_evaluators"].assert_not_called()

    async def test_no_budget_configured_skips_consume_budget_call(self):
        agent_config = {
            "sample_rate": None,
            "budget_monthly": None,
            "evaluators": [],
            "semantic_critical": True,
        }
        with _mocked_deps(agent_config=agent_config) as mocks:
            sampled, _signals = await process_run("run-3", "agent-3", "v1", "org-1")

        self.assertTrue(sampled)
        mocks["budget"].assert_not_called()

    async def test_skipped_run_does_not_consume_budget(self):
        # sample_rate=0.0 deterministically forces "not sampled" regardless of bucket.
        agent_config = {
            "sample_rate": 0.0,
            "budget_monthly": 100,
            "evaluators": [],
            "semantic_critical": False,
        }
        with _mocked_deps(agent_config=agent_config) as mocks:
            sampled, _signals = await process_run("run-4", "agent-4", "v1", "org-1")

        self.assertFalse(sampled)
        mocks["budget"].assert_not_called()
        mocks["mark"].assert_called_once_with(
            "run-4", "agent-4", "v1", "org-1", False, "agent_override"
        )

    async def test_org_quota_exceeded_downgrades_sampling_and_emits_signal(self):
        with _mocked_deps(structural=True, consume_org_quota_result=False) as mocks:
            sampled, signals = await process_run("run-5", "agent-5", "v1", "org-1")

        self.assertFalse(sampled)
        self.assertEqual(signals, 0)
        mocks["mark"].assert_called_once_with(
            "run-5", "agent-5", "v1", "org-1", False, "org_quota_exceeded"
        )
        mocks["quota"].assert_called_once_with(
            "org-1", "agent-5", "v1", "run-5", ANY, "org_quota", 1000
        )
        # Downgraded before the per-agent budget check even runs, and before
        # any evaluator is invoked.
        mocks["budget"].assert_not_called()
        mocks["run_evaluators"].assert_not_called()

    async def test_org_quota_allowed_when_under_limit(self):
        with _mocked_deps(structural=True, consume_org_quota_result=True) as mocks:
            sampled, _signals = await process_run("run-6", "agent-6", "v1", "org-1")

        self.assertTrue(sampled)
        mocks["org_quota_consume"].assert_called_once()
        mocks["quota"].assert_not_called()

    async def test_org_quota_settings_read_before_consuming(self):
        custom_settings = {"quota": 50, "allow_overage": True}
        with _mocked_deps(
            structural=True, org_quota_settings=custom_settings, consume_org_quota_result=True
        ) as mocks:
            await process_run("run-7", "agent-7", "v1", "org-1")

        mocks["org_quota_settings"].assert_called_once_with("org-1")
        mocks["org_quota_consume"].assert_called_once_with("org-1", ANY, 50, True)

    async def test_both_org_and_agent_counters_increment_for_the_same_evaluation(self):
        """Integration-style check: a single sampled run with BOTH an org
        quota and a per-agent budget configured must touch both counters —
        consume_org_semantic_quota (org-wide) and consume_budget (this
        agent's own sub-limit) — not just one or the other. These are two
        independent ceilings (Phase 1.5 org quota vs. Phase 1.2 agent
        budget); a real gap here would be either counter silently not being
        called, which would let usage drift from what's actually enforced.
        """
        agent_config = {
            "sample_rate": None,
            "budget_monthly": 500,
            "evaluators": [],
            "semantic_critical": True,
        }
        with _mocked_deps(
            agent_config=agent_config,
            consume_budget_result=True,
            consume_org_quota_result=True,
        ) as mocks:
            sampled, _signals = await process_run("run-8", "agent-8", "v1", "org-1")

        self.assertTrue(sampled)
        # Same run, same month, same org — both counters called exactly once.
        mocks["org_quota_consume"].assert_called_once()
        mocks["budget"].assert_called_once()
        org_call_args = mocks["org_quota_consume"].call_args.args
        agent_call_args = mocks["budget"].call_args.args
        self.assertEqual(org_call_args[0], "org-1")  # org_id
        self.assertEqual(agent_call_args[0], "org-1")  # org_id
        self.assertEqual(agent_call_args[1], "agent-8")  # agent_id
        self.assertEqual(org_call_args[1], agent_call_args[2])  # same month bucket
        # Neither counter's own failure was involved — this run's success
        # depended on both having been consulted, not just one.
        mocks["quota"].assert_not_called()

    async def test_conversation_evaluator_runs_regardless_of_per_run_sampling(self):
        """Phase 3.2 — conversation-level evaluation is an independent
        decision from this run's own per-run sampling outcome."""
        agent_config = {"sample_rate": 0.0, "budget_monthly": None, "semantic_critical": False}
        with _mocked_deps(agent_config=agent_config) as mocks:
            sampled, _signals = await process_run("run-skip", "agent-1", "v1", "org-1")

        self.assertFalse(sampled)
        mocks["conversation_eval"].assert_called_once_with("run-skip", "agent-1", "v1", "org-1")

    async def test_conversation_signals_added_to_total(self):
        with _mocked_deps(
            structural=True, signals_written=1, conversation_signals_written=1
        ) as mocks:
            _sampled, signals = await process_run("run-1", "agent-1", "v1", "org-1")

        self.assertEqual(signals, 2)

    async def test_conversation_evaluator_failure_does_not_block_mark_processed(self):
        with _mocked_deps(structural=True) as mocks:
            mocks["conversation_eval"].side_effect = RuntimeError("boom")
            sampled, signals = await process_run("run-1", "agent-1", "v1", "org-1")

        self.assertTrue(sampled)
        self.assertEqual(signals, 0)
        mocks["mark"].assert_called_once()


class TestMaybeRunConversationEvaluator(unittest.IsolatedAsyncioTestCase):
    async def _run(self, **overrides):
        from semantic_svc.worker import _maybe_run_conversation_evaluator

        return await _maybe_run_conversation_evaluator(
            overrides.get("run_id", "run-4"),
            overrides.get("agent_id", "agent-1"),
            overrides.get("agent_version", "v1"),
            overrides.get("org_id", "org-1"),
        )

    async def test_returns_zero_when_no_conversation_evaluator_built(self):
        with patch.object(semantic_svc.worker, "_conversation_evaluators", {}):
            result = await self._run()
        self.assertEqual(result, 0)

    async def test_returns_zero_when_run_has_no_conversation_id(self):
        with (
            patch.object(
                semantic_svc.worker, "_conversation_evaluators", {"USER_FRUSTRATION": MagicMock()}
            ),
            patch("semantic_svc.worker.fetch_run_conversation_id", AsyncMock(return_value=None)),
            patch("semantic_svc.worker.fetch_conversation_run_ids", AsyncMock()) as siblings_mock,
        ):
            result = await self._run()

        self.assertEqual(result, 0)
        siblings_mock.assert_not_called()

    async def test_returns_zero_when_not_sampled(self):
        with (
            patch.object(
                semantic_svc.worker, "_conversation_evaluators", {"USER_FRUSTRATION": MagicMock()}
            ),
            patch(
                "semantic_svc.worker.fetch_run_conversation_id",
                AsyncMock(return_value="conv-1"),
            ),
            patch(
                "semantic_svc.worker.fetch_conversation_run_ids",
                AsyncMock(return_value=["run-1", "run-2", "run-3"]),
            ),
            patch(
                "semantic_svc.worker.decide_conversation_sampling",
                return_value=(False, "not_sampled"),
            ),
            patch(
                "semantic_svc.worker.fetch_org_conversation_quota_settings", AsyncMock()
            ) as quota_settings_mock,
        ):
            result = await self._run()

        self.assertEqual(result, 0)
        quota_settings_mock.assert_not_called()

    async def test_returns_zero_when_quota_exhausted(self):
        with (
            patch.object(
                semantic_svc.worker, "_conversation_evaluators", {"USER_FRUSTRATION": MagicMock()}
            ),
            patch(
                "semantic_svc.worker.fetch_run_conversation_id",
                AsyncMock(return_value="conv-1"),
            ),
            patch(
                "semantic_svc.worker.fetch_conversation_run_ids",
                AsyncMock(return_value=["run-1", "run-2", "run-3"]),
            ),
            patch(
                "semantic_svc.worker.decide_conversation_sampling",
                return_value=(True, "conversation_sample_rate"),
            ),
            patch(
                "semantic_svc.worker.fetch_org_conversation_quota_settings",
                AsyncMock(return_value={"quota": 200, "allow_overage": False}),
            ),
            patch(
                "semantic_svc.worker.consume_org_conversation_quota",
                AsyncMock(return_value=False),
            ),
            patch("semantic_svc.worker.fetch_run_events", AsyncMock()) as events_mock,
        ):
            result = await self._run()

        self.assertEqual(result, 0)
        events_mock.assert_not_called()

    async def test_fired_evaluation_writes_signal_with_correct_evidence(self):
        from semantic_svc.evaluators.base import EvalResult

        fake_evaluator = MagicMock()
        fake_evaluator.evaluate.return_value = EvalResult(
            evaluator="USER_FRUSTRATION",
            fired=True,
            confidence=0.82,
            reasoning="User repeated the same complaint three times.",
            prompt_tokens=500,
            completion_tokens=120,
            cost_usd=0.01,
        )
        events_by_run = {
            "run-1": [
                {"event_type": "run.started", "payload": {"input_text": "help me"}},
                {"event_type": "llm.responded", "payload": {"output": "sure"}},
            ],
            "run-2": [
                {"event_type": "run.started", "payload": {"input_text": "still broken"}},
                {"event_type": "llm.responded", "payload": {"output": "trying again"}},
            ],
            "run-3": [
                {"event_type": "run.started", "payload": {"input_text": "this is useless"}},
                {"event_type": "llm.responded", "payload": {"output": "sorry"}},
            ],
        }

        async def fake_fetch_run_events(run_id):
            return events_by_run[run_id]

        with (
            patch.object(
                semantic_svc.worker,
                "_conversation_evaluators",
                {"USER_FRUSTRATION": fake_evaluator},
            ),
            patch(
                "semantic_svc.worker.fetch_run_conversation_id",
                AsyncMock(return_value="conv-xyz"),
            ),
            patch(
                "semantic_svc.worker.fetch_conversation_run_ids",
                AsyncMock(return_value=["run-1", "run-2", "run-3"]),
            ),
            patch(
                "semantic_svc.worker.decide_conversation_sampling",
                return_value=(True, "conversation_sample_rate"),
            ),
            patch(
                "semantic_svc.worker.fetch_org_conversation_quota_settings",
                AsyncMock(return_value={"quota": 200, "allow_overage": False}),
            ),
            patch(
                "semantic_svc.worker.consume_org_conversation_quota", AsyncMock(return_value=True)
            ),
            patch("semantic_svc.worker.fetch_run_events", side_effect=fake_fetch_run_events),
            patch("semantic_svc.worker.log_semantic_evaluation", AsyncMock()) as log_mock,
            patch(
                "semantic_svc.worker.write_semantic_signal", AsyncMock(return_value=99)
            ) as write_mock,
        ):
            result = await self._run(run_id="run-3", agent_id="agent-9", agent_version="v9")

        self.assertEqual(result, 1)
        log_mock.assert_called_once_with(
            "org-1", "agent-9", "USER_FRUSTRATION", True, 500, 120, 0.01
        )
        write_mock.assert_called_once()
        kwargs = write_mock.call_args.kwargs
        self.assertEqual(kwargs["evaluator"], "USER_FRUSTRATION")
        self.assertEqual(kwargs["severity"], "HIGH")  # confidence 0.82 -> HIGH
        self.assertEqual(kwargs["run_id"], "run-3")  # the triggering/current run
        self.assertEqual(kwargs["agent_id"], "agent-9")
        self.assertEqual(kwargs["agent_version"], "v9")
        self.assertEqual(kwargs["confidence"], 0.82)
        self.assertEqual(kwargs["org_id"], "org-1")
        evidence = kwargs["evidence"]
        self.assertEqual(evidence["reasoning"], "User repeated the same complaint three times.")
        self.assertEqual(evidence["prompt_tokens"], 500)
        self.assertEqual(evidence["completion_tokens"], 120)
        self.assertEqual(evidence["cost_usd"], 0.01)
        self.assertEqual(evidence["conversation_id"], "conv-xyz")
        self.assertEqual(evidence["run_ids_considered"], ["run-1", "run-2", "run-3"])

    async def test_not_fired_logs_but_does_not_write_signal(self):
        from semantic_svc.evaluators.base import EvalResult

        fake_evaluator = MagicMock()
        fake_evaluator.evaluate.return_value = EvalResult(
            evaluator="USER_FRUSTRATION",
            fired=False,
            confidence=0.1,
            reasoning="User seems satisfied.",
            prompt_tokens=400,
            completion_tokens=100,
            cost_usd=0.008,
        )
        events = [
            {"event_type": "run.started", "payload": {"input_text": "hi"}},
            {"event_type": "llm.responded", "payload": {"output": "hello"}},
        ]

        with (
            patch.object(
                semantic_svc.worker,
                "_conversation_evaluators",
                {"USER_FRUSTRATION": fake_evaluator},
            ),
            patch(
                "semantic_svc.worker.fetch_run_conversation_id",
                AsyncMock(return_value="conv-1"),
            ),
            patch(
                "semantic_svc.worker.fetch_conversation_run_ids",
                AsyncMock(return_value=["run-1", "run-2", "run-3"]),
            ),
            patch(
                "semantic_svc.worker.decide_conversation_sampling",
                return_value=(True, "conversation_sample_rate"),
            ),
            patch(
                "semantic_svc.worker.fetch_org_conversation_quota_settings",
                AsyncMock(return_value={"quota": 200, "allow_overage": False}),
            ),
            patch(
                "semantic_svc.worker.consume_org_conversation_quota", AsyncMock(return_value=True)
            ),
            patch("semantic_svc.worker.fetch_run_events", AsyncMock(return_value=events)),
            patch("semantic_svc.worker.log_semantic_evaluation", AsyncMock()) as log_mock,
            patch("semantic_svc.worker.write_semantic_signal", AsyncMock()) as write_mock,
        ):
            result = await self._run()

        self.assertEqual(result, 0)
        log_mock.assert_called_once()
        write_mock.assert_not_called()

    async def test_no_extractable_content_in_any_sibling_run_writes_nothing(self):
        with (
            patch.object(
                semantic_svc.worker, "_conversation_evaluators", {"USER_FRUSTRATION": MagicMock()}
            ),
            patch(
                "semantic_svc.worker.fetch_run_conversation_id",
                AsyncMock(return_value="conv-1"),
            ),
            patch(
                "semantic_svc.worker.fetch_conversation_run_ids",
                AsyncMock(return_value=["run-1", "run-2", "run-3"]),
            ),
            patch(
                "semantic_svc.worker.decide_conversation_sampling",
                return_value=(True, "conversation_sample_rate"),
            ),
            patch(
                "semantic_svc.worker.fetch_org_conversation_quota_settings",
                AsyncMock(return_value={"quota": 200, "allow_overage": False}),
            ),
            patch(
                "semantic_svc.worker.consume_org_conversation_quota", AsyncMock(return_value=True)
            ),
            patch("semantic_svc.worker.fetch_run_events", AsyncMock(return_value=[])),
            patch("semantic_svc.worker.log_semantic_evaluation", AsyncMock()) as log_mock,
            patch("semantic_svc.worker.write_semantic_signal", AsyncMock()) as write_mock,
        ):
            result = await self._run()

        self.assertEqual(result, 0)
        log_mock.assert_not_called()
        write_mock.assert_not_called()


class TestEvaluatorFailureContainment(unittest.IsolatedAsyncioTestCase):
    """One evaluator raising must cost that finding only. Uncontained it
    propagates through process_run into poll_once's asyncio.gather and takes the
    whole batch down — and every evaluator already run for the run is billable,
    so the retry pays for them a second time."""

    async def test_one_failing_evaluator_does_not_stop_the_others(self):
        fake_run = MagicMock()
        fired_result = MagicMock(
            evaluator="TASK_COMPLETION",
            fired=True,
            confidence=0.9,
            reasoning="incomplete",
            prompt_tokens=10,
            completion_tokens=5,
            cost_usd=0.01,
        )
        exploding = MagicMock(evaluate=MagicMock(side_effect=RuntimeError("429 rate limited")))
        healthy = MagicMock(evaluate=MagicMock(return_value=fired_result))

        with (
            patch("semantic_svc.worker.fetch_run_events", AsyncMock(return_value=[{"x": 1}])),
            patch("semantic_svc.worker.build_evaluation_input", return_value=fake_run),
            patch(
                "semantic_svc.worker._evaluators",
                {"HALLUCINATION": exploding, "TASK_COMPLETION": healthy},
            ),
            patch(
                "semantic_svc.worker.write_semantic_signal", AsyncMock(return_value=101)
            ) as write_mock,
            patch("semantic_svc.worker.record_signal_group_membership", AsyncMock()),
            patch("semantic_svc.worker.fetch_signal_group_fp_count", AsyncMock(return_value=0)),
            patch("semantic_svc.worker.log_semantic_evaluation", AsyncMock()),
        ):
            count = await _run_evaluators("run-1", "agent-1", "v1", "org-1", None)

        self.assertEqual(count, 1)
        write_mock.assert_called_once()
        self.assertEqual(write_mock.call_args.kwargs["evaluator"], "TASK_COMPLETION")

    async def test_failed_second_opinion_keeps_the_primary_finding(self):
        """Confirming a HIGH finding is an enhancement — it must not be able to
        discard the finding it was meant to confirm."""
        fake_run = MagicMock()
        high_result = MagicMock(
            evaluator="HALLUCINATION",
            fired=True,
            confidence=0.75,  # HIGH band, so a second opinion is requested
            reasoning="maybe",
            prompt_tokens=10,
            completion_tokens=5,
            cost_usd=0.01,
        )
        primary = MagicMock(evaluate=MagicMock(return_value=high_result))
        second = MagicMock(evaluate=MagicMock(side_effect=RuntimeError("provider down")))

        with (
            patch("semantic_svc.worker.fetch_run_events", AsyncMock(return_value=[{"x": 1}])),
            patch("semantic_svc.worker.build_evaluation_input", return_value=fake_run),
            patch("semantic_svc.worker._evaluators", {"HALLUCINATION": primary}),
            patch("semantic_svc.worker._second_opinion_evaluators", {"HALLUCINATION": second}),
            patch(
                "semantic_svc.worker.write_semantic_signal", AsyncMock(return_value=101)
            ) as write_mock,
            patch("semantic_svc.worker.record_signal_group_membership", AsyncMock()),
            patch("semantic_svc.worker.fetch_signal_group_fp_count", AsyncMock(return_value=0)),
            patch("semantic_svc.worker.log_semantic_evaluation", AsyncMock()),
        ):
            count = await _run_evaluators("run-1", "agent-1", "v1", "org-1", None)

        self.assertEqual(count, 1)
        self.assertEqual(write_mock.call_args.kwargs["severity"], "HIGH")


class TestRunEvaluators(unittest.IsolatedAsyncioTestCase):
    async def test_returns_zero_when_no_evaluation_input(self):
        with (
            patch("semantic_svc.worker.fetch_run_events", AsyncMock(return_value=[])),
            patch("semantic_svc.worker.build_evaluation_input", return_value=None),
        ):
            count = await _run_evaluators("run-1", "agent-1", "v1", "org-1", None)
        self.assertEqual(count, 0)

    async def test_writes_signal_for_each_fired_evaluator(self):
        fake_run = MagicMock()
        fired_result = MagicMock(
            evaluator="HALLUCINATION",
            fired=True,
            confidence=0.9,
            reasoning="bad",
            prompt_tokens=10,
            completion_tokens=5,
            cost_usd=0.01,
        )
        not_fired_result = MagicMock(fired=False)
        evaluator_a = MagicMock(evaluate=MagicMock(return_value=fired_result))
        evaluator_b = MagicMock(evaluate=MagicMock(return_value=not_fired_result))

        with (
            patch("semantic_svc.worker.fetch_run_events", AsyncMock(return_value=[{"x": 1}])),
            patch("semantic_svc.worker.build_evaluation_input", return_value=fake_run),
            patch(
                "semantic_svc.worker._evaluators",
                {"HALLUCINATION": evaluator_a, "TASK_COMPLETION": evaluator_b},
            ),
            patch(
                "semantic_svc.worker.write_semantic_signal", AsyncMock(return_value=101)
            ) as write_mock,
            patch("semantic_svc.worker.record_signal_group_membership", AsyncMock()) as group_mock,
            patch("semantic_svc.worker.fetch_signal_group_fp_count", AsyncMock(return_value=0)),
            patch("semantic_svc.worker.log_semantic_evaluation", AsyncMock()) as log_mock,
        ):
            count = await _run_evaluators("run-1", "agent-1", "v1", "org-1", None)

        self.assertEqual(count, 1)
        write_mock.assert_called_once()
        # Both evaluators get logged (fired and not-fired alike) — the whole
        # point of semantic_evaluation_log vs. failure_signals.
        self.assertEqual(log_mock.call_count, 2)
        kwargs = write_mock.call_args.kwargs
        self.assertEqual(kwargs["evaluator"], "HALLUCINATION")
        self.assertEqual(kwargs["severity"], "CRITICAL")
        self.assertEqual(kwargs["confidence"], 0.9)
        self.assertEqual(kwargs["evidence"]["reasoning"], "bad")
        group_mock.assert_called_once_with("org-1", "agent-1", "HALLUCINATION", "bad", 101, "run-1")

    async def test_evaluator_name_override_restricts_which_run(self):
        fake_run = MagicMock()
        fired_result = MagicMock(
            evaluator="TASK_COMPLETION",
            fired=True,
            confidence=0.6,
            reasoning="",
            prompt_tokens=1,
            completion_tokens=1,
            cost_usd=0.0,
        )
        evaluator_a = MagicMock()
        evaluator_b = MagicMock(evaluate=MagicMock(return_value=fired_result))

        with (
            patch("semantic_svc.worker.fetch_run_events", AsyncMock(return_value=[{"x": 1}])),
            patch("semantic_svc.worker.build_evaluation_input", return_value=fake_run),
            patch(
                "semantic_svc.worker._evaluators",
                {"HALLUCINATION": evaluator_a, "TASK_COMPLETION": evaluator_b},
            ),
            patch("semantic_svc.worker.write_semantic_signal", AsyncMock(return_value=1)),
            patch("semantic_svc.worker.record_signal_group_membership", AsyncMock()),
            patch("semantic_svc.worker.fetch_signal_group_fp_count", AsyncMock(return_value=0)),
            patch("semantic_svc.worker.log_semantic_evaluation", AsyncMock()),
        ):
            count = await _run_evaluators("run-1", "agent-1", "v1", "org-1", ["TASK_COMPLETION"])

        self.assertEqual(count, 1)
        evaluator_a.evaluate.assert_not_called()

    async def test_unknown_evaluator_name_skipped_gracefully(self):
        with (
            patch("semantic_svc.worker.fetch_run_events", AsyncMock(return_value=[{"x": 1}])),
            patch("semantic_svc.worker.build_evaluation_input", return_value=MagicMock()),
            patch("semantic_svc.worker._evaluators", {}),
        ):
            count = await _run_evaluators("run-1", "agent-1", "v1", "org-1", ["NOT_REAL"])
        self.assertEqual(count, 0)

    async def _run_single_evaluator(self, fp_count, feedback_settings=None, confidence=0.9):
        fake_run = MagicMock()
        fired_result = MagicMock(
            evaluator="HALLUCINATION",
            fired=True,
            confidence=confidence,
            reasoning="bad",
            prompt_tokens=10,
            completion_tokens=5,
            cost_usd=0.01,
        )
        evaluator_a = MagicMock(evaluate=MagicMock(return_value=fired_result))

        with (
            patch("semantic_svc.worker.fetch_run_events", AsyncMock(return_value=[{"x": 1}])),
            patch("semantic_svc.worker.build_evaluation_input", return_value=fake_run),
            patch("semantic_svc.worker._evaluators", {"HALLUCINATION": evaluator_a}),
            patch(
                "semantic_svc.worker.write_semantic_signal", AsyncMock(return_value=101)
            ) as write_mock,
            patch("semantic_svc.worker.record_signal_group_membership", AsyncMock()),
            patch(
                "semantic_svc.worker.fetch_signal_group_fp_count",
                AsyncMock(return_value=fp_count),
            ),
            patch(
                "semantic_svc.worker.fetch_org_semantic_feedback_settings",
                AsyncMock(return_value=feedback_settings or {"auto_suppress": False}),
            ) as settings_mock,
            patch("semantic_svc.worker.log_semantic_evaluation", AsyncMock()),
        ):
            count = await _run_evaluators("run-1", "agent-1", "v1", "org-1", None)
        return count, write_mock, settings_mock

    async def test_below_fp_threshold_does_not_check_feedback_settings(self):
        count, write_mock, settings_mock = await self._run_single_evaluator(fp_count=2)
        self.assertEqual(count, 1)
        settings_mock.assert_not_called()
        self.assertEqual(write_mock.call_args.kwargs["confidence"], 0.9)

    async def test_at_threshold_without_auto_suppress_lowers_confidence(self):
        count, write_mock, _settings_mock = await self._run_single_evaluator(
            fp_count=3, feedback_settings={"auto_suppress": False}
        )
        self.assertEqual(count, 1)
        self.assertAlmostEqual(write_mock.call_args.kwargs["confidence"], 0.6)
        self.assertEqual(write_mock.call_args.kwargs["severity"], "MEDIUM")

    async def test_confidence_never_goes_negative(self):
        count, write_mock, _settings_mock = await self._run_single_evaluator(
            fp_count=10, feedback_settings={"auto_suppress": False}, confidence=0.2
        )
        self.assertEqual(count, 1)
        self.assertEqual(write_mock.call_args.kwargs["confidence"], 0.0)

    async def test_at_threshold_with_auto_suppress_writes_nothing(self):
        count, write_mock, _settings_mock = await self._run_single_evaluator(
            fp_count=3, feedback_settings={"auto_suppress": True}
        )
        self.assertEqual(count, 0)
        write_mock.assert_not_called()

    async def _run_with_second_opinion(
        self, confidence, second_opinion_evaluator=None, evaluator_name="HALLUCINATION"
    ):
        fake_run = MagicMock()
        fired_result = MagicMock(
            evaluator=evaluator_name,
            fired=True,
            confidence=confidence,
            reasoning="primary reasoning",
            prompt_tokens=10,
            completion_tokens=5,
            cost_usd=0.01,
        )
        evaluator = MagicMock(evaluate=MagicMock(return_value=fired_result))
        second_opinions = (
            {evaluator_name: second_opinion_evaluator} if second_opinion_evaluator else {}
        )

        with (
            patch("semantic_svc.worker.fetch_run_events", AsyncMock(return_value=[{"x": 1}])),
            patch("semantic_svc.worker.build_evaluation_input", return_value=fake_run),
            patch("semantic_svc.worker._evaluators", {evaluator_name: evaluator}),
            patch("semantic_svc.worker._second_opinion_evaluators", second_opinions),
            patch(
                "semantic_svc.worker.write_semantic_signal", AsyncMock(return_value=101)
            ) as write_mock,
            patch("semantic_svc.worker.record_signal_group_membership", AsyncMock()),
            patch("semantic_svc.worker.fetch_signal_group_fp_count", AsyncMock(return_value=0)),
            patch("semantic_svc.worker.log_semantic_evaluation", AsyncMock()),
        ):
            await _run_evaluators("run-1", "agent-1", "v1", "org-1", None)
        return write_mock

    async def test_high_severity_no_second_opinion_configured_stays_high(self):
        write_mock = await self._run_with_second_opinion(confidence=0.75)
        self.assertEqual(write_mock.call_args.kwargs["severity"], "HIGH")
        self.assertNotIn("second_opinion", write_mock.call_args.kwargs["evidence"])

    async def test_high_severity_second_opinion_agrees_stays_high(self):
        second_result = MagicMock(
            fired=True, reasoning="confirmed", prompt_tokens=20, completion_tokens=8, cost_usd=0.02
        )
        second_evaluator = MagicMock(evaluate=MagicMock(return_value=second_result))
        write_mock = await self._run_with_second_opinion(
            confidence=0.75, second_opinion_evaluator=second_evaluator
        )
        self.assertEqual(write_mock.call_args.kwargs["severity"], "HIGH")
        so = write_mock.call_args.kwargs["evidence"]["second_opinion"]
        self.assertTrue(so["ran"])
        self.assertTrue(so["agreed"])

    async def test_high_severity_second_opinion_disagrees_downgrades_to_medium(self):
        second_result = MagicMock(
            fired=False,
            reasoning="not convinced",
            prompt_tokens=20,
            completion_tokens=8,
            cost_usd=0.02,
        )
        second_evaluator = MagicMock(evaluate=MagicMock(return_value=second_result))
        write_mock = await self._run_with_second_opinion(
            confidence=0.75, second_opinion_evaluator=second_evaluator
        )
        self.assertEqual(write_mock.call_args.kwargs["severity"], "MEDIUM")
        so = write_mock.call_args.kwargs["evidence"]["second_opinion"]
        self.assertFalse(so["agreed"])

    async def test_medium_severity_never_triggers_second_opinion(self):
        second_evaluator = MagicMock(evaluate=MagicMock())
        write_mock = await self._run_with_second_opinion(
            confidence=0.55, second_opinion_evaluator=second_evaluator
        )
        self.assertEqual(write_mock.call_args.kwargs["severity"], "MEDIUM")
        second_evaluator.evaluate.assert_not_called()

    async def test_critical_severity_never_triggers_second_opinion(self):
        # Brief scopes second-opinion strictly to HIGH severity findings.
        second_evaluator = MagicMock(evaluate=MagicMock())
        write_mock = await self._run_with_second_opinion(
            confidence=0.9, second_opinion_evaluator=second_evaluator
        )
        self.assertEqual(write_mock.call_args.kwargs["severity"], "CRITICAL")
        second_evaluator.evaluate.assert_not_called()


class TestOppositeProvider(unittest.TestCase):
    def test_openai_becomes_anthropic(self):
        self.assertEqual(_opposite_provider("openai"), "anthropic")

    def test_anthropic_becomes_openai(self):
        self.assertEqual(_opposite_provider("anthropic"), "openai")

    def test_unknown_provider_falls_back_to_openai(self):
        self.assertEqual(_opposite_provider("some-other-provider"), "openai")


class TestBuildSecondOpinionEvaluators(unittest.TestCase):
    def test_no_config_returns_empty(self):
        with (
            patch("semantic_svc.worker._EVALUATOR_CONFIG", {}),
            patch("semantic_svc.worker.settings") as mock_settings,
        ):
            mock_settings.SEMANTIC_LLM_PROVIDER = "openai"
            built = _build_second_opinion_evaluators()
        self.assertEqual(built, {})

    def test_require_second_opinion_false_excluded(self):
        config = {"HALLUCINATION": {"require_second_opinion": False}}
        with (
            patch("semantic_svc.worker._EVALUATOR_CONFIG", config),
            patch("semantic_svc.worker.settings") as mock_settings,
        ):
            mock_settings.SEMANTIC_LLM_PROVIDER = "openai"
            built = _build_second_opinion_evaluators()
        self.assertEqual(built, {})

    def test_explicit_provider_and_model_used(self):
        config = {
            "HALLUCINATION": {
                "require_second_opinion": True,
                "second_opinion_provider": "anthropic",
                "second_opinion_model": "claude-sonnet-4-6",
            }
        }
        mock_cls = MagicMock()
        with (
            patch("semantic_svc.worker._EVALUATOR_CONFIG", config),
            patch("semantic_svc.worker.settings") as mock_settings,
            patch.dict("semantic_svc.worker._EVALUATOR_CLASSES", {"HALLUCINATION": mock_cls}),
        ):
            mock_settings.SEMANTIC_LLM_PROVIDER = "openai"
            _build_second_opinion_evaluators()
        mock_cls.assert_called_once_with("anthropic", "claude-sonnet-4-6")

    def test_provider_defaults_to_opposite_of_primary_when_unset(self):
        config = {"HALLUCINATION": {"require_second_opinion": True}}
        with (
            patch("semantic_svc.worker._EVALUATOR_CONFIG", config),
            patch("semantic_svc.worker.settings") as mock_settings,
        ):
            mock_settings.SEMANTIC_LLM_PROVIDER = "openai"
            mock_cls = MagicMock()
            with patch.dict("semantic_svc.worker._EVALUATOR_CLASSES", {"HALLUCINATION": mock_cls}):
                _build_second_opinion_evaluators()
        mock_cls.assert_called_once_with("anthropic", None)

    def test_unknown_evaluator_name_in_config_skipped(self):
        config = {"NOT_A_REAL_EVALUATOR": {"require_second_opinion": True}}
        with (
            patch("semantic_svc.worker._EVALUATOR_CONFIG", config),
            patch("semantic_svc.worker.settings") as mock_settings,
        ):
            mock_settings.SEMANTIC_LLM_PROVIDER = "openai"
            built = _build_second_opinion_evaluators()
        self.assertEqual(built, {})


class TestSecondOpinionResidency(unittest.TestCase):
    """A mistral primary is a residency choice. The confirming call must not be
    the thing that ships the run text to a US vendor."""

    # The shipped docs/config/semantic-evaluators.yml block, verbatim.
    SHIPPED_HALLUCINATION_CONFIG = {
        "HALLUCINATION": {
            "require_second_opinion": True,
            "second_opinion_provider": "anthropic",
            "second_opinion_model": "claude-sonnet-4-6",
        }
    }

    @staticmethod
    @contextlib.contextmanager
    def _worker(config, *, allow_cross_provider=False):
        mock_cls = MagicMock()
        with (
            patch("semantic_svc.worker._EVALUATOR_CONFIG", config),
            patch("semantic_svc.worker.settings") as mock_settings,
            patch.dict("semantic_svc.worker._EVALUATOR_CLASSES", {"HALLUCINATION": mock_cls}),
        ):
            mock_settings.SEMANTIC_LLM_PROVIDER = "mistral"
            mock_settings.MISTRAL_API_KEY = "mistral-fake"
            mock_settings.HALLUCINATION_MODEL = ""
            mock_settings.SEMANTIC_ALLOW_CROSS_PROVIDER_SECOND_OPINION = allow_cross_provider
            yield mock_cls

    def test_provider_has_key_tracks_mistral_key(self):
        """Without this branch the only in-region configuration an operator can
        write is skipped with a warning telling them to set a key they already
        set."""
        with patch("semantic_svc.worker.settings") as s:
            s.MISTRAL_API_KEY = "mistral-fake"
            self.assertTrue(_provider_has_key("mistral"))
            s.MISTRAL_API_KEY = ""
            self.assertFalse(_provider_has_key("mistral"))

    def test_opposite_provider_keeps_mistral_in_region(self):
        self.assertEqual(_opposite_provider("mistral"), "mistral")

    def test_second_opinion_model_differs_from_a_large_primary(self):
        # Asking the identical model twice is not a second opinion.
        self.assertEqual(_mistral_second_opinion_model(None), "mistral-large-latest")
        self.assertEqual(
            _mistral_second_opinion_model("mistral-large-latest"), "mistral-medium-latest"
        )

    def test_shipped_anthropic_config_is_suppressed_for_a_mistral_primary(self):
        with self._worker(self.SHIPPED_HALLUCINATION_CONFIG) as mock_cls:
            _build_second_opinion_evaluators()
        mock_cls.assert_called_once_with("mistral", "mistral-large-latest")

    def test_opt_in_flag_restores_the_cross_provider_second_opinion(self):
        with self._worker(self.SHIPPED_HALLUCINATION_CONFIG, allow_cross_provider=True) as mock_cls:
            _build_second_opinion_evaluators()
        mock_cls.assert_called_once_with("anthropic", "claude-sonnet-4-6")

    def test_unconfigured_mistral_primary_gets_an_in_region_second_opinion(self):
        config = {"HALLUCINATION": {"require_second_opinion": True}}
        with self._worker(config) as mock_cls:
            _build_second_opinion_evaluators()
        mock_cls.assert_called_once_with("mistral", "mistral-large-latest")

    def test_openai_primary_still_crosses_to_anthropic(self):
        """The residency rule is scoped to mistral — it must not change the
        existing single-vendor deployments."""
        config = {"HALLUCINATION": {"require_second_opinion": True}}
        mock_cls = MagicMock()
        with (
            patch("semantic_svc.worker._EVALUATOR_CONFIG", config),
            patch("semantic_svc.worker.settings") as mock_settings,
            patch.dict("semantic_svc.worker._EVALUATOR_CLASSES", {"HALLUCINATION": mock_cls}),
        ):
            mock_settings.SEMANTIC_LLM_PROVIDER = "openai"
            mock_settings.ANTHROPIC_API_KEY = "sk-ant-fake"
            mock_settings.HALLUCINATION_MODEL = ""
            mock_settings.SEMANTIC_ALLOW_CROSS_PROVIDER_SECOND_OPINION = False
            _build_second_opinion_evaluators()
        mock_cls.assert_called_once_with("anthropic", None)


class TestPollOnce(unittest.IsolatedAsyncioTestCase):
    async def test_returns_zero_when_no_work(self):
        with patch("semantic_svc.worker.fetch_unevaluated_runs", AsyncMock(return_value=[])):
            runs, sampled, signals = await poll_once()
        self.assertEqual((runs, sampled, signals), (0, 0, 0))

    async def test_processes_every_fetched_run(self):
        fetched = [
            {"run_id": "r1", "agent_id": "a1", "agent_version": "v1", "org_id": "org-1"},
            {"run_id": "r2", "agent_id": "a1", "agent_version": "v1", "org_id": "org-1"},
        ]
        with (
            patch("semantic_svc.worker.fetch_unevaluated_runs", AsyncMock(return_value=fetched)),
            _mocked_deps(structural=True, signals_written=1),
        ):
            runs, sampled, signals = await poll_once()

        self.assertEqual(runs, 2)
        self.assertEqual(sampled, 2)  # structural_signal forces 100% for both
        self.assertEqual(signals, 2)  # 1 signal per run


class TestRunWorkerDisabled(unittest.IsolatedAsyncioTestCase):
    async def test_exits_without_opening_pool_when_disabled(self):
        init_mock = AsyncMock()
        with (
            patch("semantic_svc.worker.settings") as mock_settings,
            patch("semantic_svc.worker.init_pool", init_mock),
        ):
            mock_settings.SEMANTIC_WORKER_ENABLED = False
            await run_worker()

        init_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
