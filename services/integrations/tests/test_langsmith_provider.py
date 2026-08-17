"""
Tests for the LangSmith provider client. Mocks httpx.AsyncClient. No network
required. Field shapes and the required-session-filter behavior were
verified directly against LangSmith's live public OpenAPI spec AND a real
test project during Phase 2.2 discovery (see BACKLOG.md) — the spec alone
was misleading (marks session/run/key as independently optional; the live
API actually requires at least one of them).
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from integrations_svc.providers.langsmith import (
    LangSmithProvider,
    _numeric_score,
    _parse_timestamp,
    _string_value,
)


def _feedback(
    id="fb-1",
    trace_id="trace-abc",
    run_id="run-abc",
    key="hallucination",
    score=0.85,
    value=None,
    comment="test comment",
    created_at="2026-07-11T14:23:54.556711",
):
    return {
        "id": id,
        "run_id": run_id,
        "session_id": "session-1",
        "trace_id": trace_id,
        "key": key,
        "score": score,
        "value": value,
        "comment": comment,
        "created_at": created_at,
    }


def _sessions_response(sessions):
    resp = MagicMock()
    resp.json.return_value = sessions
    resp.raise_for_status = MagicMock()
    return resp


def _feedback_response(items):
    resp = MagicMock()
    resp.json.return_value = items
    resp.raise_for_status = MagicMock()
    return resp


def _mock_client(responses):
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=responses)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


class TestNumericScore(unittest.TestCase):
    def test_true_maps_to_one(self):
        self.assertEqual(_numeric_score(True), 1.0)

    def test_false_maps_to_zero(self):
        self.assertEqual(_numeric_score(False), 0.0)

    def test_int_and_float_pass_through(self):
        self.assertEqual(_numeric_score(1), 1.0)
        self.assertEqual(_numeric_score(0.75), 0.75)

    def test_none_returns_none(self):
        self.assertIsNone(_numeric_score(None))


class TestStringValue(unittest.TestCase):
    def test_numeric_score_present_means_no_string_value(self):
        self.assertIsNone(_string_value(0.5, "ignored"))

    def test_string_value_used_when_score_absent(self):
        self.assertEqual(_string_value(None, "positive"), "positive")

    def test_non_string_value_stringified(self):
        self.assertEqual(_string_value(None, {"a": 1}), "{'a': 1}")

    def test_both_absent_returns_none(self):
        self.assertIsNone(_string_value(None, None))


class TestParseTimestamp(unittest.TestCase):
    def test_naive_iso_treated_as_utc(self):
        ts = _parse_timestamp("2026-07-11T14:23:54.556711")
        self.assertGreater(ts, 1_700_000_000)

    def test_none_defaults_to_now(self):
        before = time.time()
        ts = _parse_timestamp(None)
        after = time.time()
        self.assertTrue(before <= ts <= after)


class TestLangSmithProviderFetchNewEvaluations(unittest.IsolatedAsyncioTestCase):
    async def test_resolves_session_id_before_fetching_feedback(self):
        client = _mock_client(
            [
                _sessions_response([{"id": "session-1", "name": "Dunetrace"}]),
                _feedback_response([_feedback()]),
            ]
        )
        with patch("httpx.AsyncClient", return_value=client):
            provider = LangSmithProvider("https://api.smith.langchain.com", "key", "Dunetrace")
            evaluations = await provider.fetch_new_evaluations(since=time.time())

        self.assertEqual(len(evaluations), 1)
        # First call resolves the session, second queries feedback scoped to it.
        first_call = client.get.call_args_list[0]
        self.assertIn("sessions", first_call.args[0])
        second_call = client.get.call_args_list[1]
        self.assertEqual(second_call.kwargs["params"]["session"], "session-1")

    async def test_session_id_cached_across_pagination(self):
        page1 = _feedback_response([_feedback(id="fb-1")] * 100)
        page2 = _feedback_response([_feedback(id="fb-2")])
        client = _mock_client(
            [_sessions_response([{"id": "session-1", "name": "Dunetrace"}]), page1, page2]
        )
        with patch("httpx.AsyncClient", return_value=client):
            provider = LangSmithProvider("https://api.smith.langchain.com", "key", "Dunetrace")
            await provider.fetch_new_evaluations(since=time.time())

        # Exactly one sessions call despite two feedback pages.
        session_calls = [c for c in client.get.call_args_list if "sessions" in c.args[0]]
        self.assertEqual(len(session_calls), 1)

    async def test_project_not_found_returns_empty_list(self):
        client = _mock_client([_sessions_response([])])
        with patch("httpx.AsyncClient", return_value=client):
            provider = LangSmithProvider("https://api.smith.langchain.com", "key", "NoSuchProject")
            evaluations = await provider.fetch_new_evaluations(since=time.time())
        self.assertEqual(evaluations, [])

    async def test_maps_feedback_fields_to_external_evaluation(self):
        client = _mock_client(
            [
                _sessions_response([{"id": "session-1", "name": "Dunetrace"}]),
                _feedback_response([_feedback()]),
            ]
        )
        with patch("httpx.AsyncClient", return_value=client):
            provider = LangSmithProvider("https://api.smith.langchain.com", "key", "Dunetrace")
            evaluations = await provider.fetch_new_evaluations(since=time.time())

        ev = evaluations[0]
        self.assertEqual(ev.external_id, "fb-1")
        self.assertEqual(ev.trace_id, "trace-abc")
        self.assertEqual(ev.name, "hallucination")
        self.assertEqual(ev.value, 0.85)
        self.assertIsNone(ev.string_value)
        self.assertEqual(ev.comment, "test comment")
        self.assertIsNone(ev.source_url)

    async def test_feedback_without_trace_id_is_skipped(self):
        no_trace = _feedback(id="fb-2", trace_id=None)
        client = _mock_client(
            [
                _sessions_response([{"id": "session-1", "name": "Dunetrace"}]),
                _feedback_response([_feedback(), no_trace]),
            ]
        )
        with patch("httpx.AsyncClient", return_value=client):
            provider = LangSmithProvider("https://api.smith.langchain.com", "key", "Dunetrace")
            evaluations = await provider.fetch_new_evaluations(since=time.time())

        self.assertEqual(len(evaluations), 1)
        self.assertEqual(evaluations[0].external_id, "fb-1")

    async def test_boolean_score_feedback(self):
        bool_fb = _feedback(key="passed", score=True, value=None)
        client = _mock_client(
            [
                _sessions_response([{"id": "session-1", "name": "Dunetrace"}]),
                _feedback_response([bool_fb]),
            ]
        )
        with patch("httpx.AsyncClient", return_value=client):
            provider = LangSmithProvider("https://api.smith.langchain.com", "key", "Dunetrace")
            evaluations = await provider.fetch_new_evaluations(since=time.time())

        self.assertEqual(evaluations[0].value, 1.0)

    async def test_uses_x_api_key_header(self):
        client = _mock_client([_sessions_response([])])
        with patch("httpx.AsyncClient", return_value=client) as client_cls:
            provider = LangSmithProvider("https://api.smith.langchain.com", "my-key", "Dunetrace")
            await provider.fetch_new_evaluations(since=time.time())

        _, kwargs = client_cls.call_args
        self.assertEqual(kwargs["headers"], {"X-Api-Key": "my-key"})

    async def test_paginates_across_multiple_pages(self):
        page1 = _feedback_response([_feedback(id=f"fb-{i}") for i in range(100)])
        page2 = _feedback_response([_feedback(id="fb-100")])
        client = _mock_client(
            [_sessions_response([{"id": "session-1", "name": "Dunetrace"}]), page1, page2]
        )
        with patch("httpx.AsyncClient", return_value=client):
            provider = LangSmithProvider("https://api.smith.langchain.com", "key", "Dunetrace")
            evaluations = await provider.fetch_new_evaluations(since=time.time())

        self.assertEqual(len(evaluations), 101)


if __name__ == "__main__":
    unittest.main()
