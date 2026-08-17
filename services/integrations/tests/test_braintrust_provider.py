"""
Tests for the Braintrust provider client. Mocks httpx.AsyncClient. No network
required. Field shapes and behavior (embedded scores dict, no since= filter,
descending cursor pagination, no indexing lag) were verified directly against
Braintrust's live public OpenAPI spec AND a real test project during Phase
2.3 discovery (see BACKLOG.md) — the spec alone was enough here (unlike
LangSmith, where only live testing caught the undocumented required-filter
behavior), but live testing confirmed no surprises diverged from it.
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from integrations_svc.providers.braintrust import BraintrustProvider, _parse_timestamp


def _event(
    id="event-1",
    root_span_id="span-abc",
    scores=None,
    created="2026-07-11T14:46:21.074Z",
):
    return {
        "id": id,
        "root_span_id": root_span_id,
        "scores": scores if scores is not None else {"hallucination": 0.9},
        "created": created,
    }


def _fetch_response(events, cursor=None):
    resp = MagicMock()
    body = {"events": events}
    if cursor is not None:
        body["cursor"] = cursor
    resp.json.return_value = body
    resp.raise_for_status = MagicMock()
    return resp


def _mock_client(responses):
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=responses)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


class TestParseTimestamp(unittest.TestCase):
    def test_parses_z_suffix(self):
        ts = _parse_timestamp("2026-07-11T14:46:21.074Z")
        self.assertGreater(ts, 1_700_000_000)

    def test_none_defaults_to_now(self):
        before = time.time()
        ts = _parse_timestamp(None)
        after = time.time()
        self.assertTrue(before <= ts <= after)


class TestBraintrustProviderFetchNewEvaluations(unittest.IsolatedAsyncioTestCase):
    async def test_maps_a_single_score_to_external_evaluation(self):
        client = _mock_client([_fetch_response([_event()])])
        with patch("httpx.AsyncClient", return_value=client):
            provider = BraintrustProvider("https://api-eu.braintrust.dev", "key", "proj-1")
            evaluations = await provider.fetch_new_evaluations(since=0)

        self.assertEqual(len(evaluations), 1)
        ev = evaluations[0]
        self.assertEqual(ev.external_id, "event-1:hallucination")
        self.assertEqual(ev.trace_id, "span-abc")
        self.assertEqual(ev.name, "hallucination")
        self.assertEqual(ev.value, 0.9)
        self.assertIsNone(ev.string_value)
        self.assertIsNone(ev.comment)
        self.assertIsNone(ev.source_url)

    async def test_one_event_with_multiple_scores_yields_multiple_evaluations(self):
        multi = _event(scores={"hallucination": 0.9, "relevance": 0.75})
        client = _mock_client([_fetch_response([multi])])
        with patch("httpx.AsyncClient", return_value=client):
            provider = BraintrustProvider("https://api-eu.braintrust.dev", "key", "proj-1")
            evaluations = await provider.fetch_new_evaluations(since=0)

        self.assertEqual(len(evaluations), 2)
        ids = {e.external_id for e in evaluations}
        self.assertEqual(ids, {"event-1:hallucination", "event-1:relevance"})

    async def test_event_without_scores_yields_nothing(self):
        no_scores = _event(id="event-2", scores={})
        client = _mock_client([_fetch_response([no_scores])])
        with patch("httpx.AsyncClient", return_value=client):
            provider = BraintrustProvider("https://api-eu.braintrust.dev", "key", "proj-1")
            evaluations = await provider.fetch_new_evaluations(since=0)
        self.assertEqual(evaluations, [])

    async def test_event_without_root_span_id_is_skipped(self):
        no_trace = _event(id="event-3", root_span_id=None)
        client = _mock_client([_fetch_response([no_trace])])
        with patch("httpx.AsyncClient", return_value=client):
            provider = BraintrustProvider("https://api-eu.braintrust.dev", "key", "proj-1")
            evaluations = await provider.fetch_new_evaluations(since=0)
        self.assertEqual(evaluations, [])

    async def test_stops_paginating_once_older_than_since(self):
        # Descending order: newest first, then an event older than `since`.
        since = _parse_timestamp("2026-07-11T14:00:00.000Z")
        newer = _event(id="event-new", created="2026-07-11T14:46:21.074Z")
        older = _event(id="event-old", created="2026-07-11T13:00:00.000Z")
        client = _mock_client([_fetch_response([newer, older], cursor="next-page")])
        with patch("httpx.AsyncClient", return_value=client):
            provider = BraintrustProvider("https://api-eu.braintrust.dev", "key", "proj-1")
            evaluations = await provider.fetch_new_evaluations(since=since)

        # Only the newer event's score should be included, and pagination
        # must not continue past the watermark even though a cursor exists.
        self.assertEqual({e.external_id for e in evaluations}, {"event-new:hallucination"})
        self.assertEqual(client.get.call_count, 1)

    async def test_paginates_across_multiple_pages_when_no_watermark_hit(self):
        page1 = _fetch_response([_event(id="event-1")], cursor="page-2-cursor")
        page2 = _fetch_response([_event(id="event-2")])
        client = _mock_client([page1, page2])
        with patch("httpx.AsyncClient", return_value=client):
            provider = BraintrustProvider("https://api-eu.braintrust.dev", "key", "proj-1")
            evaluations = await provider.fetch_new_evaluations(since=0)

        ids = {e.external_id for e in evaluations}
        self.assertEqual(ids, {"event-1:hallucination", "event-2:hallucination"})
        self.assertEqual(client.get.call_count, 2)
        second_call = client.get.call_args_list[1]
        self.assertEqual(second_call.kwargs["params"]["cursor"], "page-2-cursor")

    async def test_no_cursor_in_response_stops_pagination(self):
        client = _mock_client([_fetch_response([_event()])])  # no cursor key at all
        with patch("httpx.AsyncClient", return_value=client):
            provider = BraintrustProvider("https://api-eu.braintrust.dev", "key", "proj-1")
            await provider.fetch_new_evaluations(since=0)
        self.assertEqual(client.get.call_count, 1)

    async def test_uses_bearer_auth_header(self):
        client = _mock_client([_fetch_response([])])
        with patch("httpx.AsyncClient", return_value=client) as client_cls:
            provider = BraintrustProvider("https://api-eu.braintrust.dev", "my-key", "proj-1")
            await provider.fetch_new_evaluations(since=0)

        _, kwargs = client_cls.call_args
        self.assertEqual(kwargs["headers"], {"Authorization": "Bearer my-key"})

    async def test_requests_correct_project_logs_fetch_url(self):
        client = _mock_client([_fetch_response([])])
        with patch("httpx.AsyncClient", return_value=client):
            provider = BraintrustProvider("https://api-eu.braintrust.dev", "key", "proj-xyz")
            await provider.fetch_new_evaluations(since=0)

        call = client.get.call_args_list[0]
        self.assertEqual(
            call.args[0], "https://api-eu.braintrust.dev/v1/project_logs/proj-xyz/fetch"
        )


if __name__ == "__main__":
    unittest.main()
