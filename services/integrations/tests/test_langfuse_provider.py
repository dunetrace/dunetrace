"""
Tests for the Langfuse provider client. Mocks httpx.AsyncClient (this
codebase's established pattern — see services/api/tests/test_custom_detector_translator.py).
No network required. Field shapes match what was verified directly against
a real Langfuse project's API during Phase 2.1 discovery (see BACKLOG.md).
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from integrations_svc.providers.langfuse import (
    LangfuseProvider,
    _format_from_timestamp,
    _parse_timestamp,
)


def _score(
    id="score-1",
    trace_id="trace-abc",
    name="hallucination",
    value=0.85,
    string_value=None,
    comment="test comment",
    timestamp="2026-07-11T13:08:09.876Z",
    project_id="proj-1",
):
    return {
        "id": id,
        "traceId": trace_id,
        "name": name,
        "value": value,
        "stringValue": string_value,
        "comment": comment,
        "timestamp": timestamp,
        "projectId": project_id,
    }


def _response(data, page=1, total_pages=1):
    resp = MagicMock()
    resp.json.return_value = {
        "data": data,
        "meta": {"page": page, "limit": 100, "totalItems": len(data), "totalPages": total_pages},
    }
    resp.raise_for_status = MagicMock()
    return resp


def _mock_client(responses):
    """responses: list of mock Response objects, one per expected .get() call, in order."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=responses)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


class TestParseTimestamp(unittest.TestCase):
    def test_parses_z_suffix(self):
        ts = _parse_timestamp("2026-07-11T13:08:09.876Z")
        # Sanity: round-trips to a real, sane epoch (not 0, not NaN).
        self.assertGreater(ts, 1_700_000_000)

    def test_none_defaults_to_now(self):
        before = time.time()
        ts = _parse_timestamp(None)
        after = time.time()
        self.assertTrue(before <= ts <= after)


class TestFormatFromTimestamp(unittest.TestCase):
    def test_produces_millisecond_iso_with_z(self):
        formatted = _format_from_timestamp(1_752_235_689.0)
        self.assertTrue(formatted.endswith("Z"))
        self.assertIn("T", formatted)


class TestLangfuseProviderFetchNewEvaluations(unittest.IsolatedAsyncioTestCase):
    async def test_maps_score_fields_to_external_evaluation(self):
        client = _mock_client([_response([_score()])])
        with patch("httpx.AsyncClient", return_value=client):
            provider = LangfuseProvider("https://cloud.langfuse.com", "pk", "sk")
            evaluations = await provider.fetch_new_evaluations(since=time.time())

        self.assertEqual(len(evaluations), 1)
        ev = evaluations[0]
        self.assertEqual(ev.external_id, "score-1")
        self.assertEqual(ev.trace_id, "trace-abc")
        self.assertEqual(ev.name, "hallucination")
        self.assertEqual(ev.value, 0.85)
        self.assertIsNone(ev.string_value)
        self.assertEqual(ev.comment, "test comment")
        self.assertEqual(
            ev.source_url, "https://cloud.langfuse.com/project/proj-1/traces/trace-abc"
        )

    async def test_score_without_trace_id_is_skipped(self):
        no_trace = _score(id="score-2", trace_id=None)
        client = _mock_client([_response([_score(), no_trace])])
        with patch("httpx.AsyncClient", return_value=client):
            provider = LangfuseProvider("https://cloud.langfuse.com", "pk", "sk")
            evaluations = await provider.fetch_new_evaluations(since=time.time())

        self.assertEqual(len(evaluations), 1)
        self.assertEqual(evaluations[0].external_id, "score-1")

    async def test_paginates_across_multiple_pages(self):
        page1 = _response([_score(id="score-1")], page=1, total_pages=2)
        page2 = _response([_score(id="score-2")], page=2, total_pages=2)
        client = _mock_client([page1, page2])
        with patch("httpx.AsyncClient", return_value=client):
            provider = LangfuseProvider("https://cloud.langfuse.com", "pk", "sk")
            evaluations = await provider.fetch_new_evaluations(since=time.time())

        self.assertEqual({e.external_id for e in evaluations}, {"score-1", "score-2"})
        self.assertEqual(client.get.call_count, 2)

    async def test_empty_data_returns_empty_list(self):
        client = _mock_client([_response([])])
        with patch("httpx.AsyncClient", return_value=client):
            provider = LangfuseProvider("https://cloud.langfuse.com", "pk", "sk")
            evaluations = await provider.fetch_new_evaluations(since=time.time())
        self.assertEqual(evaluations, [])

    async def test_categorical_score_has_string_value_no_numeric(self):
        cat_score = _score(name="tone", value=None, string_value="positive")
        client = _mock_client([_response([cat_score])])
        with patch("httpx.AsyncClient", return_value=client):
            provider = LangfuseProvider("https://cloud.langfuse.com", "pk", "sk")
            evaluations = await provider.fetch_new_evaluations(since=time.time())

        self.assertIsNone(evaluations[0].value)
        self.assertEqual(evaluations[0].string_value, "positive")

    async def test_basic_auth_uses_public_and_secret_key(self):
        client = _mock_client([_response([])])
        with patch("httpx.AsyncClient", return_value=client) as client_cls:
            provider = LangfuseProvider("https://cloud.langfuse.com", "my-pk", "my-sk")
            await provider.fetch_new_evaluations(since=time.time())

        _, kwargs = client_cls.call_args
        self.assertEqual(kwargs["auth"], ("my-pk", "my-sk"))


if __name__ == "__main__":
    unittest.main()
