"""
Tests for Phase 4.3's real-diff-capable GitHub client (api_svc/github_client.py).
Mocks httpx.AsyncClient entirely — no network.
"""

from __future__ import annotations

import base64
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from api_svc.github_client import create_fix_pr, fetch_file_content


def _resp(status_code=200, json_data=None):
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = json_data or {}

    def _raise():
        if status_code >= 400:
            raise httpx.HTTPStatusError("error", request=MagicMock(), response=r)

    r.raise_for_status.side_effect = _raise
    return r


class _FakeAsyncClient:
    """A minimal stand-in for httpx.AsyncClient whose .get/.post/.put are
    driven by a caller-supplied side-effect queue keyed by method."""

    def __init__(self, get_responses=None, post_responses=None, put_responses=None):
        self._get_responses = list(get_responses or [])
        self._post_responses = list(post_responses or [])
        self._put_responses = list(put_responses or [])
        self.get_calls = []
        self.post_calls = []
        self.put_calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return self._get_responses.pop(0)

    async def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return self._post_responses.pop(0)

    async def put(self, url, **kwargs):
        self.put_calls.append((url, kwargs))
        return self._put_responses.pop(0)


class TestFetchFileContent(unittest.IsolatedAsyncioTestCase):
    async def test_returns_decoded_content(self):
        raw = "def hello():\n    pass\n"
        encoded = base64.b64encode(raw.encode()).decode()
        client = _FakeAsyncClient(
            get_responses=[_resp(200, {"encoding": "base64", "content": encoded})]
        )

        with patch("httpx.AsyncClient", return_value=client):
            result = await fetch_file_content("tok", "acme/bot", "src/a.py", "main")

        self.assertEqual(result, raw)

    async def test_returns_none_on_404(self):
        client = _FakeAsyncClient(get_responses=[_resp(404)])

        with patch("httpx.AsyncClient", return_value=client):
            result = await fetch_file_content("tok", "acme/bot", "src/missing.py", "main")

        self.assertIsNone(result)

    async def test_returns_none_when_encoding_not_base64(self):
        client = _FakeAsyncClient(get_responses=[_resp(200, {"encoding": "none", "content": ""})])

        with patch("httpx.AsyncClient", return_value=client):
            result = await fetch_file_content("tok", "acme/bot", "src/a.py", "main")

        self.assertIsNone(result)

    async def test_raises_on_other_http_errors(self):
        client = _FakeAsyncClient(get_responses=[_resp(500)])

        with patch("httpx.AsyncClient", return_value=client):
            with self.assertRaises(httpx.HTTPStatusError):
                await fetch_file_content("tok", "acme/bot", "src/a.py", "main")


class TestCreateFixPr(unittest.IsolatedAsyncioTestCase):
    async def test_real_file_path_commits_actual_content_not_summary(self):
        client = _FakeAsyncClient(
            get_responses=[
                _resp(200, {"object": {"sha": "base-sha"}}),  # _get_branch_sha
                _resp(404),  # _commit_real_file: no existing file
            ],
            post_responses=[
                _resp(201),  # _create_branch
                _resp(
                    201, {"html_url": "https://github.com/acme/bot/pull/7", "number": 7}
                ),  # _open_pr
            ],
            put_responses=[_resp(200)],  # commit content
        )

        with patch("httpx.AsyncClient", return_value=client):
            result = await create_fix_pr(
                token="tok",
                repo="acme/bot",
                base_branch="main",
                signal_id=1,
                agent_id="agent-1",
                failure_type="TOOL_LOOP",
                root_cause="looped",
                fix_content="add a limit",
                fix_patch="--- a\n+++ b\n",
                real_file={"file_path": "src/a.py", "new_content": "def fixed(): pass\n"},
            )

        self.assertTrue(result["applied_to_real_file"])
        self.assertEqual(result["pr_url"], "https://github.com/acme/bot/pull/7")
        self.assertEqual(result["pr_number"], 7)
        put_url, put_kwargs = client.put_calls[0]
        self.assertEqual(put_url, "/repos/acme/bot/contents/src/a.py")
        decoded = base64.b64decode(put_kwargs["json"]["content"]).decode()
        self.assertEqual(decoded, "def fixed(): pass\n")

    async def test_no_real_file_falls_back_to_summary_markdown(self):
        client = _FakeAsyncClient(
            get_responses=[
                _resp(200, {"object": {"sha": "base-sha"}}),
                _resp(404),
            ],
            post_responses=[
                _resp(201),
                _resp(201, {"html_url": "https://github.com/acme/bot/pull/8", "number": 8}),
            ],
            put_responses=[_resp(200)],
        )

        with patch("httpx.AsyncClient", return_value=client):
            result = await create_fix_pr(
                token="tok",
                repo="acme/bot",
                base_branch="main",
                signal_id=2,
                agent_id="agent-1",
                failure_type="RETRY_STORM",
                root_cause="retried too much",
                fix_content="back off",
                fix_patch="",
            )

        self.assertFalse(result["applied_to_real_file"])
        put_url, put_kwargs = client.put_calls[0]
        self.assertEqual(put_url, "/repos/acme/bot/contents/dunetrace-fixes/signal-2.md")
        decoded = base64.b64decode(put_kwargs["json"]["content"]).decode()
        self.assertIn("Signal #2", decoded)

    async def test_reviewers_requested_after_pr_created(self):
        client = _FakeAsyncClient(
            get_responses=[
                _resp(200, {"object": {"sha": "base-sha"}}),
                _resp(404),
            ],
            post_responses=[
                _resp(201),  # branch create
                _resp(201, {"html_url": "https://github.com/acme/bot/pull/9", "number": 9}),  # PR
                _resp(201),  # request reviewers
            ],
            put_responses=[_resp(200)],
        )

        with patch("httpx.AsyncClient", return_value=client):
            await create_fix_pr(
                token="tok",
                repo="acme/bot",
                base_branch="main",
                signal_id=3,
                agent_id="agent-1",
                failure_type="TOOL_LOOP",
                root_cause="x",
                fix_content="y",
                fix_patch="",
                reviewers=["octocat"],
            )

        reviewer_call_url, reviewer_call_kwargs = client.post_calls[-1]
        self.assertEqual(reviewer_call_url, "/repos/acme/bot/pulls/9/requested_reviewers")
        self.assertEqual(reviewer_call_kwargs["json"]["reviewers"], ["octocat"])

    async def test_reviewer_request_failure_does_not_fail_pr_creation(self):
        client = _FakeAsyncClient(
            get_responses=[
                _resp(200, {"object": {"sha": "base-sha"}}),
                _resp(404),
            ],
            post_responses=[
                _resp(201),
                _resp(201, {"html_url": "https://github.com/acme/bot/pull/10", "number": 10}),
            ],
            put_responses=[_resp(200)],
        )

        async def _failing_post(url, **kwargs):
            client.post_calls.append((url, kwargs))
            if "requested_reviewers" in url:
                raise RuntimeError("user has no repo access")
            return client._post_responses.pop(0)

        client.post = _failing_post

        with patch("httpx.AsyncClient", return_value=client):
            result = await create_fix_pr(
                token="tok",
                repo="acme/bot",
                base_branch="main",
                signal_id=4,
                agent_id="agent-1",
                failure_type="TOOL_LOOP",
                root_cause="x",
                fix_content="y",
                fix_patch="",
                reviewers=["ghost-user"],
            )

        self.assertEqual(result["pr_number"], 10)

    async def test_existing_pr_returned_on_422(self):
        client = _FakeAsyncClient(
            get_responses=[
                _resp(200, {"object": {"sha": "base-sha"}}),
                _resp(404),
                _resp(
                    200,
                    [{"html_url": "https://github.com/acme/bot/pull/11", "number": 11}],
                ),  # list PRs after 422
            ],
            post_responses=[
                _resp(201),
                _resp(422),
            ],
            put_responses=[_resp(200)],
        )

        with patch("httpx.AsyncClient", return_value=client):
            result = await create_fix_pr(
                token="tok",
                repo="acme/bot",
                base_branch="main",
                signal_id=5,
                agent_id="agent-1",
                failure_type="TOOL_LOOP",
                root_cause="x",
                fix_content="y",
                fix_patch="",
            )

        self.assertEqual(result["pr_number"], 11)
        self.assertEqual(result["pr_url"], "https://github.com/acme/bot/pull/11")


if __name__ == "__main__":
    unittest.main()
