"""Tests for the ElevenLabs provider client. Mocks httpx.AsyncClient (the
established pattern here — see test_langfuse_provider.py). No network. Field
shapes and the character-count-as-delta rule match what was verified against
the live ElevenLabs API during Phase 0 discovery.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from integrations_svc.providers.elevenlabs import (
    ElevenLabsProvider,
    _char_count,
)


def _item(
    history_item_id="hist-1",
    voice_id="voice-abc",
    voice_name="Rachel",
    model_id="eleven_multilingual_v2",
    change_from=1000,
    change_to=1042,
    text="your order shipped",
    source="TTS",
    date_unix=1_752_000_000,
):
    return {
        "history_item_id": history_item_id,
        "voice_id": voice_id,
        "voice_name": voice_name,
        "model_id": model_id,
        "character_count_change_from": change_from,
        "character_count_change_to": change_to,
        "text": text,
        "source": source,
        "date_unix": date_unix,
    }


def _resp(status=200, body=None, headers=None, raises=False):
    r = MagicMock()
    r.status_code = status
    r.headers = headers or {}
    r.json.return_value = body if body is not None else {"history": [], "has_more": False}
    if raises:
        r.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("err", request=MagicMock(), response=MagicMock())
        )
    else:
        r.raise_for_status = MagicMock()
    return r


def _mock_client(responses):
    """responses: mock Response objects, one per expected .get() call, in order."""
    client = AsyncMock()
    client.get = AsyncMock(side_effect=responses)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


class TestCharCount(unittest.TestCase):
    def test_delta_of_running_markers(self):
        self.assertEqual(_char_count(_item(change_from=1000, change_to=1042)), 42)

    def test_falls_back_to_text_length_when_markers_missing(self):
        item = _item()
        del item["character_count_change_to"]
        self.assertEqual(_char_count(item), len(item["text"]))

    def test_zero_when_markers_and_text_absent(self):
        self.assertEqual(_char_count({"history_item_id": "x"}), 0)

    def test_negative_delta_falls_back(self):
        # A change_to < change_from is nonsensical; must not yield a negative.
        self.assertEqual(_char_count(_item(change_from=100, change_to=50, text="hi")), 2)


class TestFetchGenerations(unittest.IsolatedAsyncioTestCase):
    async def test_maps_item_fields(self):
        client = _mock_client([_resp(body={"history": [_item()], "has_more": False})])
        with patch("httpx.AsyncClient", return_value=client):
            provider = ElevenLabsProvider("key")
            gens = await provider.fetch_generations(since_unix=0)

        self.assertEqual(len(gens), 1)
        g = gens[0]
        self.assertEqual(g.generation_id, "hist-1")
        self.assertEqual(g.voice_id, "voice-abc")
        self.assertEqual(g.voice_name, "Rachel")
        self.assertEqual(g.model, "eleven_multilingual_v2")
        self.assertEqual(g.character_count, 42)
        self.assertEqual(g.text, "your order shipped")
        self.assertEqual(g.source, "TTS")
        self.assertEqual(g.generated_at, 1_752_000_000.0)

    async def test_sends_xi_api_key_header(self):
        client = _mock_client([_resp()])
        with patch("httpx.AsyncClient", return_value=client):
            provider = ElevenLabsProvider("my-secret-key")
            await provider.fetch_generations(since_unix=0)
        _, kwargs = client.get.call_args
        self.assertEqual(kwargs["headers"], {"xi-api-key": "my-secret-key"})

    async def test_item_without_id_is_skipped(self):
        good = _item(history_item_id="hist-1")
        bad = _item(history_item_id=None)
        client = _mock_client([_resp(body={"history": [good, bad], "has_more": False})])
        with patch("httpx.AsyncClient", return_value=client):
            provider = ElevenLabsProvider("key")
            gens = await provider.fetch_generations(since_unix=0)
        self.assertEqual([g.generation_id for g in gens], ["hist-1"])

    async def test_paginates_with_cursor_until_has_more_false(self):
        page1 = _resp(
            body={
                "history": [_item(history_item_id="h1", date_unix=200)],
                "has_more": True,
                "last_history_item_id": "h1",
            }
        )
        page2 = _resp(
            body={
                "history": [_item(history_item_id="h2", date_unix=150)],
                "has_more": False,
            }
        )
        client = _mock_client([page1, page2])
        with patch("httpx.AsyncClient", return_value=client):
            provider = ElevenLabsProvider("key")
            gens = await provider.fetch_generations(since_unix=100)

        self.assertEqual({g.generation_id for g in gens}, {"h1", "h2"})
        self.assertEqual(client.get.call_count, 2)
        # Second call must carry the cursor from page 1.
        _, kwargs2 = client.get.call_args_list[1]
        self.assertEqual(kwargs2["params"]["start_after_history_item_id"], "h1")

    async def test_stops_when_page_crosses_below_since(self):
        # Desc order: once an item is older than `since`, everything after it is
        # too, so pagination must stop even though has_more is True.
        page1 = _resp(
            body={
                "history": [
                    _item(history_item_id="new", date_unix=500),
                    _item(history_item_id="old", date_unix=50),
                ],
                "has_more": True,
                "last_history_item_id": "old",
            }
        )
        client = _mock_client([page1, _resp()])
        with patch("httpx.AsyncClient", return_value=client):
            provider = ElevenLabsProvider("key")
            gens = await provider.fetch_generations(since_unix=100)

        self.assertEqual([g.generation_id for g in gens], ["new"])  # "old" dropped
        self.assertEqual(client.get.call_count, 1)  # did not fetch page 2

    async def test_empty_history_returns_empty(self):
        client = _mock_client([_resp(body={"history": [], "has_more": False})])
        with patch("httpx.AsyncClient", return_value=client):
            provider = ElevenLabsProvider("key")
            gens = await provider.fetch_generations(since_unix=0)
        self.assertEqual(gens, [])


class TestRateLimitBackoff(unittest.IsolatedAsyncioTestCase):
    async def test_429_then_success_retries_after_backoff(self):
        rate_limited = _resp(status=429, headers={})
        ok = _resp(body={"history": [_item()], "has_more": False})
        client = _mock_client([rate_limited, ok])
        with (
            patch("httpx.AsyncClient", return_value=client),
            patch("integrations_svc.providers.elevenlabs.asyncio.sleep", AsyncMock()) as sleep_mock,
        ):
            provider = ElevenLabsProvider("key")
            gens = await provider.fetch_generations(since_unix=0)

        self.assertEqual(len(gens), 1)
        self.assertEqual(client.get.call_count, 2)  # retried once
        sleep_mock.assert_awaited()  # backed off before retry

    async def test_429_honors_retry_after_header(self):
        rate_limited = _resp(status=429, headers={"retry-after": "7"})
        ok = _resp(body={"history": [], "has_more": False})
        client = _mock_client([rate_limited, ok])
        with (
            patch("httpx.AsyncClient", return_value=client),
            patch("integrations_svc.providers.elevenlabs.asyncio.sleep", AsyncMock()) as sleep_mock,
        ):
            provider = ElevenLabsProvider("key")
            await provider.fetch_generations(since_unix=0)

        sleep_mock.assert_awaited_once()
        self.assertEqual(sleep_mock.await_args.args[0], 7.0)  # exact Retry-After honored

    async def test_persistent_429_gives_up_and_raises(self):
        # Six 429s: retries exhaust at attempt 6 (_MAX_RETRIES=5) and raise.
        responses = [_resp(status=429, raises=True) for _ in range(6)]
        client = _mock_client(responses)
        with (
            patch("httpx.AsyncClient", return_value=client),
            patch("integrations_svc.providers.elevenlabs.asyncio.sleep", AsyncMock()),
        ):
            provider = ElevenLabsProvider("key")
            with self.assertRaises(httpx.HTTPStatusError):
                await provider.fetch_generations(since_unix=0)

    async def test_5xx_is_retried(self):
        server_err = _resp(status=503)
        ok = _resp(body={"history": [], "has_more": False})
        client = _mock_client([server_err, ok])
        with (
            patch("httpx.AsyncClient", return_value=client),
            patch("integrations_svc.providers.elevenlabs.asyncio.sleep", AsyncMock()),
        ):
            provider = ElevenLabsProvider("key")
            await provider.fetch_generations(since_unix=0)
        self.assertEqual(client.get.call_count, 2)


if __name__ == "__main__":
    unittest.main()
