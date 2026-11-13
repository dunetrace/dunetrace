"""
Content caps and secret redaction on the SDK path (dunetrace.redaction, and
its wiring through RunContext and Dunetrace.run).

No network required: the client's `_ship` is replaced with a list collector,
the same seam tests/test_client.py uses, so every assertion is against the
exact AgentEvent objects that would have gone on the wire.

Run: python -m pytest tests/test_redaction.py -v
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import statistics
import time
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from dunetrace import run_context as run_context_module
from dunetrace.client import Dunetrace, _resolve_max_field_chars
from dunetrace.models import EventType
from dunetrace.redaction import (
    DEFAULT_DENYLIST,
    DEFAULT_MAX_FIELD_CHARS,
    MAX_REDACT_DEPTH,
    REDACTED,
    cap_text,
    compile_denylist,
    key_is_sensitive,
    put_capped,
    redact_dict,
    serialize_capped,
)
from dunetrace.run_context import RunContext

MARKER_SUFFIXES = ("_truncated", "_original_length")


def _make_client(**kwargs):
    """A client whose shipped batches land in the returned list."""
    defaults = dict(api_key="dt_test", debug=False)
    defaults.update(kwargs)
    client = Dunetrace(**defaults)
    emitted: list = []
    client._ship = lambda batch: emitted.extend(batch)
    return client, emitted


def _wire(client, emitted, event_type):
    """Flush and return the shipped events of one type, in order."""
    client.shutdown(timeout=2)
    return [e for e in emitted if e.event_type == event_type]


def _marker_keys(payload: dict) -> list:
    return sorted(k for k in payload if k.endswith(MARKER_SUFFIXES))


@contextmanager
def _capture(logger_name: str):
    """Collect every record a logger emits (any level), unlike assertLogs
    this does not fail when nothing is logged."""
    records: list = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record)

    lg = logging.getLogger(logger_name)
    h = _H(level=logging.DEBUG)
    old_level = lg.level
    lg.addHandler(h)
    lg.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        lg.removeHandler(h)
        lg.setLevel(old_level)


# ── redact_dict / key matching ────────────────────────────────────────────────


class TestKeyMatching(unittest.TestCase):
    def test_default_denylist_contents(self):
        self.assertEqual(
            DEFAULT_DENYLIST,
            frozenset(
                {
                    "authorization",
                    "api_key",
                    "apikey",
                    "token",
                    "secret",
                    "password",
                    "cookie",
                    "set-cookie",
                }
            ),
        )

    def test_exact_match_is_case_insensitive_and_hyphen_normalised(self):
        for key in (
            "Authorization",
            "AUTHORIZATION",
            "Set-Cookie",
            "set_cookie",
            "apiKey",
            "API-KEY",
        ):
            self.assertTrue(key_is_sensitive(key), key)

    def test_suffix_match(self):
        for key in (
            "access_token",
            "refresh-token",
            "client_secret",
            "db_password",
            "X-Api-Key",
            "x_apikey",
            "Proxy-Authorization",
        ):
            self.assertTrue(key_is_sensitive(key), key)

    def test_non_matching_keys(self):
        # Substring is not enough: only equality or an `_<entry>` suffix matches.
        for key in (
            "tokenizer",
            "secret_sauce",
            "passwords_seen",
            "username",
            "cookies_enabled",
            "q",
        ):
            self.assertFalse(key_is_sensitive(key), key)

    def test_non_string_keys_never_match(self):
        self.assertFalse(key_is_sensitive(1))
        self.assertFalse(key_is_sensitive(("token",)))
        self.assertFalse(key_is_sensitive(None))

    def test_compile_denylist_extends_and_normalises(self):
        compiled = compile_denylist(["X-Session-Id", "  ssn "])
        self.assertTrue(key_is_sensitive("x_session_id", compiled))
        self.assertTrue(key_is_sensitive("user_ssn", compiled))
        self.assertTrue(key_is_sensitive("password", compiled))  # defaults kept
        self.assertFalse(key_is_sensitive("x_session_id"))  # not in the default set


class TestRedactDict(unittest.TestCase):
    def test_nested_dict(self):
        args = {
            "headers": {"Authorization": "Bearer abc", "X-Api-Key": "k1", "Accept": "json"},
            "body": {"access_token": "t", "client_secret": "s", "name": "ok"},
        }
        out = redact_dict(args)
        self.assertEqual(out["headers"]["Authorization"], REDACTED)
        self.assertEqual(out["headers"]["X-Api-Key"], REDACTED)
        self.assertEqual(out["headers"]["Accept"], "json")
        self.assertEqual(out["body"]["access_token"], REDACTED)
        self.assertEqual(out["body"]["client_secret"], REDACTED)
        self.assertEqual(out["body"]["name"], "ok")

    def test_returns_new_structure_and_never_mutates_input(self):
        args = {"password": "p", "nested": {"token": "t"}, "items": [{"secret": "s"}]}
        snapshot = json.dumps(args, sort_keys=True)
        out = redact_dict(args)
        self.assertIsNot(out, args)
        self.assertIsNot(out["nested"], args["nested"])
        self.assertEqual(json.dumps(args, sort_keys=True), snapshot)

    def test_list_of_dicts_and_tuple_of_dicts(self):
        out = redact_dict([{"password": "p"}, {"ok": 1}, "plain", 3])
        self.assertEqual(out, [{"password": REDACTED}, {"ok": 1}, "plain", 3])
        out_t = redact_dict(({"api_key": "k"},))
        self.assertIsInstance(out_t, tuple)
        self.assertEqual(out_t[0]["api_key"], REDACTED)

    def test_top_level_list_input(self):
        out = redact_dict([{"cookie": "c=1"}])
        self.assertEqual(out[0]["cookie"], REDACTED)

    def test_scalar_input_passes_through(self):
        self.assertEqual(redact_dict("password=hunter2"), "password=hunter2")
        self.assertIsNone(redact_dict(None))

    def test_keys_are_kept_only_values_replaced(self):
        out = redact_dict({"token": {"deep": "x"}})
        self.assertEqual(list(out), ["token"])
        self.assertEqual(out["token"], REDACTED)

    def test_extra_keys_iterable(self):
        out = redact_dict({"ssn": "1", "password": "p"}, ["ssn"])
        self.assertEqual(out, {"ssn": REDACTED, "password": REDACTED})

    def test_precompiled_denylist_accepted(self):
        compiled = compile_denylist(["ssn"])
        self.assertEqual(redact_dict({"ssn": "1"}, compiled), {"ssn": REDACTED})

    def test_depth_limited_does_not_raise_and_passes_deep_nodes_through(self):
        node: dict = {"password": "bottom"}
        for _ in range(MAX_REDACT_DEPTH + 5):
            node = {"child": node, "token": "t"}
        out = redact_dict(node)
        # Top level was inspected...
        self.assertEqual(out["token"], REDACTED)
        # ...and nothing raised; the node past the depth limit is passed
        # through untouched (documented: the limit stops inspecting, it does
        # not hide that data was sent).
        cur = out
        depth = 0
        while isinstance(cur, dict) and "child" in cur:
            cur = cur["child"]
            depth += 1
        self.assertEqual(cur, {"password": "bottom"})
        self.assertGreater(depth, MAX_REDACT_DEPTH - 1)

    def test_never_raises_returns_original_on_internal_failure(self):
        class Hostile(dict):
            def items(self):
                raise RuntimeError("no")

        h = Hostile(password="p")
        self.assertIs(redact_dict(h), h)

    def test_non_string_dict_keys_are_structure(self):
        out = redact_dict({1: "x", ("token",): "y", "token": "z"})
        self.assertEqual(out[1], "x")
        self.assertEqual(out[("token",)], "y")
        self.assertEqual(out["token"], REDACTED)


# ── cap_text / serialize_capped / put_capped ──────────────────────────────────


class TestCapping(unittest.TestCase):
    def test_cap_text_cuts_and_reports_original_length(self):
        text, truncated, n = cap_text("x" * 100, 10)
        self.assertEqual((text, truncated, n), ("x" * 10, True, 100))

    def test_cap_text_small_unchanged(self):
        self.assertEqual(cap_text("abc", 10), ("abc", False, 3))
        self.assertEqual(cap_text("abc", 3), ("abc", False, 3))

    def test_cap_text_zero_and_none_disable(self):
        big = "x" * 50_000
        self.assertEqual(cap_text(big, 0), (big, False, 50_000))
        self.assertEqual(cap_text(big, None), (big, False, 50_000))
        self.assertEqual(cap_text(big, -5), (big, False, 50_000))

    def test_cap_text_coerces_non_str(self):
        self.assertEqual(cap_text(12345, 3), ("123", True, 5))

    def test_serialize_capped_json_default_str(self):
        class Thing:
            def __str__(self):
                return "<thing>"

        when = datetime.datetime(2026, 9, 10, 12, 0, 0)
        text, truncated, n = serialize_capped({"when": when, "obj": Thing(), "n": 1}, 0)
        self.assertFalse(truncated)
        self.assertEqual(n, len(text))
        parsed = json.loads(text)
        self.assertEqual(parsed, {"when": "2026-09-10 12:00:00", "obj": "<thing>", "n": 1})

    def test_serialize_capped_keeps_unicode(self):
        text, _, _ = serialize_capped({"q": "café ☕"}, 0)
        self.assertIn("café ☕", text)

    def test_serialize_capped_circular_falls_back_to_repr(self):
        a: dict = {}
        a["self"] = a
        text, truncated, n = serialize_capped(a, 0)
        self.assertIsInstance(text, str)
        self.assertIn("...", text)  # repr of a recursive dict
        self.assertFalse(truncated)

    def test_serialize_capped_never_raises(self):
        class Cursed:
            def __str__(self):
                raise RuntimeError("str")

            def __repr__(self):
                raise RuntimeError("repr")

        text, _, _ = serialize_capped({("t",): Cursed()}, 0)  # tuple key → dumps fails
        self.assertIsInstance(text, str)
        self.assertTrue(text)

    def test_serialize_capped_caps(self):
        text, truncated, n = serialize_capped({"blob": "y" * 1000}, 16)
        self.assertEqual(len(text), 16)
        self.assertTrue(truncated)
        self.assertEqual(n, len(json.dumps({"blob": "y" * 1000})))

    def test_put_capped_markers_only_when_cut(self):
        payload: dict = {}
        self.assertEqual(put_capped(payload, "output", "short", 10), "short")
        self.assertEqual(payload, {"output": "short"})
        payload = {}
        self.assertEqual(put_capped(payload, "output", "x" * 20, 10), "x" * 10)
        self.assertEqual(
            payload, {"output": "x" * 10, "output_truncated": True, "output_original_length": 20}
        )

    def test_put_capped_non_str_stored_untouched(self):
        payload: dict = {}
        put_capped(payload, "value", 42, 1)
        self.assertEqual(payload, {"value": 42})


# ── Wire format: truncation markers ───────────────────────────────────────────


class TestWireTruncationMarkers(unittest.TestCase):
    CAP = 64

    def test_tool_called_args(self):
        client, emitted = _make_client(max_field_chars=self.CAP)
        args = {"query": "q" * 500, "n": 1}
        expected_len = len(json.dumps(args))
        with client.run("agent") as run:
            run.tool_called("search", args)
            tc = run.state.tool_calls[0]
        ev = _wire(client, emitted, EventType.TOOL_CALLED)[0]
        self.assertEqual(len(ev.payload["args"]), self.CAP)
        self.assertIs(ev.payload["args_truncated"], True)
        self.assertEqual(ev.payload["args_original_length"], expected_len)
        self.assertEqual(ev.payload["args_length"], expected_len)
        # In-path detectors see exactly what shipped, plus the real length.
        self.assertEqual(tc.args, ev.payload["args"])
        self.assertEqual(tc.args_length, expected_len)

    def test_tool_responded_output_length_is_real_length(self):
        client, emitted = _make_client(max_field_chars=self.CAP)
        out = "o" * 300
        with client.run("agent") as run:
            run.tool_called("search", {"q": 1})
            run.tool_responded("search", success=True, output=out)
            tc = run.state.tool_calls[0]
        ev = _wire(client, emitted, EventType.TOOL_RESPONDED)[0]
        self.assertEqual(ev.payload["output"], "o" * self.CAP)
        self.assertIs(ev.payload["output_truncated"], True)
        self.assertEqual(ev.payload["output_original_length"], 300)
        self.assertEqual(ev.payload["output_length"], 300)
        self.assertEqual(tc.output, "o" * self.CAP)
        self.assertEqual(tc.output_length, 300)

    def test_tool_responded_explicit_output_length_kept(self):
        client, emitted = _make_client(max_field_chars=self.CAP)
        with client.run("agent") as run:
            run.tool_called("search", {"q": 1})
            run.tool_responded("search", output="o" * 300, output_length=9999)
        ev = _wire(client, emitted, EventType.TOOL_RESPONDED)[0]
        self.assertEqual(ev.payload["output_length"], 9999)
        self.assertEqual(ev.payload["output_original_length"], 300)

    def test_llm_responded_output(self):
        client, emitted = _make_client(max_field_chars=self.CAP)
        out = "l" * 200
        with client.run("agent") as run:
            run.llm_called("gpt-4o", prompt_tokens=10)
            run.llm_responded(output=out, output_length=len(out), finish_reason="stop")
            lc = run.state.llm_calls[0]
        ev = _wire(client, emitted, EventType.LLM_RESPONDED)[0]
        self.assertEqual(ev.payload["output"], "l" * self.CAP)
        self.assertIs(ev.payload["output_truncated"], True)
        self.assertEqual(ev.payload["output_original_length"], 200)
        self.assertEqual(ev.payload["output_length"], 200)
        self.assertEqual(lc.output_text, "l" * self.CAP)
        self.assertEqual(lc.output_length, 200)

    def test_retrieval_query_and_content(self):
        client, emitted = _make_client(max_field_chars=self.CAP)
        with client.run("agent") as run:
            run.retrieval_called("docs", "q" * 100)
            run.retrieval_responded("docs", 3, 0.9, content="c" * 100)
            rr = run.state.retrievals[0]
        called = _wire(client, emitted, EventType.RETRIEVAL_CALLED)[0]
        responded = [e for e in emitted if e.event_type == EventType.RETRIEVAL_RESPONDED][0]
        self.assertEqual(called.payload["query"], "q" * self.CAP)
        self.assertIs(called.payload["query_truncated"], True)
        self.assertEqual(called.payload["query_original_length"], 100)
        self.assertEqual(responded.payload["content"], "c" * self.CAP)
        self.assertIs(responded.payload["content_truncated"], True)
        self.assertEqual(responded.payload["content_original_length"], 100)
        self.assertEqual(rr.content, "c" * self.CAP)

    def test_memory_written_value(self):
        client, emitted = _make_client(max_field_chars=self.CAP)
        with client.run("agent") as run:
            run.memory_written("prefs", "v" * 100, source="user_input")
            me = run.state.memory_events[0]
        ev = _wire(client, emitted, EventType.MEMORY_WRITTEN)[0]
        self.assertEqual(ev.payload["value"], "v" * self.CAP)
        self.assertIs(ev.payload["value_truncated"], True)
        self.assertEqual(ev.payload["value_original_length"], 100)
        self.assertEqual(ev.payload["source"], "user_input")
        self.assertEqual(me.value, "v" * self.CAP)

    def test_run_started_input_text_and_system_prompt(self):
        client, emitted = _make_client(max_field_chars=self.CAP)
        with client.run("agent", user_input="u" * 100, system_prompt="s" * 100) as run:
            state = run.state
        ev = _wire(client, emitted, EventType.RUN_STARTED)[0]
        self.assertEqual(ev.payload["input_text"], "u" * self.CAP)
        self.assertIs(ev.payload["input_text_truncated"], True)
        self.assertEqual(ev.payload["input_text_original_length"], 100)
        self.assertEqual(ev.payload["system_prompt"], "s" * self.CAP)
        self.assertIs(ev.payload["system_prompt_truncated"], True)
        self.assertEqual(ev.payload["system_prompt_original_length"], 100)
        self.assertEqual(state.input_text, "u" * self.CAP)
        self.assertEqual(state.system_prompt, "s" * self.CAP)

    def test_run_started_only_marks_the_field_that_was_cut(self):
        client, emitted = _make_client(max_field_chars=self.CAP)
        with client.run("agent", user_input="short", system_prompt="s" * 100):
            pass
        ev = _wire(client, emitted, EventType.RUN_STARTED)[0]
        self.assertEqual(
            _marker_keys(ev.payload), ["system_prompt_original_length", "system_prompt_truncated"]
        )

    def test_approval_request_tool_args(self):
        client, emitted = _make_client(max_field_chars=self.CAP)
        client._create_approval_request = MagicMock(return_value={"id": 1})
        client._get_approval = MagicMock(return_value={"status": "granted"})
        with client.run("agent") as run:
            run.request_approval("wire", {"password": "hunter2", "memo": "m" * 500}, timeout_s=60)
        client.shutdown(timeout=2)
        tool_args = client._create_approval_request.call_args.kwargs["tool_args"]
        self.assertEqual(len(tool_args), self.CAP)
        self.assertIn(REDACTED, tool_args)
        self.assertNotIn("hunter2", tool_args)

    def test_approval_request_none_args(self):
        client, emitted = _make_client()
        client._create_approval_request = MagicMock(return_value={"id": 1})
        client._get_approval = MagicMock(return_value={"status": "granted"})
        with client.run("agent") as run:
            run.request_approval("wire", timeout_s=60)
        client.shutdown(timeout=2)
        self.assertIsNone(client._create_approval_request.call_args.kwargs["tool_args"])


class TestWireSmallPayloadsUnchanged(unittest.TestCase):
    """Ordinary payloads carry no marker keys and are byte-for-byte what they
    were before caps existed."""

    def test_no_markers_anywhere_on_a_small_run(self):
        client, emitted = _make_client()
        args = {"expr": "1+1", "opts": {"precise": True}}
        with client.run("agent", user_input="hi", system_prompt="be terse") as run:
            run.llm_called("gpt-4o", prompt_tokens=5)
            run.llm_responded(output="2", output_length=1)
            run.tool_called("calc", args)
            run.tool_responded("calc", output="2")
            run.retrieval_called("docs", "refunds")
            run.retrieval_responded("docs", 1, 0.5, content="Refunds within 5 days.")
            run.memory_written("k", "v")
            tc = run.state.tool_calls[0]
        client.shutdown(timeout=2)
        for ev in emitted:
            self.assertEqual(_marker_keys(ev.payload), [], ev.event_type)
        called = next(e for e in emitted if e.event_type == EventType.TOOL_CALLED)
        self.assertEqual(called.payload, {"tool_name": "calc", "args": json.dumps(args)})
        self.assertEqual(tc.args, json.dumps(args))
        self.assertIsNone(tc.args_length)
        responded = next(e for e in emitted if e.event_type == EventType.TOOL_RESPONDED)
        self.assertEqual(responded.payload["output"], "2")
        self.assertEqual(responded.payload["output_length"], 1)
        started = next(e for e in emitted if e.event_type == EventType.RUN_STARTED)
        self.assertEqual(started.payload["input_text"], "hi")
        self.assertEqual(started.payload["system_prompt"], "be terse")

    def test_field_exactly_at_cap_is_not_marked(self):
        client, emitted = _make_client(max_field_chars=8)
        with client.run("agent") as run:
            run.tool_called("t", {"q": 1})
            run.tool_responded("t", output="x" * 8)
        ev = _wire(client, emitted, EventType.TOOL_RESPONDED)[0]
        self.assertEqual(ev.payload["output"], "x" * 8)
        self.assertEqual(_marker_keys(ev.payload), [])

    def test_none_args_ship_as_empty_object(self):
        client, emitted = _make_client()
        with client.run("agent") as run:
            run.tool_called("t")
        ev = _wire(client, emitted, EventType.TOOL_CALLED)[0]
        self.assertEqual(ev.payload["args"], "{}")

    def test_string_args_ship_verbatim(self):
        client, emitted = _make_client()
        with client.run("agent") as run:
            run.tool_called("handoff", "plain text args")  # type: ignore[arg-type]
        ev = _wire(client, emitted, EventType.TOOL_CALLED)[0]
        self.assertEqual(ev.payload["args"], "plain text args")

    def test_list_args_are_json_and_redacted(self):
        client, emitted = _make_client()
        with client.run("agent") as run:
            run.tool_called("batch", [{"token": "t", "id": 1}])  # type: ignore[arg-type]
        ev = _wire(client, emitted, EventType.TOOL_CALLED)[0]
        self.assertEqual(json.loads(ev.payload["args"]), [{"token": REDACTED, "id": 1}])

    def test_cap_zero_disables_everywhere(self):
        client, emitted = _make_client(max_field_chars=0)
        self.assertEqual(client._max_field_chars, 0)
        big = "z" * 100_000
        with client.run("agent", user_input=big, system_prompt=big) as run:
            run.tool_called("t", {"blob": big})
            run.tool_responded("t", output=big)
            run.llm_called("gpt-4o", prompt_tokens=1)
            run.llm_responded(output=big, output_length=len(big))
            run.retrieval_called("d", big)
            run.retrieval_responded("d", 1, 0.1, content=big)
            run.memory_written("k", big)
        client.shutdown(timeout=2)
        for ev in emitted:
            self.assertEqual(_marker_keys(ev.payload), [], ev.event_type)
        started = next(e for e in emitted if e.event_type == EventType.RUN_STARTED)
        self.assertEqual(len(started.payload["input_text"]), 100_000)
        called = next(e for e in emitted if e.event_type == EventType.TOOL_CALLED)
        self.assertGreater(len(called.payload["args"]), 100_000)


# ── Wire format: redaction ────────────────────────────────────────────────────


class TestWireRedaction(unittest.TestCase):
    def setUp(self):
        self._warned = run_context_module._redact_hook_warned
        run_context_module._redact_hook_warned = False

    def tearDown(self):
        run_context_module._redact_hook_warned = self._warned

    def test_default_denylist_applied_to_tool_args(self):
        client, emitted = _make_client()
        args = {
            "url": "https://x",
            "headers": {"Authorization": "Bearer abc", "X-Api-Key": "k"},
            "body": {"access_token": "t", "client_secret": "s", "safe": "yes"},
            "cookies": [{"Set-Cookie": "sid=1"}],
        }
        with client.run("agent") as run:
            run.tool_called("http", args)
            tc = run.state.tool_calls[0]
        ev = _wire(client, emitted, EventType.TOOL_CALLED)[0]
        parsed = json.loads(ev.payload["args"])
        self.assertEqual(parsed["headers"], {"Authorization": REDACTED, "X-Api-Key": REDACTED})
        self.assertEqual(
            parsed["body"], {"access_token": REDACTED, "client_secret": REDACTED, "safe": "yes"}
        )
        self.assertEqual(parsed["cookies"], [{"Set-Cookie": REDACTED}])
        self.assertEqual(parsed["url"], "https://x")
        for secret in ("Bearer abc", '"k"', '"t"', '"s"', "sid=1"):
            self.assertNotIn(secret, ev.payload["args"])
        self.assertEqual(tc.args, ev.payload["args"])
        # The caller's dict is untouched — the tool is about to run on it.
        self.assertEqual(args["headers"]["Authorization"], "Bearer abc")

    def test_redact_keys_extends_denylist(self):
        client, emitted = _make_client(redact_keys=["X-Session-Id", "ssn"])
        with client.run("agent") as run:
            run.tool_called("t", {"x_session_id": "abc", "user_ssn": "123", "name": "n"})
        ev = _wire(client, emitted, EventType.TOOL_CALLED)[0]
        self.assertEqual(
            json.loads(ev.payload["args"]),
            {"x_session_id": REDACTED, "user_ssn": REDACTED, "name": "n"},
        )

    def test_hook_runs_before_denylist_and_sees_raw_args(self):
        seen: list = []

        def hook(args: dict) -> dict:
            seen.append(dict(args))
            out = dict(args)
            out["account"] = "***"
            return out

        client, emitted = _make_client(redact=hook)
        with client.run("agent") as run:
            run.tool_called("pay", {"account": "1234", "password": "p", "amount": 5})
        ev = _wire(client, emitted, EventType.TOOL_CALLED)[0]
        # The hook saw the raw password (it runs first) and the denylist still
        # scrubbed it afterwards; the hook's own edit survived.
        self.assertEqual(seen, [{"account": "1234", "password": "p", "amount": 5}])
        self.assertEqual(
            json.loads(ev.payload["args"]), {"account": "***", "password": REDACTED, "amount": 5}
        )

    def test_hook_receives_a_copy(self):
        def hook(args: dict) -> dict:
            args["injected"] = True  # mutating the copy must not reach the caller
            return args

        client, emitted = _make_client(redact=hook)
        original = {"q": "x"}
        with client.run("agent") as run:
            run.tool_called("t", original)
        self.assertEqual(original, {"q": "x"})
        ev = _wire(client, emitted, EventType.TOOL_CALLED)[0]
        self.assertEqual(json.loads(ev.payload["args"]), {"q": "x", "injected": True})

    def test_hook_raising_is_fail_safe_with_one_warning(self):
        def hook(args: dict) -> dict:
            raise RuntimeError("boom")

        client, emitted = _make_client(redact=hook)
        with _capture("dunetrace.run") as records:
            with client.run("agent") as run:
                run.tool_called("t", {"password": "p", "q": 1})
                run.tool_called("t", {"password": "p2", "q": 2})
                run.tool_called("t", {"password": "p3", "q": 3})
        warnings = [
            r for r in records if r.levelno == logging.WARNING and "redact hook" in r.getMessage()
        ]
        debugs = [
            r for r in records if r.levelno == logging.DEBUG and "redact hook" in r.getMessage()
        ]
        self.assertEqual(len(warnings), 1)
        self.assertIn("RuntimeError: boom", warnings[0].getMessage())
        self.assertEqual(len(debugs), 2)
        # All three calls still shipped, with the denylist applied.
        events = _wire(client, emitted, EventType.TOOL_CALLED)
        self.assertEqual(len(events), 3)
        for ev, q in zip(events, (1, 2, 3)):
            self.assertEqual(json.loads(ev.payload["args"]), {"password": REDACTED, "q": q})

    def test_hook_returning_non_dict_is_fail_safe(self):
        client, emitted = _make_client(redact=lambda args: "nope")  # type: ignore[arg-type,return-value]
        with self.assertLogs("dunetrace.run", level="WARNING") as cm:
            with client.run("agent") as run:
                run.tool_called("t", {"token": "t", "q": 1})
        self.assertEqual(len(cm.records), 1)
        self.assertIn("expected dict", cm.records[0].getMessage())
        ev = _wire(client, emitted, EventType.TOOL_CALLED)[0]
        self.assertEqual(json.loads(ev.payload["args"]), {"token": REDACTED, "q": 1})

    def test_hook_must_be_callable(self):
        with self.assertRaises(TypeError):
            Dunetrace(api_key="dt_test", redact="strip")  # type: ignore[arg-type]

    def test_hook_also_covers_approval_requests(self):
        client, emitted = _make_client(redact=lambda a: {**a, "memo": "***"})
        client._create_approval_request = MagicMock(return_value={"id": 1})
        client._get_approval = MagicMock(return_value={"status": "granted"})
        with client.run("agent") as run:
            run.request_approval("wire", {"memo": "private", "api_key": "k"}, timeout_s=60)
        client.shutdown(timeout=2)
        tool_args = client._create_approval_request.call_args.kwargs["tool_args"]
        self.assertEqual(json.loads(tool_args), {"memo": "***", "api_key": REDACTED})


# ── Configuration: constructor + env ──────────────────────────────────────────


class TestMaxFieldCharsResolution(unittest.TestCase):
    def test_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DUNETRACE_MAX_FIELD_CHARS", None)
            client, _ = _make_client()
        self.assertEqual(client._max_field_chars, DEFAULT_MAX_FIELD_CHARS)
        self.assertEqual(DEFAULT_MAX_FIELD_CHARS, 8192)

    def test_env_override(self):
        with patch.dict(os.environ, {"DUNETRACE_MAX_FIELD_CHARS": "32"}):
            client, emitted = _make_client()
            self.assertEqual(client._max_field_chars, 32)
            with client.run("agent") as run:
                run.tool_called("t", {"q": "x" * 100})
        ev = _wire(client, emitted, EventType.TOOL_CALLED)[0]
        self.assertEqual(len(ev.payload["args"]), 32)
        self.assertIs(ev.payload["args_truncated"], True)

    def test_constructor_beats_env(self):
        with patch.dict(os.environ, {"DUNETRACE_MAX_FIELD_CHARS": "32"}):
            client, _ = _make_client(max_field_chars=1000)
        self.assertEqual(client._max_field_chars, 1000)

    def test_env_zero_disables(self):
        with patch.dict(os.environ, {"DUNETRACE_MAX_FIELD_CHARS": "0"}):
            self.assertEqual(_resolve_max_field_chars(None), 0)

    def test_bad_values_fall_back_with_warning(self):
        with patch.dict(os.environ, {"DUNETRACE_MAX_FIELD_CHARS": "lots"}):
            with self.assertLogs("dunetrace", level="WARNING"):
                self.assertEqual(_resolve_max_field_chars(None), DEFAULT_MAX_FIELD_CHARS)
        with self.assertLogs("dunetrace", level="WARNING"):
            self.assertEqual(_resolve_max_field_chars(-1), DEFAULT_MAX_FIELD_CHARS)
        with self.assertLogs("dunetrace", level="WARNING"):
            self.assertEqual(_resolve_max_field_chars("abc"), DEFAULT_MAX_FIELD_CHARS)  # type: ignore[arg-type]

    def test_explicit_zero_disables(self):
        self.assertEqual(_resolve_max_field_chars(0), 0)

    def test_env_whitespace_is_default(self):
        with patch.dict(os.environ, {"DUNETRACE_MAX_FIELD_CHARS": "  "}):
            self.assertEqual(_resolve_max_field_chars(None), DEFAULT_MAX_FIELD_CHARS)


# ── Duck-typed / mock clients (regression) ────────────────────────────────────


class TestDuckTypedClient(unittest.TestCase):
    """The framework callback handlers are unit-tested with MagicMock()
    clients, which answer every attribute lookup — so ``getattr(client, name,
    default)`` never falls back and a mock read as a configured client with a
    MagicMock cap and a MagicMock hook. A RunContext must treat such a client
    as unconfigured: default cap, no hook, built-in denylist, no warnings."""

    def setUp(self):
        self._warned = run_context_module._redact_hook_warned
        run_context_module._redact_hook_warned = False

    def tearDown(self):
        run_context_module._redact_hook_warned = self._warned

    def _ctx(self, client):
        return RunContext(
            client=client,
            agent_id="a",
            agent_version="v",
            available_tools=[],
            input_text="",
        )

    def test_magicmock_client_reads_as_unconfigured(self):
        ctx = self._ctx(MagicMock())
        self.assertEqual(ctx._max_field_chars, DEFAULT_MAX_FIELD_CHARS)
        self.assertIsNone(ctx._redact_hook)
        self.assertIsNone(ctx._redact_denylist)

    def test_magicmock_client_tool_called_caps_and_redacts_without_warning(self):
        ctx = self._ctx(MagicMock())
        with _capture("dunetrace.run") as records:
            ctx.tool_called("t", {"password": "p", "blob": "x" * 20_000})
            ctx.tool_responded("t", output="o" * 20_000)
        self.assertEqual([r for r in records if r.levelno >= logging.WARNING], [])
        called = next(e for e in ctx.state.events if e.event_type == EventType.TOOL_CALLED)
        self.assertEqual(len(called.payload["args"]), DEFAULT_MAX_FIELD_CHARS)
        self.assertIs(called.payload["args_truncated"], True)
        self.assertIn(REDACTED, called.payload["args"])
        self.assertNotIn('"p"', called.payload["args"])
        responded = next(e for e in ctx.state.events if e.event_type == EventType.TOOL_RESPONDED)
        self.assertEqual(len(responded.payload["output"]), DEFAULT_MAX_FIELD_CHARS)
        self.assertEqual(responded.payload["output_length"], 20_000)

    def test_bare_object_client_reads_as_unconfigured(self):
        class Stub:
            def _emit(self, event):
                pass

        ctx = self._ctx(Stub())
        self.assertEqual(ctx._max_field_chars, DEFAULT_MAX_FIELD_CHARS)
        self.assertIsNone(ctx._redact_hook)
        self.assertIsNone(ctx._redact_denylist)

    def test_wrong_typed_attributes_read_as_unconfigured(self):
        class Stub:
            _max_field_chars = "8192"  # str, not int
            _redact_hook = "not callable"
            _redact_denylist = ["token"]  # not the compiled pair

            def _emit(self, event):
                pass

        ctx = self._ctx(Stub())
        self.assertEqual(ctx._max_field_chars, DEFAULT_MAX_FIELD_CHARS)
        self.assertIsNone(ctx._redact_hook)
        self.assertIsNone(ctx._redact_denylist)

    def test_bool_cap_is_not_an_int_cap(self):
        class Stub:
            _max_field_chars = True

            def _emit(self, event):
                pass

        self.assertEqual(self._ctx(Stub())._max_field_chars, DEFAULT_MAX_FIELD_CHARS)

    def test_real_client_settings_are_read(self):
        client, _ = _make_client(max_field_chars=77, redact=lambda a: a, redact_keys=["ssn"])
        ctx = self._ctx(client)
        self.assertEqual(ctx._max_field_chars, 77)
        self.assertIs(ctx._redact_hook, client._redact_hook)
        self.assertIs(ctx._redact_denylist, client._redact_denylist)

    def test_run_on_client_built_without_init(self):
        """Framework tests build the client via __new__; run() must still start."""
        from dunetrace.policies import PolicyEngine

        dt = Dunetrace.__new__(Dunetrace)
        dt._ingest_url = "http://localhost:8001/v1/ingest"
        dt._api_key = ""
        dt._otel_exporter = None
        dt._policy_engine = PolicyEngine()
        emitted: list = []
        dt._emit = lambda event: emitted.append(event)  # type: ignore[method-assign]
        dt.flush = lambda: None  # type: ignore[method-assign]
        with dt.run("agent", user_input="u" * (DEFAULT_MAX_FIELD_CHARS + 10)):
            pass
        started = next(e for e in emitted if e.event_type == EventType.RUN_STARTED)
        self.assertEqual(len(started.payload["input_text"]), DEFAULT_MAX_FIELD_CHARS)
        self.assertIs(started.payload["input_text_truncated"], True)


# ── Cost ──────────────────────────────────────────────────────────────────────


class TestTiming(unittest.TestCase):
    """Loose, non-flaky bounds: the hook must stay sub-millisecond for
    ordinary args, and a pathological value must be cut, not serialised in
    full on every emit."""

    def test_ordinary_4kb_args_median_under_2ms(self):
        client, _ = _make_client()
        args = {f"field_{i}": "v" * 100 for i in range(38)}  # ~4KB of JSON
        args["headers"] = {"Authorization": "Bearer x", "Accept": "json"}
        self.assertGreater(len(json.dumps(args)), 3800)
        samples = []
        with client.run("agent") as run:
            for _ in range(200):
                t0 = time.perf_counter()
                run.tool_called("t", args)
                samples.append(time.perf_counter() - t0)
        client.shutdown(timeout=2)
        self.assertLess(statistics.median(samples) * 1000, 2.0)

    def test_1mb_args_capped_well_under_100ms(self):
        client, emitted = _make_client()
        args = {"blob": "x" * 1_000_000}
        with client.run("agent") as run:
            t0 = time.perf_counter()
            run.tool_called("t", args)
            elapsed_ms = (time.perf_counter() - t0) * 1000
        ev = _wire(client, emitted, EventType.TOOL_CALLED)[0]
        self.assertLess(elapsed_ms, 100)
        self.assertEqual(len(ev.payload["args"]), DEFAULT_MAX_FIELD_CHARS)
        self.assertEqual(ev.payload["args_original_length"], len(json.dumps(args)))


if __name__ == "__main__":
    unittest.main()
