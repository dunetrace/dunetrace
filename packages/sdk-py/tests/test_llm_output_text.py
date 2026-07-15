"""
LlmCall.output_text — typed access to raw LLM output text, plus the
DUNETRACE_OMIT_LLM_OUTPUT_TEXT bandwidth opt-out.

Run: python -m unittest tests.test_llm_output_text -v
"""

from __future__ import annotations

import os
import unittest

from dunetrace import Dunetrace
from dunetrace.models import EventType


def _client():
    c = Dunetrace(api_key="k")
    c._ship = lambda batch: None
    return c


def _last_llm_responded(run):
    return [e for e in run.state.events if e.event_type == EventType.LLM_RESPONDED][-1]


class TestLlmOutputText(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("DUNETRACE_OMIT_LLM_OUTPUT_TEXT", None)

    def test_output_text_populated_on_struct(self):
        c = _client()
        with c.run("a") as run:
            run.llm_called("gpt-4o")
            run.llm_responded(output="the answer is 42", output_length=16)
            self.assertEqual(run.state.llm_calls[-1].output_text, "the answer is 42")
        c.shutdown(timeout=1)

    def test_output_text_none_when_no_output_passed(self):
        c = _client()
        with c.run("a") as run:
            run.llm_called("gpt-4o")
            run.llm_responded(output_length=0)  # no output=
            self.assertIsNone(run.state.llm_calls[-1].output_text)
        c.shutdown(timeout=1)

    def test_output_transmitted_by_default(self):
        c = _client()
        with c.run("a") as run:
            run.llm_called("gpt-4o")
            run.llm_responded(output="hello", output_length=5)
            payload = _last_llm_responded(run).payload
        c.shutdown(timeout=1)
        self.assertEqual(payload.get("output"), "hello")
        self.assertEqual(payload.get("output_length"), 5)

    def test_opt_out_omits_output_from_payload_but_keeps_struct(self):
        os.environ["DUNETRACE_OMIT_LLM_OUTPUT_TEXT"] = "1"
        c = _client()
        with c.run("a") as run:
            run.llm_called("gpt-4o")
            run.llm_responded(output="sensitive text", output_length=14)
            payload = _last_llm_responded(run).payload
            struct_text = run.state.llm_calls[-1].output_text
        c.shutdown(timeout=1)
        # Wire omits the text (bandwidth), but output_length still travels…
        self.assertNotIn("output", payload)
        self.assertEqual(payload.get("output_length"), 14)
        # …and the in-process struct keeps full fidelity for local detectors.
        self.assertEqual(struct_text, "sensitive text")


if __name__ == "__main__":
    unittest.main(verbosity=2)
