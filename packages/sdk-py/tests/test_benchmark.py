"""
Microbenchmark for SDK instrumentation overhead.

Confirms the <500µs per-hook overhead claim holds after the Phase A refactor
(raw content passthrough instead of SHA-256 hashing). Network I/O is excluded
from these numbers — the background drain thread ships events asynchronously,
so "overhead" here measures the synchronous, in-process cost of each hook
call: building the event and appending it to the buffer.

Run: python -m unittest tests.test_benchmark -v
"""

from __future__ import annotations

import time
import unittest

from dunetrace.client import DunetraceClient
from dunetrace.detectors import PROMPT_INJECTION_DETECTOR

MAX_OVERHEAD_US = 500
ITERATIONS = 5_000
WARMUP = 200


def _time_hook(fn, iterations: int = ITERATIONS, warmup: int = WARMUP) -> float:
    """Mean wall-clock cost per call, in microseconds."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - t0
    return (elapsed / iterations) * 1_000_000


class TestSdkOverhead(unittest.TestCase):
    def setUp(self) -> None:
        self.client = DunetraceClient(api_key="dt_bench")
        self.client._ship = lambda batch: None  # exclude network I/O from the measurement

    def tearDown(self) -> None:
        self.client.shutdown(timeout=1)

    def test_tool_called_overhead(self) -> None:
        with self.client.run("bench-agent") as run:
            us = _time_hook(lambda: run.tool_called("search", {"query": "benchmark input text"}))
        print(f"\n  tool_called:       {us:6.1f}µs")
        self.assertLess(us, MAX_OVERHEAD_US)

    def test_tool_responded_overhead(self) -> None:
        with self.client.run("bench-agent") as run:
            us = _time_hook(
                lambda: run.tool_responded("search", success=True, output_length=256, latency_ms=5)
            )
        print(f"  tool_responded:    {us:6.1f}µs")
        self.assertLess(us, MAX_OVERHEAD_US)

    def test_tool_responded_with_error_overhead(self) -> None:
        """Error path — exercises the raw error-text field (was hash_content() pre-refactor)."""
        with self.client.run("bench-agent") as run:
            us = _time_hook(
                lambda: run.tool_responded(
                    "search", success=False, latency_ms=5, error="ConnectionError: upstream timeout"
                )
            )
        print(f"  tool_responded(err):{us:6.1f}µs")
        self.assertLess(us, MAX_OVERHEAD_US)

    def test_llm_called_overhead(self) -> None:
        with self.client.run("bench-agent") as run:
            us = _time_hook(lambda: run.llm_called("gpt-4o", prompt_tokens=100))
        print(f"  llm_called:        {us:6.1f}µs")
        self.assertLess(us, MAX_OVERHEAD_US)

    def test_llm_responded_overhead(self) -> None:
        """Exercises the raw output-text field (was hash_content() pre-refactor)."""
        output_text = "This is a representative LLM completion of moderate length. " * 3
        with self.client.run("bench-agent") as run:
            us = _time_hook(
                lambda: run.llm_responded(
                    completion_tokens=50,
                    latency_ms=10,
                    finish_reason="stop",
                    output=output_text,
                )
            )
        print(f"  llm_responded:     {us:6.1f}µs")
        self.assertLess(us, MAX_OVERHEAD_US)

    def test_retrieval_called_overhead(self) -> None:
        with self.client.run("bench-agent") as run:
            us = _time_hook(lambda: run.retrieval_called("docs", query="benchmark query text"))
        print(f"  retrieval_called:  {us:6.1f}µs")
        self.assertLess(us, MAX_OVERHEAD_US)

    def test_full_run_overhead(self) -> None:
        """A representative run: llm + tool + retrieval, called then responded."""

        def run_once() -> None:
            with self.client.run("bench-agent", model="gpt-4o", tools=["search"]) as run:
                run.llm_called("gpt-4o", prompt_tokens=100)
                run.llm_responded(
                    completion_tokens=50, latency_ms=10, finish_reason="tool_calls", output="answer"
                )
                run.tool_called("search", {"query": "benchmark"})
                run.tool_responded("search", success=True, output_length=256, latency_ms=5)
                run.retrieval_called("docs", query="benchmark")
                run.retrieval_responded("docs", result_count=3, top_score=0.9, latency_ms=8)
                run.final_answer()

        n_events = 8  # run.started + 6 hooks + run.completed
        us_per_run = _time_hook(run_once, iterations=1000)
        us_per_event = us_per_run / n_events
        print(
            f"\n  Full run ({n_events} events): {us_per_run:6.1f}µs/run  ({us_per_event:5.1f}µs/event)"
        )
        self.assertLess(us_per_event, MAX_OVERHEAD_US)


class TestSignalPolicyOverhead(unittest.TestCase):
    """Benchmark the detector path enabled by a signal-trigger policy."""

    def setUp(self) -> None:
        self.client = DunetraceClient(api_key="dt_bench")
        self.client._ship = lambda batch: None
        self.client.add_policy(
            name="bench-scattershot",
            condition={
                "trigger": "signal",
                "operator": "contains",
                "value": "SCATTERSHOT_TOOL_USE",
            },
            action={"type": "log"},
        )

    def tearDown(self) -> None:
        self.client.shutdown(timeout=1)

    def _tool_called_us(self, names: tuple[str, ...]) -> float:
        """Cost of ONE tool_called hook on a run already holding `names`.

        Times the hook directly, like every other benchmark in this file. The
        previous version timed a whole run and divided by len(names), which
        amortises the fixed run-start/run-completion cost across the tool calls
        — so the larger the case, the more per-hook regression it could hide
        (at 32 names, ~16ms of run-level overhead still passed a 500µs
        assertion). It also meant the assertion got weaker precisely as the
        case got more demanding.
        """
        with self.client.run("bench-agent") as run:
            for name in names:
                run.tool_called(name, {"query": "benchmark"})
            return _time_hook(
                lambda: run.tool_called("probe", {"query": "benchmark"}),
                iterations=200,
                warmup=20,
            )

    def test_scattershot_signal_policy_overhead(self) -> None:
        cases = {
            "six-distinct-at-threshold": ("a", "b", "c", "d", "e", "f", "a", "b"),
            "one-tool-repeated": ("search",) * 32,
            # The worst case for SCATTERSHOT_TOOL_USE specifically: many distinct
            # names AND a long tool_calls list, which is what its per-call set
            # build scales with. Neither case above exercises it — one has 8
            # calls, the other a single distinct name — so the O(n) scan that
            # blew MAX_COST_NS in review went unmeasured.
            "wide-and-long": tuple(f"tool_{i % 40}" for i in range(512)),
        }
        for label, names in cases.items():
            with self.subTest(label=label):
                us = self._tool_called_us(names)
                print(f"\n  scattershot signal ({label}): {us:6.1f}µs/hook")
                self.assertLess(us, MAX_OVERHEAD_US)


class TestRunStartOverheadWithInput(unittest.TestCase):
    """dt.run()'s own enter cost, with user_input supplied.

    Every other benchmark in this file opens the run with no ``user_input``,
    which skips the in-path prompt-injection scan entirely — so the hook budget
    above was being enforced by measurements that never touched the one code
    path whose cost scales with caller data. It did: the scan was unbounded, and
    a 1 MB input cost ~1.5s of synchronous time before the agent ran. These
    tests cover that path directly.
    """

    def setUp(self) -> None:
        self.client = DunetraceClient(api_key="dt_bench")
        self.client._ship = lambda batch: None

    def tearDown(self) -> None:
        self.client.shutdown(timeout=1)

    def _run_enter_us(self, text: str) -> float:
        def once() -> None:
            with self.client.run("bench-agent", user_input=text) as run:
                run.final_answer()

        return _time_hook(once, iterations=500)

    def test_run_start_with_small_input(self) -> None:
        """Typical inputs are far below the scan window, so the per-hook budget
        still applies to them."""
        us = self._run_enter_us("what is the weather in London?")
        print(f"\n  run() enter, 30B input:      {us:7.1f}µs")
        self.assertLess(us, MAX_OVERHEAD_US)

    def test_run_start_cost_is_bounded_for_large_input(self) -> None:
        """The invariant the scan window exists to guarantee: past the window,
        cost stops tracking input size.

        Deliberately *not* asserted against MAX_OVERHEAD_US. That budget is a
        per-hook figure, and the run-start scan is sized against the fraction of
        a real agent step it consumes (an LLM call is 500ms+), not a fixed
        microsecond ceiling. What must hold is the scaling property: an input
        well past the window costs about the same as one at the window. Before
        the window, 1 MB cost ~1000x a 32K input.
        """
        detector = PROMPT_INJECTION_DETECTOR
        window = detector.SCAN_HEAD_CHARS + detector.SCAN_TAIL_CHARS

        at_window = self._run_enter_us("lorem ipsum dolor sit amet " * (window // 27))
        way_over = self._run_enter_us("lorem ipsum dolor sit amet " * (window // 27 * 30))
        print(
            f"  run() enter, {window // 1024}K input:      {at_window:7.1f}µs\n"
            f"  run() enter, {window * 30 // 1024}K input:    {way_over:7.1f}µs"
        )
        self.assertLess(
            way_over,
            at_window * 3,
            "dt.run() enter cost is still scaling with user_input size — the scan "
            "window (PromptInjectionDetector.SCAN_HEAD_CHARS/SCAN_TAIL_CHARS) is "
            "not bounding the scan.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
