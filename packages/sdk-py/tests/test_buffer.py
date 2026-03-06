"""Tests for the RingBuffer."""
import threading
import unittest

from dunetrace.buffer import RingBuffer


class TestRingBuffer(unittest.TestCase):

    def test_push_and_drain(self):
        buf = RingBuffer(maxsize=10)
        buf.push("a")
        buf.push("b")
        buf.push("c")
        result = buf.drain(10)
        self.assertEqual(result, ["a", "b", "c"])
        self.assertEqual(len(buf), 0)

    def test_drain_respects_n(self):
        buf = RingBuffer(maxsize=10)
        for i in range(5):
            buf.push(i)
        result = buf.drain(3)
        self.assertEqual(result, [0, 1, 2])
        self.assertEqual(len(buf), 2)

    def test_ring_drops_oldest_when_full(self):
        buf = RingBuffer(maxsize=3)
        buf.push(1)
        buf.push(2)
        buf.push(3)
        buf.push(4)  # drops 1
        result = buf.drain_all()
        self.assertEqual(result, [2, 3, 4])

    def test_drain_all_empties_buffer(self):
        buf = RingBuffer(maxsize=100)
        for i in range(50):
            buf.push(i)
        result = buf.drain_all()
        self.assertEqual(len(result), 50)
        self.assertEqual(len(buf), 0)

    def test_bool_and_len(self):
        buf = RingBuffer(maxsize=5)
        self.assertFalse(buf)
        buf.push("x")
        self.assertTrue(buf)
        self.assertEqual(len(buf), 1)

    def test_concurrent_push_drain(self):
        """Multiple threads pushing should not crash or lose the buffer invariant."""
        buf: RingBuffer[int] = RingBuffer(maxsize=1000)
        errors: list = []

        def pusher(start: int) -> None:
            try:
                for i in range(start, start + 100):
                    buf.push(i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=pusher, args=(i * 100,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertLessEqual(len(buf), 1000)


if __name__ == "__main__":
    unittest.main()
